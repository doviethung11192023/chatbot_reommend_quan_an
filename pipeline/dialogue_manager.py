
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from dialogue_policy.rule_based_policy import RuleBasedPolicy
from dialogue_state_tracking.dst import DialogueStateTracker
from dialogue_state_tracking.state_schema import DialogueState, IntentType


LABEL_ID_TO_INTENT = {
    0: "ASK_DIRECTION",
    1: "ASK_FOOD_TYPE",
    2: "ASK_LOCATION",
    3: "ASK_OPEN_TIME",
    4: "ASK_PRICE",
    5: "ASK_REVIEW",
    6: "COMPARE_PLACES",
    7: "NO_CLEAR_INTENT",
    8: "OUT_OF_SCOPE",
    9: "RECOMMEND_FOOD",
    10: "RECOMMEND_PLACE_NEARBY",
    11: "SMALL_TALK",
}


class IntentPredictor(Protocol):
    def predict(self, text: str) -> Dict[str, Any]:
        ...


class SlotPredictor(Protocol):
    def extract_slots(self, text: str) -> List[Dict[str, Any]]:
        ...


# @dataclass
# class OrchestratorResult:
#     session_id: str
#     user_text: str
#     intent_raw: str
#     intent_resolved: str
#     intent_confidence: float
#     slots: List[Dict[str, Any]]
#     action_type: str
#     action_slot: Optional[str]
#     action_template: str
#     state_summary: Dict[str, Any]
#     policy_log: Optional[Dict[str, Any]] = None

#     def to_dict(self) -> Dict[str, Any]:
#         out = {
#             "session_id": self.session_id,
#             "user_text": self.user_text,
#             "intent": {
#                 "raw": self.intent_raw,
#                 "resolved": self.intent_resolved,
#                 "confidence": self.intent_confidence,
#             },
#             "slots": self.slots,
#             "action": {
#                 "type": self.action_type,
#                 "slot": self.action_slot,
#                 "template": self.action_template,
#             },
#             "state": self.state_summary,
#         }
#         if self.policy_log is not None:
#             out["policy"] = self.policy_log
#         return out

@dataclass
class OrchestratorResult:
    session_id: str
    user_text: str
    intent_raw: str
    intent_resolved: str
    intent_confidence: float
    slots: List[Dict[str, Any]]
    action_type: str
    action_slot: Optional[str]
    action_template: str
    state_summary: Dict[str, Any]
    state_quality: Optional[float] = None
    slot_conflicts: int = 0
    policy_log: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "session_id": self.session_id,
            "user_text": self.user_text,
            "intent": {
                "raw": self.intent_raw,
                "resolved": self.intent_resolved,
                "confidence": self.intent_confidence,
            },
            "slots": self.slots,
            "action": {
                "type": self.action_type,
                "slot": self.action_slot,
                "template": self.action_template,
            },
            "state": self.state_summary,
            "state_quality": self.state_quality,
            "slot_conflicts": self.slot_conflicts,
        }
        if self.policy_log is not None:
            out["policy"] = self.policy_log
        return out
class IntentClassifierHF:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)

        pred_id = int(torch.argmax(probs).item())
        return {"intent": str(self.id2label[pred_id]), "confidence": float(probs[pred_id].item())}


class SlotExtractorHF:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def extract_slots(self, text: str) -> List[Dict[str, Any]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0].detach().cpu()
            probs = torch.softmax(logits, dim=-1)

        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_labels = [str(self.id2label[i]) for i in pred_ids]
        token_confs = [float(probs[i, pred_ids[i]].item()) for i in range(len(pred_ids))]

        return self._decode_entities(text, pred_labels, offsets, token_confs)

    @staticmethod
    def _decode_entities(
        text: str,
        labels: List[str],
        offsets: List[List[int]],
        token_confs: List[float],
    ) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        cur_type: Optional[str] = None
        cur_start = -1
        cur_end = -1
        cur_confs: List[float] = []

        def flush():
            nonlocal cur_type, cur_start, cur_end, cur_confs
            if cur_type is not None and cur_start >= 0 and cur_end > cur_start:
                value = text[cur_start:cur_end].strip()
                if value:
                    entities.append(
                        {
                            "type": cur_type,
                            "value": value,
                            "confidence": float(sum(cur_confs) / len(cur_confs)),
                            "start": cur_start,
                            "end": cur_end,
                        }
                    )
            cur_type = None
            cur_start = -1
            cur_end = -1
            cur_confs = []

        for label, (start, end), conf in zip(labels, offsets, token_confs):
            if start == end:
                continue
            if label.startswith("B-"):
                flush()
                cur_type = label[2:]
                cur_start = start
                cur_end = end
                cur_confs = [conf]
                continue
            if label.startswith("I-"):
                i_type = label[2:]
                if cur_type == i_type:
                    cur_end = end
                    cur_confs.append(conf)
                else:
                    flush()
                    cur_type = i_type
                    cur_start = start
                    cur_end = end
                    cur_confs = [conf]
                continue
            flush()

        flush()
        return entities

logger = logging.getLogger(__name__)
class DialogueOrchestrator:
    FOLLOWUP_INTENTS = {"ASK_LOCATION", "ASK_PRICE", "ASK_OPEN_TIME", "NO_CLEAR_INTENT"}

    def __init__(
        self,
        intent_model: IntentPredictor,
        slot_model: SlotPredictor,
        dst: Optional[DialogueStateTracker] = None,
        policy: Optional[Any] = None,
        intent_conf_threshold: float = 0.5,
        debug: bool = False,    
    ):
        self.intent_model = intent_model
        self.slot_model = slot_model
        self.dst = dst or DialogueStateTracker()
        self.policy = policy or RuleBasedPolicy()
        self.intent_conf_threshold = intent_conf_threshold
        self.debug = debug  # <-- added

    def _dbg(self, msg: str, *args: Any) -> None:
        if self.debug:
            logger.info("[DialogueOrchestrator] " + msg, *args)
    # def create_session(self, user_id: Optional[str] = None) -> str:

    #     return self.dst.create_session(user_id=user_id)
    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = self.dst.create_session(user_id=user_id)
        self._dbg("create_session user_id=%s -> session_id=%s", user_id, session_id)
        return session_id
    # def process_user_message(self, session_id: str, user_text: str) -> Dict[str, Any]:
    #     state_before = self.dst.get_state(session_id)
    #     if state_before is None:
    #         raise ValueError(f"Session {session_id} not found. Call create_session() first.")

    #     intent_pred = self.intent_model.predict(user_text)
    #     raw_label = intent_pred.get("intent", "NO_CLEAR_INTENT")
    #     raw_intent = self._normalize_intent_label(raw_label)
    #     confidence = float(intent_pred.get("confidence", 0.0))

    #     slots = self._normalize_slots(self.slot_model.extract_slots(user_text))
    #     resolved_intent = self._resolve_intent(raw_intent, confidence, state_before, slots)

    #     state_after = self.dst.update_state(
    #         session_id=session_id,
    #         user_utterance=user_text,
    #         intent=resolved_intent,
    #         intent_confidence=confidence,
    #         slots=slots,
    #     )

    #     action = self.policy.decide_action(state_after)
    #     state_after.turns[-1].bot_action = action.type
    #     state_after.turns[-1].bot_response = action.template

    #     policy_log = None
    #     if hasattr(self.policy, "get_last_decision_log"):
    #         policy_log = self.policy.get_last_decision_log()

    #     result = OrchestratorResult(
    #         session_id=session_id,
    #         user_text=user_text,
    #         intent_raw=raw_intent,
    #         intent_resolved=resolved_intent,
    #         intent_confidence=confidence,
    #         slots=slots,
    #         action_type=action.type,
    #         action_slot=action.slot,
    #         action_template=action.template or "",
    #         state_summary=state_after.get_context_summary(),
    #         policy_log=policy_log,
    #     )
    #     return result.to_dict()
    # def process_user_message(self, session_id: str, user_text: str) -> Dict[str, Any]:
    #     self._dbg("process_user_message(session_id=%s, user_text=%r)", session_id, user_text)

    #     state_before = self.dst.get_state(session_id)
    #     if state_before is None:
    #         raise ValueError(f"Session {session_id} not found. Call create_session() first.")

    #     self._dbg("state_before=%s", state_before.get_context_summary())

    #     # 1) Intent prediction
    #     intent_pred = self.intent_model.predict(user_text)
    #     raw_label = intent_pred.get("intent", "NO_CLEAR_INTENT")
    #     raw_intent = self._normalize_intent_label(raw_label)
    #     confidence = float(intent_pred.get("confidence", 0.0))
    #     self._dbg(
    #         "intent_pred raw_label=%s normalized=%s confidence=%.4f",
    #         raw_label,
    #         raw_intent,
    #         confidence,
    #     )

    #     # 2) Slot extraction
    #     raw_slots = self.slot_model.extract_slots(user_text)
    #     slots = self._normalize_slots(raw_slots)
    #     self._dbg("slots raw=%s normalized=%s", raw_slots, slots)

    #     # 3) Resolve intent with dialogue context
    #     resolved_intent = self._resolve_intent(raw_intent, confidence, state_before, slots)
    #     self._dbg("resolved_intent=%s", resolved_intent)

    #     # 4) Update DST
    #     state_after = self.dst.update_state(
    #         session_id=session_id,
    #         user_utterance=user_text,
    #         intent=resolved_intent,
    #         intent_confidence=confidence,
    #         slots=slots,
    #     )
    #     self._dbg("state_after=%s", state_after.get_context_summary())

    #     # 5) Policy decision
    #     action = self.policy.decide_action(state_after)
    #     state_after.turns[-1].bot_action = action.type
    #     state_after.turns[-1].bot_response = action.template
    #     self._dbg(
    #         "policy_action type=%s slot=%s template=%r",
    #         action.type,
    #         action.slot,
    #         action.template,
    #     )

    #     policy_log = None
    #     if hasattr(self.policy, "get_last_decision_log"):
    #         policy_log = self.policy.get_last_decision_log()
    #         self._dbg("policy_log=%s", policy_log)

    #     result = OrchestratorResult(
    #         session_id=session_id,
    #         user_text=user_text,
    #         intent_raw=raw_intent,
    #         intent_resolved=resolved_intent,
    #         intent_confidence=confidence,
    #         slots=slots,
    #         action_type=action.type,
    #         action_slot=action.slot,
    #         action_template=action.template or "",
    #         state_summary=state_after.get_context_summary(),
    #         policy_log=policy_log,
    #     )
    #     out = result.to_dict()
    #     self._dbg("result=%s", out)
    #     return out

    def process_user_message(self, session_id: str, user_text: str) -> Dict[str, Any]:
        self._dbg("process_user_message(session_id=%s, user_text=%r)", session_id, user_text)

        state_before = self.dst.get_state(session_id)
        if state_before is None:
            raise ValueError(f"Session {session_id} not found. Call create_session() first.")

        self._dbg("state_before=%s", state_before.get_context_summary())

        # 1) Intent prediction
        intent_pred = self.intent_model.predict(user_text)
        raw_label = intent_pred.get("intent", "NO_CLEAR_INTENT")
        raw_intent = self._normalize_intent_label(raw_label)
        confidence = float(intent_pred.get("confidence", 0.0))
        self._dbg(
            "intent_pred raw_label=%s normalized=%s confidence=%.4f",
            raw_label,
            raw_intent,
            confidence,
        )

        # 2) Slot extraction
        raw_slots = self.slot_model.extract_slots(user_text)
        slots = self._normalize_slots(raw_slots)
        self._dbg("slots raw=%s normalized=%s", raw_slots, slots)

        # 3) Resolve intent with dialogue context
        resolved_intent = self._resolve_intent(raw_intent, confidence, state_before, slots)
        self._dbg("resolved_intent=%s", resolved_intent)

        # 4) Update DST
        state_after = self.dst.update_state(
            session_id=session_id,
            user_utterance=user_text,
            intent=resolved_intent,
            intent_confidence=confidence,
            slots=slots,
        )
        self._dbg("state_after=%s", state_after.get_context_summary())

        # 5) Policy decision
        action = self.policy.decide_action(state_after)
        state_after.turns[-1].bot_action = action.type
        state_after.turns[-1].bot_response = action.template
        self._dbg(
            "policy_action type=%s slot=%s template=%r",
            action.type,
            action.slot,
            action.template,
        )

        policy_log = None
        if hasattr(self.policy, "get_last_decision_log"):
            policy_log = self.policy.get_last_decision_log()
            self._dbg("policy_log=%s", policy_log)

        state_quality = None
        if hasattr(state_after, "get_state_quality"):
            try:
                state_quality = float(state_after.get_state_quality())
            except Exception:
                state_quality = None
        else:
            state_quality = state_after.context.get("state_quality") if hasattr(state_after, "context") else None

        slot_conflicts = 0
        if hasattr(state_after, "slot_conflicts"):
            try:
                slot_conflicts = len(state_after.slot_conflicts)
            except Exception:
                slot_conflicts = 0
        elif hasattr(state_after, "context") and isinstance(state_after.context, dict):
            slot_conflicts = int(state_after.context.get("slot_conflicts", 0) or 0)

        result = OrchestratorResult(
            session_id=session_id,
            user_text=user_text,
            intent_raw=raw_intent,
            intent_resolved=resolved_intent,
            intent_confidence=confidence,
            slots=slots,
            action_type=action.type,
            action_slot=action.slot,
            action_template=action.template or "",
            state_summary=state_after.get_context_summary(),
            state_quality=state_quality,
            slot_conflicts=slot_conflicts,
            policy_log=policy_log,
        )
        out = result.to_dict()
        self._dbg("result=%s", out)
        return out
    def _normalize_intent_label(self, label: str) -> str:
        raw = str(label).strip().upper().replace(" ", "_")

        # LABEL_10 -> RECOMMEND_PLACE_NEARBY
        m = re.search(r"(\d+)$", raw)
        if raw.startswith("LABEL_") and m:
            mapped = LABEL_ID_TO_INTENT.get(int(m.group(1)))
            if mapped:
                raw = mapped

        alias_map = {
            "RECOMMEND_RESTAURANT": "RECOMMEND_PLACE_NEARBY",
            "RECOMMEND_PLACE": "RECOMMEND_PLACE_NEARBY",
            "OTHER": "NO_CLEAR_INTENT",
        }
        normalized = alias_map.get(raw, raw)
        return normalized if normalized in IntentType.__members__ else "NO_CLEAR_INTENT"

    def _normalize_slots(self, slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for slot in slots:
            slot_type = str(slot.get("type", "")).strip().upper()
            value = str(slot.get("value", "")).strip()
            if not slot_type or not value:
                continue
            conf = max(0.0, min(1.0, float(slot.get("confidence", 0.0))))
            out.append({"type": slot_type, "value": value, "confidence": conf})
        return out

    def _resolve_intent(
        self,
        predicted_intent: str,
        confidence: float,
        state_before: DialogueState,
        slots: List[Dict[str, Any]],
    ) -> str:
        if confidence < self.intent_conf_threshold and state_before.current_intent:
            return state_before.current_intent.name

        if (
            state_before.current_intent in {IntentType.RECOMMEND_PLACE_NEARBY, IntentType.RECOMMEND_FOOD}
            and predicted_intent in self.FOLLOWUP_INTENTS
            and len(slots) > 0
        ):
            return state_before.current_intent.name

        return predicted_intent