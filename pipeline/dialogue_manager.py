from __future__ import annotations

from cProfile import label
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

id2intent = {
    "9": "RECOMMEND_FOOD",
    "7": "NO_CLEAR_INTENT",
    "10": "RECOMMEND_PLACE_NEARBY",
    "1": "ASK_FOOD_TYPE",
    "2": "ASK_LOCATION",
    "4": "ASK_PRICE",
    "3": "ASK_OPEN_TIME",
    "5": "ASK_REVIEW",
    "0": "ASK_DIRECTION",
    "6": "COMPARE_PLACES",
    "11": "SMALL_TALK",
    "8": "OUT_OF_SCOPE"
}
class IntentPredictor(Protocol):
    def predict(self, text: str) -> Dict[str, Any]:
        ...


class SlotPredictor(Protocol):
    def extract_slots(self, text: str) -> List[Dict[str, Any]]:
        ...


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

    def to_dict(self) -> Dict[str, Any]:
        return {
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
        }


class IntentClassifierHF:
    """Adapter cho intent model (SequenceClassification)."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)

        pred_id = int(torch.argmax(probs).item())
        return {
            "intent": str(self.id2label[pred_id]),
            "confidence": float(probs[pred_id].item()),
        }


class SlotExtractorHF:
    """Adapter cho slot model (TokenClassification)."""

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

        return self._decode_entities(
            text=text,
            labels=pred_labels,
            offsets=offsets,
            token_confs=token_confs,
        )

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
            # Special tokens
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


class DialogueOrchestrator:
    """Ghép Intent + Slot + DST + Policy thành pipeline một lượt hội thoại."""

    FOLLOWUP_INTENTS = {"ASK_LOCATION", "ASK_PRICE", "ASK_OPEN_TIME", "NO_CLEAR_INTENT"}

    def __init__(
        self,
        intent_model: IntentPredictor,
        slot_model: SlotPredictor,
        dst: Optional[DialogueStateTracker] = None,
        policy: Optional[RuleBasedPolicy] = None,
        intent_conf_threshold: float = 0.5,
    ):
        self.intent_model = intent_model
        self.slot_model = slot_model
        self.dst = dst or DialogueStateTracker()
        self.policy = policy or RuleBasedPolicy()
        self.intent_conf_threshold = intent_conf_threshold

    def create_session(self, user_id: Optional[str] = None) -> str:
        return self.dst.create_session(user_id=user_id)
    def convert_label_to_intent(self, label: str, mapping: dict) -> str:
        label_id = label.split("_")[-1]
        return mapping.get(label_id, "UNKNOWN")
    def process_user_message(self, session_id: str, user_text: str) -> Dict[str, Any]:
        state_before = self.dst.get_state(session_id)
        if state_before is None:
            raise ValueError(f"Session {session_id} not found. Call create_session() first.")

        intent_pred = self.intent_model.predict(user_text)
        print(f"DEBUG: Intent prediction: {intent_pred}")
        raw_label = intent_pred.get("intent", "NO_CLEAR_INTENT")
        intent = self.convert_label_to_intent(raw_label, id2intent)
        raw_intent = self._normalize_intent_label(intent)
        print(f"DEBUG: Raw intent: {raw_intent}")
        confidence = float(intent_pred.get("confidence", 0.0))
        print(f"DEBUG: Intent confidence: {confidence}")

        slots = self._normalize_slots(self.slot_model.extract_slots(user_text))
        print(f"DEBUG: Extracted slots: {slots}")
        resolved_intent = self._resolve_intent(raw_intent, confidence, state_before, slots)
        print(f"DEBUG: Resolved intent: {resolved_intent}")
        state_after = self.dst.update_state(
            session_id=session_id,
            user_utterance=user_text,
            intent=resolved_intent,
            intent_confidence=confidence,
            slots=slots,
        )

        action = self.policy.decide_action(state_after)
        state_after.turns[-1].bot_action = action.type
        state_after.turns[-1].bot_response = action.template

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
        )
        return result.to_dict()

    def _normalize_intent_label(self, label: str) -> str:
        normalized = str(label).strip().upper().replace(" ", "_")
        alias_map = {
            "RECOMMEND_RESTAURANT": "RECOMMEND_PLACE_NEARBY",
            "RECOMMEND_PLACE": "RECOMMEND_PLACE_NEARBY",
            "OTHER": "NO_CLEAR_INTENT",
        }
        normalized = alias_map.get(normalized, normalized)
        return normalized if normalized in IntentType.__members__ else "NO_CLEAR_INTENT"

    def _normalize_slots(self, slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_slots: List[Dict[str, Any]] = []

        for slot in slots:
            slot_type = str(slot.get("type", "")).strip().upper()
            value = str(slot.get("value", "")).strip()
            if not slot_type or not value:
                continue

            confidence = float(slot.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            normalized_slots.append(
                {
                    "type": slot_type,
                    "value": value,
                    "confidence": confidence,
                }
            )

        return normalized_slots

    def _resolve_intent(
        self,
        predicted_intent: str,
        confidence: float,
        state_before: DialogueState,
        slots: List[Dict[str, Any]],
    ) -> str:
        # Intent confidence quá thấp -> giữ intent task trước đó nếu có
        if confidence < self.intent_conf_threshold and state_before.current_intent:
            return state_before.current_intent.name

        # Follow-up turn (ví dụ user chỉ trả lời "quận 1") -> giữ task chính trước đó
        if (
            state_before.current_intent in {IntentType.RECOMMEND_PLACE_NEARBY, IntentType.RECOMMEND_FOOD}
            and predicted_intent in self.FOLLOWUP_INTENTS
            and len(slots) > 0
        ):
            return state_before.current_intent.name

        return predicted_intent