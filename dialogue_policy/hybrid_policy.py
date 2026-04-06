

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from dialogue_policy.rule_based_policy import Action, RuleBasedPolicy
from dialogue_state_tracking.state_schema import DialogueState, IntentType

logger = logging.getLogger(__name__)

ACTION_SPACE = {"ASK_SLOT", "CLARIFY", "CONFIRM", "RECOMMEND", "RESPOND", "FALLBACK"}
ACTION_ALIASES = {
    "ASK_LOCATION": "ASK_SLOT",
    "ASK_PRICE": "ASK_SLOT",
    "ASK_OPEN_TIME": "ASK_SLOT",
    "ASK_FOOD_TYPE": "ASK_SLOT",
    "SMALL_TALK": "RESPOND",
    "OUT_OF_SCOPE": "FALLBACK",
}

DEFAULT_TEMPLATES: Dict[str, List[str]] = {
    "ASK_SLOT": [
        "Bạn cho mình thêm thông tin được không?",
        "Bạn bổ sung giúp mình khu vực hoặc mức giá nhé.",
        "Mình cần thêm một chút thông tin để gợi ý chính xác hơn.",
    ],
    "CLARIFY": [
        "Bạn có thể nói rõ hơn giúp mình không?",
        "Ý bạn là gần khu vực nào vậy?",
    ],
    "RECOMMEND": [
        "Để mình tìm quán phù hợp cho bạn nhé! 🔍",
        "Ok, mình sẽ gợi ý vài quán hợp với nhu cầu của bạn.",
    ],
    "CONFIRM": [
        "Mình hiểu đúng ý bạn rồi chứ? Nếu đúng, mình gợi ý ngay nhé.",
        "Mình gần đủ thông tin rồi, bạn xác nhận giúp mình một chút nhé.",
    ],
    "RESPOND": [
        "Chào bạn! Mình có thể giúp bạn tìm quán ăn ngon nè 😊",
        "Mình đây, bạn muốn tìm món gì hôm nay?",
    ],
    "FALLBACK": [
        "Xin lỗi, mình chưa hiểu rõ. Bạn có thể nói lại giúp mình không?",
        "Mình chưa bắt đúng ý, bạn diễn đạt lại ngắn hơn giúp mình nhé.",
    ],
}


class MLPolicyProtocol(Protocol):
    def predict_action(self, state: DialogueState) -> Dict[str, Any]:
        ...


class LLMPolicyProtocol(Protocol):
    def decide_action(self, state_summary: Dict[str, Any], action_space: List[str]) -> str:
        ...

    def decide_decision(self, state_summary: Dict[str, Any], action_space: List[str]) -> Dict[str, Any]:
        ...


@dataclass
class PolicyDecisionLog:
    source: str  # RULE | ML | LLM | FALLBACK
    action: str
    confidence: float = 0.0
    note: str = ""


class HybridPolicy:
    """Option 3: Rule safety layer + LLM decision maker + DST memory."""

    def __init__(
        self,
        rule_policy: Optional[RuleBasedPolicy] = None,
        ml_policy: Optional[MLPolicyProtocol] = None,
        llm_policy: Optional[LLMPolicyProtocol] = None,
        ml_conf_threshold: float = 0.7,
        confirm_threshold: float = 0.72,
        templates: Optional[Dict[str, List[str]]] = None,
        rng: Optional[random.Random] = None,
        debug: bool = False,
        repeat_window: int = 3,
        allow_ml_llm_escape: bool = True,
        state_quality_threshold: float = 0.65,
    ):
        self.rule_policy = rule_policy or RuleBasedPolicy()
        self.ml_policy = ml_policy
        self.llm_policy = llm_policy
        self.ml_conf_threshold = ml_conf_threshold
        self.confirm_threshold = confirm_threshold
        self.templates = templates or DEFAULT_TEMPLATES
        self.rng = rng or random.Random(42)
        self.debug = debug
        self.repeat_window = repeat_window
        self.allow_ml_llm_escape = allow_ml_llm_escape
        self.state_quality_threshold = state_quality_threshold
        self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")

    def _dbg(self, msg: str, *args: Any) -> None:
        if self.debug:
            logger.info("[HybridPolicy] " + msg, *args)


    def decide_action(self, state: DialogueState) -> Action:
        dialogue_act = str(getattr(state, "context", {}).get("dialogue_act", "") or "").upper()
        if dialogue_act in {"CANCEL", "GOODBYE"}:
            self._last_log = PolicyDecisionLog(source="RULE", action="RESPOND", confidence=1.0, note=f"dialogue_act={dialogue_act}")
            return self._build_action("RESPOND", state)

        if bool(getattr(state, "context", {}).get("block_recommend", False)):
            self._dbg("block_recommend=True -> suppress direct recommend")
        quality = getattr(state, "get_state_quality", lambda: 1.0)()
        self._dbg(
            "decide_action state_quality=%.4f complete=%s missing=%s intent=%s",
            quality,
            state.is_complete(),
            state.get_missing_slots(),
            state.current_intent.value if state.current_intent else None,
        )

        # Rule layer is safety-only in Option 3.
        safety_rule = self._select_safety_rule(state)
        if safety_rule is not None:
            safety_action = self._build_action_from_rule(safety_rule, state)
            self._last_log = PolicyDecisionLog(source="RULE", action=self._normalize_action(safety_action.type), confidence=1.0, note="safety_rule")
            return self._apply_state_quality_guard(state, safety_action)

        llm_action = self._decide_with_llm(state)
        if llm_action is not None:
            return self._apply_state_quality_guard(state, llm_action)

        # Fallback deterministic policy if no LLM available.
        if state.get_missing_slots():
            self._last_log = PolicyDecisionLog(source="FALLBACK", action="ASK_SLOT", confidence=0.0, note="llm_unavailable")
            return self._build_action("ASK_SLOT", state)
        self._last_log = PolicyDecisionLog(source="FALLBACK", action="RECOMMEND", confidence=0.0, note="llm_unavailable")
        return self._build_action("RECOMMEND", state)

        self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")
        return self._build_action("FALLBACK", state)
    
    def _apply_state_quality_guard(self, state: DialogueState, action: Action) -> Action:
        quality = getattr(state, "get_state_quality", lambda: 1.0)()
        block_recommend = bool(getattr(state, "context", {}).get("block_recommend", False))

        if action.type == "RECOMMEND" and state.get_missing_slots():
            self._dbg("downgrade RECOMMEND -> ASK_SLOT due missing_slots=%s", state.get_missing_slots())
            return self._build_action("ASK_SLOT", state)

        if block_recommend and action.type == "RECOMMEND":
            self._dbg("downgrade RECOMMEND -> CLARIFY due block_recommend flag")
            return self._build_action("CLARIFY", state)

        if quality < self.state_quality_threshold and action.type == "RECOMMEND":
            self._dbg("downgrade RECOMMEND -> CLARIFY due low state quality=%.4f", quality)
            return self._build_action("CLARIFY", state)

        if quality < self.state_quality_threshold and action.type == "ASK_SLOT" and not state.get_missing_slots():
            self._dbg("downgrade ASK_SLOT -> CLARIFY due low state quality=%.4f", quality)
            return self._build_action("CLARIFY", state)

        return action

    def _select_best_rule(self, state: DialogueState) -> tuple[Optional[Dict[str, Any]], int]:
        matched: List[Dict[str, Any]] = []
        for idx, rule in enumerate(getattr(self.rule_policy, "rules", [])):
            cond = rule.get("condition", {})
            ok = self.rule_policy._evaluate_condition(cond, state)
            self._dbg("rule[%d] matched=%s condition=%s", idx, ok, cond)
            if ok:
                matched.append(rule)
        if not matched:
            return None, 0
        matched.sort(key=lambda r: r.get("priority", 0), reverse=True)
        selected = matched[0]
        return selected, int(selected.get("priority", 0) or 0)

    def _select_safety_rule(self, state: DialogueState) -> Optional[Dict[str, Any]]:
        matched: List[Dict[str, Any]] = []
        for rule in getattr(self.rule_policy, "rules", []):
            cond = rule.get("condition", {})
            try:
                ok = self.rule_policy._evaluate_condition(cond, state)
            except Exception:
                ok = False
            if not ok:
                continue

            action_type = self._normalize_action(str(rule.get("action", {}).get("type", "FALLBACK")).upper())
            intent_cond = str(cond.get("intent", "")).upper()
            is_safety_intent = intent_cond in {"SMALL_TALK", "OUT_OF_SCOPE", "NO_CLEAR_INTENT"}
            is_safety_action = action_type in {"RESPOND", "FALLBACK"}
            if is_safety_intent or is_safety_action:
                matched.append(rule)

        if not matched:
            return None

        matched.sort(key=lambda r: r.get("priority", 0), reverse=True)
        return matched[0]

    def _build_action_from_rule(self, rule: Dict[str, Any], state: DialogueState) -> Action:
        action_cfg = rule.get("action", {})
        action_type = self._normalize_action(str(action_cfg.get("type", "FALLBACK")).upper())
        slot = action_cfg.get("slot_to_ask") or action_cfg.get("slot_to_clarify")
        if action_type in {"ASK_SLOT", "CLARIFY"} and not slot:
            slot = self._choose_target_slot(state)

        templates = action_cfg.get("templates") or self._get_templates_for_action(action_type, slot)
        template = self._pick_non_repeated_template(templates, state)
        return Action(action_type=action_type, slot=slot, template=template)

    def _decide_with_llm(self, state: DialogueState) -> Optional[Action]:
        if self.llm_policy is None:
            return None

        payload = self._build_llm_payload(state)
        self._dbg("llm_payload=%s", payload)

        try:
            if hasattr(self.llm_policy, "decide_decision"):
                decision = self.llm_policy.decide_decision(payload, sorted(ACTION_SPACE)) or {}
            else:
                raw_action = self.llm_policy.decide_action(payload, sorted(ACTION_SPACE))
                decision = {"action": raw_action}

            action_type = self._normalize_action(str(decision.get("action", "FALLBACK")).upper())
            if action_type not in ACTION_SPACE:
                action_type = "FALLBACK"

            slot = decision.get("slot")
            if slot is not None:
                slot = str(slot).upper().strip()
            if action_type in {"ASK_SLOT", "CLARIFY"} and not slot:
                slot = self._choose_target_slot(state)

            response = str(decision.get("response", "") or "").strip()
            next_action = str(decision.get("next_action", "") or "").upper().strip() or None
            reason = str(decision.get("reason", "") or "")

            if action_type == "RECOMMEND" and state.get_missing_slots():
                action_type = "ASK_SLOT"
                slot = self._choose_target_slot(state)

            if action_type == "CONFIRM" and not next_action and state.is_complete():
                next_action = "RECOMMEND"

            action = self._build_action(action_type, state, planned_next_action=next_action)
            if slot and action.type in {"ASK_SLOT", "CLARIFY"}:
                action.slot = slot
            if response:
                action.template = response

            self._last_log = PolicyDecisionLog(source="LLM", action=action.type, confidence=0.0, note=reason)
            return action
        except Exception as ex:
            self._dbg("LLM error=%s", ex)
            self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK", note=f"LLM error: {ex}")
        return None

    def _build_llm_payload(self, state: DialogueState) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        for t in getattr(state, "turns", [])[-4:]:
            history.append(
                {
                    "user": t.user_utterance,
                    "bot_action": t.bot_action,
                    "bot_response": t.bot_response,
                }
            )
        return {
            "intent": state.current_intent.value if state.current_intent else None,
            "filled_slots": {k: v.value for k, v in state.filled_slots.items()},
            "missing_slots": state.get_missing_slots(),
            "is_complete": state.is_complete(),
            "state_quality": getattr(state, "get_state_quality", lambda: 1.0)(),
            "turn_count": len(getattr(state, "turns", [])),
            "dialogue_act": str(getattr(state, "context", {}).get("dialogue_act", "") or ""),
            "block_recommend": bool(getattr(state, "context", {}).get("block_recommend", False)),
            "policy_plan": getattr(state, "context", {}).get("policy_plan", {}),
            "history": history,
        }

    def _build_action(
        self,
        action_type: str,
        state: DialogueState,
        planned_next_action: Optional[str] = None,
    ) -> Action:
        slot = self._choose_target_slot(state) if action_type in {"ASK_SLOT", "CLARIFY"} else None
        templates = self._get_templates_for_action(action_type, slot)
        template = self._pick_non_repeated_template(templates, state)
        if action_type == "CONFIRM":
            template = self._build_confirm_template(state)
        if planned_next_action:
            state.context["policy_plan"] = {
                "current_action": action_type,
                "next_action": planned_next_action,
            }
        elif action_type != "CONFIRM":
            state.context.pop("policy_plan", None)
        return Action(action_type=action_type, slot=slot, template=template)

    def _choose_target_slot(self, state: DialogueState) -> Optional[str]:
        missing = state.get_missing_slots()
        if missing:
            return missing[0]
        return None

    def _get_templates_for_action(self, action_type: str, slot: Optional[str]) -> List[str]:
        if action_type == "ASK_SLOT":
            if slot == "LOCATION":
                return [
                    "Bạn đang ở khu vực nào để mình tìm quán gần bạn?",
                    "Bạn cho mình biết quận hoặc khu vực cụ thể nhé.",
                ]
            if slot == "PRICE":
                return [
                    "Bạn muốn mức giá khoảng bao nhiêu ạ?",
                    "Bạn thích tầm giá bình dân hay cao hơn một chút?",
                ]
            if slot == "DISH":
                return [
                    "Bạn đang muốn ăn món gì để mình gợi ý đúng hơn?",
                    "Bạn cho mình biết món bạn muốn ăn nhé.",
                ]

        if action_type == "CLARIFY" and slot == "LOCATION":
            return [
                "Bạn có thể cho mình quận hoặc địa chỉ cụ thể được không?",
                "Mình cần khu vực cụ thể hơn để gợi ý chính xác cho bạn.",
            ]

        if action_type == "CONFIRM":
            return self.templates.get("CONFIRM", DEFAULT_TEMPLATES["CONFIRM"])

        return self.templates.get(action_type, self.templates["FALLBACK"])

    def _build_confirm_template(self, state: DialogueState) -> str:
        filled = state.filled_slots
        dish = getattr(filled.get("DISH"), "value", None)
        location = getattr(filled.get("LOCATION"), "value", None)
        pieces: List[str] = []
        if dish:
            pieces.append(f"món {dish}")
        if location and location != "__NEARBY__":
            pieces.append(f"khu vực {location}")

        if pieces:
            summary = " và ".join(pieces)
            return f"Mình hiểu bạn đang muốn tìm quán cho {summary}. Mình gợi ý ngay nhé, đúng không?"

        return self._pick_non_repeated_template(self.templates.get("CONFIRM", DEFAULT_TEMPLATES["CONFIRM"]), state)

    def _build_planned_confirm_action(self, state: DialogueState, score: float, source: str, note: str) -> Action:
        action = self._build_action("CONFIRM", state, planned_next_action="RECOMMEND")
        self._dbg("planned confirm action source=%s score=%.4f note=%s", source, score, note)
        return action

    def _pick_non_repeated_template(self, templates: List[str], state: DialogueState) -> str:
        if not templates:
            return ""
        recent = []
        for t in getattr(state, "turns", [])[-self.repeat_window:]:
            br = getattr(t, "bot_response", None)
            if br:
                recent.append(br.strip().lower())

        candidates = [x for x in templates if x.strip().lower() not in recent]
        return self.rng.choice(candidates if candidates else templates)

    def _is_repetitive_prompt(self, state: DialogueState, next_action_type: str) -> bool:
        if next_action_type not in {"ASK_SLOT", "CLARIFY"}:
            return False
        recent_actions = [getattr(t, "bot_action", None) for t in getattr(state, "turns", [])[-self.repeat_window:]]
        repeats = sum(1 for a in recent_actions if a in {"ASK_SLOT", "CLARIFY"})
        return repeats >= 2

    @staticmethod
    def _normalize_action(action: str) -> str:
        a = action.strip().upper()
        return ACTION_ALIASES.get(a, a)

    def get_last_decision_log(self) -> Dict[str, Any]:
        return {
            "source": self._last_log.source,
            "action": self._last_log.action,
            "confidence": self._last_log.confidence,
            "note": self._last_log.note,
        }

    @staticmethod
    def is_out_of_scope(state: DialogueState) -> bool:
        return state.current_intent == IntentType.OUT_OF_SCOPE