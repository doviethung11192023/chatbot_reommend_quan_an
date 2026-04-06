

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


@dataclass
class PolicyDecisionLog:
    source: str  # RULE | ML | LLM | FALLBACK
    action: str
    confidence: float = 0.0
    note: str = ""


class HybridPolicy:
    """Priority mềm: Rule (safe) -> ML -> LLM -> fallback, có anti-repeat."""

    def __init__(
        self,
        rule_policy: Optional[RuleBasedPolicy] = None,
        ml_policy: Optional[MLPolicyProtocol] = None,
        llm_policy: Optional[LLMPolicyProtocol] = None,
        ml_conf_threshold: float = 0.7,
        fusion_rule_weight: float = 0.6,
        fusion_ml_weight: float = 0.4,
        recommend_direct_threshold: float = 0.85,
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
        self.fusion_rule_weight = fusion_rule_weight
        self.fusion_ml_weight = fusion_ml_weight
        self.recommend_direct_threshold = recommend_direct_threshold
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

        rule, rule_priority = self._select_best_rule(state)
        rule_action = self._build_action_from_rule(rule, state) if rule else None
        ml_action, ml_conf = self._try_ml(state)

        should_escape = self._should_try_model_escape(state, rule_action)
        if should_escape and self.allow_ml_llm_escape:
            self._dbg("rule escape condition met -> try ML/LLM")
            escaped = ml_action or self._try_llm(state)
            if escaped:
                return self._apply_state_quality_guard(state, escaped)

        final_action: Optional[Action] = None
        final_source = "FALLBACK"
        final_score = 0.0
        final_note = ""

        if rule_action and ml_action:
            final_action, final_source, final_score, final_note = self._fuse_rule_and_ml(
                state=state,
                rule=rule,
                rule_priority=rule_priority,
                rule_action=rule_action,
                ml_action=ml_action,
                ml_conf=ml_conf,
            )
        elif rule_action:
            final_action = rule_action
            final_source = "RULE"
            final_score = self._score_rule_action(rule_action, rule, state)
        elif ml_action:
            final_action = ml_action
            final_source = "ML"
            final_score = ml_conf

        if final_action is not None:
            if self._should_plan_confirm_then_recommend(state, final_action, final_score):
                self._last_log = PolicyDecisionLog(
                    source=final_source,
                    action="CONFIRM",
                    confidence=final_score,
                    note=(final_note + " | " if final_note else "") + "planned_next=RECOMMEND",
                )
                return self._build_planned_confirm_action(state, final_score, final_source, final_note)

            self._last_log = PolicyDecisionLog(source=final_source, action=self._normalize_action(final_action.type), confidence=final_score, note=final_note)
            return self._apply_state_quality_guard(state, final_action)

        ml_action, ml_conf = self._try_ml(state)
        if ml_action:
            if self._should_plan_confirm_then_recommend(state, ml_action, ml_conf):
                self._last_log = PolicyDecisionLog(source="ML", action="CONFIRM", confidence=ml_conf, note="planned_next=RECOMMEND")
                return self._build_planned_confirm_action(state, ml_conf, "ML", "planned_next=RECOMMEND")
            self._last_log = PolicyDecisionLog(source="ML", action=self._normalize_action(ml_action.type), confidence=ml_conf)
            return self._apply_state_quality_guard(state, ml_action)

        llm_action = self._try_llm(state)
        if llm_action:
            return self._apply_state_quality_guard(state, llm_action)

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

    def _should_try_model_escape(self, state: DialogueState, rule_action: Optional[Action]) -> bool:
        if rule_action is None:
            return False

        quality = getattr(state, "get_state_quality", lambda: 1.0)()
        slot_conflicts = int(getattr(state, "context", {}).get("slot_conflicts", 0) or 0)
        repetitive = self._is_repetitive_prompt(state, rule_action.type)
        return repetitive or slot_conflicts > 1 or quality < 0.5
    
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

    def _build_action_from_rule(self, rule: Dict[str, Any], state: DialogueState) -> Action:
        action_cfg = rule.get("action", {})
        action_type = self._normalize_action(str(action_cfg.get("type", "FALLBACK")).upper())
        slot = action_cfg.get("slot_to_ask") or action_cfg.get("slot_to_clarify")
        if action_type in {"ASK_SLOT", "CLARIFY"} and not slot:
            slot = self._choose_target_slot(state)

        templates = action_cfg.get("templates") or self._get_templates_for_action(action_type, slot)
        template = self._pick_non_repeated_template(templates, state)
        return Action(action_type=action_type, slot=slot, template=template)

    def _try_ml(self, state: DialogueState) -> tuple[Optional[Action], float]:
        if self.ml_policy is None:
            return None, 0.0
        pred = self.ml_policy.predict_action(state) or {}
        ml_action = self._normalize_action(str(pred.get("action", "FALLBACK")).upper())
        ml_conf = float(pred.get("confidence", 0.0))
        self._dbg("ml_pred=%s normalized=%s conf=%.4f", pred, ml_action, ml_conf)
        if ml_action in ACTION_SPACE and ml_conf >= self.ml_conf_threshold:
            return self._build_action(ml_action, state), ml_conf
        return None, ml_conf

    def _try_llm(self, state: DialogueState) -> Optional[Action]:
        if self.llm_policy is None:
            return None
        try:
            pred = self.llm_policy.decide_action(state.get_context_summary(), sorted(ACTION_SPACE))
            llm_action = self._normalize_action(str(pred).upper())
            self._dbg("llm_pred=%s normalized=%s", pred, llm_action)
            if llm_action in ACTION_SPACE:
                self._last_log = PolicyDecisionLog(source="LLM", action=llm_action, confidence=0.0)
                return self._build_action(llm_action, state)
        except Exception as ex:
            self._dbg("LLM error=%s", ex)
            self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK", note=f"LLM error: {ex}")
        return None

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

    def _score_rule_action(self, action: Action, rule: Optional[Dict[str, Any]], state: DialogueState) -> float:
        priority = float(rule.get("priority", 0) if rule else 0)
        quality = getattr(state, "get_state_quality", lambda: 1.0)()
        if action.type in {"ASK_SLOT", "CLARIFY", "RESPOND", "FALLBACK"}:
            return 0.90 + min(0.05, priority / 1000.0)
        if action.type == "RECOMMEND":
            completeness = 1.0 if state.is_complete() else 0.0
            return min(0.92, 0.62 + 0.12 * completeness + 0.12 * quality + min(0.06, priority / 200.0))
        if action.type == "CONFIRM":
            return min(0.90, 0.55 + 0.15 * quality + min(0.08, priority / 100.0))
        return 0.60

    @staticmethod
    def _actions_compatible(rule_action: Action, ml_action: Action) -> bool:
        if rule_action.type == ml_action.type:
            return True
        if {rule_action.type, ml_action.type} <= {"RECOMMEND", "CONFIRM"}:
            return True
        return False

    def _fuse_rule_and_ml(
        self,
        state: DialogueState,
        rule: Optional[Dict[str, Any]],
        rule_priority: int,
        rule_action: Action,
        ml_action: Action,
        ml_conf: float,
    ) -> tuple[Action, str, float, str]:
        rule_score = self._score_rule_action(rule_action, rule, state)
        ml_score = float(ml_conf)
        compatible = self._actions_compatible(rule_action, ml_action)

        if compatible:
            fused_score = (self.fusion_rule_weight * rule_score) + (self.fusion_ml_weight * ml_score)
            if ml_action.type == "RECOMMEND" and rule_action.type in {"ASK_SLOT", "CLARIFY"} and state.is_complete():
                if fused_score >= self.recommend_direct_threshold:
                    return ml_action, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"
                if fused_score >= self.confirm_threshold:
                    confirm_action = self._build_action("CONFIRM", state, planned_next_action="RECOMMEND")
                    return confirm_action, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"
            if fused_score >= self.recommend_direct_threshold:
                chosen = rule_action if rule_score >= ml_score else ml_action
                return chosen, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"
            if fused_score >= self.confirm_threshold and ml_action.type == "RECOMMEND" and state.is_complete():
                confirm_action = self._build_action("CONFIRM", state, planned_next_action="RECOMMEND")
                return confirm_action, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"

            chosen = rule_action if rule_score >= ml_score else ml_action
            return chosen, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"

        if state.is_complete() and rule_action.type in {"ASK_SLOT", "CLARIFY"} and ml_action.type == "RECOMMEND":
            fused_score = (self.fusion_rule_weight * rule_score) + (self.fusion_ml_weight * ml_score)
            if fused_score >= self.recommend_direct_threshold:
                return ml_action, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"
            if fused_score >= self.confirm_threshold:
                confirm_action = self._build_action("CONFIRM", state, planned_next_action="RECOMMEND")
                return confirm_action, "FUSED", fused_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"

        if ml_score >= self.ml_conf_threshold and ml_score >= rule_score + 0.10:
            return ml_action, "ML", ml_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"

        return rule_action, "RULE", rule_score, f"rule_score={rule_score:.3f}|ml_score={ml_score:.3f}|priority={rule_priority}"

    def _should_plan_confirm_then_recommend(self, state: DialogueState, action: Action, score: float) -> bool:
        if action.type != "RECOMMEND":
            return False
        if not state.is_complete():
            return False
        if bool(getattr(state, "context", {}).get("block_recommend", False)):
            return False
        return self.confirm_threshold <= score < self.recommend_direct_threshold

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