     

# from __future__ import annotations

# import logging
# import random
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Protocol

# from dialogue_policy.rule_based_policy import Action, RuleBasedPolicy
# from dialogue_state_tracking.state_schema import DialogueState, IntentType

# logger = logging.getLogger(__name__)

# ACTION_SPACE = {"ASK_SLOT", "CLARIFY", "RECOMMEND", "RESPOND", "FALLBACK"}
# ACTION_ALIASES = {
#     "ASK_LOCATION": "ASK_SLOT",
#     "ASK_PRICE": "ASK_SLOT",
#     "ASK_OPEN_TIME": "ASK_SLOT",
#     "ASK_FOOD_TYPE": "ASK_SLOT",
#     "SMALL_TALK": "RESPOND",
#     "OUT_OF_SCOPE": "FALLBACK",
# }

# DEFAULT_TEMPLATES: Dict[str, List[str]] = {
#     "ASK_SLOT": ["Bạn có thể cho mình thêm thông tin được không?"],
#     "CLARIFY": ["Bạn có thể nói rõ hơn giúp mình được không?"],
#     "RECOMMEND": ["Để mình tìm quán phù hợp cho bạn nhé! 🔍"],
#     "RESPOND": ["Chào bạn! Mình có thể giúp bạn tìm quán ăn ngon nè 😊"],
#     "FALLBACK": ["Xin lỗi, mình chưa hiểu rõ. Bạn có thể nói lại giúp mình không?"],
# }


# class MLPolicyProtocol(Protocol):
#     def predict_action(self, state: DialogueState) -> Dict[str, Any]:
#         ...


# class LLMPolicyProtocol(Protocol):
#     def decide_action(self, state_summary: Dict[str, Any], action_space: List[str]) -> str:
#         ...


# @dataclass
# class PolicyDecisionLog:
#     source: str  # RULE | ML | LLM | FALLBACK
#     action: str
#     confidence: float = 0.0
#     note: str = ""


# class HybridPolicy:
#     """Priority: Rule -> ML -> LLM -> fallback"""

#     def __init__(
#         self,
#         rule_policy: Optional[RuleBasedPolicy] = None,
#         ml_policy: Optional[MLPolicyProtocol] = None,
#         llm_policy: Optional[LLMPolicyProtocol] = None,
#         ml_conf_threshold: float = 0.7,
#         templates: Optional[Dict[str, List[str]]] = None,
#         rng: Optional[random.Random] = None,
#         debug: bool = False,
#     ):
#         self.rule_policy = rule_policy or RuleBasedPolicy()
#         self.ml_policy = ml_policy
#         self.llm_policy = llm_policy
#         self.ml_conf_threshold = ml_conf_threshold
#         self.templates = templates or DEFAULT_TEMPLATES
#         self.rng = rng or random.Random(42)
#         self.debug = debug
#         self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")

#     def _dbg(self, msg: str, *args: Any) -> None:
#         if self.debug:
#             logger.info("[HybridPolicy] " + msg, *args)

#     def decide_action(self, state: DialogueState) -> Action:
#         state_summary = state.get_context_summary()
#         self._dbg("decide_action state=%s", state_summary)

#         # 1) Rule
#         matched_rule = None
#         for idx, rule in enumerate(self.rule_policy.rules):
#             cond = rule.get("condition", {})
#             try:
#                 ok = self.rule_policy._evaluate_condition(cond, state)
#             except Exception as ex:
#                 self._dbg("rule[%d] eval error: %s | rule=%s", idx, ex, rule)
#                 continue

#             self._dbg("rule[%d] condition=%s -> matched=%s", idx, cond, ok)
#             if ok:
#                 matched_rule = rule
#                 break

#         if matched_rule is not None:
#             self._dbg("matched rule=%s", matched_rule)
#             rule_action = self.rule_policy.decide_action(state)
#             a = self._normalize_action(rule_action.type)
#             self._last_log = PolicyDecisionLog(source="RULE", action=a, confidence=1.0)
#             self._dbg("selected RULE action=%s", rule_action)
#             return rule_action

#         self._dbg("no rule matched, trying ML")

#         # 2) ML
#         if self.ml_policy is not None:
#             try:
#                 pred = self.ml_policy.predict_action(state) or {}
#                 ml_action = self._normalize_action(str(pred.get("action", "FALLBACK")).upper())
#                 ml_conf = float(pred.get("confidence", 0.0))
#                 self._dbg("ml_pred=%s normalized_action=%s confidence=%.4f", pred, ml_action, ml_conf)

#                 if ml_action in ACTION_SPACE and ml_conf >= self.ml_conf_threshold:
#                     self._last_log = PolicyDecisionLog(
#                         source="ML", action=ml_action, confidence=ml_conf
#                     )
#                     action = self._build_action(ml_action)
#                     self._dbg("selected ML action=%s", action)
#                     return action
#                 self._dbg(
#                     "ML below threshold or invalid (threshold=%.2f), fallback to LLM",
#                     self.ml_conf_threshold,
#                 )
#             except Exception as ex:
#                 self._dbg("ML error: %s", ex)

#         # 3) LLM
#         if self.llm_policy is not None:
#             try:
#                 llm_action = self.llm_policy.decide_action(
#                     state.get_context_summary(),
#                     sorted(ACTION_SPACE),
#                 )
#                 llm_action = self._normalize_action(str(llm_action).upper())
#                 self._dbg("llm_pred=%s normalized_action=%s", llm_action, llm_action)

#                 if llm_action in ACTION_SPACE:
#                     self._last_log = PolicyDecisionLog(source="LLM", action=llm_action, confidence=0.0)
#                     action = self._build_action(llm_action)
#                     self._dbg("selected LLM action=%s", action)
#                     return action

#                 self._dbg("LLM returned invalid action=%s", llm_action)
#             except Exception as ex:
#                 self._last_log = PolicyDecisionLog(
#                     source="FALLBACK",
#                     action="FALLBACK",
#                     note=f"LLM error: {ex}",
#                 )
#                 self._dbg("LLM error: %s", ex)

#         # 4) fallback
#         self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")
#         action = self._build_action("FALLBACK")
#         self._dbg("selected FALLBACK action=%s", action)
#         return action

#     def _build_action(self, action_type: str) -> Action:
#         templates = self.templates.get(action_type, self.templates["FALLBACK"])
#         template = self.rng.choice(templates) if templates else ""
#         slot = "LOCATION" if action_type in {"ASK_SLOT", "CLARIFY"} else None
#         return Action(action_type=action_type, slot=slot, template=template)

#     @staticmethod
#     def _normalize_action(action: str) -> str:
#         a = action.strip().upper()
#         return ACTION_ALIASES.get(a, a)

#     def get_last_decision_log(self) -> Dict[str, Any]:
#         return {
#             "source": self._last_log.source,
#             "action": self._last_log.action,
#             "confidence": self._last_log.confidence,
#             "note": self._last_log.note,
#         }

#     @staticmethod
#     def is_out_of_scope(state: DialogueState) -> bool:
#         return state.current_intent == IntentType.OUT_OF_SCOPE

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from dialogue_policy.rule_based_policy import Action, RuleBasedPolicy
from dialogue_state_tracking.state_schema import DialogueState, IntentType

logger = logging.getLogger(__name__)

ACTION_SPACE = {"ASK_SLOT", "CLARIFY", "RECOMMEND", "RESPOND", "FALLBACK"}
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

    # def decide_action(self, state: DialogueState) -> Action:
       
    #     rule = self._select_best_rule(state)
    #     rule_action = self._build_action_from_rule(rule, state) if rule else None

    #     # Nếu rule bị lặp ASK_SLOT/CLARIFY nhiều lần thì mở đường cho ML/LLM
    #     if rule_action and self._is_repetitive_prompt(state, rule_action.type) and self.allow_ml_llm_escape:
    #         self._dbg("rule repetitive detected -> try ML/LLM escape")
    #         ml_action = self._try_ml(state)
    #         if ml_action:
    #             return ml_action
    #         llm_action = self._try_llm(state)
    #         if llm_action:
    #             return llm_action

    #     # Rule vẫn là safe default
    #     if rule_action:
    #         self._last_log = PolicyDecisionLog(source="RULE", action=self._normalize_action(rule_action.type), confidence=1.0)
    #         self._dbg("selected RULE action=%s", rule_action)
    #         return rule_action

    #     ml_action = self._try_ml(state)
    #     if ml_action:
    #         return ml_action

    #     llm_action = self._try_llm(state)
    #     if llm_action:
    #         return llm_action

    #     self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")
    #     return self._build_action("FALLBACK", state)
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

        rule = self._select_best_rule(state)
        rule_action = self._build_action_from_rule(rule, state) if rule else None

        if rule_action and self._is_repetitive_prompt(state, rule_action.type) and self.allow_ml_llm_escape:
            self._dbg("rule repetitive detected -> try ML/LLM escape")
            escaped = self._try_ml(state) or self._try_llm(state)
            if escaped:
                return self._apply_state_quality_guard(state, escaped)

        if rule_action:
            self._last_log = PolicyDecisionLog(source="RULE", action=self._normalize_action(rule_action.type), confidence=1.0)
            return self._apply_state_quality_guard(state, rule_action)

        ml_action = self._try_ml(state)
        if ml_action:
            return self._apply_state_quality_guard(state, ml_action)

        llm_action = self._try_llm(state)
        if llm_action:
            return self._apply_state_quality_guard(state, llm_action)

        self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")
        return self._build_action("FALLBACK", state)
    
    def _apply_state_quality_guard(self, state: DialogueState, action: Action) -> Action:
        quality = getattr(state, "get_state_quality", lambda: 1.0)()
        block_recommend = bool(getattr(state, "context", {}).get("block_recommend", False))

        if block_recommend and action.type == "RECOMMEND":
            self._dbg("downgrade RECOMMEND -> CLARIFY due block_recommend flag")
            return self._build_action("CLARIFY", state)

        if quality < self.state_quality_threshold and action.type == "RECOMMEND":
            self._dbg("downgrade RECOMMEND -> CLARIFY due low state quality=%.4f", quality)
            return self._build_action("CLARIFY", state)
        if quality < self.state_quality_threshold and action.type == "RECOMMEND":
            self._dbg("downgrade RECOMMEND -> CLARIFY due low state quality=%.4f", quality)
            return self._build_action("CLARIFY", state)

        if quality < self.state_quality_threshold and action.type == "ASK_SLOT" and not state.get_missing_slots():
            self._dbg("downgrade ASK_SLOT -> CLARIFY due low state quality=%.4f", quality)
            return self._build_action("CLARIFY", state)

        return action
    
    def _select_best_rule(self, state: DialogueState) -> Optional[Dict[str, Any]]:
        matched: List[Dict[str, Any]] = []
        for idx, rule in enumerate(getattr(self.rule_policy, "rules", [])):
            cond = rule.get("condition", {})
            ok = self.rule_policy._evaluate_condition(cond, state)
            self._dbg("rule[%d] matched=%s condition=%s", idx, ok, cond)
            if ok:
                matched.append(rule)
        if not matched:
            return None
        matched.sort(key=lambda r: r.get("priority", 0), reverse=True)
        return matched[0]

    def _build_action_from_rule(self, rule: Dict[str, Any], state: DialogueState) -> Action:
        action_cfg = rule.get("action", {})
        action_type = self._normalize_action(str(action_cfg.get("type", "FALLBACK")).upper())
        templates = action_cfg.get("templates") or self.templates.get(action_type, self.templates["FALLBACK"])
        template = self._pick_non_repeated_template(templates, state)
        slot = action_cfg.get("slot_to_ask") or action_cfg.get("slot_to_clarify")
        return Action(action_type=action_type, slot=slot, template=template)

    def _try_ml(self, state: DialogueState) -> Optional[Action]:
        if self.ml_policy is None:
            return None
        pred = self.ml_policy.predict_action(state) or {}
        ml_action = self._normalize_action(str(pred.get("action", "FALLBACK")).upper())
        ml_conf = float(pred.get("confidence", 0.0))
        self._dbg("ml_pred=%s normalized=%s conf=%.4f", pred, ml_action, ml_conf)
        if ml_action in ACTION_SPACE and ml_conf >= self.ml_conf_threshold:
            self._last_log = PolicyDecisionLog(source="ML", action=ml_action, confidence=ml_conf)
            return self._build_action(ml_action, state)
        return None

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

    def _build_action(self, action_type: str, state: DialogueState) -> Action:
        templates = self.templates.get(action_type, self.templates["FALLBACK"])
        template = self._pick_non_repeated_template(templates, state)
        slot = "LOCATION" if action_type in {"ASK_SLOT", "CLARIFY"} else None
        return Action(action_type=action_type, slot=slot, template=template)

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