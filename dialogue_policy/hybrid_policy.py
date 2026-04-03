# from __future__ import annotations

# import random
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Protocol

# from dialogue_policy.rule_based_policy import Action, RuleBasedPolicy
# from dialogue_state_tracking.state_schema import DialogueState, IntentType


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
#     ):
#         self.rule_policy = rule_policy or RuleBasedPolicy()
#         self.ml_policy = ml_policy
#         self.llm_policy = llm_policy
#         self.ml_conf_threshold = ml_conf_threshold
#         self.templates = templates or DEFAULT_TEMPLATES
#         self.rng = rng or random.Random(42)
#         self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")

#     def decide_action(self, state: DialogueState) -> Action:
#         # 1) Rule
#         if self._has_matched_rule(state):
#             rule_action = self.rule_policy.decide_action(state)
#             a = self._normalize_action(rule_action.type)
#             self._last_log = PolicyDecisionLog(source="RULE", action=a, confidence=1.0)
#             return rule_action

#         # 2) ML
#         if self.ml_policy is not None:
#             pred = self.ml_policy.predict_action(state) or {}
#             ml_action = self._normalize_action(str(pred.get("action", "FALLBACK")).upper())
#             ml_conf = float(pred.get("confidence", 0.0))
#             if ml_action in ACTION_SPACE and ml_conf >= self.ml_conf_threshold:
#                 self._last_log = PolicyDecisionLog(source="ML", action=ml_action, confidence=ml_conf)
#                 return self._build_action(ml_action)

#         # 3) LLM
#         if self.llm_policy is not None:
#             try:
#                 llm_action = self.llm_policy.decide_action(state.get_context_summary(), sorted(ACTION_SPACE))
#                 llm_action = self._normalize_action(str(llm_action).upper())
#                 if llm_action in ACTION_SPACE:
#                     self._last_log = PolicyDecisionLog(source="LLM", action=llm_action, confidence=0.0)
#                     return self._build_action(llm_action)
#             except Exception as ex:
#                 self._last_log = PolicyDecisionLog(
#                     source="FALLBACK", action="FALLBACK", note=f"LLM error: {ex}"
#                 )

#         # 4) fallback cuối
#         self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")
#         return self._build_action("FALLBACK")

#     def _has_matched_rule(self, state: DialogueState) -> bool:
#         for rule in self.rule_policy.rules:
#             cond = rule.get("condition", {})
#             if self.rule_policy._evaluate_condition(cond, state):
#                 return True
#         return False

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
    "ASK_SLOT": ["Bạn có thể cho mình thêm thông tin được không?"],
    "CLARIFY": ["Bạn có thể nói rõ hơn giúp mình được không?"],
    "RECOMMEND": ["Để mình tìm quán phù hợp cho bạn nhé! 🔍"],
    "RESPOND": ["Chào bạn! Mình có thể giúp bạn tìm quán ăn ngon nè 😊"],
    "FALLBACK": ["Xin lỗi, mình chưa hiểu rõ. Bạn có thể nói lại giúp mình không?"],
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
    """Priority: Rule -> ML -> LLM -> fallback"""

    def __init__(
        self,
        rule_policy: Optional[RuleBasedPolicy] = None,
        ml_policy: Optional[MLPolicyProtocol] = None,
        llm_policy: Optional[LLMPolicyProtocol] = None,
        ml_conf_threshold: float = 0.7,
        templates: Optional[Dict[str, List[str]]] = None,
        rng: Optional[random.Random] = None,
        debug: bool = False,
    ):
        self.rule_policy = rule_policy or RuleBasedPolicy()
        self.ml_policy = ml_policy
        self.llm_policy = llm_policy
        self.ml_conf_threshold = ml_conf_threshold
        self.templates = templates or DEFAULT_TEMPLATES
        self.rng = rng or random.Random(42)
        self.debug = debug
        self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")

    def _dbg(self, msg: str, *args: Any) -> None:
        if self.debug:
            logger.info("[HybridPolicy] " + msg, *args)

    def decide_action(self, state: DialogueState) -> Action:
        state_summary = state.get_context_summary()
        self._dbg("decide_action state=%s", state_summary)

        # 1) Rule
        matched_rule = None
        for idx, rule in enumerate(self.rule_policy.rules):
            cond = rule.get("condition", {})
            try:
                ok = self.rule_policy._evaluate_condition(cond, state)
            except Exception as ex:
                self._dbg("rule[%d] eval error: %s | rule=%s", idx, ex, rule)
                continue

            self._dbg("rule[%d] condition=%s -> matched=%s", idx, cond, ok)
            if ok:
                matched_rule = rule
                break

        if matched_rule is not None:
            self._dbg("matched rule=%s", matched_rule)
            rule_action = self.rule_policy.decide_action(state)
            a = self._normalize_action(rule_action.type)
            self._last_log = PolicyDecisionLog(source="RULE", action=a, confidence=1.0)
            self._dbg("selected RULE action=%s", rule_action)
            return rule_action

        self._dbg("no rule matched, trying ML")

        # 2) ML
        if self.ml_policy is not None:
            try:
                pred = self.ml_policy.predict_action(state) or {}
                ml_action = self._normalize_action(str(pred.get("action", "FALLBACK")).upper())
                ml_conf = float(pred.get("confidence", 0.0))
                self._dbg("ml_pred=%s normalized_action=%s confidence=%.4f", pred, ml_action, ml_conf)

                if ml_action in ACTION_SPACE and ml_conf >= self.ml_conf_threshold:
                    self._last_log = PolicyDecisionLog(
                        source="ML", action=ml_action, confidence=ml_conf
                    )
                    action = self._build_action(ml_action)
                    self._dbg("selected ML action=%s", action)
                    return action
                self._dbg(
                    "ML below threshold or invalid (threshold=%.2f), fallback to LLM",
                    self.ml_conf_threshold,
                )
            except Exception as ex:
                self._dbg("ML error: %s", ex)

        # 3) LLM
        if self.llm_policy is not None:
            try:
                llm_action = self.llm_policy.decide_action(
                    state.get_context_summary(),
                    sorted(ACTION_SPACE),
                )
                llm_action = self._normalize_action(str(llm_action).upper())
                self._dbg("llm_pred=%s normalized_action=%s", llm_action, llm_action)

                if llm_action in ACTION_SPACE:
                    self._last_log = PolicyDecisionLog(source="LLM", action=llm_action, confidence=0.0)
                    action = self._build_action(llm_action)
                    self._dbg("selected LLM action=%s", action)
                    return action

                self._dbg("LLM returned invalid action=%s", llm_action)
            except Exception as ex:
                self._last_log = PolicyDecisionLog(
                    source="FALLBACK",
                    action="FALLBACK",
                    note=f"LLM error: {ex}",
                )
                self._dbg("LLM error: %s", ex)

        # 4) fallback
        self._last_log = PolicyDecisionLog(source="FALLBACK", action="FALLBACK")
        action = self._build_action("FALLBACK")
        self._dbg("selected FALLBACK action=%s", action)
        return action

    def _build_action(self, action_type: str) -> Action:
        templates = self.templates.get(action_type, self.templates["FALLBACK"])
        template = self.rng.choice(templates) if templates else ""
        slot = "LOCATION" if action_type in {"ASK_SLOT", "CLARIFY"} else None
        return Action(action_type=action_type, slot=slot, template=template)

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