
# """
# Rule-Based Dialogue Policy
# Quyết định hành động tiếp theo dựa trên rules
# """

# import json
# import random
# import re
# from pathlib import Path
# from typing import Dict, List, Optional

# from dialogue_state_tracking.state_schema import DialogueState, IntentType


# class Action:
#     """Hành động bot sẽ thực hiện"""

#     def __init__(self, action_type: str, slot: Optional[str] = None, template: Optional[str] = None):
#         self.type = action_type  # ASK_SLOT, CLARIFY, RECOMMEND, RESPOND, FALLBACK
#         self.slot = slot
#         self.template = template

#     def __repr__(self):
#         return f"Action({self.type}, slot={self.slot})"


# class RuleBasedPolicy:
#     """
#     Policy engine dựa trên luật

#     Workflow:
#     1. Load rules từ JSON
#     2. Evaluate conditions dựa trên dialogue state
#     3. Select action với priority cao nhất
#     4. Return action
#     """

#     def __init__(self, rules_path: Optional[str] = None, rng: Optional[random.Random] = None):
#         if rules_path is None:
#             rules_path = str(Path(__file__).resolve().parent / "policy_rules.json")
#         self.rules = self._load_rules(rules_path)
#         self.rng = rng or random.Random()

#     def _load_rules(self, path: str) -> List[Dict]:
#         """Load rules từ JSON/JSONC (hỗ trợ dòng comment bắt đầu bằng //)."""
#         raw_text = Path(path).read_text(encoding="utf-8")
#         cleaned_text = "\n".join(
#             line for line in raw_text.splitlines()
#             if not line.strip().startswith("//")
#         )
#         data = json.loads(cleaned_text)

#         if "rules" not in data or not isinstance(data["rules"], list):
#             raise ValueError(f"Invalid policy file: missing 'rules' list -> {path}")

#         return data["rules"]

#     def decide_action(self, state: DialogueState) -> Action:
#         """
#         Quyết định action tiếp theo

#         Args:
#             state: Current dialogue state

#         Returns:
#             Action object
#         """
#         matched_rules = []

#         for rule in self.rules:
#             if self._evaluate_condition(rule.get("condition", {}), state):
#                 matched_rules.append(rule)

#         if not matched_rules:
#             return Action(
#                 action_type="FALLBACK",
#                 template="Xin lỗi, mình chưa hiểu yêu cầu của bạn. Bạn có thể nói rõ hơn được không?"
#             )

#         matched_rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
#         selected_rule = matched_rules[0]
#         action_config = selected_rule.get("action", {})

#         templates = action_config.get("templates") or [
#             "Mình đã nhận yêu cầu của bạn."
#         ]
#         template = self.rng.choice(templates)

#         return Action(
#             action_type=action_config.get("type", "FALLBACK"),
#             slot=action_config.get("slot_to_ask") or action_config.get("slot_to_clarify"),
#             template=template
#         )

#     def _evaluate_condition(self, condition: Dict, state: DialogueState) -> bool:
#         """
#         Evaluate condition của rule

#         Conditions có thể bao gồm:
#         - intent
#         - missing_slots
#         - slots (value pattern, confidence)
#         - turn_count
#         - state_complete
#         """
#         # Check intent
#         if "intent" in condition:
#             intent_name = str(condition["intent"]).strip().upper()

#             if intent_name not in IntentType.__members__:
#                 return False

#             expected_intent = IntentType[intent_name]
#             latest_intent = state.turns[-1].intent if state.turns else state.current_intent

#             # SMALL_TALK là intent ngắn hạn -> check theo turn mới nhất
#             if expected_intent == IntentType.SMALL_TALK:
#                 if latest_intent != expected_intent:
#                     return False
#             else:
#                 if state.current_intent != expected_intent:
#                     return False

#         # Check missing slots
#         if "missing_slots" in condition:
#             required_missing = set(condition["missing_slots"])
#             actual_missing = set(state.get_missing_slots())
#             if not required_missing.issubset(actual_missing):
#                 return False

#         # Check state complete
#         if "state_complete" in condition:
#             if state.is_complete() != condition["state_complete"]:
#                 return False

#         # Check turn count
#         if "turn_count" in condition:
#             turn_count = len(state.turns)
#             tc = condition["turn_count"]

#             if "max" in tc and turn_count > tc["max"]:
#                 return False
#             if "min" in tc and turn_count < tc["min"]:
#                 return False

#         # Check specific slots
#         if "slots" in condition:
#             for slot_type, slot_cond in condition["slots"].items():
#                 if slot_type not in state.filled_slots:
#                     return False

#                 slot = state.filled_slots[slot_type]

#                 # Check value pattern
#                 if "value_pattern" in slot_cond:
#                     if not re.match(slot_cond["value_pattern"], slot.value, flags=re.IGNORECASE):
#                         return False

#                 # Check confidence
#                 if "confidence" in slot_cond:
#                     conf_cond = slot_cond["confidence"]
#                     if "min" in conf_cond and slot.confidence < conf_cond["min"]:
#                         return False

#         return True
"""
Rule-Based Dialogue Policy
Quyết định hành động tiếp theo dựa trên rules
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from dialogue_state_tracking.state_schema import DialogueState, IntentType

logger = logging.getLogger(__name__)


class Action:
    """Hành động bot sẽ thực hiện"""

    def __init__(self, action_type: str, slot: Optional[str] = None, template: Optional[str] = None):
        self.type = action_type  # ASK_SLOT, CLARIFY, RECOMMEND, RESPOND, FALLBACK
        self.slot = slot
        self.template = template

    def __repr__(self):
        return f"Action({self.type}, slot={self.slot})"


class RuleBasedPolicy:
    """
    Policy engine dựa trên luật
    """

    def __init__(
        self,
        rules_path: Optional[str] = None,
        rng: Optional[random.Random] = None,
        debug: bool = False,
    ):
        self.debug = debug
        if rules_path is None:
            rules_path = str(Path(__file__).resolve().parent / "policy_rules.json")
        self.rules = self._load_rules(rules_path)
        self.rng = rng or random.Random()
        self._dbg("initialized rules_path=%s rules_count=%d", rules_path, len(self.rules))

    def _dbg(self, msg: str, *args) -> None:
        if self.debug:
            logger.info("[RuleBasedPolicy] " + msg, *args)

    def _load_rules(self, path: str) -> List[Dict]:
        """Load rules từ JSON/JSONC (hỗ trợ dòng comment bắt đầu bằng //)."""
        raw_text = Path(path).read_text(encoding="utf-8")
        cleaned_text = "\n".join(
            line for line in raw_text.splitlines()
            if not line.strip().startswith("//")
        )
        data = json.loads(cleaned_text)

        if "rules" not in data or not isinstance(data["rules"], list):
            raise ValueError(f"Invalid policy file: missing 'rules' list -> {path}")

        return data["rules"]

    def decide_action(self, state: DialogueState) -> Action:
        """
        Quyết định action tiếp theo
        """
        self._dbg(
            "decide_action start current_intent=%s turns=%d filled_slots=%s",
            getattr(state.current_intent, "name", state.current_intent),
            len(state.turns),
            list(state.filled_slots.keys()) if hasattr(state, "filled_slots") else None,
        )

        matched_rules = []

        for idx, rule in enumerate(self.rules):
            cond = rule.get("condition", {})
            try:
                ok = self._evaluate_condition(cond, state)
            except Exception as ex:
                self._dbg("rule[%d] eval error=%s rule=%s", idx, ex, rule)
                continue

            self._dbg("rule[%d] condition=%s matched=%s priority=%s", idx, cond, ok, rule.get("priority", 0))
            if ok:
                matched_rules.append(rule)

        if not matched_rules:
            self._dbg("no rules matched -> FALLBACK")
            return Action(
                action_type="FALLBACK",
                template="Xin lỗi, mình chưa hiểu yêu cầu của bạn. Bạn có thể nói rõ hơn được không?"
            )

        matched_rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
        selected_rule = matched_rules[0]
        action_config = selected_rule.get("action", {})

        templates = action_config.get("templates") or [
            "Mình đã nhận yêu cầu của bạn."
        ]
        template = self.rng.choice(templates)

        action = Action(
            action_type=action_config.get("type", "FALLBACK"),
            slot=action_config.get("slot_to_ask") or action_config.get("slot_to_clarify"),
            template=template
        )

        self._dbg("selected_rule=%s", selected_rule)
        self._dbg("selected_action=%s", action)
        return action

    def _evaluate_condition(self, condition: Dict, state: DialogueState) -> bool:
        """
        Evaluate condition của rule
        """
        self._dbg("evaluate_condition start=%s", condition)

        # Check intent
        if "intent" in condition:
            intent_name = str(condition["intent"]).strip().upper()

            if intent_name not in IntentType.__members__:
                self._dbg("intent check failed: invalid intent=%s", intent_name)
                return False

            expected_intent = IntentType[intent_name]
            latest_intent = state.turns[-1].intent if state.turns else state.current_intent

            if expected_intent == IntentType.SMALL_TALK:
                if latest_intent != expected_intent:
                    self._dbg(
                        "intent check failed: expected SMALL_TALK latest=%s",
                        getattr(latest_intent, "name", latest_intent),
                    )
                    return False
            else:
                if state.current_intent != expected_intent:
                    self._dbg(
                        "intent check failed: expected=%s current=%s",
                        expected_intent.name,
                        getattr(state.current_intent, "name", state.current_intent),
                    )
                    return False

        # Check missing slots
        if "missing_slots" in condition:
            required_missing = set(condition["missing_slots"])
            actual_missing = set(state.get_missing_slots())
            if not required_missing.issubset(actual_missing):
                self._dbg(
                    "missing_slots failed required=%s actual=%s",
                    required_missing,
                    actual_missing,
                )
                return False

        # Check state complete
        if "state_complete" in condition:
            actual_complete = state.is_complete()
            expected_complete = condition["state_complete"]
            if actual_complete != expected_complete:
                self._dbg(
                    "state_complete failed expected=%s actual=%s",
                    expected_complete,
                    actual_complete,
                )
                return False

        # Check turn count
        if "turn_count" in condition:
            turn_count = len(state.turns)
            tc = condition["turn_count"]

            if "max" in tc and turn_count > tc["max"]:
                self._dbg("turn_count failed max=%s actual=%s", tc["max"], turn_count)
                return False
            if "min" in tc and turn_count < tc["min"]:
                self._dbg("turn_count failed min=%s actual=%s", tc["min"], turn_count)
                return False

        # Check specific slots
        if "slots" in condition:
            for slot_type, slot_cond in condition["slots"].items():
                if slot_type not in state.filled_slots:
                    self._dbg("slot check failed missing slot_type=%s", slot_type)
                    return False

                slot = state.filled_slots[slot_type]

                if "value_pattern" in slot_cond:
                    if not re.match(slot_cond["value_pattern"], slot.value, flags=re.IGNORECASE):
                        self._dbg(
                            "slot value_pattern failed slot=%s value=%r pattern=%s",
                            slot_type,
                            slot.value,
                            slot_cond["value_pattern"],
                        )
                        return False

                if "confidence" in slot_cond:
                    conf_cond = slot_cond["confidence"]
                    if "min" in conf_cond and slot.confidence < conf_cond["min"]:
                        self._dbg(
                            "slot confidence failed slot=%s conf=%.4f min=%.4f",
                            slot_type,
                            slot.confidence,
                            conf_cond["min"],
                        )
                        return False

        self._dbg("evaluate_condition passed=%s", condition)
        return True