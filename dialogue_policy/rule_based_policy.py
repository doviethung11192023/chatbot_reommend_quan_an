
"""
Rule-Based Dialogue Policy
Quyết định hành động tiếp theo dựa trên rules
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from dialogue_state_tracking.state_schema import DialogueState, IntentType


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

    Workflow:
    1. Load rules từ JSON
    2. Evaluate conditions dựa trên dialogue state
    3. Select action với priority cao nhất
    4. Return action
    """

    def __init__(self, rules_path: Optional[str] = None, rng: Optional[random.Random] = None):
        if rules_path is None:
            rules_path = str(Path(__file__).resolve().parent / "policy_rules.json")
        self.rules = self._load_rules(rules_path)
        self.rng = rng or random.Random()

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

        Args:
            state: Current dialogue state

        Returns:
            Action object
        """
        matched_rules = []

        for rule in self.rules:
            if self._evaluate_condition(rule.get("condition", {}), state):
                matched_rules.append(rule)

        if not matched_rules:
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

        return Action(
            action_type=action_config.get("type", "FALLBACK"),
            slot=action_config.get("slot_to_ask") or action_config.get("slot_to_clarify"),
            template=template
        )

    def _evaluate_condition(self, condition: Dict, state: DialogueState) -> bool:
        """
        Evaluate condition của rule

        Conditions có thể bao gồm:
        - intent
        - missing_slots
        - slots (value pattern, confidence)
        - turn_count
        - state_complete
        """
        # Check intent
        if "intent" in condition:
            intent_name = str(condition["intent"]).strip().upper()

            if intent_name not in IntentType.__members__:
                return False

            expected_intent = IntentType[intent_name]
            latest_intent = state.turns[-1].intent if state.turns else state.current_intent

            # SMALL_TALK là intent ngắn hạn -> check theo turn mới nhất
            if expected_intent == IntentType.SMALL_TALK:
                if latest_intent != expected_intent:
                    return False
            else:
                if state.current_intent != expected_intent:
                    return False

        # Check missing slots
        if "missing_slots" in condition:
            required_missing = set(condition["missing_slots"])
            actual_missing = set(state.get_missing_slots())
            if not required_missing.issubset(actual_missing):
                return False

        # Check state complete
        if "state_complete" in condition:
            if state.is_complete() != condition["state_complete"]:
                return False

        # Check turn count
        if "turn_count" in condition:
            turn_count = len(state.turns)
            tc = condition["turn_count"]

            if "max" in tc and turn_count > tc["max"]:
                return False
            if "min" in tc and turn_count < tc["min"]:
                return False

        # Check specific slots
        if "slots" in condition:
            for slot_type, slot_cond in condition["slots"].items():
                if slot_type not in state.filled_slots:
                    return False

                slot = state.filled_slots[slot_type]

                # Check value pattern
                if "value_pattern" in slot_cond:
                    if not re.match(slot_cond["value_pattern"], slot.value, flags=re.IGNORECASE):
                        return False

                # Check confidence
                if "confidence" in slot_cond:
                    conf_cond = slot_cond["confidence"]
                    if "min" in conf_cond and slot.confidence < conf_cond["min"]:
                        return False

        return True