from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_policy.rule_based_policy import Action
from dialogue_state_tracking.state_schema import DialogueState, IntentType, Slot


class DummyRulePolicy:
    def __init__(self, rules=None):
        self.rules = rules or []

    def _evaluate_condition(self, condition, state):
        return bool(condition.get("always", False))


class DummyMLPolicy:
    def __init__(self, action="RECOMMEND", confidence=0.95):
        self.action = action
        self.confidence = confidence

    def predict_action(self, state):
        return {"action": self.action, "confidence": self.confidence}


class DummyLLMPolicy:
    def __init__(self, action="FALLBACK"):
        self.action = action

    def decide_action(self, state_summary, action_space):
        return self.action


def _state(intent=IntentType.RECOMMEND_PLACE_NEARBY, missing_conflicts=0):
    state = DialogueState(session_id="s1")
    state.current_intent = intent
    state.context["slot_conflicts"] = missing_conflicts
    return state


def test_recommend_is_downgraded_to_ask_slot_when_missing_slots():
    state = _state()
    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[]),
        ml_policy=DummyMLPolicy(action="RECOMMEND", confidence=0.99),
        llm_policy=None,
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "ASK_SLOT"
    assert action.slot == "DISH"


def test_dynamic_slot_target_uses_missing_slot_order():
    state = _state()
    policy = HybridPolicy(rule_policy=DummyRulePolicy(rules=[]), debug=False)

    ask_action = policy._build_action("ASK_SLOT", state)
    clarify_action = policy._build_action("CLARIFY", state)

    assert ask_action.slot == "DISH"
    assert clarify_action.slot == "DISH"


def test_rule_escape_to_ml_when_slot_conflicts_high():
    state = _state(missing_conflicts=2)

    rule = {
        "priority": 10,
        "condition": {"always": True},
        "action": {
            "type": "ASK_SLOT",
            "slot_to_ask": "LOCATION",
            "templates": ["rule ask"],
        },
    }

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[rule]),
        ml_policy=DummyMLPolicy(action="RESPOND", confidence=0.9),
        llm_policy=DummyLLMPolicy(action="FALLBACK"),
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "RESPOND"


def test_rule_is_prioritized_for_ask_slot_when_incomplete():
    state = _state()
    state.filled_slots["DISH"] = Slot(type="DISH", value="pho", confidence=0.95, turn_index=0, is_confirmed=True)

    rule = {
        "priority": 10,
        "condition": {"always": True},
        "action": {
            "type": "ASK_SLOT",
            "slot_to_ask": "LOCATION",
            "templates": ["rule ask"],
        },
    }

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[rule]),
        ml_policy=DummyMLPolicy(action="RESPOND", confidence=0.99),
        llm_policy=None,
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "ASK_SLOT"
    assert action.slot == "LOCATION"


def test_rule_ml_fusion_can_choose_ml_recommendation():
    state = _state()
    state.filled_slots["DISH"] = Slot(type="DISH", value="pho", confidence=0.95, turn_index=0, is_confirmed=True)
    state.filled_slots["LOCATION"] = Slot(type="LOCATION", value="quận 1", confidence=0.95, turn_index=0, is_confirmed=True)

    rule = {
        "priority": 5,
        "condition": {"always": True},
        "action": {
            "type": "ASK_SLOT",
            "slot_to_ask": "PRICE",
            "templates": ["rule ask price"],
        },
    }

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[rule]),
        ml_policy=DummyMLPolicy(action="RECOMMEND", confidence=0.98),
        llm_policy=None,
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type in {"RECOMMEND", "CONFIRM"}


def test_confirm_then_recommend_is_planned_when_confidence_midrange():
    state = _state()
    state.filled_slots["DISH"] = Slot(type="DISH", value="pho", confidence=0.95, turn_index=0, is_confirmed=True)
    state.filled_slots["LOCATION"] = Slot(type="LOCATION", value="quận 1", confidence=0.95, turn_index=0, is_confirmed=True)

    rule = {
        "priority": 8,
        "condition": {"always": True},
        "action": {
            "type": "RECOMMEND",
            "templates": ["rule recommend"],
        },
    }

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[rule]),
        ml_policy=DummyMLPolicy(action="RECOMMEND", confidence=0.75),
        llm_policy=None,
        debug=False,
        confirm_threshold=0.72,
        recommend_direct_threshold=0.85,
    )

    action = policy.decide_action(state)

    assert action.type == "CONFIRM"
    assert state.context["policy_plan"]["next_action"] == "RECOMMEND"
