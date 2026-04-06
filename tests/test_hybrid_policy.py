from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_state_tracking.state_schema import DialogueState, IntentType, Slot


class DummyRulePolicy:
    def __init__(self, rules=None):
        self.rules = rules or []

    def _evaluate_condition(self, condition, state):
        return bool(condition.get("always", False))


class DummyLLMPolicy:
    def __init__(self, action="FALLBACK", slot=None, response="", next_action=None, reason=""):
        self.action = action
        self.slot = slot
        self.response = response
        self.next_action = next_action
        self.reason = reason

    def decide_action(self, state_summary, action_space):
        return self.action

    def decide_decision(self, state_summary, action_space):
        return {
            "action": self.action,
            "slot": self.slot,
            "response": self.response,
            "next_action": self.next_action,
            "reason": self.reason,
        }


def _state(intent=IntentType.RECOMMEND_PLACE_NEARBY, missing_conflicts=0):
    state = DialogueState(session_id="s1")
    state.current_intent = intent
    state.context["slot_conflicts"] = missing_conflicts
    return state


def test_recommend_is_downgraded_to_ask_slot_when_missing_slots():
    state = _state()
    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[]),
        llm_policy=DummyLLMPolicy(action="RECOMMEND"),
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


def test_safety_rule_handles_small_talk_before_llm():
    state = _state(intent=IntentType.SMALL_TALK)

    rule = {
        "priority": 10,
        "condition": {"always": True},
        "action": {
            "type": "RESPOND",
            "templates": ["rule small talk"],
        },
    }

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[rule]),
        llm_policy=DummyLLMPolicy(action="RECOMMEND", response="llm should not run"),
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "RESPOND"
    assert action.template == "rule small talk"


def test_llm_can_choose_slot_with_specific_response():
    state = _state()
    state.filled_slots["DISH"] = Slot(type="DISH", value="pho", confidence=0.95, turn_index=0, is_confirmed=True)

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[]),
        llm_policy=DummyLLMPolicy(
            action="ASK_SLOT",
            slot="LOCATION",
            response="Bạn đang ở khu vực nào để mình gợi ý quán gần nhất?",
        ),
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "ASK_SLOT"
    assert action.slot == "LOCATION"
    assert "khu vực" in action.template


def test_confirm_then_recommend_plan_is_persisted():
    state = _state()
    state.filled_slots["DISH"] = Slot(type="DISH", value="pho", confidence=0.95, turn_index=0, is_confirmed=True)
    state.filled_slots["LOCATION"] = Slot(type="LOCATION", value="quận 1", confidence=0.95, turn_index=0, is_confirmed=True)

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[]),
        llm_policy=DummyLLMPolicy(action="CONFIRM", next_action="RECOMMEND", response="Mình xác nhận lại trước khi gợi ý nhé."),
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "CONFIRM"
    assert state.context["policy_plan"]["next_action"] == "RECOMMEND"


def test_change_dialogue_act_prompts_for_missing_slot():
    state = _state(intent=IntentType.RECOMMEND_FOOD)
    state.context["dialogue_act"] = "CHANGE"

    policy = HybridPolicy(
        rule_policy=DummyRulePolicy(rules=[]),
        llm_policy=DummyLLMPolicy(action="RECOMMEND"),
        debug=False,
    )

    action = policy.decide_action(state)

    assert action.type == "ASK_SLOT"
    assert action.slot == "DISH"
