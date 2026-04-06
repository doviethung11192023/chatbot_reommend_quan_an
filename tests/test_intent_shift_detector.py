from dialogue_state_tracking.intent_shift_detector import IntentShiftDetector
from dialogue_state_tracking.state_schema import IntentType


def test_trailing_thoi_is_not_cancel():
    detector = IntentShiftDetector()
    decision = detector.detect(
        user_text="tim cho minh quan nao re binh dan thoi",
        current_intent=IntentType.RECOMMEND_FOOD,
        predicted_intent=IntentType.RECOMMEND_PLACE_NEARBY,
        accepted_slots=[{"type": "PRICE", "value": "re binh dan", "confidence": 0.95}],
    )

    assert decision.dialogue_act != "CANCEL"
    assert decision.block_recommend is False


def test_change_preference_not_cancel_and_force_replace():
    detector = IntentShiftDetector()
    decision = detector.detect(
        user_text="thoi gio minh doi mon khac an thit cho",
        current_intent=IntentType.RECOMMEND_FOOD,
        predicted_intent=IntentType.RECOMMEND_FOOD,
        accepted_slots=[{"type": "DISH", "value": "thit cho", "confidence": 0.99}],
    )

    assert decision.dialogue_act == "CHANGE"
    assert decision.override_intent == IntentType.RECOMMEND_FOOD
    assert "DISH" in decision.force_replace_slots
    assert decision.block_recommend is False


def test_change_cue_without_slots_blocks_recommend_and_targets_dish():
    detector = IntentShiftDetector()
    decision = detector.detect(
        user_text="thoi minh doi mon khac",
        current_intent=IntentType.RECOMMEND_FOOD,
        predicted_intent=IntentType.RECOMMEND_FOOD,
        accepted_slots=[],
    )

    assert decision.dialogue_act == "CHANGE"
    assert decision.override_intent == IntentType.RECOMMEND_FOOD
    assert "DISH" in decision.force_replace_slots
    assert decision.block_recommend is True
