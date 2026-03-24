import random

from dialogue_policy.rule_based_policy import RuleBasedPolicy
from dialogue_state_tracking.dst import DialogueStateTracker
from pipeline.dialogue_manager import DialogueOrchestrator


class FakeIntentModel:
    def predict(self, text: str):
        t = text.lower().strip()

        if "xin chào" in t or "hello" in t:
            return {"intent": "SMALL_TALK", "confidence": 0.98}
        if "quận" in t or "district" in t:
            return {"intent": "ASK_LOCATION", "confidence": 0.93}
        if "tìm" in t or "quán" in t:
            return {"intent": "RECOMMEND_PLACE_NEARBY", "confidence": 0.95}
        return {"intent": "NO_CLEAR_INTENT", "confidence": 0.20}


class FakeSlotModel:
    def extract_slots(self, text: str):
        t = text.lower()
        slots = []

        if "phở" in t:
            slots.append({"type": "DISH", "value": "phở", "confidence": 0.99})
        if "gần đây" in t:
            slots.append({"type": "LOCATION", "value": "gần đây", "confidence": 0.9})
        if "quận 1" in t:
            slots.append({"type": "LOCATION", "value": "quận 1", "confidence": 0.95})
        if "rẻ" in t:
            slots.append({"type": "PRICE", "value": "rẻ", "confidence": 0.91})

        return slots


def _build_manager(rules_path: str) -> DialogueOrchestrator:
    return DialogueOrchestrator(
        intent_model=FakeIntentModel(),
        slot_model=FakeSlotModel(),
        dst=DialogueStateTracker(),
        policy=RuleBasedPolicy(rules_path=rules_path, rng=random.Random(7)),
        intent_conf_threshold=0.5,
    )


def test_multiturn_followup_intent_resolution(rules_path):
    manager = _build_manager(str(rules_path))
    session_id = manager.create_session(user_id="u1")

    r1 = manager.process_user_message(session_id, "Tìm quán phở")
    assert r1["action"]["type"] == "ASK_SLOT"
    assert r1["action"]["slot"] == "LOCATION"

    r2 = manager.process_user_message(session_id, "quận 1")
    assert r2["intent"]["raw"] == "ASK_LOCATION"
    assert r2["intent"]["resolved"] == "RECOMMEND_PLACE_NEARBY"
    assert r2["action"]["type"] == "RECOMMEND"


def test_vague_location_triggers_clarify(rules_path):
    manager = _build_manager(str(rules_path))
    session_id = manager.create_session()

    result = manager.process_user_message(session_id, "Tìm quán phở gần đây")
    assert result["action"]["type"] == "CLARIFY"
    assert result["action"]["slot"] == "LOCATION"


def test_low_confidence_intent_keep_previous_intent(rules_path):
    manager = _build_manager(str(rules_path))
    session_id = manager.create_session()

    manager.process_user_message(session_id, "Tìm quán phở")
    result = manager.process_user_message(session_id, "ừm...")

    assert result["intent"]["raw"] == "NO_CLEAR_INTENT"
    assert result["intent"]["resolved"] == "RECOMMEND_PLACE_NEARBY"
    assert result["action"]["type"] == "ASK_SLOT"


def test_small_talk_action(rules_path):
    manager = _build_manager(str(rules_path))
    session_id = manager.create_session()

    result = manager.process_user_message(session_id, "xin chào")
    assert result["action"]["type"] == "RESPOND"