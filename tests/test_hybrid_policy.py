from dialogue_state_tracking.dst import DialogueStateTracker


def test_dst_update_and_missing_slots():
    dst = DialogueStateTracker()
    session_id = dst.create_session(user_id="u1")

    state = dst.update_state(
        session_id=session_id,
        user_utterance="Tìm quán bún bò",
        intent="RECOMMEND_PLACE_NEARBY",
        intent_confidence=0.95,
        slots=[{"type": "DISH", "value": "bún bò", "confidence": 0.9}],
    )

    assert state.current_intent.name == "RECOMMEND_PLACE_NEARBY"
    assert "LOCATION" in state.get_missing_slots()

    state = dst.update_state(
        session_id=session_id,
        user_utterance="quận 1",
        intent="ASK_LOCATION",
        intent_confidence=0.85,
        slots=[{"type": "LOCATION", "value": "quận 1", "confidence": 0.95}],
    )

    assert "LOCATION" in state.filled_slots
    assert state.is_complete() is True