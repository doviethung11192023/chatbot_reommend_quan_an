# """
# Dialogue State Tracker
# Quản lý trạng thái hội thoại
# """

# from typing import Dict, List, Optional
# import uuid
# from .state_schema import DialogueState, Turn, Slot, IntentType


# class DialogueStateTracker:
#     """
#     Tracker quản lý dialogue states của nhiều sessions
    
#     Features:
#     - Multi-session support
#     - State persistence
#     - Slot merging logic
#     - Context resolution
#     """
    
#     def __init__(self):
#         self.sessions: Dict[str, DialogueState] = {}
    
#     def create_session(self, user_id: Optional[str] = None) -> str:
#         """Tạo session mới"""
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = DialogueState(
#             session_id=session_id,
#             user_id=user_id
#         )
#         return session_id
    
#     def get_state(self, session_id: str) -> Optional[DialogueState]:
#         """Lấy state của session"""
#         return self.sessions.get(session_id)
    
#     def update_state(
#         self,
#         session_id: str,
#         user_utterance: str,
#         intent: str,
#         intent_confidence: float,
#         slots: List[Dict]
#     ) -> DialogueState:
#         """
#         Update state với turn mới
        
#         Args:
#             session_id: ID của session
#             user_utterance: Câu nói của user
#             intent: Intent từ classifier
#             intent_confidence: Confidence score
#             slots: List slots từ slot extractor
        
#         Returns:
#             Updated DialogueState
#         """
#         state = self.get_state(session_id)
#         if not state:
#             raise ValueError(f"Session {session_id} not found")
        
#         # Convert intent string to enum
#         intent_enum = IntentType[intent]
        
#         # Convert slots dict to Slot objects
#         slot_objects = [
#             Slot(
#                 type=s['type'],
#                 value=s['value'],
#                 confidence=s['confidence'],
#                 turn_index=len(state.turns)
#             )
#             for s in slots
#         ]
        
#         # Create new turn
#         turn = Turn(
#             turn_index=len(state.turns),
#             user_utterance=user_utterance,
#             intent=intent_enum,
#             intent_confidence=intent_confidence,
#             slots_extracted=slot_objects
#         )

        
#         # Add turn to state
#         state.add_turn(turn)
#         print(f"xem state sau khi add turn: {state}")
#         return state
    
#     def resolve_coreference(self, state: DialogueState) -> DialogueState:
#         """
#         Xử lý đại từ chỉ định (it, that, đó, này)
        
#         Example:
#         Turn 1: "Tìm quán phở"
#         Turn 2: "Nó ở đâu?" → resolve "nó" -> "quán phở"
#         """
#         # TODO: Implement coreference resolution
#         # Placeholder for now
#         return state
    
#     def clear_session(self, session_id: str):
#         """Xóa session"""
#         if session_id in self.sessions:
#             del self.sessions[session_id]
    
#     def get_active_sessions(self) -> List[str]:
#         """Lấy danh sách sessions đang active"""
#         return list(self.sessions.keys())


# # ============================================================
# # USAGE EXAMPLE
# # ============================================================

# if __name__ == "__main__":
#     # Initialize tracker
#     dst = DialogueStateTracker()
    
#     # Create session
#     session_id = dst.create_session(user_id="user_123")
#     print(f"Created session: {session_id}")
    
#     # Turn 1
#     state = dst.update_state(
#         session_id=session_id,
#         user_utterance="Tìm quán phở gần đây",
#         intent="RECOMMEND_PLACE_NEARBY",
#         intent_confidence=0.95,
#         slots=[
#             {'type': 'DISH', 'value': 'phở', 'confidence': 0.98},
#             {'type': 'LOCATION', 'value': 'gần đây', 'confidence': 0.85}
#         ]
#     )
    
#     print("\n=== Turn 1 ===")
#     print(f"Intent: {state.current_intent}")
#     print(f"Filled slots: {state.filled_slots}")
#     print(f"Missing slots: {state.get_missing_slots()}")
#     print(f"Is complete: {state.is_complete()}")
    
#     # Turn 2: Add more info
#     state = dst.update_state(
#         session_id=session_id,
#         user_utterance="Giá rẻ nhé",
#         intent="RECOMMEND_PLACE_NEARBY",
#         intent_confidence=0.90,
#         slots=[
#             {'type': 'PRICE', 'value': 'rẻ', 'confidence': 0.92}
#         ]
#     )
    
#     print("\n=== Turn 2 ===")
#     print(f"Filled slots: {state.filled_slots}")
#     print(f"Context: {state.get_context_summary()}")

# from __future__ import annotations

# import uuid
# from typing import Dict, List, Optional

# from .state_schema import DialogueState, IntentType, Slot, Turn


# class DialogueStateTracker:
#     def __init__(self):
#         self.sessions: Dict[str, DialogueState] = {}

#     def create_session(self, user_id: Optional[str] = None) -> str:
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = DialogueState(session_id=session_id, user_id=user_id)
#         return session_id

#     def get_state(self, session_id: str) -> Optional[DialogueState]:
#         return self.sessions.get(session_id)

#     def clear_session(self, session_id: str) -> None:
#         self.sessions.pop(session_id, None)

#     def get_active_sessions(self) -> List[str]:
#         return list(self.sessions.keys())

#     def update_state(
#         self,
#         session_id: str,
#         user_utterance: str,
#         intent: str,
#         intent_confidence: float,
#         slots: List[Dict],
#     ) -> DialogueState:
#         state = self.get_state(session_id)
#         if not state:
#             raise ValueError(f"Session {session_id} not found")

#         intent_enum = self._to_intent(intent)

#         slot_objects = [
#             Slot(
#                 type=str(s["type"]).upper(),
#                 value=str(s["value"]).strip(),
#                 confidence=float(s.get("confidence", 0.0)),
#                 turn_index=len(state.turns),
#             )
#             for s in slots
#             if s.get("type") and s.get("value")
#         ]

#         turn = Turn(
#             turn_index=len(state.turns),
#             user_utterance=user_utterance,
#             intent=intent_enum,
#             intent_confidence=float(intent_confidence),
#             slots_extracted=slot_objects,
#         )

#         state.add_turn(turn)
#         return state

#     @staticmethod
#     def _to_intent(intent: str) -> IntentType:
#         normalized = str(intent).strip().upper()
#         if normalized in IntentType.__members__:
#             return IntentType[normalized]
#         return IntentType.NO_CLEAR_INTENT


# from __future__ import annotations

# import logging
# import uuid
# from typing import Dict, List, Optional

# from .state_schema import DialogueState, IntentType, Slot, Turn

# logger = logging.getLogger(__name__)


# class DialogueStateTracker:
#     def __init__(self, debug: bool = False):
#         self.sessions: Dict[str, DialogueState] = {}
#         self.debug = debug

#     def _dbg(self, msg: str, *args) -> None:
#         if self.debug:
#             logger.info("[DST] " + msg, *args)

#     def create_session(self, user_id: Optional[str] = None) -> str:
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = DialogueState(session_id=session_id, user_id=user_id)
#         self._dbg("create_session user_id=%s -> session_id=%s", user_id, session_id)
#         return session_id

#     def get_state(self, session_id: str) -> Optional[DialogueState]:
#         return self.sessions.get(session_id)

#     def clear_session(self, session_id: str) -> None:
#         self.sessions.pop(session_id, None)
#         self._dbg("clear_session session_id=%s", session_id)

#     def get_active_sessions(self) -> List[str]:
#         return list(self.sessions.keys())

#     def update_state(
#         self,
#         session_id: str,
#         user_utterance: str,
#         intent: str,
#         intent_confidence: float,
#         slots: List[Dict],
#     ) -> DialogueState:
#         state = self.get_state(session_id)
#         if not state:
#             raise ValueError(f"Session {session_id} not found")

#         before_summary = state.get_context_summary()
#         self._dbg(
#             "update_state start session_id=%s turn=%d utterance=%r",
#             session_id,
#             len(state.turns),
#             user_utterance,
#         )
#         self._dbg("before_state=%s", before_summary)

#         intent_enum = self._to_intent(intent)
#         self._dbg(
#             "intent raw=%s -> enum=%s (confidence=%.4f)",
#             intent,
#             intent_enum.name,
#             float(intent_confidence),
#         )

#         slot_objects = [
#             Slot(
#                 type=str(s["type"]).upper(),
#                 value=str(s["value"]).strip(),
#                 confidence=float(s.get("confidence", 0.0)),
#                 turn_index=len(state.turns),
#             )
#             for s in slots
#             if s.get("type") and s.get("value")
#         ]
#         self._dbg("slots_in=%s", slots)
#         self._dbg(
#             "slots_normalized=%s",
#             [
#                 {
#                     "type": x.type,
#                     "value": x.value,
#                     "confidence": x.confidence,
#                     "turn_index": x.turn_index,
#                 }
#                 for x in slot_objects
#             ],
#         )

#         turn = Turn(
#             turn_index=len(state.turns),
#             user_utterance=user_utterance,
#             intent=intent_enum,
#             intent_confidence=float(intent_confidence),
#             slots_extracted=slot_objects,
#         )

#         state.add_turn(turn)

#         after_summary = state.get_context_summary()
#         self._dbg("after_state=%s", after_summary)
#         self._dbg(
#             "filled_slots_delta before=%s -> after=%s",
#             before_summary.get("filled_slots", {}),
#             after_summary.get("filled_slots", {}),
#         )

#         return state

#     @staticmethod
#     def _to_intent(intent: str) -> IntentType:
#         normalized = str(intent).strip().upper()
#         if normalized in IntentType.__members__:
#             return IntentType[normalized]
#         return IntentType.NO_CLEAR_INTENT


# ...existing code...
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from .state_schema import DialogueState, IntentType, Slot, Turn
from .intent_shift_detector import IntentShiftDetector
from .semantic_slot_ranking import SemanticSlotRanker
try:
    from .slot_guard import SlotValidator
except Exception:
    SlotValidator = None

logger = logging.getLogger(__name__)

class DialogueStateTracker:
    def __init__(self, debug: bool = False, enable_validator: bool = True):
        self.sessions: Dict[str, DialogueState] = {}
        self.debug = debug
        self.enable_validator = enable_validator and SlotValidator is not None
        self.slot_validator = SlotValidator(debug=debug) if self.enable_validator else None
        self.intent_shift_detector = IntentShiftDetector()
        self.slot_ranker = SemanticSlotRanker()

    def _dbg(self, msg: str, *args: Any) -> None:
        if self.debug:
            logger.info("[DST] " + msg, *args)

    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = DialogueState(session_id=session_id, user_id=user_id)
        self._dbg("create_session user_id=%s -> session_id=%s", user_id, session_id)
        return session_id

    def get_state(self, session_id: str) -> Optional[DialogueState]:
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self._dbg("clear_session session_id=%s", session_id)

    def update_state(
        self,
        session_id: str,
        user_utterance: str,
        intent: str,
        intent_confidence: float,
        slots: List[Dict],
    ) -> DialogueState:
        state = self.get_state(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        turn_index = len(state.turns)
        before_summary = state.get_context_summary()
        intent_enum = self._to_intent(intent)
        accepted_slots = self._validate_slots(slots, intent_enum, user_utterance, turn_index)
        self._dbg("accepted_slots=%s", [s.snapshot() for s in accepted_slots])

        # intent shift detector
        shift = self.intent_shift_detector.detect(
            user_text=user_utterance,
            current_intent=state.current_intent,
            predicted_intent=intent_enum,
            accepted_slots=[{"type": s.type, "value": s.value, "confidence": s.confidence} for s in accepted_slots],
        )
        if shift.override_intent is not None:
            intent_enum = shift.override_intent

        if shift.reset_mode == "hard":
            state.reset_slots(preserve=shift.preserve_slots, reason=f"intent_shift:{shift.note}")
        elif shift.reset_mode == "soft":
            # soft reset: giữ slot cốt lõi
            state.reset_slots(preserve=["LOCATION"], reason=f"intent_shift:{shift.note}")

        state.context["dialogue_act"] = shift.dialogue_act
        state.context["block_recommend"] = bool(shift.block_recommend)
        state.context["intent_shift_note"] = shift.note
        self._dbg(
            "update_state start session_id=%s turn=%d utterance=%r intent=%s",
            session_id, turn_index, user_utterance, intent_enum.name
        )
        self._dbg("before_state=%s", before_summary)
        self._dbg("slots_raw=%s", slots)

        # 1) Validator: guardrail cuối trước khi commit state
        accepted_slots = self._validate_slots(slots, intent_enum, user_utterance, turn_index)
        self._dbg("accepted_slots=%s", [s.snapshot() for s in accepted_slots])

        # 2) Intent shift / reset
        shift_log = self._maybe_reset_on_intent_shift(state, intent_enum, user_utterance)
        if shift_log:
            self._dbg("intent_shift_reset=%s", shift_log)

        # 3) Semantic merge
        merged_slots, merge_logs = self._semantic_merge(
            state=state,
            new_slots=accepted_slots,
            force_replace_slots=set(shift.force_replace_slots),
        )
        self._dbg("merge_logs=%s", merge_logs)

        # 4) Commit turn
        turn = Turn(
            turn_index=turn_index,
            user_utterance=user_utterance,
            intent=intent_enum,
            intent_confidence=float(intent_confidence),
            slots_extracted=accepted_slots,
        )

        
        state.context["state_quality"] = state.get_state_quality()
        state.context["slot_conflicts"] = len(state.slot_conflicts)
        state.add_turn(turn, merge_slots=False)
        state.filled_slots = merged_slots
        state.context["state_quality"] = state.get_state_quality()
        state.context["last_merge_logs"] = merge_logs[-10:]
        state.context["last_shift_log"] = shift_log

        after_summary = state.get_context_summary()
        self._dbg("after_state=%s", after_summary)
        self._dbg(
            "filled_slots_delta before=%s -> after=%s",
            before_summary.get("filled_slots", {}),
            after_summary.get("filled_slots", {}),
        )
        return state

    def _validate_slots(self, slots: List[Dict], intent_enum: IntentType, utterance: str, turn_index: int) -> List[Slot]:
        if not self.slot_validator:
            return [
                Slot(
                    type=str(s["type"]).upper(),
                    value=str(s["value"]).strip(),
                    confidence=float(s.get("confidence", 0.0)),
                    turn_index=turn_index,
                    source_intent=intent_enum,
                    source_utterance=utterance,
                )
                for s in slots
                if s.get("type") and s.get("value")
            ]

        result = self.slot_validator.validate(slots)
        return [
            Slot(
                type=s.type,
                value=s.value,
                confidence=s.confidence,
                turn_index=turn_index,
                source_intent=intent_enum,
                source_utterance=utterance,
            )
            for s in result.accepted
        ]

    def _maybe_reset_on_intent_shift(
        self,
        state: DialogueState,
        new_intent: IntentType,
        utterance: str,
    ) -> Optional[Dict[str, Any]]:
        current = state.current_intent
        text = utterance.lower()

        reset_cues = ("thôi", "đổi", "món khác", "quán khác", "không muốn", "bỏ qua")
        has_reset_cue = any(cue in text for cue in reset_cues)
        intent_changed = current is not None and new_intent != current

        if not has_reset_cue and not intent_changed:
            return None

        preserve: List[str] = []
        if new_intent == IntentType.RECOMMEND_FOOD:
            preserve = ["LOCATION"]
        elif new_intent == IntentType.RECOMMEND_PLACE_NEARBY:
            preserve = ["DISH"]

        removed = state.reset_slots(preserve=preserve if has_reset_cue or intent_changed else [], reason="intent_shift_reset")
        if current != new_intent:
            state.current_intent = new_intent

        return {
            "current_intent": current.value if current else None,
            "new_intent": new_intent.value,
            "has_reset_cue": has_reset_cue,
            "preserve": preserve,
            "removed": {k: v.snapshot() for k, v in removed.items()},
        }
    def _semantic_merge(
        self,
        state: DialogueState,
        new_slots: List[Slot],
        force_replace_slots: set[str],
    ) -> tuple[Dict[str, Slot], List[Dict[str, Any]]]:
        merged = dict(state.filled_slots)
        logs: List[Dict[str, Any]] = []

        for slot in new_slots:
            prev = merged.get(slot.type)
            if prev is None:
                merged[slot.type] = slot
                logs.append({"type": slot.type, "action": "add", "value": slot.value, "confidence": slot.confidence})
                continue

            replace, reason = self.slot_ranker.should_replace(
                old=prev,
                new=slot,
                force=slot.type in force_replace_slots,
            )

            if replace:
                slot.history = list(prev.history) + [prev.snapshot()]
                merged[slot.type] = slot
                state.record_conflict(slot.type, prev, slot, reason)
                logs.append({"type": slot.type, "action": "replace", "reason": reason, "old": prev.snapshot(), "new": slot.snapshot()})
            else:
                # vẫn ghi conflict nếu value khác để tracking
                if prev.value.strip().lower() != slot.value.strip().lower():
                    state.record_conflict(slot.type, prev, slot, reason)
                logs.append({"type": slot.type, "action": "keep", "reason": reason, "old": prev.snapshot(), "new": slot.snapshot()})

        return merged, logs
    # def _semantic_merge(self, state: DialogueState, new_slots: List[Slot]) -> tuple[Dict[str, Slot], List[Dict[str, Any]]]:
    #     merged = dict(state.filled_slots)
    #     logs: List[Dict[str, Any]] = []

    #     for slot in new_slots:
    #         prev = merged.get(slot.type)
    #         if prev is None:
    #             merged[slot.type] = slot
    #             logs.append({"type": slot.type, "action": "add", "value": slot.value, "confidence": slot.confidence})
    #             continue

    #         replace, reason = self._should_replace(prev, slot, state.current_intent)
    #         if replace:
    #             slot.history = list(prev.history) + [prev.snapshot()]
    #             merged[slot.type] = slot
    #             logs.append({
    #                 "type": slot.type,
    #                 "action": "replace",
    #                 "reason": reason,
    #                 "old": prev.snapshot(),
    #                 "new": slot.snapshot(),
    #             })
    #             state.record_conflict(slot.type, prev, slot, reason)
    #         else:
    #             logs.append({
    #                 "type": slot.type,
    #                 "action": "keep",
    #                 "reason": reason,
    #                 "old": prev.snapshot(),
    #                 "new": slot.snapshot(),
    #             })

    #     return merged, logs

    def _should_replace(self, old: Slot, new: Slot, current_intent: Optional[IntentType]) -> tuple[bool, str]:
        if old.value.strip().lower() == new.value.strip().lower():
            return False, "same_value"

        if new.confidence >= old.confidence + 0.08:
            return True, "higher_confidence"

        junk_values = {"cảm ơn", "cho tôi", "giúp tôi", "giúp mình", "hay", "ok", "okay"}
        if old.value.strip().lower() in junk_values and new.value.strip().lower() not in junk_values:
            return True, "replace_junk"

        # cùng intent recommend nhưng user đổi món rõ ràng
        if current_intent in {IntentType.RECOMMEND_FOOD, IntentType.RECOMMEND_PLACE_NEARBY}:
            if new.confidence >= old.confidence:
                return True, "intent_sensitive_update"

        return False, "lower_confidence_or_ambiguous"

    @staticmethod
    def _to_intent(intent: str) -> IntentType:
        normalized = str(intent).strip().upper()
        if normalized in IntentType.__members__:
            return IntentType[normalized]
        return IntentType.NO_CLEAR_INTENT