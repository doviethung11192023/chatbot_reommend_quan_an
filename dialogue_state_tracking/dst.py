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
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from .state_schema import DialogueState, IntentType, Slot, Turn


class DialogueStateTracker:
    def __init__(self):
        self.sessions: Dict[str, DialogueState] = {}

    def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = DialogueState(session_id=session_id, user_id=user_id)
        return session_id

    def get_state(self, session_id: str) -> Optional[DialogueState]:
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def get_active_sessions(self) -> List[str]:
        return list(self.sessions.keys())

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

        intent_enum = self._to_intent(intent)

        slot_objects = [
            Slot(
                type=str(s["type"]).upper(),
                value=str(s["value"]).strip(),
                confidence=float(s.get("confidence", 0.0)),
                turn_index=len(state.turns),
            )
            for s in slots
            if s.get("type") and s.get("value")
        ]

        turn = Turn(
            turn_index=len(state.turns),
            user_utterance=user_utterance,
            intent=intent_enum,
            intent_confidence=float(intent_confidence),
            slots_extracted=slot_objects,
        )

        state.add_turn(turn)
        return state

    @staticmethod
    def _to_intent(intent: str) -> IntentType:
        normalized = str(intent).strip().upper()
        if normalized in IntentType.__members__:
            return IntentType[normalized]
        return IntentType.NO_CLEAR_INTENT