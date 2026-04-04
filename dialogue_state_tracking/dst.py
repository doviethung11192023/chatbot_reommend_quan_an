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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

class IntentType(Enum):
    ASK_DIRECTION = "ASK_DIRECTION"
    ASK_FOOD_TYPE = "ASK_FOOD_TYPE"
    ASK_LOCATION = "ASK_LOCATION"
    ASK_OPEN_TIME = "ASK_OPEN_TIME"
    ASK_PRICE = "ASK_PRICE"
    ASK_REVIEW = "ASK_REVIEW"
    COMPARE_PLACES = "COMPARE_PLACES"
    NO_CLEAR_INTENT = "NO_CLEAR_INTENT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    RECOMMEND_FOOD = "RECOMMEND_FOOD"
    RECOMMEND_PLACE_NEARBY = "RECOMMEND_PLACE_NEARBY"
    SMALL_TALK = "SMALL_TALK"

EPHEMERAL_INTENTS = {IntentType.SMALL_TALK, IntentType.NO_CLEAR_INTENT}

RESET_CUE_PATTERNS = (
    "thôi",
    "đổi",
    "món khác",
    "quán khác",
    "không muốn",
    "bỏ qua",
)

@dataclass
class Slot:
    type: str
    value: str
    confidence: float
    turn_index: int
    source_intent: Optional[IntentType] = None
    source_utterance: Optional[str] = None
    is_confirmed: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "confidence": self.confidence,
            "turn_index": self.turn_index,
            "source_intent": self.source_intent.value if self.source_intent else None,
            "source_utterance": self.source_utterance,
            "is_confirmed": self.is_confirmed,
        }

@dataclass
class Turn:
    turn_index: int
    user_utterance: str
    intent: IntentType
    intent_confidence: float
    slots_extracted: List[Slot]
    bot_response: Optional[str] = None
    bot_action: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DialogueState:
    session_id: str
    user_id: Optional[str] = None
    turns: List[Turn] = field(default_factory=list)
    filled_slots: Dict[str, Slot] = field(default_factory=dict)
    current_intent: Optional[IntentType] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    slot_conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def get_required_slots(self) -> List[str]:
        required_slots_map = {
            IntentType.RECOMMEND_PLACE_NEARBY: ["LOCATION"],
            IntentType.RECOMMEND_FOOD: ["DISH"],
            IntentType.ASK_PRICE: ["DISH"],
            IntentType.ASK_OPEN_TIME: ["LOCATION"],
        }
        return required_slots_map.get(self.current_intent, [])

    def get_missing_slots(self) -> List[str]:
        required = set(self.get_required_slots())
        filled = set(self.filled_slots.keys())
        return sorted(list(required - filled))

    def is_complete(self) -> bool:
        return len(self.get_missing_slots()) == 0

    def reset_slots(self, preserve: Optional[List[str]] = None, reason: str = "") -> Dict[str, Slot]:
        preserve_set = set(preserve or [])
        removed: Dict[str, Slot] = {}
        for key in list(self.filled_slots.keys()):
            if key not in preserve_set:
                removed[key] = self.filled_slots.pop(key)
        if reason:
            self.context.setdefault("reset_log", []).append({
                "reason": reason,
                "preserve": sorted(list(preserve_set)),
                "removed": {k: v.snapshot() for k, v in removed.items()},
                "timestamp": datetime.now().isoformat(),
            })
        return removed

    def record_conflict(self, slot_type: str, old_slot: Slot, new_slot: Slot, reason: str) -> None:
        self.slot_conflicts.append({
            "slot_type": slot_type,
            "old": old_slot.snapshot(),
            "new": new_slot.snapshot(),
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def get_state_quality(self) -> float:
        if not self.filled_slots and not self.get_required_slots():
            return 1.0
        if not self.filled_slots:
            return 0.0

        score = 1.0
        low_conf = sum(1 for s in self.filled_slots.values() if s.confidence < 0.70)
        unconfirmed = sum(1 for s in self.filled_slots.values() if not s.is_confirmed)
        conflicts = len(self.slot_conflicts)

        score -= 0.12 * low_conf
        score -= 0.08 * unconfirmed
        score -= 0.10 * conflicts

        return max(0.0, min(1.0, score))

    def add_turn(self, turn: Turn, merge_slots: bool = True) -> None:
        self.turns.append(turn)
        self.updated_at = datetime.now()

        if turn.intent not in EPHEMERAL_INTENTS:
            self.current_intent = turn.intent

        if merge_slots:
            for slot in turn.slots_extracted:
                prev = self.filled_slots.get(slot.type)
                if prev is None or slot.confidence >= prev.confidence:
                    self.filled_slots[slot.type] = slot

    def get_context_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "current_intent": self.current_intent.value if self.current_intent else None,
            "turn_count": len(self.turns),
            "filled_slots": {k: v.value for k, v in self.filled_slots.items()},
            "missing_slots": self.get_missing_slots(),
            "is_complete": self.is_complete(),
            "state_quality": self.get_state_quality(),
            "slot_conflicts": len(self.slot_conflicts),
            "last_utterance": self.turns[-1].user_utterance if self.turns else None,
        }