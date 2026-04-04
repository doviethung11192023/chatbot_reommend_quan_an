# """
# Định nghĩa schema cho dialogue state
# """

# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Any
# from enum import Enum
# from datetime import datetime


# class IntentType(Enum):
#     """Các intent types từ classifier"""
#     ASK_DIRECTION = "ASK_DIRECTION"
#     ASK_FOOD_TYPE = "ASK_FOOD_TYPE"
#     ASK_LOCATION = "ASK_LOCATION"
#     ASK_OPEN_TIME = "ASK_OPEN_TIME"
#     ASK_PRICE = "ASK_PRICE"
#     ASK_REVIEW = "ASK_REVIEW"
#     COMPARE_PLACES = "COMPARE_PLACES"
#     NO_CLEAR_INTENT = "NO_CLEAR_INTENT"
#     OUT_OF_SCOPE = "OUT_OF_SCOPE"
#     RECOMMEND_FOOD = "RECOMMEND_FOOD"
#     RECOMMEND_PLACE_NEARBY = "RECOMMEND_PLACE_NEARBY"
#     SMALL_TALK = "SMALL_TALK"


# @dataclass
# class Slot:
#     """Một slot entity"""
#     type: str  # DISH, LOCATION, PRICE, TASTE, DIET, CUISINE, AMBIANCE, TIME
#     value: str
#     confidence: float
#     turn_index: int  # Ở turn nào được điền
    
#     def __repr__(self):
#         return f"{self.type}={self.value} ({self.confidence:.2f})"


# @dataclass
# class Turn:
#     """Một lượt hội thoại"""
#     turn_index: int
#     user_utterance: str
#     intent: IntentType
#     intent_confidence: float
#     slots_extracted: List[Slot]
#     bot_response: Optional[str] = None
#     bot_action: Optional[str] = None
#     timestamp: datetime = field(default_factory=datetime.now)


# @dataclass
# class DialogueState:
#     """
#     Trạng thái hội thoại đầy đủ
    
#     Lưu trữ:
#     - Lịch sử turns
#     - Các slots đã điền (tích lũy)
#     - Slots còn thiếu
#     - Context hiện tại
#     """
#     session_id: str
#     user_id: Optional[str] = None
    
#     # Lịch sử hội thoại
#     turns: List[Turn] = field(default_factory=list)
    
#     # Slots tích lũy qua các turns
#     filled_slots: Dict[str, Slot] = field(default_factory=dict)
    
#     # Metadata
#     current_intent: Optional[IntentType] = None
#     context: Dict[str, Any] = field(default_factory=dict)
    
#     # Tracking
#     created_at: datetime = field(default_factory=datetime.now)
#     updated_at: datetime = field(default_factory=datetime.now)
    
#     def get_required_slots(self) -> List[str]:
#         """Slots bắt buộc dựa trên intent hiện tại"""
#         required_slots_map = {
#             IntentType.RECOMMEND_PLACE_NEARBY: ["LOCATION"],
#             IntentType.RECOMMEND_FOOD: ["DISH"],
#             IntentType.ASK_PRICE: ["DISH"],
#             IntentType.ASK_OPEN_TIME: ["LOCATION"],
#         }
        
#         return required_slots_map.get(self.current_intent, [])
    
#     def get_missing_slots(self) -> List[str]:
#         """Slots còn thiếu"""
#         required = set(self.get_required_slots())
#         filled = set(self.filled_slots.keys())
#         print(f"required: {required}, filled: {filled}")
#         print(f"missing: {required - filled}")
#         return list(required - filled)
    
#     def is_complete(self) -> bool:
#         """Kiểm tra state đã đủ thông tin chưa"""
#         return len(self.get_missing_slots()) == 0
    
#     def add_turn(self, turn: Turn):
#         """Thêm turn mới"""
#         self.turns.append(turn)
#         self.updated_at = datetime.now()
        
#         # Update current intent
#         if turn.intent != IntentType.SMALL_TALK:
#             self.current_intent = turn.intent
        
#         # Merge slots (ưu tiên slots mới hơn)
#         for slot in turn.slots_extracted:
#             if slot.type not in self.filled_slots or slot.confidence > self.filled_slots[slot.type].confidence:
#                 self.filled_slots[slot.type] = slot
    
#     def get_context_summary(self) -> Dict[str, Any]:
#         """Tóm tắt context để truyền cho các thành phần khác"""
#         return {
#             'session_id': self.session_id,
#             'current_intent': self.current_intent.value if self.current_intent else None,
#             'turn_count': len(self.turns),
#             'filled_slots': {k: v.value for k, v in self.filled_slots.items()},
#             'missing_slots': self.get_missing_slots(),
#             'is_complete': self.is_complete(),
#             'last_utterance': self.turns[-1].user_utterance if self.turns else None
#         }
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Serialize cho logging/storage"""
#         return {
#             'session_id': self.session_id,
#             'user_id': self.user_id,
#             'turns': [
#                 {
#                     'turn_index': t.turn_index,
#                     'user_utterance': t.user_utterance,
#                     'intent': t.intent.value,
#                     'intent_confidence': t.intent_confidence,
#                     'slots': [{'type': s.type, 'value': s.value, 'confidence': s.confidence} for s in t.slots_extracted],
#                     'bot_response': t.bot_response,
#                     'bot_action': t.bot_action,
#                     'timestamp': t.timestamp.isoformat()
#                 }
#                 for t in self.turns
#             ],
#             'filled_slots': {k: {'value': v.value, 'confidence': v.confidence} for k, v in self.filled_slots.items()},
#             'current_intent': self.current_intent.value if self.current_intent else None,
#             'created_at': self.created_at.isoformat(),
#             'updated_at': self.updated_at.isoformat()
#         }

# from dataclasses import dataclass, field
# from datetime import datetime
# from enum import Enum
# from typing import Any, Dict, List, Optional


# class IntentType(Enum):
#     ASK_DIRECTION = "ASK_DIRECTION"
#     ASK_FOOD_TYPE = "ASK_FOOD_TYPE"
#     ASK_LOCATION = "ASK_LOCATION"
#     ASK_OPEN_TIME = "ASK_OPEN_TIME"
#     ASK_PRICE = "ASK_PRICE"
#     ASK_REVIEW = "ASK_REVIEW"
#     COMPARE_PLACES = "COMPARE_PLACES"
#     NO_CLEAR_INTENT = "NO_CLEAR_INTENT"
#     OUT_OF_SCOPE = "OUT_OF_SCOPE"
#     RECOMMEND_FOOD = "RECOMMEND_FOOD"
#     RECOMMEND_PLACE_NEARBY = "RECOMMEND_PLACE_NEARBY"
#     SMALL_TALK = "SMALL_TALK"


# EPHEMERAL_INTENTS = {IntentType.SMALL_TALK, IntentType.NO_CLEAR_INTENT}


# @dataclass
# class Slot:
#     type: str
#     value: str
#     confidence: float
#     turn_index: int


# @dataclass
# class Turn:
#     turn_index: int
#     user_utterance: str
#     intent: IntentType
#     intent_confidence: float
#     slots_extracted: List[Slot]
#     bot_response: Optional[str] = None
#     bot_action: Optional[str] = None
#     timestamp: datetime = field(default_factory=datetime.now)


# @dataclass
# class DialogueState:
#     session_id: str
#     user_id: Optional[str] = None
#     turns: List[Turn] = field(default_factory=list)
#     filled_slots: Dict[str, Slot] = field(default_factory=dict)
#     current_intent: Optional[IntentType] = None
#     context: Dict[str, Any] = field(default_factory=dict)
#     created_at: datetime = field(default_factory=datetime.now)
#     updated_at: datetime = field(default_factory=datetime.now)

#     def get_required_slots(self) -> List[str]:
#         required_slots_map = {
#             IntentType.RECOMMEND_PLACE_NEARBY: ["LOCATION"],
#             IntentType.RECOMMEND_FOOD: ["DISH"],
#             IntentType.ASK_PRICE: ["DISH"],
#             IntentType.ASK_OPEN_TIME: ["LOCATION"],
#         }
#         return required_slots_map.get(self.current_intent, [])

#     def get_missing_slots(self) -> List[str]:
#         required = set(self.get_required_slots())
#         filled = set(self.filled_slots.keys())
#         return sorted(list(required - filled))

#     def is_complete(self) -> bool:
#         return len(self.get_missing_slots()) == 0

#     def add_turn(self, turn: Turn) -> None:
#         self.turns.append(turn)
#         self.updated_at = datetime.now()

#         if turn.intent not in EPHEMERAL_INTENTS:
#             self.current_intent = turn.intent

#         for slot in turn.slots_extracted:
#             prev = self.filled_slots.get(slot.type)
#             if prev is None or slot.confidence >= prev.confidence:
#                 self.filled_slots[slot.type] = slot

#     def get_context_summary(self) -> Dict[str, Any]:
#         return {
#             "session_id": self.session_id,
#             "current_intent": self.current_intent.value if self.current_intent else None,
#             "turn_count": len(self.turns),
#             "filled_slots": {k: v.value for k, v in self.filled_slots.items()},
#             "missing_slots": self.get_missing_slots(),
#             "is_complete": self.is_complete(),
#             "last_utterance": self.turns[-1].user_utterance if self.turns else None,
#         }


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