from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state_schema import IntentType


@dataclass
class IntentShiftDecision:
    override_intent: Optional[IntentType] = None
    reset_mode: str = "none"  # none | soft | hard
    preserve_slots: List[str] = field(default_factory=list)
    force_replace_slots: List[str] = field(default_factory=list)
    dialogue_act: Optional[str] = None
    block_recommend: bool = False
    note: str = ""


class IntentShiftDetector:
    CANCEL_PATTERNS = [
        r"\b(thôi|không muốn|dừng|bỏ qua|hủy|cancel)\b",
    ]
    GOODBYE_PATTERNS = [
        r"\b(tạm biệt|bye|goodbye|hẹn gặp lại)\b",
    ]
    NEGATIVE_SHORT = {"không", "no", "khong"}

    VAGUE_LOCATION = {"gần đây", "nearby", "around here"}

    def detect(
        self,
        user_text: str,
        current_intent: Optional[IntentType],
        predicted_intent: IntentType,
        accepted_slots: List[Dict[str, Any]],
    ) -> IntentShiftDecision:
        text = (user_text or "").strip().lower()
        d = IntentShiftDecision()

        if any(re.search(p, text) for p in self.CANCEL_PATTERNS):
            d.override_intent = IntentType.NO_CLEAR_INTENT
            d.dialogue_act = "CANCEL"
            d.reset_mode = "hard"
            d.block_recommend = True
            d.note = "cancel_cue"
            return d

        if any(re.search(p, text) for p in self.GOODBYE_PATTERNS):
            d.override_intent = IntentType.NO_CLEAR_INTENT
            d.dialogue_act = "GOODBYE"
            d.reset_mode = "soft"
            d.block_recommend = True
            d.note = "goodbye_cue"
            return d

        if text in self.NEGATIVE_SHORT:
            d.dialogue_act = "NEGATE"
            d.block_recommend = True
            d.note = "short_negative_reply"

        # follow-up location cụ thể cho luồng recommend nearby
        if current_intent == IntentType.RECOMMEND_PLACE_NEARBY:
            for s in accepted_slots:
                if str(s.get("type", "")).upper() == "LOCATION":
                    val = str(s.get("value", "")).strip().lower()
                    if val and val not in self.VAGUE_LOCATION:
                        d.override_intent = IntentType.RECOMMEND_PLACE_NEARBY
                        d.force_replace_slots.append("LOCATION")
                        d.note = d.note or "concrete_location_followup"

        # giữ intent hiện tại khi classifier dao động ở ASK_* nhưng user đang follow-up
        if (
            current_intent in {IntentType.RECOMMEND_FOOD, IntentType.RECOMMEND_PLACE_NEARBY}
            and predicted_intent in {IntentType.ASK_LOCATION, IntentType.ASK_PRICE, IntentType.ASK_OPEN_TIME, IntentType.NO_CLEAR_INTENT}
            and accepted_slots
        ):
            d.override_intent = current_intent
            d.note = d.note or "followup_keep_current_intent"

        return d