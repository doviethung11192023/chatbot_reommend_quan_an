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
    drop_slot_types: List[str] = field(default_factory=list)


class IntentShiftDetector:
    CANCEL_PATTERNS = [
        r"\b(không muốn ăn nữa|không ăn nữa|dừng lại|bỏ qua|hủy|cancel|stop)\b",
    ]
    GOODBYE_PATTERNS = [
        r"\b(tạm biệt|bye|goodbye|hẹn gặp lại)\b",
    ]
    NEGATIVE_SHORT = {"không", "no", "khong"}

    VAGUE_LOCATION = {"gần đây", "nearby", "around here"}
    CHANGE_CUES = [
        "đổi món",
        "doi mon",
        "món khác",
        "mon khac",
        "đổi quán",
        "doi quan",
        "quán khác",
        "quan khac",
        "đổi sang",
        "doi sang",
        "ăn món khác",
        "an mon khac",
    ]
    NEGATE_DISH_PATTERNS = [
        r"\b(không|khong)\s*muốn\s*ăn\b",
        r"\b(không|khong)\s*ăn\b",
    ]
    NEGATE_GENERIC_PATTERNS = [
        r"\b(không|khong)\s*muốn\s*ăn\s*gì\b",
        r"\b(không|khong)\s*muốn\s*ăn\s*gi\b",
    ]

    def detect(
        self,
        user_text: str,
        current_intent: Optional[IntentType],
        predicted_intent: IntentType,
        accepted_slots: List[Dict[str, Any]],
    ) -> IntentShiftDecision:
        text = (user_text or "").strip().lower()
        d = IntentShiftDecision()

        is_cancel = any(re.search(p, text) for p in self.CANCEL_PATTERNS)
        is_goodbye = any(re.search(p, text) for p in self.GOODBYE_PATTERNS)
        is_negate_generic = any(re.search(p, text) for p in self.NEGATE_GENERIC_PATTERNS)
        is_negate_dish = any(re.search(p, text) for p in self.NEGATE_DISH_PATTERNS)
        has_dish_slot = any(str(s.get("type", "")).upper() == "DISH" for s in accepted_slots)
        has_change_cue = any(phrase in text for phrase in self.CHANGE_CUES)
        if not has_change_cue and current_intent in {IntentType.RECOMMEND_FOOD, IntentType.RECOMMEND_PLACE_NEARBY}:
            if re.search(r"\bthôi\b.*\bmuốn\s*ăn\b", text) and not re.search(r"\bkhông\s*muốn\s*ăn\b", text):
                has_change_cue = True

        if not is_cancel and not is_goodbye and is_negate_dish and not is_negate_generic:
            d.dialogue_act = "CHANGE"
            d.block_recommend = True
            d.note = "negate_dish"
            if current_intent in {IntentType.RECOMMEND_FOOD, IntentType.RECOMMEND_PLACE_NEARBY}:
                d.override_intent = current_intent
            d.force_replace_slots.append("DISH")
            if has_dish_slot:
                d.drop_slot_types.append("DISH")
            return d

        # Explicit preference change should not be treated as cancel.
        if not is_cancel and not is_goodbye and has_change_cue:
            d.dialogue_act = "CHANGE"
            d.block_recommend = not bool(accepted_slots)
            d.note = "change_preference"
            if current_intent in {IntentType.RECOMMEND_FOOD, IntentType.RECOMMEND_PLACE_NEARBY}:
                d.override_intent = current_intent
            target_slots = {
                str(s.get("type", "")).upper().strip()
                for s in accepted_slots
                if s.get("type")
            }
            if not target_slots and current_intent == IntentType.RECOMMEND_FOOD:
                target_slots.add("DISH")
            if not target_slots and current_intent == IntentType.RECOMMEND_PLACE_NEARBY:
                target_slots.add("LOCATION")
            d.force_replace_slots.extend(sorted(target_slots))
            return d

        # Keep terse stop messages as cancel, but avoid matching trailing polite '... thôi'.
        if text in {"thôi", "thôi nhé", "thôi nha", "thôi ạ"}:
            d.override_intent = IntentType.NO_CLEAR_INTENT
            d.dialogue_act = "CANCEL"
            d.reset_mode = "hard"
            d.block_recommend = True
            d.note = "cancel_short"
            return d

        if is_cancel:
            d.override_intent = IntentType.NO_CLEAR_INTENT
            d.dialogue_act = "CANCEL"
            d.reset_mode = "hard"
            d.block_recommend = True
            d.note = "cancel_cue"
            return d

        if is_goodbye:
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