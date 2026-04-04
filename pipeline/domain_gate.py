from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from dialogue_state_tracking.state_schema import IntentType


@dataclass
class DomainGateDecision:
    intent: str
    suppress_slots: bool
    reason: str = ""


class DomainGate:
    SMALL_TALK_PATTERNS = [
        r"\b(hello|hi|xin chào|chào|hey)\b",
        r"\b(cảm ơn|thank you|thanks)\b",
        r"\b(bạn khỏe không|how are you)\b",
    ]

    OUT_OF_SCOPE_PATTERNS = [
        r"\b(code|python|java|c\+\+|javascript)\b",
        r"\b(học tiếng anh|english|toán|vật lý)\b",
        r"\b(thời tiết|weather|tin tức|news)\b",
    ]

    FOOD_HINT_PATTERNS = [
        r"\b(quán|nhà hàng|ăn|món|phở|bún|cơm|lẩu|nướng)\b",
        r"\b(quận|huyện|phường|đường|gần đây|nearby)\b",
    ]

    def apply(
        self,
        user_text: str,
        predicted_intent: str,
        current_intent: Optional[IntentType],
    ) -> DomainGateDecision:
        text = (user_text or "").strip().lower()
        intent = str(predicted_intent or "NO_CLEAR_INTENT").strip().upper()

        if self._matches_any(text, self.SMALL_TALK_PATTERNS):
            return DomainGateDecision(intent="SMALL_TALK", suppress_slots=True, reason="small_talk_pattern")

        if self._matches_any(text, self.OUT_OF_SCOPE_PATTERNS) and not self._matches_any(text, self.FOOD_HINT_PATTERNS):
            return DomainGateDecision(intent="OUT_OF_SCOPE", suppress_slots=True, reason="out_of_scope_pattern")

        if intent in {"SMALL_TALK", "OUT_OF_SCOPE"}:
            return DomainGateDecision(intent=intent, suppress_slots=True, reason="predicted_non_domain")

        if intent == "NO_CLEAR_INTENT" and current_intent is None:
            return DomainGateDecision(intent="NO_CLEAR_INTENT", suppress_slots=True, reason="first_turn_no_clear")

        return DomainGateDecision(intent=intent, suppress_slots=False, reason="in_domain_or_followup")

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)