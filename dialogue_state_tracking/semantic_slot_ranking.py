from __future__ import annotations

import re
from typing import Tuple

from .state_schema import Slot


class SemanticSlotRanker:
    VAGUE_LOCATION = {"gần đây", "nearby", "around here"}

    def should_replace(self, old: Slot, new: Slot, force: bool = False) -> Tuple[bool, str]:
        if force:
            return True, "forced_by_intent_shift_detector"

        old_v = old.value.strip().lower()
        new_v = new.value.strip().lower()

        if old_v == new_v:
            return False, "same_value"

        if old.type == "LOCATION":
            if old_v in self.VAGUE_LOCATION and new_v not in self.VAGUE_LOCATION:
                return True, "concrete_over_vague_location"

            old_spec = self._location_specificity(old_v)
            new_spec = self._location_specificity(new_v)

            old_score = 0.70 * old.confidence + 0.30 * old_spec
            new_score = 0.70 * new.confidence + 0.30 * new_spec

            if new_score >= old_score + 0.01:
                return True, "higher_semantic_score_location"
            return False, "lower_semantic_score_location"

        # generic fallback
        if new.confidence >= old.confidence + 0.08:
            return True, "higher_confidence"
        return False, "lower_confidence_or_ambiguous"

    @staticmethod
    def _location_specificity(v: str) -> float:
        if v in {"gần đây", "nearby", "around here"}:
            return 0.1
        if re.search(r"\b(quận|q\.?)\s*\d+\b", v):
            return 1.0
        if re.search(r"\b(phường|đường|huyện)\b", v):
            return 0.9
        if len(v) >= 6:
            return 0.7
        return 0.4