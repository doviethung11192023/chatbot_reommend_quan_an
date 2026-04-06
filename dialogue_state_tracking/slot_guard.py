from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NormalizedSlot:
    type: str
    value: str
    confidence: float
    turn_index: Optional[int] = None


@dataclass
class SlotValidationResult:
    accepted: List[NormalizedSlot]
    rejected: List[Dict[str, Any]]


class SlotValidator:
    """
    Validate + normalize raw slots trước khi ghi vào DST.
    """

    DEFAULT_MIN_CONF = {
        "DISH": 0.70,
        "LOCATION": 0.65,
        "PRICE": 0.60,
        "AMBIANCE": 0.60,
        "TIME": 0.60,
        "TASTE": 0.60,
    }

    INVALID_PHRASES = {
        "cảm ơn",
        "thank you",
        "thanks",
        "ok",
        "okay",
        "ừ",
        "uh",
        "à",
        "á",
        "cho tôi",
        "giúp tôi",
        "giúp mình",
        "nha",
        "nhé",
        "được không",
    }

    LOCATION_PATTERNS = [
        r"^gần đây$",
        r"^nearby$",
        r"^around here$",
        r"^quận\s+\d+$",
        r"^q\.?\s*\d+$",
        r"^huyện\s+.+$",
        r"^phường\s+.+$",
        r"^đường\s+.+$",
        r"^nam định$",
    ]

    DISH_PATTERNS = [
        r"^[\wÀ-ỹ\s]+$",
    ]

    def __init__(self, debug: bool = False):
        self.debug = debug

    def _dbg(self, msg: str, *args: Any) -> None:
        if self.debug:
            logger.info("[SlotValidator] " + msg, *args)

    def validate(self, raw_slots: List[Dict[str, Any]]) -> SlotValidationResult:
        accepted: List[NormalizedSlot] = []
        rejected: List[Dict[str, Any]] = []

        self._dbg("validate start raw_slots=%s", raw_slots)

        for idx, slot in enumerate(raw_slots or []):
            slot_type = str(slot.get("type", "")).strip().upper()
            value = str(slot.get("value", "")).strip()
            confidence = float(slot.get("confidence", 0.0))
            turn_index = slot.get("turn_index")

            norm_value = self.normalize_value(slot_type, value)
            reason = self._reject_reason(slot_type, norm_value, confidence)

            if reason is not None:
                rejected.append({
                    "index": idx,
                    "type": slot_type,
                    "value": value,
                    "normalized_value": norm_value,
                    "confidence": confidence,
                    "reason": reason,
                })
                self._dbg(
                    "reject slot idx=%d type=%s value=%r normalized=%r conf=%.4f reason=%s",
                    idx, slot_type, value, norm_value, confidence, reason
                )
                continue

            accepted.append(
                NormalizedSlot(
                    type=slot_type,
                    value=norm_value,
                    confidence=confidence,
                    turn_index=turn_index,
                )
            )
            self._dbg(
                "accept slot idx=%d type=%s value=%r normalized=%r conf=%.4f",
                idx, slot_type, value, norm_value, confidence
            )

        accepted = self._deduplicate(accepted)
        self._dbg("validate done accepted=%s rejected=%s", accepted, rejected)
        return SlotValidationResult(accepted=accepted, rejected=rejected)

    def normalize_value(self, slot_type: str, value: str) -> str:
        v = (value or "").strip().lower()
        v = re.sub(r"\s+", " ", v)

        # bỏ hậu tố lịch sự
        v = re.sub(r"\s+(cho tôi|giúp tôi|giúp mình|nha+|nhé+|nhe+|được không)$", "", v).strip()

        if slot_type == "LOCATION":
            v = v.replace("gấn đây", "gần đây")
            v = v.replace("quận nhất", "quận 1")
            v = v.replace("q1", "quận 1")
            v = v.replace("q.1", "quận 1")

        if slot_type == "TASTE":
            if re.search(r"\bkhông\s*cay\b", v) or re.search(r"\bkhong\s*cay\b", v):
                return "không cay"
            if re.search(r"\bcay+\b", v):
                return "cay"
            if re.search(r"\bngọt\b", v) or re.search(r"\bngot\b", v):
                return "ngọt"
            if re.search(r"\bmặn\b", v) or re.search(r"\bman\b", v):
                return "mặn"
            if re.search(r"\bchua\b", v) or re.search(r"\bchua\b", v):
                return "chua"
            if re.search(r"\bbeo\b", v):
                return "béo"
            if v == "ca":
                return "cay"

        return v

    def _reject_reason(self, slot_type: str, value: str, confidence: float) -> Optional[str]:
        if not slot_type:
            return "missing_type"
        if not value:
            return "empty_value"
        if value in self.INVALID_PHRASES:
            return "invalid_phrase"
        if confidence < self.DEFAULT_MIN_CONF.get(slot_type, 0.60):
            return "low_confidence"

        if slot_type == "LOCATION":
            if not any(re.match(p, value, flags=re.IGNORECASE) for p in self.LOCATION_PATTERNS):
                # vẫn cho phép các location tự do nếu đủ dài và không phải junk
                if len(value) < 3:
                    return "invalid_location"
        elif slot_type == "DISH":
            if len(value) < 2:
                return "invalid_dish"
        elif slot_type == "TASTE":
            if len(value) < 2:
                return "invalid_taste"

        return None

    def _deduplicate(self, slots: List[NormalizedSlot]) -> List[NormalizedSlot]:
        best: Dict[str, NormalizedSlot] = {}
        for s in slots:
            prev = best.get(s.type)
            if prev is None:
                best[s.type] = s
                continue

            # ưu tiên confidence cao hơn, nếu bằng nhau thì giá trị dài hơn
            if s.confidence > prev.confidence:
                best[s.type] = s
            elif s.confidence == prev.confidence and len(s.value) > len(prev.value):
                best[s.type] = s

        result = list(best.values())
        self._dbg("deduplicated slots=%s", result)
        return result


class SlotMergePolicy:
    """
    Quyết định slot nào được ghi vào state.
    """

    def __init__(
        self,
        debug: bool = False,
        overwrite_margin: float = 0.08,
    ):
        self.debug = debug
        self.overwrite_margin = overwrite_margin

    def _dbg(self, msg: str, *args: Any) -> None:
        if self.debug:
            logger.info("[SlotMergePolicy] " + msg, *args)

    def merge(
        self,
        existing_slots: Dict[str, Any],
        new_slots: List[NormalizedSlot],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Return:
            merged_slots, merge_log
        """
        merged = dict(existing_slots or {})
        logs: List[Dict[str, Any]] = []

        self._dbg("merge start existing=%s new=%s", existing_slots, new_slots)

        for slot in new_slots:
            old = merged.get(slot.type)
            action = "add"

            if old is not None:
                old_value = getattr(old, "value", old)
                old_conf = getattr(old, "confidence", 0.0)

                if self._should_replace(slot.value, slot.confidence, old_value, old_conf):
                    merged[slot.type] = slot
                    action = "replace"
                else:
                    action = "keep"

                logs.append({
                    "type": slot.type,
                    "new_value": slot.value,
                    "new_confidence": slot.confidence,
                    "old_value": old_value,
                    "old_confidence": old_conf,
                    "action": action,
                })
                self._dbg(
                    "slot type=%s old=%r(%.4f) new=%r(%.4f) action=%s",
                    slot.type, old_value, old_conf, slot.value, slot.confidence, action
                )
            else:
                merged[slot.type] = slot
                logs.append({
                    "type": slot.type,
                    "new_value": slot.value,
                    "new_confidence": slot.confidence,
                    "old_value": None,
                    "old_confidence": None,
                    "action": action,
                })
                self._dbg(
                    "slot type=%s old=None new=%r(%.4f) action=%s",
                    slot.type, slot.value, slot.confidence, action
                )

        self._dbg("merge done merged=%s", merged)
        return merged, logs

    def _should_replace(
        self,
        new_value: str,
        new_conf: float,
        old_value: str,
        old_conf: float,
    ) -> bool:
        if not old_value:
            return True

        # cùng value thì giữ nguyên
        if str(new_value).strip().lower() == str(old_value).strip().lower():
            return False

        # overwrite nếu confidence cao hơn đủ nhiều
        if new_conf >= old_conf + self.overwrite_margin:
            return True

        # nếu old là junk mà new sạch hơn thì cho replace
        if self._looks_junk(old_value) and not self._looks_junk(new_value):
            return True

        return False

    @staticmethod
    def _looks_junk(value: str) -> bool:
        v = str(value).strip().lower()
        return v in {
            "cảm ơn",
            "cho tôi",
            "giúp tôi",
            "giúp mình",
            "hay",
            "ok",
            "okay",
        }