


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
    VAGUE_LOCATION_VALUES = {"gần đây", "nearby", "around here", "gan day"}

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
        if self._should_suppress_slots(intent_enum, state):
            self._dbg("suppress slots due intent=%s current_intent=%s", intent_enum.name, getattr(state.current_intent, "name", None))
            slots = []

        # 1) Validator: guardrail cuối trước khi commit state
        accepted_slots = self._validate_slots(slots, intent_enum, user_utterance, turn_index)
        self._dbg("accepted_slots=%s", [s.snapshot() for s in accepted_slots])

        # 2) Intent shift detector (single source of truth)
        shift = self.intent_shift_detector.detect(
            user_text=user_utterance,
            current_intent=state.current_intent,
            predicted_intent=intent_enum,
            accepted_slots=[{"type": s.type, "value": s.value, "confidence": s.confidence} for s in accepted_slots],
        )
        if shift.override_intent is not None:
            intent_enum = shift.override_intent

        if shift.dialogue_act in {"CANCEL", "GOODBYE"}:
            state.current_intent = None
            state.context["session_status"] = "closed"

        if shift.dialogue_act == "CHANGE" and shift.force_replace_slots and shift.reset_mode == "none":
            preserve = [
                key for key in state.filled_slots.keys()
                if key not in set(shift.force_replace_slots)
            ]
            state.reset_slots(preserve=preserve, reason=f"intent_shift:{shift.note or 'change'}")

        if shift.reset_mode == "hard":
            state.reset_slots(preserve=shift.preserve_slots, reason=f"intent_shift:{shift.note}")
        elif shift.reset_mode == "soft":
            # soft reset: giữ slot cốt lõi
            state.reset_slots(preserve=["LOCATION"], reason=f"intent_shift:{shift.note}")

        if shift.drop_slot_types:
            drop_types = {t for t in shift.drop_slot_types if t}
            if drop_types:
                accepted_slots = [s for s in accepted_slots if s.type not in drop_types]

        state.context["dialogue_act"] = shift.dialogue_act
        state.context["block_recommend"] = bool(shift.block_recommend)
        state.context["intent_shift_note"] = shift.note
        state.context["last_shift_decision"] = {
            "override_intent": shift.override_intent.value if shift.override_intent else None,
            "reset_mode": shift.reset_mode,
            "preserve_slots": list(shift.preserve_slots),
            "force_replace_slots": list(shift.force_replace_slots),
            "drop_slot_types": list(shift.drop_slot_types),
            "dialogue_act": shift.dialogue_act,
            "block_recommend": bool(shift.block_recommend),
            "note": shift.note,
        }
        self._dbg(
            "update_state start session_id=%s turn=%d utterance=%r intent=%s",
            session_id, turn_index, user_utterance, intent_enum.name
        )
        self._dbg("before_state=%s", before_summary)
        self._dbg("slots_raw=%s", slots)

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

        # Set current intent before add_turn to keep state transitions deterministic.
        if intent_enum not in {IntentType.SMALL_TALK, IntentType.NO_CLEAR_INTENT}:
            state.current_intent = intent_enum

        state.add_turn(turn, merge_slots=False)
        state.filled_slots = merged_slots

        use_geo = any(
            slot.type == "LOCATION" and slot.value == "__NEARBY__"
            for slot in state.filled_slots.values()
        )
        state.context["use_geo"] = use_geo
        state.context["state_quality"] = state.get_state_quality()
        state.context["slot_conflicts"] = len(state.slot_conflicts)
        state.context["missing_slots"] = state.get_missing_slots()
        state.context["is_complete"] = state.is_complete()
        state.context["state_quality"] = state.get_state_quality()
        state.context["last_merge_logs"] = merge_logs[-10:]
        state.context["last_shift_log"] = state.context.get("last_shift_decision")

        after_summary = state.get_context_summary()
        self._dbg("after_state=%s", after_summary)
        self._dbg(
            "filled_slots_delta before=%s -> after=%s",
            before_summary.get("filled_slots", {}),
            after_summary.get("filled_slots", {}),
        )
        return state
    def _should_suppress_slots(self, intent_enum: IntentType, state: DialogueState) -> bool:
        if intent_enum in {IntentType.SMALL_TALK, IntentType.OUT_OF_SCOPE}:
            return True
        if intent_enum == IntentType.NO_CLEAR_INTENT and state.current_intent is None:
            return True
        return False
    def _validate_slots(self, slots: List[Dict], intent_enum: IntentType, utterance: str, turn_index: int) -> List[Slot]:
        if not self.slot_validator:
            validated = [
                Slot(
                    type=str(s["type"]).upper(),
                    value=str(s["value"]).strip(),
                    confidence=float(s.get("confidence", 0.0)),
                    turn_index=turn_index,
                    source_intent=intent_enum,
                    source_utterance=utterance,
                    is_confirmed=float(s.get("confidence", 0.0)) >= 0.70,
                )
                for s in slots
                if s.get("type") and s.get("value")
            ]
            return [self._normalize_semantic_slot(s) for s in validated]

        result = self.slot_validator.validate(slots)
        validated = [
            Slot(
                type=s.type,
                value=s.value,
                confidence=s.confidence,
                turn_index=turn_index,
                source_intent=intent_enum,
                source_utterance=utterance,
                is_confirmed=s.confidence >= 0.70,
            )
            for s in result.accepted
        ]
        return [self._normalize_semantic_slot(s) for s in validated]

    def _normalize_semantic_slot(self, slot: Slot) -> Slot:
        if slot.type == "LOCATION" and slot.value.strip().lower() in self.VAGUE_LOCATION_VALUES:
            slot.value = "__NEARBY__"
        return slot
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