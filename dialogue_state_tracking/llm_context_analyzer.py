from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLMContextAnalyzer:
    """
    Dùng LLM để hiểu semantic context + intent thật sự của user.
    Phân tích:
      1. User intent thật sự (đổi món, hủy, tiếp tục, v.v.)
      2. Slots nào là "negation" (không muốn), slots nào là "positive" (muốn)
      3. Có nên reset state hay không
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_tokens: int = 150,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        dtype = "auto"
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32

        device_map = "auto" if device == "auto" else {"": device}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("[LLMContextAnalyzer] loaded model=%s", model_name)

    def analyze_user_intent(
        self,
        user_utterance: str,
        current_intent: Optional[str],
        filled_slots: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Phân tích ý định thật sự của user.

        Returns:
            {
                "real_intent": "change_dish" | "cancel" | "continue" | "clarify",
                "negated_slots": ["DISH"],  # slots user KHÔNG muốn
                "positive_slots": ["DISH"],  # slots user muốn
                "confidence": 0.9,
                "reasoning": "User nói không muốn ăn phở nữa, muốn ăn thịt chó"
            }
        """
        prompt = self._build_intent_analysis_prompt(
            user_utterance=user_utterance,
            current_intent=current_intent,
            filled_slots=filled_slots,
        )

        result = self._call_llm(prompt)
        return self._parse_intent_analysis(result, user_utterance)

    def analyze_slot_sentiment(
        self,
        user_utterance: str,
        extracted_slots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Phân tích xem slot nào là positive (user muốn), negative (user không muốn).

        Returns:
            {
                "slot_sentiment": {
                    "DISH": {
                        "phở": "negative",  # user nói "không muốn ăn phở"
                        "thịt chó": "positive"  # user nói "muốn ăn thịt chó"
                    }
                },
                "confidence_per_slot": {...}
            }
        """
        prompt = self._build_slot_sentiment_prompt(user_utterance, extracted_slots)
        result = self._call_llm(prompt)
        return self._parse_slot_sentiment(result)

    def should_reset_state(
        self,
        user_utterance: str,
        current_intent: Optional[str],
        filled_slots: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Quyết định có nên reset state hay không dựa trên semantic understanding.

        Returns:
            {
                "should_reset": True | False,
                "reason": "user_cancel" | "intent_change" | "slot_update" | "none",
                "preserve_slots": ["LOCATION"],
                "confidence": 0.95
            }
        """
        prompt = self._build_reset_decision_prompt(
            user_utterance=user_utterance,
            current_intent=current_intent,
            filled_slots=filled_slots,
        )
        result = self._call_llm(prompt)
        return self._parse_reset_decision(result)

    # ============ PRIVATE METHODS ============

    def _build_intent_analysis_prompt(
        self,
        user_utterance: str,
        current_intent: Optional[str],
        filled_slots: Dict[str, str],
    ) -> str:
        slots_str = ", ".join([f"{k}={v}" for k, v in filled_slots.items()]) or "none"
        return (
            "Bạn là trợ lý phân tích ý định người dùng trong chatbot gợi ý quán ăn.\n\n"
            f"Tình huống:\n"
            f"- User nói: {user_utterance}\n"
            f"- Intent hiện tại: {current_intent or 'none'}\n"
            f"- Slots đã có: {slots_str}\n\n"
            "Hãy phân tích:\n"
            "1. Real intent của user (change_dish | cancel | continue | clarify)\n"
            "2. Slots nào user muốn (positive)\n"
            "3. Slots nào user KHÔNG muốn (negative)\n"
            "4. Lý do\n\n"
            "Format JSON:\n"
            '{"real_intent": "...", "positive_slots": [...], '
            '"negated_slots": [...], "confidence": 0.0-1.0, "reasoning": "..."}'
        )

    def _build_slot_sentiment_prompt(
        self,
        user_utterance: str,
        extracted_slots: List[Dict[str, Any]],
    ) -> str:
        slots_str = json.dumps(extracted_slots, ensure_ascii=False)
        return (
            "Bạn là chuyên gia phân tích sentiment của slots trong hội thoại.\n\n"
            f"User nói: {user_utterance}\n"
            f"Extracted slots: {slots_str}\n\n"
            "Hãy phân tích:\n"
            "- Slot nào user muốn (positive)?\n"
            "- Slot nào user KHÔNG muốn (negative)?\n"
            "- Confidence của mỗi phán đoán?\n\n"
            "Format JSON:\n"
            '{"slot_sentiment": {"DISH": {"phở": "negative", "thịt chó": "positive"}}, '
            '"confidence_per_slot": {...}}'
        )

    def _build_reset_decision_prompt(
        self,
        user_utterance: str,
        current_intent: Optional[str],
        filled_slots: Dict[str, str],
    ) -> str:
        slots_str = ", ".join([f"{k}={v}" for k, v in filled_slots.items()]) or "none"
        return (
            "Bạn là trợ lý quyết định khi nào nên xoá slots cũ (reset state).\n\n"
            f"Tình huống:\n"
            f"- User nói: {user_utterance}\n"
            f"- Intent: {current_intent or 'none'}\n"
            f"- Slots cũ: {slots_str}\n\n"
            "Hãy quyết định:\n"
            "1. Có nên reset slots hay không?\n"
            "2. Lý do?\n"
            "3. Slots nếu có nên giữ lại?\n\n"
            "Format JSON:\n"
            '{"should_reset": true/false, "reason": "...", '
            '"preserve_slots": [...], "confidence": 0.0-1.0}'
        )

    def _call_llm(self, prompt: str) -> str:
        with torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,  # deterministic
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    @staticmethod
    def _parse_intent_analysis(result: str, user_utterance: str) -> Dict[str, Any]:
        try:
            # tìm JSON trong result
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "real_intent": data.get("real_intent", "continue"),
                    "negative_slots": set(data.get("negated_slots", [])),
                    "positive_slots": set(data.get("positive_slots", [])),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", ""),
                }
        except Exception as e:
            logger.warning("[LLMContextAnalyzer] parse intent failed: %s", e)

        return {
            "real_intent": "continue",
            "negative_slots": set(),
            "positive_slots": set(),
            "confidence": 0.0,
            "reasoning": f"parse failed: {result}",
        }

    @staticmethod
    def _parse_slot_sentiment(result: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "slot_sentiment": data.get("slot_sentiment", {}),
                    "confidence_per_slot": data.get("confidence_per_slot", {}),
                }
        except Exception as e:
            logger.warning("[LLMContextAnalyzer] parse slot sentiment failed: %s", e)

        return {"slot_sentiment": {}, "confidence_per_slot": {}}

    @staticmethod
    def _parse_reset_decision(result: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "should_reset": data.get("should_reset", False),
                    "reason": data.get("reason", ""),
                    "preserve_slots": data.get("preserve_slots", []),
                    "confidence": float(data.get("confidence", 0.5)),
                }
        except Exception as e:
            logger.warning("[LLMContextAnalyzer] parse reset decision failed: %s", e)

        return {
            "should_reset": False,
            "reason": "parse failed",
            "preserve_slots": [],
            "confidence": 0.0,
        }