from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceLLMPolicy:
    """
    Local LLM policy (HF Transformers), phù hợp fallback cho Hybrid Policy.
    Khuyến nghị:
      - Qwen/Qwen2.5-7B-Instruct (GPU >= 16GB)
      - Qwen/Qwen2.5-3B-Instruct (GPU thấp hơn)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",  # "auto" | "cuda" | "cpu"
        torch_dtype: str = "auto",  # "auto" | "float16" | "float32"
        max_new_tokens: int = 12,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        dtype = "auto"
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32

        device_map = "auto" if device == "auto" else {"": device}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def decide_action(self, state_summary: Dict[str, Any], action_space: List[str]) -> str:
        decision = self.decide_decision(state_summary, action_space)
        return str(decision.get("action", "FALLBACK")).upper()

    def decide_decision(self, state_summary: Dict[str, Any], action_space: List[str]) -> Dict[str, Any]:
        prompt = self._build_structured_prompt(state_summary, action_space)

        with torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        parsed = self._extract_decision(generated, action_space)
        return parsed

    @staticmethod
    def _build_prompt(state_summary: Dict[str, Any], action_space: List[str]) -> str:
        return (
            "Bạn là dialogue policy cho chatbot gợi ý quán ăn.\n"
            "Chỉ chọn 1 hành động hợp lệ.\n\n"
            f"State: {state_summary}\n"
            f"Action hợp lệ: {', '.join(action_space)}\n\n"
            "Quy tắc trả lời:\n"
            "- Chỉ trả về đúng 1 token action\n"
            "- Không giải thích\n\n"
            "Action:"
        )

    @staticmethod
    def _build_structured_prompt(state_summary: Dict[str, Any], action_space: List[str]) -> str:
        return (
            "Bạn là bộ quyết định action cho chatbot gợi ý quán ăn.\n"
            "Bạn PHẢI trả về duy nhất 1 JSON object hợp lệ, không thêm text khác.\n"
            "Schema bắt buộc:\n"
            "{\n"
            '  "action": "ASK_SLOT|CLARIFY|CONFIRM|RECOMMEND|RESPOND|FALLBACK",\n'
            '  "slot": "DISH|LOCATION|PRICE|null",\n'
            '  "response": "string",\n'
            '  "next_action": "RECOMMEND|null",\n'
            '  "reason": "string"\n'
            "}\n\n"
            f"State: {state_summary}\n"
            f"Action hop le: {', '.join(action_space)}\n"
            "Rang buoc:\n"
            "- Neu missing_slots khong rong thi khong duoc RECOMMEND.\n"
            "- Neu action la ASK_SLOT/CLARIFY thi uu tien slot dang thieu.\n"
            "- Neu state da du thong tin nhung can chac chan, co the tra ve CONFIRM va next_action='RECOMMEND'.\n"
            "- response phai cu the theo context, khong duoc generic.\n"
            "JSON:"
        )

    @staticmethod
    def _extract_action(text: str, action_space: List[str]) -> str:
        upper_text = text.upper()

        # Ưu tiên match exact action token
        for action in action_space:
            if re.search(rf"\b{re.escape(action)}\b", upper_text):
                return action

        # fallback
        return "FALLBACK"

    @staticmethod
    def _extract_decision(text: str, action_space: List[str]) -> Dict[str, Any]:
        raw = (text or "").strip()
        candidate = raw

        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            candidate = m.group(0)

        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        action = str(data.get("action", "") or "").upper().strip()
        if action not in action_space:
            action = HuggingFaceLLMPolicy._extract_action(raw, action_space)

        slot = data.get("slot")
        if slot is not None:
            slot = str(slot).upper().strip()
            if slot in {"", "NONE", "NULL"}:
                slot = None

        response = str(data.get("response", "") or "").strip()
        next_action = data.get("next_action")
        if next_action is not None:
            next_action = str(next_action).upper().strip()
            if next_action in {"", "NONE", "NULL"}:
                next_action = None

        reason = str(data.get("reason", "") or "").strip()
        return {
            "action": action,
            "slot": slot,
            "response": response,
            "next_action": next_action,
            "reason": reason,
        }