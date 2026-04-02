from __future__ import annotations

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
        prompt = self._build_prompt(state_summary, action_space)

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
        action = self._extract_action(generated, action_space)
        return action

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
    def _extract_action(text: str, action_space: List[str]) -> str:
        upper_text = text.upper()

        # Ưu tiên match exact action token
        for action in action_space:
            if re.search(rf"\b{re.escape(action)}\b", upper_text):
                return action

        # fallback
        return "FALLBACK"