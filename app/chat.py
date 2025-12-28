import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod
import requests
import json
import re

class ChatGenerator(ABC):    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]],  enable_thinking: bool, max_new_tokens: int):
        pass

class QwenLocalGenerator(ChatGenerator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
        )

    def _split_thinking(self, text: str) -> Tuple[str, str]:
        """
        Extracts <think>...</think> if present.
        Never assumes it exists.
        """

        if "<think>" in text and "</think>" in text:
            thinking = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            content = text.split("</think>", 1)[1].strip()
            return thinking, content

        # No thinking block found
        return "", text

    def generate(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        max_new_tokens: int = 32768,
    ) -> Tuple[str, str]:
        """
        Returns:
            thinking: str
            content: str
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        raw_output = self.tokenizer.decode(
            ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        thinking, content = self._split_thinking(raw_output)
        return thinking, content


class MistralGenerator(ChatGenerator):
    def __init__(self, api_key: str, temperature: float = 0.1, model="mistralai/mistral-small-3.1-24b-instruct:free"):
        self.api_key = api_key
        self.temperature = temperature
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        max_new_tokens: int = 1400,
    ) -> Tuple[str, str]:

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                # "model": "deepseek/deepseek-r1-0528:free",
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": self.temperature,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]

        return msg.get("reasoning", ""), msg.get("content", "")

class QwenGenerator(ChatGenerator):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        max_new_tokens: int = 300,
    ) -> Tuple[str, str]:

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "qwen/qwen3-235b-a22b-2507",
                "messages": messages,
                "max_tokens": max_new_tokens,
              }),
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        print(data)
        msg = data["choices"][0]["message"]

        # Qwen does NOT return a `reasoning` field
        return "", msg.get("content", "")


class DeepSeekGenerator(ChatGenerator):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, str]:

        payload = {
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.1,  # IMPORTANT for faithfulness
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]

        # DeepSeek-R1 exposes chain-of-thought as `reasoning`
        reasoning = msg.get("reasoning", "") if enable_thinking else ""
        content = msg.get("content", "")

        return reasoning, content.strip()