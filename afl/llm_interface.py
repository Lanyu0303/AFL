from typing import Any

from afl.local_qwen import LocalQwen


class LLMInterface:
    def __init__(self, args: Any):
        self.temperature = float(getattr(args, "temperature", 0.0))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 256))
        self.top_p = float(getattr(args, "top_p", 1.0))
        model_name = getattr(args, "local_model_name", "Qwen/Qwen2-1.5B-Instruct")
        load_in_4bit = bool(getattr(args, "load_in_4bit", False))
        self.local_qwen = LocalQwen(model_name=model_name, load_in_4bit=load_in_4bit)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.local_qwen.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
        )
