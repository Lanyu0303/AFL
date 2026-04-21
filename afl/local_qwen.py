import logging
from typing import ClassVar, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


class LocalQwen:
    """Local Qwen2 inference helper with shared model/tokenizer cache."""

    _cached: ClassVar[Optional[Tuple[AutoModelForCausalLM, AutoTokenizer, str, bool]]] = None

    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct", load_in_4bit: bool = True):
        if (
            LocalQwen._cached is not None
            and LocalQwen._cached[2] == model_name
            and LocalQwen._cached[3] == load_in_4bit
        ):
            self.model, self.tokenizer = LocalQwen._cached[0], LocalQwen._cached[1]
            return

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()

        LocalQwen._cached = (model, tokenizer, model_name, load_in_4bit)
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_device = self.model.get_input_embeddings().weight.device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            with torch.no_grad():
                do_sample = temperature > 0
                outputs = self.model.generate(
                    **inputs,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:
            LOGGER.warning("Local Qwen generation failed: %s", exc)
            return ""
