import logging
import os
import random
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def _get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _resolve_prompt(system_prompt: str, user_prompt: str, args: Any) -> str:
    system_prompt = (system_prompt or "").strip()
    user_prompt = (user_prompt or "").strip()
    prompt_style = str(_get_arg(args, "prompt_style", "llama2_chat")).lower()

    if prompt_style == "llama2_chat":
        return (
            "[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_prompt}\n"
            "[/INST]"
        )

    return f"{system_prompt}\n\n{user_prompt}".strip()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _attach_llara_checkpoint(model: AutoModelForCausalLM, llara_ckpt_path: str) -> AutoModelForCausalLM:
    if not llara_ckpt_path:
        return model

    if os.path.isdir(llara_ckpt_path) and os.path.exists(os.path.join(llara_ckpt_path, "adapter_config.json")):
        try:
            from peft import PeftModel

            LOGGER.info("Loading LLaRA LoRA adapter from %s", llara_ckpt_path)
            return PeftModel.from_pretrained(model, llara_ckpt_path)
        except Exception as exc:
            LOGGER.warning("PEFT adapter load failed (%s), falling back to state_dict merge", exc)

    checkpoint = torch.load(llara_ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    LOGGER.info(
        "Loaded checkpoint %s via state_dict (missing=%d, unexpected=%d)",
        llara_ckpt_path,
        len(missing_keys),
        len(unexpected_keys),
    )
    return model


@lru_cache(maxsize=2)
def _load_local_model_cached(
    llm_path: str,
    llara_ckpt_path: str,
    use_4bit: bool,
    use_8bit: bool,
    dtype_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not llm_path:
        raise ValueError("`llm_path` is required for local inference.")
    if use_4bit and use_8bit:
        raise ValueError("Only one of `use_4bit` or `use_8bit` can be true.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_name.lower(), torch.float16)

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_4bit:
        model_kwargs["load_in_4bit"] = True
    elif use_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, **model_kwargs)
    model = _attach_llara_checkpoint(model, llara_ckpt_path)
    model.eval()
    return model, tokenizer


def get_local_model(args: Any) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    return _load_local_model_cached(
        llm_path=_get_arg(args, "llm_path", ""),
        llara_ckpt_path=_get_arg(args, "llara_ckpt_path", ""),
        use_4bit=bool(_get_arg(args, "load_in_4bit", False)),
        use_8bit=bool(_get_arg(args, "load_in_8bit", False)),
        dtype_name=str(_get_arg(args, "torch_dtype", "float16")),
    )


def local_inference(
    system_prompt: str,
    user_prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: Any,
) -> Optional[str]:
    prompt = _resolve_prompt(system_prompt, user_prompt, args)

    seed = int(_get_arg(args, "seed", 333))
    _set_seed(seed)

    max_new_tokens = int(_get_arg(args, "max_new_tokens", 256))
    top_p = float(_get_arg(args, "top_p", 1.0))
    temperature = float(_get_arg(args, "temperature", 0.0))
    do_sample = temperature > 0

    encoded = tokenizer(prompt, return_tensors="pt")
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device
    encoded = {k: v.to(input_device) for k, v in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 1e-5) if do_sample else 1.0,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    current_tokens = max_new_tokens
    for _ in range(3):
        try:
            with torch.no_grad():
                output_ids = model.generate(**encoded, **generation_kwargs)
            prompt_len = encoded["input_ids"].shape[1]
            generated_ids = output_ids[0][prompt_len:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_tokens = max(32, current_tokens // 2)
            generation_kwargs["max_new_tokens"] = current_tokens
            LOGGER.warning("CUDA OOM during generation. Retrying with max_new_tokens=%s", current_tokens)
        except Exception as exc:
            LOGGER.exception("Local inference failed: %s", exc)
            return None

    LOGGER.error("Local inference failed after OOM retries.")
    return None
