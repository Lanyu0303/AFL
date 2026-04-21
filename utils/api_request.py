from typing import Any, Optional

from utils.local_model import get_local_model, local_inference


def api_request(system_prompt: str, user_prompt: str, args: Any) -> Optional[str]:
    """Backward-compatible wrapper routed to offline local inference."""
    model, tokenizer = get_local_model(args)
    return local_inference(system_prompt, user_prompt, model, tokenizer, args)
