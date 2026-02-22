"""Project-local entrypoint for verl FSDP SFT trainer with configurable attention backend.

This keeps upstream verl unchanged while allowing proof runs to bypass hardcoded
FlashAttention2 when `flash_attn` is unavailable on the remote node.
"""

from __future__ import annotations

import os
from typing import Any

from transformers import AutoModelForCausalLM

_ORIGINAL_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained


def _resolved_attn_implementation() -> str | None:
    value = os.environ.get("SMALL_SWE_RFT_ATTN_IMPL")
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _patched_from_pretrained(*args: Any, **kwargs: Any):
    attn_implementation = _resolved_attn_implementation()
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)


AutoModelForCausalLM.from_pretrained = _patched_from_pretrained

from verl.trainer.fsdp_sft_trainer import main  # noqa: E402


if __name__ == "__main__":
    main()
