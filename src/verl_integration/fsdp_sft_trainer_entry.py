"""Project-local entrypoint for verl FSDP SFT trainer with configurable attention backend.

This keeps upstream verl unchanged while allowing proof runs to bypass hardcoded
FlashAttention2 when `flash_attn` is unavailable on the remote node.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from transformers import AutoModelForCausalLM

_ORIGINAL_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained
_FLASH_ATTN_DISABLED = False


def _clear_cached_flash_attn_modules() -> None:
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


def _resolved_attn_implementation() -> str | None:
    value = os.environ.get("SMALL_SWE_RFT_ATTN_IMPL")
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _normalize_dtype_name(value: str) -> str | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    return aliases.get(normalized)


def _resolved_model_dtype():
    """Resolve the model load dtype used for FlashAttention2 compatibility."""
    requested = os.environ.get("SMALL_SWE_RFT_MODEL_DTYPE", "").strip()
    normalized = _normalize_dtype_name(requested) if requested else None
    try:
        import torch
    except Exception:
        return None

    if normalized is not None:
        return getattr(torch, normalized)

    # When FlashAttention2 is active and dtype is unspecified, default to AMP-safe
    # precision instead of float32 to avoid runtime warnings and fallback behavior.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _coerce_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "on"}


def _disable_flash_attn_availability(*, reason: str) -> None:
    global _FLASH_ATTN_DISABLED
    from transformers import utils as transformers_utils
    from transformers.utils import import_utils as transformers_import_utils
    from sitecustomize import apply_small_swe_runtime_patches

    def _not_available() -> bool:
        return False

    _clear_cached_flash_attn_modules()
    os.environ["SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN"] = "1"
    apply_small_swe_runtime_patches()
    transformers_utils.is_flash_attn_2_available = _not_available
    transformers_import_utils.is_flash_attn_2_available = _not_available
    _FLASH_ATTN_DISABLED = True
    print(
        f"[small-swe] flash-attn disabled for this run: {reason}",
        file=sys.stderr,
    )


def _ensure_flash_attn_runtime_compatibility() -> None:
    """Guard against broken flash-attn wheel/torch ABI mismatches."""
    if _coerce_bool_env("SMALL_SWE_DISABLE_FLASH_ATTN", default=False):
        _disable_flash_attn_availability(reason="SMALL_SWE_DISABLE_FLASH_ATTN=1")
        return

    try:
        # Force loading the CUDA extension early so import-time ABI issues are
        # detected before verl imports model modules that assume flash-attn works.
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_interface  # noqa: F401
    except Exception as exc:
        _disable_flash_attn_availability(reason=f"{type(exc).__name__}: {exc}")


def _patched_from_pretrained(*args: Any, **kwargs: Any):
    attn_implementation = _resolved_attn_implementation()
    current_attn_impl = str(kwargs.get("attn_implementation", "")).strip().lower()
    if _FLASH_ATTN_DISABLED:
        fallback = os.environ.get("SMALL_SWE_FALLBACK_ATTN_IMPL", "sdpa").strip()
        if (
            attn_implementation is None
            and fallback
            and (not current_attn_impl or current_attn_impl == "flash_attention_2")
        ):
            attn_implementation = fallback
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    effective_attn_impl = str(kwargs.get("attn_implementation", "")).strip().lower()
    if (
        "torch_dtype" not in kwargs
        and "dtype" not in kwargs
        and effective_attn_impl in {"", "flash_attention_2"}
    ):
        model_dtype = _resolved_model_dtype()
        if model_dtype is not None:
            kwargs["torch_dtype"] = model_dtype
    return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)


_ensure_flash_attn_runtime_compatibility()
AutoModelForCausalLM.from_pretrained = _patched_from_pretrained

from verl.trainer.fsdp_sft_trainer import main  # noqa: E402


if __name__ == "__main__":
    main()
