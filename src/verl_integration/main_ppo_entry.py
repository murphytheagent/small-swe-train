"""Project-local SDPO entrypoint for verl PPO trainer.

This module keeps upstream verl unchanged while giving small-swe-train a stable
hook for runtime registration and process-wide patching.

It also mirrors the FlashAttention2 compatibility guard used by the FSDP SFT
entrypoint so minimal proof runs can fall back to SDPA when `flash_attn` is
missing or ABI-incompatible on the target host.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from transformers import AutoModelForCausalLM

_ORIGINAL_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained
_FLASH_ATTN_DISABLED = False


def _apply_local_runtime_bootstrap() -> None:
    # Propagate runtime patch enablement into spawned worker processes where
    # sitecustomize is imported but this wrapper module is not.
    os.environ.setdefault("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    from small_swe_runtime_patches import apply_small_swe_runtime_patches
    from verl_integration.ppo_runtime_patch import apply_small_swe_sdpo_runtime_patch

    apply_small_swe_runtime_patches()
    apply_small_swe_sdpo_runtime_patch()


def _clear_cached_flash_attn_modules() -> None:
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


def _resolved_attn_implementation() -> str | None:
    value = os.environ.get("SMALL_SWE_SDPO_ATTN_IMPL")
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
    requested = os.environ.get("SMALL_SWE_SDPO_MODEL_DTYPE", "").strip()
    normalized = _normalize_dtype_name(requested) if requested else None
    try:
        import torch
    except Exception:
        return None

    if normalized is not None:
        return getattr(torch, normalized)

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
    from small_swe_runtime_patches import apply_small_swe_runtime_patches

    def _not_available() -> bool:
        return False

    _clear_cached_flash_attn_modules()
    os.environ["SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN"] = "1"
    apply_small_swe_runtime_patches()
    transformers_utils.is_flash_attn_2_available = _not_available
    transformers_import_utils.is_flash_attn_2_available = _not_available
    _FLASH_ATTN_DISABLED = True
    print(
        f"[small-swe] flash-attn disabled for this SDPO run: {reason}",
        file=sys.stderr,
    )


def _ensure_flash_attn_runtime_compatibility() -> None:
    if _coerce_bool_env("SMALL_SWE_DISABLE_FLASH_ATTN", default=False):
        _disable_flash_attn_availability(reason="SMALL_SWE_DISABLE_FLASH_ATTN=1")
        return

    try:
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
            kwargs["dtype"] = model_dtype
    return _call_from_pretrained_with_dtype_fallback(*args, **kwargs)


def _call_from_pretrained_with_dtype_fallback(*args: Any, **kwargs: Any):
    try:
        return _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)
    except TypeError as exc:
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            message = str(exc)
            if "unexpected keyword argument 'dtype'" in message:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                return _ORIGINAL_FROM_PRETRAINED(*args, **fallback_kwargs)
        raise


_apply_local_runtime_bootstrap()
_ensure_flash_attn_runtime_compatibility()
AutoModelForCausalLM.from_pretrained = _patched_from_pretrained

# Register local SDPO agent-loop integrations before trainer startup.
from verl_integration import swe_bridge_agent_loop  # noqa: F401,E402
from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    main()
