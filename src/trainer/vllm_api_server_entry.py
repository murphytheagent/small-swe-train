"""vLLM OpenAI server entrypoint with guardrails for broken external flash-attn wheels.

Grounding:
- vLLM OpenAI-compatible server entrypoint module:
  https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
  https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
"""

from __future__ import annotations

import os
import runpy
import sys


def _coerce_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "on"}


def _clear_cached_flash_attn_modules() -> None:
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


def _probe_external_flash_attn() -> tuple[bool, str]:
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_interface  # noqa: F401
    except Exception as exc:
        _clear_cached_flash_attn_modules()
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _should_hide_external_flash_attn() -> tuple[bool, str]:
    if _coerce_bool_env("SMALL_SWE_DISABLE_FLASH_ATTN", default=False):
        return True, "SMALL_SWE_DISABLE_FLASH_ATTN=1"

    is_usable, reason = _probe_external_flash_attn()
    if is_usable:
        return False, ""
    return True, reason


def _configure_flash_attn_guard_if_needed() -> None:
    hide_external, reason = _should_hide_external_flash_attn()
    if not hide_external:
        return
    _clear_cached_flash_attn_modules()
    os.environ["SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN"] = "1"
    from sitecustomize import apply_small_swe_runtime_patches

    apply_small_swe_runtime_patches()
    print(
        "[small-swe] external flash-attn hidden for vLLM server launch "
        f"(reason: {reason})",
        file=sys.stderr,
    )


def main() -> None:
    _configure_flash_attn_guard_if_needed()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
