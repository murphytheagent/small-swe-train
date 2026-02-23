"""Process-wide runtime patches for small-swe-train.

Python imports this module automatically at interpreter startup when it is
present on `PYTHONPATH`. Patches are opt-in through environment variables.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from importlib.machinery import ModuleSpec


def _coerce_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "on"}


def _install_flash_attn_find_spec_guard() -> None:
    current = importlib.util.find_spec
    if getattr(current, "__name__", "") == "_small_swe_guarded_find_spec":
        return

    original_find_spec: Callable[[str, str | None], ModuleSpec | None] = current

    def _small_swe_guarded_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = _small_swe_guarded_find_spec


def apply_small_swe_runtime_patches() -> None:
    if _coerce_bool_env("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", default=False):
        _install_flash_attn_find_spec_guard()


apply_small_swe_runtime_patches()
