"""Process-wide runtime patches for small-swe-train."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
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


def _install_flash_attn_import_guard() -> None:
    current = builtins.__import__
    if getattr(current, "__name__", "") == "_small_swe_guarded_import":
        return

    original_import = current

    def _small_swe_guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "flash_attn" or name.startswith("flash_attn."):
            raise ModuleNotFoundError(
                "No module named 'flash_attn' (hidden by SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN)"
            )
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _small_swe_guarded_import


def _clear_cached_flash_attn_modules() -> None:
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


def _try_apply_sdpo_runtime_patch() -> None:
    ray_trainer_module = sys.modules.get("verl.trainer.ppo.ray_trainer")
    if ray_trainer_module is None:
        return
    # Avoid noisy false-negative warnings during partially-initialized imports.
    # The runtime patch only becomes meaningful once RayPPOTrainer exists.
    if getattr(ray_trainer_module, "RayPPOTrainer", None) is None:
        return
    try:
        from verl_integration.ppo_runtime_patch import apply_small_swe_sdpo_runtime_patch
    except Exception:
        return
    try:
        apply_small_swe_sdpo_runtime_patch(ray_trainer_module)
    except Exception:
        return


def _install_self_distillation_config_compat_patch() -> None:
    """Accept small-swe-only SDPO keys on older verl SelfDistillationConfig."""
    try:
        from verl.workers.config.actor import SelfDistillationConfig
    except Exception:
        return

    # Newer verl versions may already expose this field natively.
    dataclass_fields = getattr(SelfDistillationConfig, "__dataclass_fields__", {})
    if isinstance(dataclass_fields, dict) and "num_recent_raw_blocks" in dataclass_fields:
        return

    if getattr(SelfDistillationConfig, "_small_swe_num_recent_raw_blocks_compat", False):
        return

    original_init = SelfDistillationConfig.__init__

    def _small_swe_self_distillation_init(self, *args, **kwargs):
        raw_num_recent_raw_blocks = kwargs.pop("num_recent_raw_blocks", None)
        original_init(self, *args, **kwargs)
        value = 3 if raw_num_recent_raw_blocks is None else raw_num_recent_raw_blocks
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = 3
        normalized = max(normalized, 0)
        # BaseConfig allows setting new fields once on frozen configs.
        setattr(self, "num_recent_raw_blocks", normalized)

    _small_swe_self_distillation_init.__name__ = "_small_swe_self_distillation_init"
    SelfDistillationConfig.__init__ = _small_swe_self_distillation_init
    setattr(SelfDistillationConfig, "_small_swe_num_recent_raw_blocks_compat", True)


def _install_sdpo_runtime_patch_import_guard() -> None:
    current = builtins.__import__
    if getattr(current, "__name__", "") == "_small_swe_sdpo_guarded_import":
        return

    original_import = current

    def _small_swe_sdpo_guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        module = original_import(name, globals, locals, fromlist, level)
        if (
            name == "verl.trainer.ppo.ray_trainer"
            or name.startswith("verl.trainer.ppo.ray_trainer.")
            or (
                name.startswith("verl.trainer.")
                and "verl.trainer.ppo.ray_trainer" in sys.modules
            )
        ):
            _try_apply_sdpo_runtime_patch()
        return module

    builtins.__import__ = _small_swe_sdpo_guarded_import


def apply_small_swe_runtime_patches() -> None:
    if _coerce_bool_env("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", default=False):
        _clear_cached_flash_attn_modules()
        _install_flash_attn_find_spec_guard()
        _install_flash_attn_import_guard()
    if _coerce_bool_env("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", default=False):
        _install_self_distillation_config_compat_patch()
        # Ray worker processes do not enter our main wrapper module.
        # Patch lazily once ray_trainer is imported in-process.
        _install_sdpo_runtime_patch_import_guard()
        _try_apply_sdpo_runtime_patch()
