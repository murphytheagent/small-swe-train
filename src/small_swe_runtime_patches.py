"""Process-wide runtime patches for small-swe-train."""

from __future__ import annotations

import builtins
import importlib.util
import json
import logging
import os
import sys
import warnings
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

_MISTRAL_MODEL_TYPES = {
    "mistral",
    "mistral3",
    "voxtral",
    "ministral",
    "pixtral",
}
_TORCH_DTYPE_DEPRECATION_MESSAGE = "`torch_dtype` is deprecated! Use `dtype` instead!"


class _TorchDtypeDeprecationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return _TORCH_DTYPE_DEPRECATION_MESSAGE not in message


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


def _resolve_local_rank_from_env() -> int | None:
    raw_rank = os.environ.get("RANK")
    if raw_rank is None:
        return None
    try:
        rank = int(raw_rank)
    except (TypeError, ValueError):
        return None

    raw_local_world_size = os.environ.get("RAY_LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE")
    try:
        local_world_size = int(raw_local_world_size) if raw_local_world_size is not None else 1
    except (TypeError, ValueError):
        local_world_size = 1
    if local_world_size <= 0:
        local_world_size = 1
    return rank % local_world_size


def _install_ray_worker_local_rank_device_patch() -> None:
    """Ensure Ray no-set mode binds each worker to its deterministic local rank."""
    try:
        from verl.single_controller.base.worker import Worker
        from verl.utils.device import get_torch_device
        from verl.utils.ray_utils import ray_noset_visible_devices
    except Exception:
        return

    if getattr(Worker, "_small_swe_local_rank_device_patch", False):
        return

    original_setup = Worker._setup_env_cuda_visible_devices

    def _small_swe_setup_env_cuda_visible_devices(self):
        original_setup(self)
        if not ray_noset_visible_devices():
            return

        local_rank = _resolve_local_rank_from_env()
        if local_rank is None:
            return

        os.environ["LOCAL_RANK"] = str(local_rank)
        try:
            get_torch_device().set_device(local_rank)
        except Exception:
            return

    _small_swe_setup_env_cuda_visible_devices.__name__ = "_small_swe_setup_env_cuda_visible_devices"
    Worker._setup_env_cuda_visible_devices = _small_swe_setup_env_cuda_visible_devices
    setattr(Worker, "_small_swe_local_rank_device_patch", True)


def _resolve_model_type_from_local_config(model_path: Any) -> str | None:
    if not isinstance(model_path, (str, os.PathLike)):
        return None

    raw_value = os.fspath(model_path)
    if not raw_value.strip():
        return None

    path = Path(raw_value).expanduser()
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_type = payload.get("model_type")
    if not isinstance(model_type, str):
        return None

    normalized = model_type.strip().lower()
    return normalized or None


def _resolve_fix_mistral_regex_default(model_path: Any) -> bool | None:
    model_type = _resolve_model_type_from_local_config(model_path)
    if model_type is None:
        return None
    return model_type in _MISTRAL_MODEL_TYPES


def _suppress_fast_tokenizer_pad_warning(tokenizer: Any) -> None:
    deprecation_warnings = getattr(tokenizer, "deprecation_warnings", None)
    if deprecation_warnings is None:
        try:
            setattr(tokenizer, "deprecation_warnings", {})
            deprecation_warnings = getattr(tokenizer, "deprecation_warnings", None)
        except Exception:
            return

    if not isinstance(deprecation_warnings, dict):
        return

    deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


def _install_verl_tokenizer_compat_patches() -> None:
    try:
        from verl.utils import tokenizer as tokenizer_module
    except Exception:
        return

    if getattr(tokenizer_module, "_small_swe_tokenizer_compat_patch", False):
        return

    original_hf_tokenizer = tokenizer_module.hf_tokenizer
    original_hf_processor = tokenizer_module.hf_processor

    def _small_swe_hf_tokenizer(name_or_path, *args, **kwargs):
        tokenizer_kwargs = dict(kwargs)
        if "fix_mistral_regex" not in tokenizer_kwargs:
            # Some merged checkpoints preserve a non-Mistral model_type in
            # config.json while shipping a tokenizer that still needs the
            # regex fix. Prefer the safe default unless explicitly disabled.
            force_fix_mistral_regex = _coerce_bool_env(
                "SMALL_SWE_FORCE_FIX_MISTRAL_REGEX",
                default=True,
            )
            if force_fix_mistral_regex:
                tokenizer_kwargs["fix_mistral_regex"] = True
            else:
                default_fix_flag = _resolve_fix_mistral_regex_default(name_or_path)
                if default_fix_flag is not None:
                    tokenizer_kwargs["fix_mistral_regex"] = default_fix_flag
        try:
            tokenizer = original_hf_tokenizer(name_or_path, *args, **tokenizer_kwargs)
        except TypeError as exc:
            if (
                "fix_mistral_regex" in tokenizer_kwargs
                and "fix_mistral_regex" in str(exc)
            ):
                tokenizer_kwargs.pop("fix_mistral_regex", None)
                tokenizer = original_hf_tokenizer(name_or_path, *args, **tokenizer_kwargs)
            else:
                raise
        _suppress_fast_tokenizer_pad_warning(tokenizer)
        return tokenizer

    def _small_swe_hf_processor(name_or_path, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Failed to create processor: Unsupported processor type: .*Tokenizer.*",
                category=UserWarning,
            )
            return original_hf_processor(name_or_path, *args, **kwargs)

    _small_swe_hf_tokenizer.__name__ = "_small_swe_hf_tokenizer"
    _small_swe_hf_processor.__name__ = "_small_swe_hf_processor"
    tokenizer_module.hf_tokenizer = _small_swe_hf_tokenizer
    tokenizer_module.hf_processor = _small_swe_hf_processor
    setattr(tokenizer_module, "_small_swe_tokenizer_compat_patch", True)

    verl_utils_module = sys.modules.get("verl.utils")
    if verl_utils_module is not None:
        setattr(verl_utils_module, "hf_tokenizer", tokenizer_module.hf_tokenizer)
        setattr(verl_utils_module, "hf_processor", tokenizer_module.hf_processor)


def _install_transformers_torch_dtype_warning_filter() -> None:
    logger_names = (
        "transformers.configuration_utils",
        "transformers.modeling_utils",
        "transformers.pipelines",
    )
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        if getattr(logger, "_small_swe_torch_dtype_filter_installed", False):
            continue
        logger.addFilter(_TorchDtypeDeprecationFilter())
        setattr(logger, "_small_swe_torch_dtype_filter_installed", True)


def _install_transformers_torch_dtype_property_patch() -> None:
    try:
        from transformers.configuration_utils import PretrainedConfig
    except Exception:
        return

    if getattr(PretrainedConfig, "_small_swe_torch_dtype_property_patch", False):
        return

    current_property = getattr(PretrainedConfig, "torch_dtype", None)
    if not isinstance(current_property, property):
        return

    def _small_swe_get_torch_dtype(self):
        return self.dtype

    def _small_swe_set_torch_dtype(self, value):
        self.dtype = value

    PretrainedConfig.torch_dtype = property(
        _small_swe_get_torch_dtype,
        _small_swe_set_torch_dtype,
        current_property.fdel,
        current_property.__doc__,
    )
    setattr(PretrainedConfig, "_small_swe_torch_dtype_property_patch", True)


def _install_known_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Torch profiler tool config is not fully supported now\.",
        category=UserWarning,
    )


def apply_small_swe_runtime_patches() -> None:
    if _coerce_bool_env("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", default=False):
        _clear_cached_flash_attn_modules()
        _install_flash_attn_find_spec_guard()
        _install_flash_attn_import_guard()
    if _coerce_bool_env("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", default=False):
        _install_self_distillation_config_compat_patch()
        _install_ray_worker_local_rank_device_patch()
        _install_verl_tokenizer_compat_patches()
        _install_transformers_torch_dtype_property_patch()
        _install_transformers_torch_dtype_warning_filter()
        _install_known_warning_filters()
        # Ray worker processes do not enter our main wrapper module.
        # Patch lazily once ray_trainer is imported in-process.
        _install_sdpo_runtime_patch_import_guard()
        _try_apply_sdpo_runtime_patch()
