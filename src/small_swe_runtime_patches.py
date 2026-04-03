"""Process-wide runtime patches for small-swe-train."""

from __future__ import annotations

import builtins
import math
import importlib.util
import json
import logging
import os
import sys
import threading
import warnings
from collections.abc import Callable, Mapping
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised in train runtime
    import torch
except ModuleNotFoundError:  # pragma: no cover - unit-test environments without train deps
    torch = None  # type: ignore[assignment]

_MISTRAL_MODEL_TYPES = {
    "mistral",
    "mistral3",
    "voxtral",
    "ministral",
    "pixtral",
}
_TORCH_DTYPE_DEPRECATION_MESSAGE = "`torch_dtype` is deprecated! Use `dtype` instead!"
_TURN_SUPERVISION_NEXT = "next_turn"
_TURN_SUPERVISION_CURRENT = "current_turn"
_TURN_SUPERVISION_MODES = {_TURN_SUPERVISION_NEXT, _TURN_SUPERVISION_CURRENT}
_WANDB_EXTRA_KEYS_ENV = "SMALL_SWE_WANDB_EXTRA_KEYS"
_WANDB_EXTRA_PREFIXES_ENV = "SMALL_SWE_WANDB_EXTRA_PREFIXES"
_WANDB_ESSENTIAL_FILTER_ENV = "SMALL_SWE_WANDB_FILTER_ESSENTIALS"

_WANDB_ESSENTIAL_EXACT_KEYS = {
    "training/global_step",
    "training/epoch",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/entropy",
    "actor/grad_norm",
    "actor/lr",
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "critic/rewards/mean",
    "critic/rewards/max",
    "critic/rewards/min",
    "global_seqlen/min",
    "global_seqlen/max",
    "global_seqlen/mean",
    "global_seqlen/balanced_min",
    "global_seqlen/balanced_max",
    "prompt_length/mean",
    "prompt_length/max",
    "prompt_length/min",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "response_length_non_aborted/mean",
    "response_length_non_aborted/max",
    "response_length_non_aborted/min",
    "response/aborted_ratio",
    "num_turns/mean",
    "num_turns/max",
    "num_turns/min",
    "self_distillation/empty_target_batch",
    "self_distillation/active_turn_pairs_in_micro_batch",
    "self_distillation/turn_pair_count_per_sample",
    "self_distillation/success_sample_fraction",
    "self_distillation/feedback_available_fraction",
    "self_distillation/reprompt_sample_fraction",
    "self_distillation/prompt_truncated_fraction",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "rollout_corr/training_ppl",
    "rollout_corr/rollout_ppl",
    "rollout_corr/log_ppl_diff",
    "rollout_corr/log_ppl_abs_diff",
    "rollout_corr/ppl_ratio",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_max",
    "training/rollout_probs_diff_std",
    "training/rollout_actor_probs_pearson_corr",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/step",
    "timing_per_token_ms/gen",
    "timing_per_token_ms/update_actor",
    "perf/time_per_step",
    "perf/throughput",
    "perf/total_num_tokens",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
}

_WANDB_ESSENTIAL_PREFIXES = (
    "val-aux/num_turns/",
)
_VERIFIER_FEEDBACK_NONE = "none"
_VERIFIER_FEEDBACK_FINAL_TURN_ONLY = "final_turn_only"
_VERIFIER_FEEDBACK_ALL_TURNS = "all_turns"
_VERIFIER_FEEDBACK_MODES = {
    _VERIFIER_FEEDBACK_NONE,
    _VERIFIER_FEEDBACK_FINAL_TURN_ONLY,
    _VERIFIER_FEEDBACK_ALL_TURNS,
}
_LEGACY_GATING_RESOLVED_ONLY = "resolved_only"
_LEGACY_GATING_FEEDBACK_PRESENT = "feedback_present"
_LEGACY_GATING_ALWAYS = "always"
_LEGACY_GATING_POLICIES = {
    _LEGACY_GATING_RESOLVED_ONLY,
    _LEGACY_GATING_FEEDBACK_PRESENT,
    _LEGACY_GATING_ALWAYS,
}
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


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
    return _coerce_bool_value(value, default=default)


def _coerce_bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default


def _normalize_turn_supervision_mode(value: Any) -> str:
    if value is None:
        return _TURN_SUPERVISION_CURRENT
    normalized = str(value).strip().lower()
    if not normalized:
        return _TURN_SUPERVISION_CURRENT
    if normalized not in _TURN_SUPERVISION_MODES:
        supported = ", ".join(sorted(_TURN_SUPERVISION_MODES))
        raise ValueError(f"turn_supervision_mode must be one of: {supported}")
    return normalized


def _normalize_verifier_feedback_mode(value: Any) -> str:
    if value is None:
        return _VERIFIER_FEEDBACK_NONE
    normalized = str(value).strip().lower()
    if not normalized:
        return _VERIFIER_FEEDBACK_NONE
    if normalized not in _VERIFIER_FEEDBACK_MODES:
        supported = ", ".join(sorted(_VERIFIER_FEEDBACK_MODES))
        raise ValueError(f"verifier_feedback_mode must be one of: {supported}")
    return normalized


def _normalize_legacy_gating_policy(value: Any) -> str:
    if value is None:
        return _LEGACY_GATING_RESOLVED_ONLY
    normalized = str(value).strip().lower()
    if not normalized:
        return _LEGACY_GATING_RESOLVED_ONLY
    if normalized not in _LEGACY_GATING_POLICIES:
        supported = ", ".join(sorted(_LEGACY_GATING_POLICIES))
        raise ValueError(f"legacy_distillation_gating_policy must be one of: {supported}")
    return normalized


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


def _resolved_flash_attn_fallback_impl() -> str | None:
    value = os.environ.get("SMALL_SWE_FALLBACK_ATTN_IMPL", "sdpa")
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _install_transformers_flash_attn_fallback_patch() -> None:
    try:
        from transformers import AutoModelForCausalLM
        from transformers import utils as transformers_utils
        from transformers.utils import import_utils as transformers_import_utils
    except Exception:
        return

    def _not_available() -> bool:
        return False

    transformers_utils.is_flash_attn_2_available = _not_available
    transformers_import_utils.is_flash_attn_2_available = _not_available

    current = AutoModelForCausalLM.from_pretrained
    if getattr(current, "_small_swe_flash_attn_fallback_patch", False):
        return

    original_from_pretrained = current

    def _small_swe_from_pretrained(*args: Any, **kwargs: Any):
        current_attn_impl = str(kwargs.get("attn_implementation", "")).strip().lower()
        fallback = _resolved_flash_attn_fallback_impl()
        if fallback and (not current_attn_impl or current_attn_impl == "flash_attention_2"):
            kwargs["attn_implementation"] = fallback
        return original_from_pretrained(*args, **kwargs)

    _small_swe_from_pretrained.__name__ = "_small_swe_from_pretrained"
    setattr(_small_swe_from_pretrained, "_small_swe_flash_attn_fallback_patch", True)
    AutoModelForCausalLM.from_pretrained = _small_swe_from_pretrained


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

    # Newer verl versions may already expose small-swe SDPO fields natively.
    dataclass_fields = getattr(SelfDistillationConfig, "__dataclass_fields__", {})
    has_native_num_recent = isinstance(dataclass_fields, dict) and "num_recent_raw_blocks" in dataclass_fields
    has_native_turn_supervision_mode = (
        isinstance(dataclass_fields, dict) and "turn_supervision_mode" in dataclass_fields
    )
    has_native_verifier_feedback_mode = (
        isinstance(dataclass_fields, dict) and "verifier_feedback_mode" in dataclass_fields
    )
    has_native_legacy_gating_policy = (
        isinstance(dataclass_fields, dict)
        and "legacy_distillation_gating_policy" in dataclass_fields
    )
    has_native_include_teacher_memory_blocks = (
        isinstance(dataclass_fields, dict) and "include_teacher_memory_blocks" in dataclass_fields
    )
    if (
        has_native_num_recent
        and has_native_turn_supervision_mode
        and has_native_verifier_feedback_mode
        and has_native_legacy_gating_policy
        and has_native_include_teacher_memory_blocks
    ):
        return

    if getattr(SelfDistillationConfig, "_small_swe_self_distillation_compat", False):
        return

    original_init = SelfDistillationConfig.__init__
    missing = object()

    def _small_swe_self_distillation_init(self, *args, **kwargs):
        raw_num_recent_raw_blocks: Any = missing
        if not has_native_num_recent:
            raw_num_recent_raw_blocks = kwargs.pop("num_recent_raw_blocks", missing)

        raw_turn_supervision_mode: Any = missing
        if not has_native_turn_supervision_mode:
            raw_turn_supervision_mode = kwargs.pop("turn_supervision_mode", missing)
        raw_verifier_feedback_mode: Any = missing
        if not has_native_verifier_feedback_mode:
            raw_verifier_feedback_mode = kwargs.pop("verifier_feedback_mode", missing)
        raw_legacy_gating_policy: Any = missing
        if not has_native_legacy_gating_policy:
            raw_legacy_gating_policy = kwargs.pop("legacy_distillation_gating_policy", missing)
        raw_include_teacher_memory_blocks: Any = missing
        if not has_native_include_teacher_memory_blocks:
            raw_include_teacher_memory_blocks = kwargs.pop("include_teacher_memory_blocks", missing)

        original_init(self, *args, **kwargs)

        if not has_native_num_recent:
            value = 3 if raw_num_recent_raw_blocks is missing else raw_num_recent_raw_blocks
            try:
                normalized = int(value)
            except (TypeError, ValueError):
                normalized = 3
            normalized = max(normalized, 0)
            # BaseConfig allows setting new fields once on frozen configs.
            setattr(self, "num_recent_raw_blocks", normalized)

        if not has_native_turn_supervision_mode:
            mode_value = _TURN_SUPERVISION_CURRENT if raw_turn_supervision_mode is missing else raw_turn_supervision_mode
            normalized_mode = _normalize_turn_supervision_mode(mode_value)
            setattr(self, "turn_supervision_mode", normalized_mode)
        if not has_native_verifier_feedback_mode:
            verifier_mode_value = (
                _VERIFIER_FEEDBACK_ALL_TURNS
                if raw_verifier_feedback_mode is missing
                else raw_verifier_feedback_mode
            )
            normalized_verifier_mode = _normalize_verifier_feedback_mode(verifier_mode_value)
            setattr(self, "verifier_feedback_mode", normalized_verifier_mode)
        if not has_native_legacy_gating_policy:
            gating_policy_value = (
                _LEGACY_GATING_RESOLVED_ONLY
                if raw_legacy_gating_policy is missing
                else raw_legacy_gating_policy
            )
            normalized_gating_policy = _normalize_legacy_gating_policy(gating_policy_value)
            setattr(self, "legacy_distillation_gating_policy", normalized_gating_policy)
        if not has_native_include_teacher_memory_blocks:
            include_memory_value = (
                True if raw_include_teacher_memory_blocks is missing else raw_include_teacher_memory_blocks
            )
            normalized_include_memory = _coerce_bool_value(include_memory_value, default=True)
            setattr(self, "include_teacher_memory_blocks", normalized_include_memory)

    _small_swe_self_distillation_init.__name__ = "_small_swe_self_distillation_init"
    SelfDistillationConfig.__init__ = _small_swe_self_distillation_init
    setattr(SelfDistillationConfig, "_small_swe_self_distillation_compat", True)


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
        if not ray_noset_visible_devices():
            original_setup(self)
            return

        local_rank = _resolve_local_rank_from_env()
        if local_rank is None:
            original_setup(self)
            return

        os.environ["LOCAL_RANK"] = str(local_rank)
        try:
            get_torch_device().set_device(local_rank)
        except Exception:
            original_setup(self)

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


def _install_tokenizer_pad_tensor_guard(tokenizer: Any) -> None:
    if torch is None or tokenizer is None:
        return
    if getattr(tokenizer, "_small_swe_pad_tensor_guard_installed", False):
        return

    original_pad = getattr(tokenizer, "pad", None)
    if not callable(original_pad):
        return

    def _small_swe_pad(*args, **kwargs):
        pad_output = original_pad(*args, **kwargs)
        if kwargs.get("return_tensors") != "pt":
            return pad_output

        for key in ("input_ids", "attention_mask"):
            try:
                value = pad_output.get(key) if hasattr(pad_output, "get") else pad_output[key]
            except Exception:
                continue
            if not isinstance(value, (list, tuple)):
                continue
            try:
                tensor_value = torch.as_tensor(value)
            except Exception:
                continue
            try:
                pad_output[key] = tensor_value
            except Exception:
                continue
        return pad_output

    _small_swe_pad.__name__ = "_small_swe_pad"
    tokenizer.pad = _small_swe_pad
    setattr(tokenizer, "_small_swe_pad_tensor_guard_installed", True)


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
        _install_tokenizer_pad_tensor_guard(tokenizer)
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


def _resolve_sequence_length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                resolved = int(shape[-1])
                if resolved >= 0:
                    return resolved
        except Exception:
            pass
    try:
        resolved = int(len(value))
    except Exception:
        return 0
    if resolved < 0:
        return 0
    return resolved


def _coerce_non_negative_index(value: Any, *, fallback: int, upper_bound: int | None = None) -> int:
    parsed: int | None = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor_value = value.detach()
        if tensor_value.numel() == 0:
            parsed = fallback
        else:
            try:
                parsed = int(tensor_value.sum().item())
            except Exception:
                try:
                    parsed = int(tensor_value.reshape(-1)[0].item())
                except Exception:
                    parsed = fallback
    elif isinstance(value, (list, tuple)):
        parsed = 0
        for item in value:
            parsed += _coerce_non_negative_index(item, fallback=0)
    elif hasattr(value, "item"):
        try:
            parsed = int(value.item())
        except Exception:
            parsed = None
    if parsed is None:
        try:
            parsed = int(value)
        except Exception:
            parsed = fallback
    if parsed < 0:
        parsed = 0
    if upper_bound is not None and parsed > upper_bound:
        parsed = upper_bound
    return parsed


def _slice_attention_tail_for_valid_response_length(attention_mask: Any, response_length: int) -> Any:
    if response_length <= 0:
        return attention_mask

    if torch is not None and isinstance(attention_mask, torch.Tensor):
        try:
            return attention_mask[..., -response_length:]
        except Exception:
            return attention_mask

    try:
        return attention_mask[-response_length:]
    except Exception:
        return attention_mask


def _slice_response_ids_for_decode(response_ids: Any, valid_response_length: int) -> Any:
    if valid_response_length <= 0:
        return response_ids[:0] if hasattr(response_ids, "__getitem__") else []

    if torch is not None and isinstance(response_ids, torch.Tensor):
        flattened = response_ids.detach().reshape(-1)
        return flattened[:valid_response_length]

    try:
        return response_ids[:valid_response_length]
    except Exception:
        pass

    if hasattr(response_ids, "tolist"):
        try:
            list_value = response_ids.tolist()
        except Exception:
            return []
        if isinstance(list_value, list):
            if list_value and isinstance(list_value[0], list):
                list_value = list_value[0]
            return list_value[:valid_response_length]
    return []


def _install_reward_loop_valid_response_length_guard() -> None:
    try:
        from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
    except Exception:
        return

    if getattr(NaiveRewardManager, "_small_swe_valid_response_length_guard", False):
        return

    original_run_single = getattr(NaiveRewardManager, "run_single", None)
    if not callable(original_run_single):
        return

    async def _small_swe_run_single(self, data):
        try:
            return await original_run_single(self, data)
        except TypeError as exc:
            if "only integer tensors of a single element can be converted to an index" not in str(exc):
                raise

        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = _resolve_sequence_length(response_ids)
        attention_mask = data_item.batch["attention_mask"]
        attention_tail = _slice_attention_tail_for_valid_response_length(attention_mask, response_length)
        try:
            raw_valid_response_length = attention_tail.sum()
        except Exception:
            raw_valid_response_length = attention_tail
        valid_response_length = _coerce_non_negative_index(
            raw_valid_response_length,
            fallback=response_length,
            upper_bound=response_length if response_length > 0 else None,
        )
        valid_response_ids = _slice_response_ids_for_decode(response_ids, valid_response_length)

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score
        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    _small_swe_run_single.__name__ = "_small_swe_run_single"
    NaiveRewardManager.run_single = _small_swe_run_single
    setattr(NaiveRewardManager, "_small_swe_valid_response_length_guard", True)


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


def _parse_csv_env_values(name: str) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return ()
    parts = [part.strip() for part in value.split(",")]
    return tuple(part for part in parts if part)


def _metric_key_is_wandb_essential(key: str) -> bool:
    if key in _WANDB_ESSENTIAL_EXACT_KEYS:
        return True
    if key.startswith("val-core/") and "/reward/mean@1" in key:
        return True
    if key.startswith("val-aux/") and (
        "/score/mean@1" in key
        or "/validation_errors/mean@1" in key
        or "/reward_verification_missing/mean@1" in key
    ):
        return True
    for prefix in _WANDB_ESSENTIAL_PREFIXES:
        if key.startswith(prefix):
            return True
    if key in _parse_csv_env_values(_WANDB_EXTRA_KEYS_ENV):
        return True
    for prefix in _parse_csv_env_values(_WANDB_EXTRA_PREFIXES_ENV):
        if key.startswith(prefix):
            return True
    return False


def _normalize_scalar_metric_value(value: Any) -> Any | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    # Handle numpy scalar-like values without importing numpy.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _normalize_scalar_metric_value(item())
        except Exception:
            return None
    return None


def _filter_wandb_metrics_for_sdpo(data: Mapping[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for raw_key, value in data.items():
        if not isinstance(raw_key, str):
            continue
        if not _metric_key_is_wandb_essential(raw_key):
            continue
        normalized = _normalize_scalar_metric_value(value)
        if normalized is None:
            continue
        filtered[raw_key] = normalized
    return filtered


def _sanitize_scalar_metrics(data: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for raw_key, value in data.items():
        if not isinstance(raw_key, str):
            continue
        normalized = _normalize_scalar_metric_value(value)
        if normalized is None:
            continue
        sanitized[raw_key] = normalized
    return sanitized


def _safe_finalize_tracking_backend(backend_name: str, logger_instance: Any) -> None:
    finish_fn = getattr(logger_instance, "finish", None)
    if not callable(finish_fn):
        return

    try:
        if backend_name == "wandb":
            # wandb.finish can throw when called inside an active async loop.
            try:
                import asyncio

                asyncio.get_running_loop()
                in_async_loop = True
            except RuntimeError:
                in_async_loop = False
            except Exception:
                in_async_loop = False

            if in_async_loop:
                thread_error: list[Exception] = []

                def _finish_in_thread() -> None:
                    try:
                        try:
                            finish_fn(exit_code=0, quiet=True)
                        except TypeError:
                            finish_fn(exit_code=0)
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        thread_error.append(exc)

                finish_thread = threading.Thread(
                    target=_finish_in_thread,
                    name="small-swe-wandb-finish",
                    daemon=True,
                )
                finish_thread.start()
                finish_thread.join(timeout=5.0)
                if finish_thread.is_alive():
                    logging.getLogger(__name__).warning(
                        "Timed out waiting for wandb.finish() in async-loop fallback thread."
                    )
                if thread_error:
                    raise thread_error[0]
                return

            try:
                finish_fn(exit_code=0, quiet=True)
            except TypeError:
                finish_fn(exit_code=0)
            return

        finish_fn()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Tracking backend %s finalization failed: %s",
            backend_name,
            exc,
        )


def _install_verl_tracking_patch() -> None:
    try:
        from verl.utils import tracking as tracking_module
    except Exception:
        return

    Tracking = getattr(tracking_module, "Tracking", None)
    if Tracking is None:
        return
    if getattr(Tracking, "_small_swe_tracking_patch", False):
        return

    original_log = Tracking.log

    def _small_swe_tracking_log(self, data, step, backend=None):
        if not isinstance(data, Mapping):
            return original_log(self, data, step, backend=backend)

        logger_map = getattr(self, "logger", {})
        if not isinstance(logger_map, dict):
            return original_log(self, data, step, backend=backend)

        payload = dict(data)
        backend_filter: set[str] | None
        if backend is None:
            backend_filter = None
        elif isinstance(backend, str):
            backend_filter = {backend}
        else:
            backend_filter = {str(item) for item in backend}

        if "wandb" in logger_map and (backend_filter is None or "wandb" in backend_filter):
            if _coerce_bool_env(_WANDB_ESSENTIAL_FILTER_ENV, default=True):
                wandb_payload = _filter_wandb_metrics_for_sdpo(payload)
            else:
                wandb_payload = _sanitize_scalar_metrics(payload)
            if wandb_payload:
                try:
                    logger_map["wandb"].log(data=wandb_payload, step=step)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "W&B logging failed at step %s: %s",
                        step,
                        exc,
                    )

        for default_backend, logger_instance in logger_map.items():
            if default_backend == "wandb":
                continue
            if backend_filter is not None and default_backend not in backend_filter:
                continue
            logger_instance.log(data=payload, step=step)

    def _small_swe_tracking_close(self) -> None:
        if getattr(self, "_small_swe_tracking_closed", False):
            return
        setattr(self, "_small_swe_tracking_closed", True)

        logger_map = getattr(self, "logger", None)
        if not isinstance(logger_map, dict):
            return

        ordered_backends = (
            "wandb",
            "swanlab",
            "vemlp_wandb",
            "tensorboard",
            "clearml",
            "trackio",
            "file",
        )
        finalized: set[str] = set()
        for backend_name in ordered_backends:
            logger_instance = logger_map.get(backend_name)
            if logger_instance is None:
                continue
            _safe_finalize_tracking_backend(backend_name, logger_instance)
            finalized.add(backend_name)

        for backend_name, logger_instance in logger_map.items():
            if backend_name in finalized:
                continue
            _safe_finalize_tracking_backend(backend_name, logger_instance)

    def _small_swe_tracking_del(self):
        close_fn = getattr(self, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                return

    _small_swe_tracking_log.__name__ = "_small_swe_tracking_log"
    _small_swe_tracking_close.__name__ = "_small_swe_tracking_close"
    _small_swe_tracking_del.__name__ = "_small_swe_tracking_del"

    Tracking.log = _small_swe_tracking_log
    Tracking.close = _small_swe_tracking_close
    Tracking.__del__ = _small_swe_tracking_del
    setattr(Tracking, "_small_swe_tracking_patch", True)


def apply_small_swe_runtime_patches() -> None:
    if _coerce_bool_env("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", default=False):
        _clear_cached_flash_attn_modules()
        _install_flash_attn_find_spec_guard()
        _install_flash_attn_import_guard()
        _install_transformers_flash_attn_fallback_patch()
    if _coerce_bool_env("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", default=False):
        _install_self_distillation_config_compat_patch()
        _install_ray_worker_local_rank_device_patch()
        _install_verl_tokenizer_compat_patches()
        _install_reward_loop_valid_response_length_guard()
        _install_transformers_torch_dtype_property_patch()
        _install_transformers_torch_dtype_warning_filter()
        _install_verl_tracking_patch()
        _install_known_warning_filters()
        # Ray worker processes do not enter our main wrapper module.
        # Patch lazily once ray_trainer is imported in-process.
        _install_sdpo_runtime_patch_import_guard()
        _try_apply_sdpo_runtime_patch()
