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


_ensure_flash_attn_runtime_compatibility()
AutoModelForCausalLM.from_pretrained = _patched_from_pretrained

import verl.trainer.fsdp_sft_trainer as _verl_sft_trainer  # noqa: E402


_ORIGINAL_RUN_SFT = _verl_sft_trainer.run_sft
_ORIGINAL_FSDP_SFT_TRAINER = _verl_sft_trainer.FSDPSFTTrainer


class _EmptyValidationDataset:
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:
        raise IndexError(index)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _has_validation_files(config: Any) -> bool:
    data_config = _config_get(config, "data")
    val_files = _config_get(data_config, "val_files")
    if val_files is None:
        return False
    if isinstance(val_files, str):
        return bool(val_files.strip())
    try:
        return any(bool(str(item).strip()) for item in val_files)
    except TypeError:
        return bool(val_files)


def _trainer_test_freq(config: Any) -> int:
    trainer_config = _config_get(config, "trainer")
    raw_value = _config_get(trainer_config, "test_freq", 0)
    return int(raw_value)


def _inner_validation_disabled(config: Any) -> bool:
    return _trainer_test_freq(config) <= 0 or not _has_validation_files(config)


def _load_hf_tokenizer(model_path: str, *, trust_remote_code: bool) -> Any:
    from verl.utils import hf_tokenizer

    return hf_tokenizer(model_path, trust_remote_code=trust_remote_code)


def _fit_without_validation(trainer: Any) -> None:
    rank = trainer.device_mesh.get_rank()

    if rank == 0:
        tracking = _verl_sft_trainer.Tracking(
            project_name=trainer.config.trainer.project_name,
            experiment_name=trainer.config.trainer.experiment_name,
            default_backend=trainer.config.trainer.logger,
            config=_verl_sft_trainer.OmegaConf.to_container(
                trainer.config,
                resolve=True,
            ),
            group_name=trainer.config.trainer.get("group_name", None),
        )

    global_step = trainer.resume_global_step
    total_training_steps = len(trainer.train_dataloader) * trainer.config.trainer.total_epochs
    if trainer.config.trainer.total_training_steps is not None:
        total_training_steps = trainer.config.trainer.total_training_steps

    trainer.total_training_steps = total_training_steps
    _verl_sft_trainer.log_with_rank(
        f"Total training steps: {trainer.total_training_steps},",
        logger=_verl_sft_trainer.logger,
        rank=trainer.device_mesh.get_rank(),
        log_only_rank_0=True,
    )

    if global_step > 0:
        _verl_sft_trainer.log_with_rank(
            f"StatefulDataLoader will automatically resume from global step: {global_step}",
            logger=_verl_sft_trainer.logger,
            rank=trainer.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

    start_epoch = global_step // trainer.steps_per_epoch

    train_time = 0
    for epoch in range(start_epoch, trainer.config.trainer.total_epochs):
        trainer.train_sampler.set_epoch(epoch=epoch)

        for _step_in_epoch, data in enumerate(
            _verl_sft_trainer.tqdm(
                trainer.train_dataloader,
                initial=global_step % trainer.steps_per_epoch if epoch == start_epoch else 0,
                total=trainer.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{trainer.config.trainer.total_epochs}",
                disable=rank != 0,
            )
        ):
            global_step += 1
            data = _verl_sft_trainer.TensorDict(
                data,
                batch_size=trainer.config.data.train_batch_size,
            ).to(trainer.device_name)
            metric = trainer.training_step(data)
            train_time += metric["train/time(s)"]
            if rank == 0:
                tracking.log(data=metric, step=global_step)

            is_last_step = global_step >= trainer.total_training_steps
            save_freq = int(trainer.config.trainer.save_freq)
            is_save_step = save_freq > 0 and global_step % save_freq == 0

            if is_last_step or is_save_step:
                trainer.save_checkpoint(step=global_step)

            if is_last_step:
                if rank == 0:
                    print(f"Total time for train steps: {train_time:.2f}s")
                    print("Final validation metrics: None")
                return


class _SmallSWEFSDPSFTTrainer(_ORIGINAL_FSDP_SFT_TRAINER):
    def fit(self) -> None:
        if _inner_validation_disabled(self.config):
            _fit_without_validation(self)
            return
        super().fit()


def _patched_run_sft(config: Any) -> None:
    if not _inner_validation_disabled(config):
        _ORIGINAL_RUN_SFT(config)
        return

    device_name = _verl_sft_trainer.get_device_name()
    _local_rank, _rank, world_size = _verl_sft_trainer.initialize_global_process_group()

    device_mesh = _verl_sft_trainer.init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = _verl_sft_trainer.init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    local_model_path = _verl_sft_trainer.copy_to_local(
        src=config.model.partial_pretrain,
        verbose=True,
    )
    tokenizer = _load_hf_tokenizer(
        local_model_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    train_dataset = _verl_sft_trainer.create_sft_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        max_samples=config.data.get("train_max_samples", -1),
    )

    trainer = _SmallSWEFSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=_EmptyValidationDataset(),
    )

    trainer.fit()
    _verl_sft_trainer.destroy_global_process_group()


_verl_sft_trainer.FSDPSFTTrainer = _SmallSWEFSDPSFTTrainer
_verl_sft_trainer.run_sft = _patched_run_sft
main = _verl_sft_trainer.main


if __name__ == "__main__":
    main()
