"""Custom verl SFT dataset that pulls on-policy rollouts and builds RFT batches in-memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME
from trainer.rft_runtime import (
    OnPolicyRFTRuntimeRequest,
    collect_onpolicy_rft_runtime_batch,
)

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}
_ONPOLICY_RFT_CACHE: dict[str, dict[str, Any]] = {}
_FORMAT_RFT_STAGE_NAME = "format_rft"
_POSITIVE_RFT_STAGE_NAME = "positive_rft"
_TASK_PARTITION_ALL = "all"
_TASK_PARTITION_TRAIN = "train"
_TASK_PARTITION_EVAL = "eval"
_TASK_PARTITION_ALIASES = {
    "": _TASK_PARTITION_ALL,
    _TASK_PARTITION_ALL: _TASK_PARTITION_ALL,
    _TASK_PARTITION_TRAIN: _TASK_PARTITION_TRAIN,
    _TASK_PARTITION_EVAL: _TASK_PARTITION_EVAL,
    "heldout": _TASK_PARTITION_EVAL,
    "held_out": _TASK_PARTITION_EVAL,
    "val": _TASK_PARTITION_EVAL,
    "validation": _TASK_PARTITION_EVAL,
}


class OnPolicyRFTDataset:
    """verl-compatible Dataset that materializes one on-policy RFT batch."""

    def __init__(
        self,
        parquet_files: str | Sequence[str],
        tokenizer: Any,
        config: Mapping[str, Any],
        processor: Any | None = None,
        max_samples: int = -1,
    ) -> None:
        del processor  # This dataset is generated on-policy, not from parquet.

        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
            raise RuntimeError(
                "OnPolicyRFTDataset requires torch. Install training extras (`pip install -e \".[train]\"`)."
            ) from exc

        config_mapping = _as_mapping(config)
        on_policy_cfg = _as_mapping(config_mapping.get("on_policy", {}))
        if not _coerce_bool(on_policy_cfg.get("enabled", True), fallback=True):
            raise ValueError("data.on_policy.enabled must be true when using OnPolicyRFTDataset.")

        data_config_name = str(
            on_policy_cfg.get("data_config_name", DEFAULT_ON_POLICY_DATA_CONFIG_NAME)
        ).strip() or DEFAULT_ON_POLICY_DATA_CONFIG_NAME
        turn_generator_mode = str(on_policy_cfg.get("turn_generator_mode", "default")).strip().lower()
        total_steps = _coerce_positive_int(on_policy_cfg.get("total_steps", 1), label="data.on_policy.total_steps")
        runtime_overrides = _as_mapping(on_policy_cfg.get("runtime_overrides", {}))
        data_overrides = _as_mapping(on_policy_cfg.get("data_overrides", {}))
        handoff_overrides = _as_mapping(on_policy_cfg.get("rft_handoff_overrides", {}))
        explicit_task_partition = ""
        explicit_task_partition_raw = on_policy_cfg.get("task_partition")
        if explicit_task_partition_raw is not None:
            explicit_task_partition = str(explicit_task_partition_raw).strip()
        stage_name = _resolve_stage_name(on_policy_cfg.get("stage_name", _FORMAT_RFT_STAGE_NAME))
        handoff_overrides = _merge_stage_handoff_overrides(
            handoff_overrides=handoff_overrides,
            stage_name=stage_name,
        )
        verify_submissions = _coerce_bool(
            on_policy_cfg.get("verify_submissions", runtime_overrides.get("verify_submissions")),
            fallback=False,
        ) or _resolve_stage_verify_submissions(stage_name)
        task_eval_split_fraction = _coerce_fraction(
            on_policy_cfg.get("task_eval_split_fraction", 0.0),
            label="data.on_policy.task_eval_split_fraction",
        )
        task_eval_min_rows = _coerce_non_negative_int(
            on_policy_cfg.get("task_eval_min_rows", 0),
            label="data.on_policy.task_eval_min_rows",
        )
        task_eval_task_count_raw = on_policy_cfg.get("task_eval_task_count")
        task_eval_task_count = (
            _coerce_non_negative_int(
                task_eval_task_count_raw,
                label="data.on_policy.task_eval_task_count",
            )
            if task_eval_task_count_raw is not None
            else None
        )
        output_dir_raw = on_policy_cfg.get("output_dir")
        output_dir = str(output_dir_raw).strip() if isinstance(output_dir_raw, str) and output_dir_raw.strip() else None
        parquet_file_fingerprint = _normalize_parquet_files(parquet_files)
        task_partition = _resolve_task_partition(
            explicit_partition=explicit_task_partition_raw,
            parquet_files=parquet_file_fingerprint,
            train_files=config_mapping.get("train_files"),
            val_files=config_mapping.get("val_files"),
            task_eval_split_fraction=task_eval_split_fraction,
            task_eval_task_count=task_eval_task_count,
        )

        def _build_cache_key_for(
            *,
            resolved_task_partition: str,
            resolved_parquet_files: Sequence[str],
        ) -> str:
            return _cache_key(
                data_config_name=data_config_name,
                turn_generator_mode=turn_generator_mode,
                total_steps=total_steps,
                runtime_overrides=runtime_overrides,
                data_overrides=data_overrides,
                handoff_overrides=handoff_overrides,
                verify_submissions=verify_submissions,
                stage_name=stage_name,
                task_partition=resolved_task_partition,
                task_eval_split_fraction=task_eval_split_fraction,
                task_eval_min_rows=task_eval_min_rows,
                task_eval_task_count=task_eval_task_count,
                parquet_files=resolved_parquet_files,
                tokenizer=tokenizer,
            )

        cache_key = _build_cache_key_for(
            resolved_task_partition=task_partition,
            resolved_parquet_files=parquet_file_fingerprint,
        )
        def _collect_runtime_batch_for_partition(resolved_task_partition: str) -> dict[str, Any]:
            request_runtime_overrides = dict(runtime_overrides)
            if (
                resolved_task_partition == _TASK_PARTITION_EVAL
                and task_eval_task_count is not None
                and task_eval_task_count > 0
            ):
                request_runtime_overrides["task_batch_size"] = task_eval_task_count
                request_runtime_overrides["attempts_per_task"] = 1
                max_in_flight = request_runtime_overrides.get("max_in_flight_tasks")
                if isinstance(max_in_flight, int) and not isinstance(max_in_flight, bool):
                    request_runtime_overrides["max_in_flight_tasks"] = min(
                        max_in_flight,
                        task_eval_task_count,
                    )
            request = OnPolicyRFTRuntimeRequest(
                data_config_name=data_config_name,
                turn_generator_mode=turn_generator_mode,
                total_steps=total_steps,
                runtime_overrides=request_runtime_overrides,
                data_overrides=data_overrides,
                handoff_overrides=handoff_overrides,
                task_partition=resolved_task_partition,
                task_eval_split_fraction=task_eval_split_fraction,
                task_eval_min_rows=task_eval_min_rows,
                task_eval_task_count=task_eval_task_count,
                verify_submissions=verify_submissions,
                stage_name=stage_name,
                output_dir=output_dir,
            )
            return collect_onpolicy_rft_runtime_batch(
                request=request,
                tokenizer=tokenizer,
            )

        cached_result = _ONPOLICY_RFT_CACHE.get(cache_key)
        if cached_result is None:
            collected_result = _collect_on_rank0_and_broadcast(
                torch_module=torch,
                collect_fn=lambda: _collect_runtime_batch_for_partition(task_partition),
            )
            if _selected_sample_count(collected_result) < 1:
                raise ValueError("OnPolicyRFTDataset produced zero selected rows for training.")
            _ONPOLICY_RFT_CACHE[cache_key] = collected_result
            cached_result = collected_result

        sft_batch = _as_mapping(cached_result["sft_batch"])
        tensors = _as_mapping(sft_batch["tensors"])
        grouping = _as_mapping(sft_batch["grouping_metadata"])
        meta_info = dict(_as_mapping(sft_batch["meta_info"]))
        rejected_rows = tuple(cached_result["rejected_rows"])

        input_ids_rows = _as_rows(tensors.get("input_ids"), label="sft_batch.tensors.input_ids")
        sample_count = len(input_ids_rows)
        if max_samples > 0:
            sample_count = min(sample_count, max_samples)
        if sample_count < 1:
            _ONPOLICY_RFT_CACHE.pop(cache_key, None)
            raise ValueError("OnPolicyRFTDataset produced zero selected rows for training.")

        self._samples: list[dict[str, Any]] = []
        for index in range(sample_count):
            self._samples.append(
                {
                    "input_ids": torch.tensor(tensors["input_ids"][index], dtype=torch.long),
                    "attention_mask": torch.tensor(tensors["attention_mask"][index], dtype=torch.long),
                    "position_ids": torch.tensor(tensors["position_ids"][index], dtype=torch.long),
                    "loss_mask": torch.tensor(tensors["loss_mask"][index], dtype=torch.long),
                }
            )

        self.grouping_metadata = {
            key: list(value[:sample_count]) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else value
            for key, value in grouping.items()
        }
        self.meta_info = {
            **meta_info,
            "selected_count_after_max_samples": sample_count,
            "rejected_count": len(rejected_rows),
        }
        self.rejected_rows = rejected_rows

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError("Expected mapping-like configuration payload.")


def _selected_sample_count(result: Mapping[str, Any]) -> int:
    try:
        sft_batch = _as_mapping(result.get("sft_batch"))
        tensors = _as_mapping(sft_batch.get("tensors"))
    except ValueError:
        return 0
    input_ids = tensors.get("input_ids")
    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        return 0
    return len(input_ids)


def _as_rows(value: Any, *, label: str) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of rows.")
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"{label} must contain sequence rows.")
        rows.append([int(item) for item in row])
    return rows


def _coerce_bool(value: Any, *, fallback: bool) -> bool:
    if value is None:
        return fallback
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
    return fallback


def _coerce_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= 1.")
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = int(stripped)
            except ValueError as exc:
                raise ValueError(f"{label} must be an integer >= 1.") from exc
            if parsed >= 1:
                return parsed
    raise ValueError(f"{label} must be an integer >= 1.")


def _coerce_non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= 0.")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = int(stripped)
            except ValueError as exc:
                raise ValueError(f"{label} must be an integer >= 0.") from exc
            if parsed >= 0:
                return parsed
    raise ValueError(f"{label} must be an integer >= 0.")


def _coerce_fraction(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must satisfy 0.0 <= value < 1.0.")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0.0
        try:
            parsed = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{label} must satisfy 0.0 <= value < 1.0.") from exc
    else:
        raise ValueError(f"{label} must satisfy 0.0 <= value < 1.0.")
    if parsed < 0.0 or parsed >= 1.0:
        raise ValueError(f"{label} must satisfy 0.0 <= value < 1.0.")
    return parsed


def _resolve_stage_name(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"", "default", "format", _FORMAT_RFT_STAGE_NAME}:
        return _FORMAT_RFT_STAGE_NAME
    if normalized in {"positive", _POSITIVE_RFT_STAGE_NAME}:
        return _POSITIVE_RFT_STAGE_NAME
    raise ValueError("data.on_policy.stage_name must be one of: format_rft, positive_rft.")


def _resolve_stage_verify_submissions(stage_name: str) -> bool:
    return _resolve_stage_name(stage_name) == _POSITIVE_RFT_STAGE_NAME


def _merge_stage_handoff_overrides(
    *,
    handoff_overrides: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    resolved = dict(handoff_overrides)
    selection = _mapping_or_empty(resolved.get("selection"))
    if _resolve_stage_name(stage_name) == _POSITIVE_RFT_STAGE_NAME:
        selection.update(
            {
                "require_terminal": False,
                "require_format_valid": False,
                "require_resolved": True,
                "reject_on_invalid_final_submit": False,
            }
        )
    else:
        selection.update(
            {
                "require_terminal": False,
                "require_format_valid": True,
                "require_resolved": False,
                "reject_on_invalid_final_submit": False,
            }
        )
    resolved["selection"] = selection
    return resolved


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _resolve_task_partition(
    *,
    explicit_partition: Any,
    parquet_files: Sequence[str],
    train_files: Any,
    val_files: Any,
    task_eval_split_fraction: float,
    task_eval_task_count: int | None,
) -> str:
    normalized_explicit = ""
    if explicit_partition is not None:
        normalized_explicit = str(explicit_partition).strip()
    if normalized_explicit:
        return _normalize_task_partition(normalized_explicit)
    task_eval_enabled = (
        (task_eval_task_count is not None and task_eval_task_count > 0)
        or task_eval_split_fraction > 0.0
    )
    if not task_eval_enabled:
        return _TASK_PARTITION_ALL

    normalized_train_files = _normalize_parquet_files(train_files)
    normalized_val_files = _normalize_parquet_files(val_files)
    if (
        parquet_files
        and normalized_train_files
        and normalized_train_files == normalized_val_files
        and parquet_files == normalized_train_files
    ):
        raise ValueError(
            "data.train_files and data.val_files must differ when "
            "data.on_policy.task_eval_task_count > 0 or "
            "data.on_policy.task_eval_split_fraction > 0 unless "
            "data.on_policy.task_partition is set explicitly."
        )
    if parquet_files and parquet_files == normalized_val_files and normalized_val_files:
        return _TASK_PARTITION_EVAL
    if parquet_files and parquet_files == normalized_train_files and normalized_train_files:
        return _TASK_PARTITION_TRAIN
    return _TASK_PARTITION_ALL


def _normalize_task_partition(value: Any) -> str:
    normalized = str(value).strip().lower()
    partition = _TASK_PARTITION_ALIASES.get(normalized)
    if partition is None:
        raise ValueError("data.on_policy.task_partition must be one of: all, train, eval.")
    return partition


def _cache_key(
    *,
    data_config_name: str,
    turn_generator_mode: str,
    total_steps: int,
    runtime_overrides: Mapping[str, Any],
    data_overrides: Mapping[str, Any],
    handoff_overrides: Mapping[str, Any],
    verify_submissions: bool,
    stage_name: str,
    task_partition: str,
    task_eval_split_fraction: float,
    task_eval_min_rows: int,
    task_eval_task_count: int | None,
    parquet_files: Sequence[str],
    tokenizer: Any,
) -> str:
    payload = {
        "data_config_name": data_config_name,
        "turn_generator_mode": turn_generator_mode,
        "total_steps": total_steps,
        "runtime_overrides": _normalize_mapping(runtime_overrides),
        "data_overrides": _normalize_mapping(data_overrides),
        "handoff_overrides": _normalize_mapping(handoff_overrides),
        "verify_submissions": bool(verify_submissions),
        "stage_name": str(stage_name),
        "task_partition": str(task_partition),
        "task_eval_split_fraction": float(task_eval_split_fraction),
        "task_eval_min_rows": int(task_eval_min_rows),
        "task_eval_task_count": (
            int(task_eval_task_count) if task_eval_task_count is not None else None
        ),
        "parquet_files": [str(path) for path in parquet_files],
        "tokenizer_fingerprint": _tokenizer_cache_fingerprint(tokenizer),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _normalize_parquet_files(parquet_files: str | Sequence[str]) -> list[str]:
    if isinstance(parquet_files, str):
        normalized = parquet_files.strip()
        return [normalized] if normalized else []

    if isinstance(parquet_files, Sequence):
        paths: list[str] = []
        for raw_path in parquet_files:
            if isinstance(raw_path, (str, Path)):
                normalized = str(raw_path).strip()
                if normalized:
                    paths.append(normalized)
        return paths

    return []


def _collect_on_rank0_and_broadcast(
    *,
    torch_module: Any,
    collect_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    distributed = _resolve_distributed_module(torch_module)
    if distributed is None:
        return collect_fn()

    rank = int(distributed.get_rank())
    payload: dict[str, Any] | None
    if rank == 0:
        try:
            payload = {"ok": True, "result": collect_fn()}
        except Exception as exc:  # pragma: no cover - defensive branch for distributed jobs.
            payload = {"ok": False, "error": f"On-policy collection failed on rank 0: {exc}"}
    else:
        payload = None

    object_list: list[Any] = [payload]
    distributed.broadcast_object_list(object_list, src=0)
    received = object_list[0]
    if not isinstance(received, Mapping):
        raise RuntimeError("Distributed on-policy collection received malformed payload.")
    if not bool(received.get("ok", False)):
        raise RuntimeError(str(received.get("error", "On-policy collection failed on rank 0.")))

    result = received.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Distributed on-policy collection returned invalid result payload.")
    return result


def _resolve_distributed_module(torch_module: Any) -> Any | None:
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None:
        return None
    is_available = getattr(distributed, "is_available", None)
    is_initialized = getattr(distributed, "is_initialized", None)
    if not callable(is_available) or not callable(is_initialized):
        return None
    if not is_available() or not is_initialized():
        return None
    broadcast_object_list = getattr(distributed, "broadcast_object_list", None)
    get_rank = getattr(distributed, "get_rank", None)
    if not callable(broadcast_object_list) or not callable(get_rank):
        return None
    return distributed


def _normalize_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json(payload)
    if isinstance(normalized, dict):
        return normalized
    return {}


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _tokenizer_cache_fingerprint(tokenizer: Any) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
    }
    for attr_name in ("name_or_path", "vocab_size", "model_max_length", "padding_side", "truncation_side"):
        attr_value = getattr(tokenizer, attr_name, None)
        if attr_value is not None and attr_value != "":
            fingerprint[attr_name] = attr_value

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        try:
            added_vocab = get_added_vocab()
        except Exception:  # pragma: no cover - defensive only
            added_vocab = None
        if isinstance(added_vocab, Mapping):
            fingerprint["added_vocab"] = {
                str(token): int(token_id) if isinstance(token_id, int) else repr(token_id)
                for token, token_id in sorted(added_vocab.items(), key=lambda item: str(item[0]))
            }

    special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_tokens_map, Mapping):
        fingerprint["special_tokens_map"] = _normalize_mapping(special_tokens_map)

    # If tokenizer metadata is unavailable, isolate cache entries by instance to avoid collisions.
    if len(fingerprint) == 1:
        fingerprint["instance_id"] = id(tokenizer)

    return _normalize_mapping(fingerprint)
