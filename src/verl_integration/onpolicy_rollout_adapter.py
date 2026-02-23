"""On-policy rollout adapters for deterministic RFT handoff into verl SFT batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import (
    DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    RFTHandoffSettings,
    resolve_on_policy_settings,
    resolve_rft_handoff_settings,
)
from data.tokenization import SupportsOffsetsTokenizer
from env.task_dataset import DatasetLoader
from rollout.onpolicy_collector import (
    AssistantTurnGenerator,
    AttemptResolver,
    ExecutorFactory,
    OnPolicyRolloutCollector,
    PoolFactory,
)
from schemas import RolloutRow
from verl_integration.data_preprocessor import preprocess_trajectories
from verl_integration.rft_rejection import (
    evaluate_rft_rejection_reason,
    select_rft_attempt_rows,
)

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


def build_onpolicy_collector(
    *,
    turn_generator: AssistantTurnGenerator | None = None,
    data_config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    runtime_overrides: Mapping[str, Any] | None = None,
    data_overrides: Mapping[str, Any] | None = None,
    dataset_loader: DatasetLoader | None = None,
    pool_factory: PoolFactory | None = None,
    executor_factory: ExecutorFactory | None = None,
    attempt_resolver: AttemptResolver | None = None,
) -> OnPolicyRolloutCollector:
    """Build a collector using centralized config authority only."""
    settings = resolve_on_policy_settings(
        data_config_name=data_config_name,
        runtime_overrides=runtime_overrides,
        data_overrides=data_overrides,
    )
    return OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=dataset_loader,
        pool_factory=pool_factory,
        executor_factory=executor_factory,
        attempt_resolver=attempt_resolver,
    )


def collect_rollouts_for_steps(
    *,
    total_steps: int,
    collector: OnPolicyRolloutCollector,
    output_dir: str | Path | None = None,
) -> list[list[RolloutRow]]:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")

    all_rows: list[list[RolloutRow]] = []
    base_dir = Path(output_dir) if output_dir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for step_index in range(total_steps):
        rows = collector.collect_step(step_index)
        all_rows.append(rows)
        if base_dir is not None:
            output_path = base_dir / f"step_{step_index:05d}.jsonl"
            write_jsonl_rows(output_path, rows)
    return all_rows


def collect_rft_sft_batch_for_steps(
    *,
    total_steps: int,
    collector: OnPolicyRolloutCollector,
    tokenizer: SupportsOffsetsTokenizer,
    handoff_overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Collect rollouts, apply centralized RFT rejection policy, and build SFT tensors."""
    rollout_steps = collect_rollouts_for_steps(
        total_steps=total_steps,
        collector=collector,
        output_dir=output_dir,
    )
    rollout_rows = _flatten_rollout_steps(rollout_steps)
    if not rollout_rows:
        raise ValueError("No on-policy rollout rows were collected.")

    preprocessed_rows = preprocess_trajectories(
        rollout_rows,
        max_tool_calls=collector.settings.runtime.max_tool_calls_per_turn,
        tokenizer=tokenizer,
    )
    merged_rows = merge_rollout_and_preprocessed_rows(rollout_rows, preprocessed_rows)

    handoff_settings = resolve_rft_handoff_settings(overrides=handoff_overrides)
    selected_rows, rejected_rows = select_rft_attempt_rows(
        merged_rows,
        selection_policy=handoff_settings.selection,
    )
    if not selected_rows:
        raise ValueError(
            "RFT handoff selected 0 attempts; all rollouts were rejected. "
            "Inspect rejection reasons in rejected rows."
        )

    sft_batch = build_verl_sft_batch(
        selected_rows,
        handoff_settings=handoff_settings,
    )
    dataproto_payload = build_dataproto_compatible_payload(sft_batch)

    result = {
        "rollout_rows": rollout_rows,
        "selected_rows": selected_rows,
        "rejected_rows": rejected_rows,
        "sft_batch": sft_batch,
        "dataproto_payload": dataproto_payload,
    }

    if output_dir is not None:
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_rows(base_dir / "rollout_rows.jsonl", rollout_rows)
        write_jsonl_rows(base_dir / "selected_rows.jsonl", selected_rows)
        write_jsonl_rows(base_dir / "rejected_rows.jsonl", rejected_rows)
        _write_json(
            base_dir / "rft_sft_meta.json",
            {
                "selected_count": len(selected_rows),
                "rejected_count": len(rejected_rows),
                "max_padded_length": sft_batch["meta_info"]["max_padded_length"],
                "max_sequence_length_limit": sft_batch["meta_info"]["max_sequence_length_limit"],
            },
        )
        _write_json(
            base_dir / "rollout_artifact_summary.json",
            _build_rollout_artifact_summary(
                rollout_rows=rollout_rows,
                total_steps=total_steps,
                selected_count=len(selected_rows),
                rejected_count=len(rejected_rows),
            ),
        )

    return result


def merge_rollout_and_preprocessed_rows(
    rollout_rows: Sequence[Mapping[str, Any]],
    preprocessed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge attempt-level rollout metadata into preprocessed rows 1:1."""
    if len(rollout_rows) != len(preprocessed_rows):
        raise ValueError(
            "rollout_rows and preprocessed_rows must have the same length "
            f"(got {len(rollout_rows)} != {len(preprocessed_rows)})."
        )

    merged_rows: list[dict[str, Any]] = []
    for index, (rollout_row, preprocessed_row) in enumerate(zip(rollout_rows, preprocessed_rows)):
        task_id = _require_non_empty_task_id(rollout_row.get("task_id"), index=index)
        merged = dict(preprocessed_row)
        merged.update(
            {
                "task_id": task_id,
                "attempt_index": rollout_row.get("attempt_index", 0),
                "turn_index": rollout_row.get("turn_index", 0),
                "step_index": rollout_row.get("step_index", 0),
                "resolved": rollout_row.get("resolved", False),
                "is_terminal": rollout_row.get("is_terminal", False),
                "collector_error": rollout_row.get("collector_error", ""),
                "bridge_error": rollout_row.get("bridge_error", ""),
                "timeout_error": rollout_row.get("timeout_error", ""),
                "executor_error": rollout_row.get("executor_error", ""),
                "exit_code": rollout_row.get("exit_code"),
                "tool_name": rollout_row.get("tool_name", ""),
                "container_id": rollout_row.get("container_id", ""),
                "image_name": rollout_row.get("image_name", ""),
                "trajectory_steps": rollout_row.get("trajectory_steps", ()),
                "trajectory_history": rollout_row.get("trajectory_history", ()),
            }
        )
        merged_rows.append(merged)
    return merged_rows


def build_verl_sft_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    handoff_settings: RFTHandoffSettings,
) -> dict[str, Any]:
    """Convert selected RFT rows into verl SFT-compatible tensor + metadata payloads."""
    if not rows:
        raise ValueError("rows must be non-empty.")

    padded_limit = handoff_settings.max_sequence_length
    if padded_limit < 2:
        raise ValueError("rft_handoff.max_sequence_length must be >= 2")

    pad_token_id = handoff_settings.pad_token_id
    packed_rows: list[dict[str, Any]] = []
    max_length = 0

    for index, row in enumerate(rows):
        input_ids = _coerce_int_sequence(
            row.get("input_ids"),
            label=f"rows[{index}].input_ids",
        )
        if not input_ids:
            raise ValueError(f"rows[{index}].input_ids is empty.")

        action_mask = _coerce_loss_mask_sequence(
            row.get("action_mask_rft"),
            label=f"rows[{index}].action_mask_rft",
        )
        token_labels = _coerce_token_labels(
            row.get("token_labels"),
            length_hint=len(input_ids),
        )

        truncated_length = min(len(input_ids), len(action_mask), len(token_labels), padded_limit)
        input_ids = input_ids[:truncated_length]
        action_mask = action_mask[:truncated_length]
        token_labels = token_labels[:truncated_length]

        if not input_ids:
            raise ValueError(
                f"rows[{index}] has no tokens after truncation; "
                "increase rft_handoff.max_sequence_length."
            )

        # Match verl SFT behavior by masking the final token target.
        action_mask[-1] = 0

        task_id = _require_non_empty_task_id(row.get("task_id"), index=index)
        attempt_index = _coerce_int(row.get("attempt_index"), fallback=0)
        max_length = max(max_length, len(input_ids))
        packed_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "position_ids": list(range(len(input_ids))),
                "loss_mask": action_mask,
                "token_labels": token_labels,
                "original_length": len(input_ids),
                "group_id": _build_group_id(task_id=task_id, attempt_index=attempt_index),
                "task_id": task_id,
                "attempt_index": attempt_index,
                "step_index": _coerce_int(row.get("step_index"), fallback=index),
                "turn_index": _coerce_int(row.get("turn_index"), fallback=0),
                "resolved": _coerce_bool(row.get("resolved"), fallback=False),
                "is_terminal": _coerce_bool(row.get("is_terminal"), fallback=False),
                "format_valid": _coerce_bool(row.get("format_valid"), fallback=False),
            }
        )

    padded_input_ids: list[list[int]] = []
    padded_attention: list[list[int]] = []
    padded_position_ids: list[list[int]] = []
    padded_loss_mask: list[list[int]] = []
    padded_token_labels: list[list[str]] = []
    original_lengths: list[int] = []
    group_ids: list[str] = []
    task_ids: list[str] = []
    attempt_indexes: list[int] = []
    step_indexes: list[int] = []
    turn_indexes: list[int] = []
    resolved_flags: list[bool] = []
    terminal_flags: list[bool] = []
    format_valid_flags: list[bool] = []

    for row in packed_rows:
        pad_size = max_length - len(row["input_ids"])
        padded_input_ids.append(row["input_ids"] + [pad_token_id] * pad_size)
        padded_attention.append(row["attention_mask"] + [0] * pad_size)
        padded_position_ids.append(row["position_ids"] + [0] * pad_size)
        padded_loss_mask.append(row["loss_mask"] + [0] * pad_size)
        padded_token_labels.append(row["token_labels"] + ["other"] * pad_size)
        original_lengths.append(row["original_length"])
        group_ids.append(row["group_id"])
        task_ids.append(row["task_id"])
        attempt_indexes.append(row["attempt_index"])
        step_indexes.append(row["step_index"])
        turn_indexes.append(row["turn_index"])
        resolved_flags.append(row["resolved"])
        terminal_flags.append(row["is_terminal"])
        format_valid_flags.append(row["format_valid"])

    return {
        "tensors": {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention,
            "position_ids": padded_position_ids,
            "loss_mask": padded_loss_mask,
        },
        "grouping_metadata": {
            "group_id": group_ids,
            "task_id": task_ids,
            "attempt_index": attempt_indexes,
            "step_index": step_indexes,
            "turn_index": turn_indexes,
            "resolved": resolved_flags,
            "is_terminal": terminal_flags,
            "format_valid": format_valid_flags,
            "token_labels": padded_token_labels,
            "original_length": original_lengths,
        },
        "meta_info": {
            "selected_count": len(padded_input_ids),
            "max_padded_length": max_length,
            "max_sequence_length_limit": padded_limit,
        },
    }


def build_dataproto_compatible_payload(sft_batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return DataProto-compatible payload buckets (tensors, non_tensors, meta_info)."""
    tensors = dict(sft_batch.get("tensors", {}))
    grouping_metadata = dict(sft_batch.get("grouping_metadata", {}))
    meta_info = dict(sft_batch.get("meta_info", {}))
    return {
        "tensors": tensors,
        "non_tensors": grouping_metadata,
        "meta_info": meta_info,
    }


def write_jsonl_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")


def _flatten_rollout_steps(steps: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for step_rows in steps:
        flattened.extend(dict(row) for row in step_rows)
    return flattened


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


def _coerce_int(value: Any, *, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return fallback
    return fallback


def _coerce_int_sequence(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of ints.")
    return [int(item) for item in value]


def _coerce_loss_mask_sequence(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of bool/int values.")
    return [1 if _coerce_bool(item, fallback=False) else 0 for item in value]


def _coerce_token_labels(value: Any, *, length_hint: int) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return ["other"] * length_hint


def _build_group_id(*, task_id: str, attempt_index: int) -> str:
    return f"{task_id}#attempt-{attempt_index}"


def _require_non_empty_task_id(value: Any, *, index: int) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    raise ValueError(f"rows[{index}].task_id must be a non-empty string.")


def _build_rollout_artifact_summary(
    *,
    rollout_rows: Sequence[Mapping[str, Any]],
    total_steps: int,
    selected_count: int,
    rejected_count: int,
) -> dict[str, Any]:
    unique_task_ids: list[str] = []
    unique_image_names: list[str] = []
    seen_task_ids: set[str] = set()
    seen_image_names: set[str] = set()
    task_image_pairs: list[dict[str, str]] = []
    seen_task_image_pairs: set[tuple[str, str]] = set()
    rows_with_trajectory_steps = 0
    trajectory_step_count = 0

    for index, row in enumerate(rollout_rows):
        task_id = _require_non_empty_task_id(row.get("task_id"), index=index)
        if task_id not in seen_task_ids:
            seen_task_ids.add(task_id)
            unique_task_ids.append(task_id)

        image_name_raw = row.get("image_name", "")
        image_name = image_name_raw.strip() if isinstance(image_name_raw, str) else ""
        if image_name and image_name not in seen_image_names:
            seen_image_names.add(image_name)
            unique_image_names.append(image_name)

        if image_name:
            pair = (task_id, image_name)
            if pair not in seen_task_image_pairs:
                seen_task_image_pairs.add(pair)
                task_image_pairs.append({"task_id": task_id, "image_name": image_name})

        trajectory_steps = row.get("trajectory_steps")
        if isinstance(trajectory_steps, Sequence) and not isinstance(trajectory_steps, (str, bytes)):
            step_count = len(trajectory_steps)
            if step_count > 0:
                rows_with_trajectory_steps += 1
                trajectory_step_count += step_count

    return {
        "total_steps": int(total_steps),
        "rollout_row_count": len(rollout_rows),
        "selected_count": int(selected_count),
        "rejected_count": int(rejected_count),
        "unique_task_ids": unique_task_ids,
        "unique_image_names": unique_image_names,
        "task_image_pairs": task_image_pairs,
        "rows_with_trajectory_steps": rows_with_trajectory_steps,
        "trajectory_step_count": trajectory_step_count,
    }
