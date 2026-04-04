"""Core on-policy RFT rollout handoff and SFT-batch assembly logic."""

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
from trainer.rft_rejection import select_rft_attempt_rows
from verl_integration.data_preprocessor import preprocess_trajectories

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


def build_onpolicy_collector(
    *,
    turn_generator: AssistantTurnGenerator | None = None,
    data_config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    runtime_overrides: Mapping[str, Any] | None = None,
    data_overrides: Mapping[str, Any] | None = None,
    task_partition: str = "all",
    task_eval_split_fraction: float = 0.0,
    task_eval_min_rows: int = 0,
    stage_name: str = "format_rft",
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
        task_partition=task_partition,
        task_eval_split_fraction=task_eval_split_fraction,
        task_eval_min_rows=task_eval_min_rows,
        stage_name=stage_name,
    )


def collect_rollouts_for_steps(
    *,
    total_steps: int,
    start_step_index: int = 0,
    collector: OnPolicyRolloutCollector,
    output_dir: str | Path | None = None,
) -> list[list[RolloutRow]]:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if start_step_index < 0:
        raise ValueError("start_step_index must be >= 0")

    all_rows: list[list[RolloutRow]] = []
    base_dir = Path(output_dir) if output_dir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for local_step_index in range(total_steps):
        rows = collector.collect_step(start_step_index + local_step_index)
        all_rows.append(rows)
        if base_dir is not None:
            output_path = base_dir / f"step_{local_step_index:05d}.jsonl"
            write_jsonl_rows(output_path, rows)
    return all_rows


def collect_rft_sft_batch_for_steps(
    *,
    total_steps: int,
    start_step_index: int = 0,
    collector: OnPolicyRolloutCollector,
    tokenizer: SupportsOffsetsTokenizer,
    handoff_overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Collect rollouts, apply centralized RFT rejection policy, and build SFT tensors."""
    rollout_steps = collect_rollouts_for_steps(
        total_steps=total_steps,
        start_step_index=start_step_index,
        collector=collector,
        output_dir=output_dir,
    )
    rollout_rows = _flatten_rollout_steps(rollout_steps)
    handoff_settings = resolve_rft_handoff_settings(overrides=handoff_overrides)
    if rollout_rows:
        preprocessed_rows = preprocess_trajectories(
            rollout_rows,
            max_tool_calls=collector.settings.runtime.max_tool_calls_per_turn,
            tokenizer=tokenizer,
        )
        merged_rows = merge_rollout_and_preprocessed_rows(rollout_rows, preprocessed_rows)

        selected_rows, rejected_rows = select_rft_attempt_rows(
            merged_rows,
            selection_policy=handoff_settings.selection,
        )
        if selected_rows:
            selected_rows, overlength_selected_rows = _partition_rows_by_handoff_length(
                selected_rows,
                padded_limit=handoff_settings.max_sequence_length,
            )
            if overlength_selected_rows:
                rejected_rows = [*rejected_rows, *overlength_selected_rows]
        if selected_rows:
            sft_batch = build_verl_sft_batch(
                selected_rows,
                handoff_settings=handoff_settings,
                tokenizer=tokenizer,
            )
        else:
            sft_batch = _build_empty_verl_sft_batch(handoff_settings=handoff_settings)
    else:
        selected_rows = []
        rejected_rows = []
        sft_batch = _build_empty_verl_sft_batch(handoff_settings=handoff_settings)
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
                "max_turn_level_generated_tokens": sft_batch["meta_info"][
                    "max_turn_level_generated_tokens"
                ],
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
        trajectory_format_valid = _coerce_bool(
            rollout_row.get("trajectory_format_valid"),
            fallback=_coerce_bool(preprocessed_row.get("format_valid"), fallback=False),
        )
        final_turn_has_submit = _coerce_bool(
            rollout_row.get("final_turn_has_submit"),
            fallback=_coerce_bool(rollout_row.get("is_terminal"), fallback=False),
        )
        final_submit_format_valid = _coerce_bool(
            rollout_row.get("terminal_format_valid"),
            fallback=trajectory_format_valid and final_turn_has_submit,
        )
        final_submit_format_valid = _coerce_bool(
            rollout_row.get("final_submit_format_valid"),
            fallback=final_submit_format_valid,
        )
        fail_to_pass = rollout_row.get("fail_to_pass", rollout_row.get("FAIL_TO_PASS"))
        pass_to_pass = rollout_row.get("pass_to_pass", rollout_row.get("PASS_TO_PASS"))
        verifier_status, verifier_resolution_source = _derive_verifier_telemetry(
            rollout_row=rollout_row,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        merged = dict(preprocessed_row)
        merged.update(
            {
                "stage": rollout_row.get("stage", preprocessed_row.get("stage", "format_rft")),
                "task_id": task_id,
                "attempt_index": rollout_row.get("attempt_index", 0),
                "turn_index": rollout_row.get("turn_index", 0),
                "step_index": rollout_row.get("step_index", 0),
                "resolved": rollout_row.get("resolved", False),
                "verifier_kind": rollout_row.get("verifier_kind", "pytest"),
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "is_terminal": rollout_row.get("is_terminal", False),
                "collector_error": rollout_row.get("collector_error", ""),
                "bridge_error": rollout_row.get("bridge_error", ""),
                "timeout_error": rollout_row.get("timeout_error", ""),
                "executor_error": rollout_row.get("executor_error", ""),
                "infra_invalid": rollout_row.get("infra_invalid", False),
                "invalid_reason": rollout_row.get("invalid_reason", ""),
                "hit_generation_cap": rollout_row.get("hit_generation_cap", False),
                "container_init_succeeded": rollout_row.get("container_init_succeeded", False),
                "exit_code": rollout_row.get("exit_code"),
                "tool_name": rollout_row.get("tool_name", ""),
                "container_id": rollout_row.get("container_id", ""),
                "image_name": rollout_row.get("image_name", ""),
                "format_valid": trajectory_format_valid,
                "trajectory_format_valid": trajectory_format_valid,
                "final_turn_has_submit": final_turn_has_submit,
                "terminal_format_valid": final_submit_format_valid,
                "final_submit_format_valid": final_submit_format_valid,
                "verifier_status": verifier_status,
                "verifier_resolution_source": verifier_resolution_source,
                "trajectory_tool_validation_errors": rollout_row.get(
                    "trajectory_tool_validation_errors",
                    (),
                ),
                "trajectory_assistant_turns": rollout_row.get("trajectory_assistant_turns", ()),
                "trajectory_steps": rollout_row.get("trajectory_steps", ()),
                "trajectory_history": rollout_row.get("trajectory_history", ()),
            }
        )
        merged_rows.append(merged)
    return merged_rows


def _partition_rows_by_handoff_length(
    rows: Sequence[Mapping[str, Any]],
    *,
    padded_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        input_ids = _coerce_int_sequence(
            row.get("input_ids"),
            label=f"rows[{index}].input_ids",
        )
        action_mask = _coerce_loss_mask_sequence(
            row.get("action_mask_rft"),
            label=f"rows[{index}].action_mask_rft",
        )
        token_labels = _coerce_token_labels(
            row.get("token_labels"),
            length_hint=len(input_ids),
        )
        if (
            len(input_ids) > padded_limit
            or len(action_mask) > padded_limit
            or len(token_labels) > padded_limit
        ):
            rejected_row = dict(row)
            rejected_row["selected_over_budget"] = True
            rejected_row["rft_rejection_reason"] = "selected_over_handoff_length"
            rejected_rows.append(rejected_row)
            continue
        kept_rows.append(dict(row))
    return kept_rows, rejected_rows


def build_verl_sft_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    handoff_settings: RFTHandoffSettings,
    tokenizer: SupportsOffsetsTokenizer | None = None,
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
    max_turn_level_generated_tokens = 0

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

        if (
            len(input_ids) > padded_limit
            or len(action_mask) > padded_limit
            or len(token_labels) > padded_limit
        ):
            raise ValueError(
                f"rows[{index}] exceeds rft_handoff.max_sequence_length={padded_limit}; "
                "selected rows should be filtered before handoff."
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
        turn_level_generated_tokens = _resolve_turn_level_generated_token_count(
            row=row,
            tokenizer=tokenizer,
        )
        max_turn_level_generated_tokens = max(
            max_turn_level_generated_tokens,
            turn_level_generated_tokens,
        )
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
            "max_turn_level_generated_tokens": max_turn_level_generated_tokens,
            "max_sequence_length_limit": padded_limit,
        },
    }


def _build_empty_verl_sft_batch(*, handoff_settings: RFTHandoffSettings) -> dict[str, Any]:
    return {
        "tensors": {
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [],
            "loss_mask": [],
        },
        "grouping_metadata": {
            "group_id": [],
            "task_id": [],
            "attempt_index": [],
            "step_index": [],
            "turn_index": [],
            "resolved": [],
            "is_terminal": [],
            "format_valid": [],
            "token_labels": [],
            "original_length": [],
        },
        "meta_info": {
            "selected_count": 0,
            "max_turn_level_generated_tokens": 0,
            "max_sequence_length_limit": int(handoff_settings.max_sequence_length),
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


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
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
    return None


def _has_expected_verifier_targets(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


def _derive_verifier_telemetry(
    *,
    rollout_row: Mapping[str, Any],
    fail_to_pass: Any,
    pass_to_pass: Any,
) -> tuple[str, str]:
    status = str(rollout_row.get("verifier_status", "")).strip()
    source = str(rollout_row.get("verifier_resolution_source", "")).strip()
    if status and source:
        return status, source

    has_expected_targets = _has_expected_verifier_targets(fail_to_pass) or _has_expected_verifier_targets(
        pass_to_pass
    )
    verification_missing = _coerce_optional_bool(rollout_row.get("verification_missing"))
    resolved = _coerce_optional_bool(rollout_row.get("resolved"))
    has_explicit_verifier_signal = any(
        key in rollout_row
        for key in (
            "fail_to_pass_verified",
            "pass_to_pass_verified",
            "verification_missing",
            "verification_error",
            "verification_feedback",
            "submission_final_response",
        )
    )
    if not has_explicit_verifier_signal or verification_missing is True or resolved is None:
        return (
            "missing",
            "missing_verifier" if has_expected_targets else "missing_verifier_targets",
        )
    return ("correct" if resolved else "incorrect", "verifiable_tests")


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


def _resolve_turn_level_generated_token_count(
    *,
    row: Mapping[str, Any],
    tokenizer: SupportsOffsetsTokenizer | None,
) -> int:
    if tokenizer is not None:
        assistant_response_raw = row.get("assistant_response")
        assistant_response = (
            assistant_response_raw
            if isinstance(assistant_response_raw, str)
            else str(assistant_response_raw or "")
        )
        if assistant_response:
            tokenized = _tokenize_text(tokenizer=tokenizer, text=assistant_response)
            if tokenized:
                return len(tokenized)
            # If tokenizer unexpectedly fails, do not silently drop to zero.
            # Fall back to row-level ids to preserve monotonic diagnostics.

    # Fallback for older tests/callers that do not pass tokenizer and for
    # defensive recovery when assistant-response tokenization is unavailable.
    input_ids = row.get("input_ids")
    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        return len(input_ids)
    return 0


def _tokenize_text(*, tokenizer: SupportsOffsetsTokenizer, text: str) -> list[int]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=False,
        )
    except TypeError:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
        )
    except Exception:
        return []

    if not isinstance(encoded, Mapping):
        return []
    raw_input_ids = encoded.get("input_ids")
    if not isinstance(raw_input_ids, Sequence) or isinstance(raw_input_ids, (str, bytes)):
        return []
    if raw_input_ids and isinstance(raw_input_ids[0], Sequence) and not isinstance(
        raw_input_ids[0],
        (str, bytes),
    ):
        raw_input_ids = raw_input_ids[0]
    return [int(token_id) for token_id in raw_input_ids]


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
