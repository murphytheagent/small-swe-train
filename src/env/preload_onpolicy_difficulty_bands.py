"""CLI helper that materializes rollout-backed on-policy difficulty bands."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import (
    DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    OnPolicyDataConfig,
    resolve_on_policy_settings,
    rft_runtime_defaults,
)
from env.task_dataset import (
    TaskSample,
    load_task_samples,
    resolve_on_policy_difficulty_band_cache_path,
)
from runtime_paths import resolve_on_policy_difficulty_band_cache_dir
from trainer.rft_handoff import build_onpolicy_collector, build_rft_handoff_result_from_rollout_rows
from trainer.rft_runtime import _resolve_turn_generator
from trainer.rft_runtime_loop import (
    resolve_rft_stage_correctness_contract,
    _load_tokenizer,
    resolve_rft_stage_handoff_overrides,
    resolve_rft_stage_name,
    resolve_rft_stage_selection_contract,
    resolve_rft_stage_verify_submissions,
)

DEFAULT_PROBE_LABEL = "positive_rft_probe"
DEFAULT_ATTEMPTS_PER_TASK = 4
PROBE_STATUS_COMPLETE = "complete"
PROBE_STATUS_INCOMPLETE = "incomplete"
PARTIAL_PROGRESS_RECORDS_SUFFIX = ".partial.records.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve or materialize rollout-backed on-policy difficulty bands.",
    )
    parser.add_argument(
        "--data-config-name",
        default=DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
        help="Named config from configs/data/<name>.yaml (default: on_policy_swe_smith).",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Directory that stores the difficulty-band cache JSON (default: data/on_policy_difficulty_band_cache).",
    )
    parser.add_argument(
        "--probe-label",
        default=DEFAULT_PROBE_LABEL,
        help="Descriptive cache label appended to the output filename.",
    )
    parser.add_argument(
        "--print-path-only",
        action="store_true",
        help="Print the resolved cache path without running rollouts.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild cache even when the resolved JSON already exists.",
    )
    parser.add_argument(
        "--initial-model",
        required=True,
        help="Model or checkpoint path used to load the tokenizer for length filtering.",
    )
    parser.add_argument(
        "--turn-generator-mode",
        default="default",
        help="Turn-generator mode passed through to the live RFT runtime (default: default).",
    )
    parser.add_argument(
        "--stage-name",
        default="positive_rft",
        help="RFT stage contract used for verifier-backed selection (default: positive_rft).",
    )
    parser.add_argument(
        "--task-partition",
        default="all",
        choices=("all", "train", "eval"),
        help="Dataset partition to probe after the deterministic held-out split logic (default: all).",
    )
    parser.add_argument(
        "--attempts-per-task",
        type=int,
        default=DEFAULT_ATTEMPTS_PER_TASK,
        help=f"Rollout attempts per task during probing (default: {DEFAULT_ATTEMPTS_PER_TASK}).",
    )
    parser.add_argument(
        "--start-task-index",
        type=int,
        default=0,
        help="Start probing from this deterministic task index (default: 0).",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Probe at most this many tasks after start-task-index.",
    )
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum number of tasks kept queued in each probe window before refilling. "
            "Active concurrency is capped separately by --env-pool-size and "
            "--max-in-flight-tasks (default: 1)."
        ),
    )
    parser.add_argument(
        "--env-pool-size",
        type=int,
        default=None,
        help=(
            "Upper bound on concurrently active task environments during probing. "
            "Defaults to --task-batch-size when omitted."
        ),
    )
    parser.add_argument(
        "--max-in-flight-tasks",
        type=int,
        default=None,
        help=(
            "Upper bound on concurrently active probe tasks. Defaults to "
            "--env-pool-size when omitted."
        ),
    )
    parser.add_argument(
        "--eval-split-fraction",
        type=float,
        default=None,
        help=(
            "Held-out split fraction used when probing the deterministic train/eval partition. "
            "Defaults to the runtime-loop setting for train/eval partitions."
        ),
    )
    parser.add_argument(
        "--min-eval-rows",
        type=int,
        default=None,
        help=(
            "Minimum held-out rows used when probing the deterministic train/eval partition. "
            "Defaults to the runtime-loop setting for train/eval partitions."
        ),
    )
    return parser


def _resolve_cache_dir(raw_value: str) -> Path:
    normalized = str(raw_value or "").strip()
    if normalized:
        return Path(normalized)
    project_root = Path(__file__).resolve().parents[2]
    return resolve_on_policy_difficulty_band_cache_dir(project_root=project_root)


def _assign_difficulty_band(*, selected_count: int, rollout_count: int) -> str:
    if selected_count <= 0:
        return "near_impossible"
    if rollout_count <= 0:
        return "near_impossible"

    easy_cutoff = max(1, math.ceil(0.75 * rollout_count))
    if selected_count >= easy_cutoff:
        return "easy"
    return "learnable"


def _count_rejection_reasons(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason_text = str(
            row.get("stage_decision_reason", row.get("rft_rejection_reason", ""))
        ).strip()
        reasons = [reason.strip() for reason in reason_text.split(",") if reason.strip()]
        if not reasons:
            reasons = ["unknown"]
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _build_band_record(
    *,
    task: TaskSample,
    probe_step_index: int,
    stage_name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    rollout_rows = _coerce_rows(result.get("rollout_rows"))
    selected_rows = _coerce_rows(result.get("selected_rows"))
    rejected_rows = _coerce_rows(result.get("rejected_rows"))

    selected_over_length_count = sum(
        1 for row in rejected_rows if bool(row.get("selected_over_budget", False))
    )
    selected_count = len(selected_rows)
    selected_count_raw = selected_count + selected_over_length_count
    rollout_count = len(rollout_rows)
    resolved_attempt_count = sum(1 for row in rollout_rows if bool(row.get("resolved", False)))
    infra_invalid_attempt_count = sum(
        1 for row in rollout_rows if bool(row.get("infra_invalid", False))
    )
    if rollout_count > 0 and infra_invalid_attempt_count >= rollout_count:
        raise ValueError(
            "Difficulty probe produced only infra-invalid attempts for "
            f"task_id={task.task_id!r}; refusing to assign a difficulty band."
        )
    difficulty_band = _assign_difficulty_band(
        selected_count=selected_count,
        rollout_count=rollout_count,
    )

    return {
        "task_id": task.task_id,
        "task_family": task.task_family,
        "difficulty_band": difficulty_band,
        "difficulty_band_source": f"rollout_probe:selected_{selected_count}_of_{rollout_count}",
        "probe_step_index": int(probe_step_index),
        "stage_name": stage_name,
        "rollout_count": rollout_count,
        "resolved_attempt_count": resolved_attempt_count,
        "selected_count_raw": selected_count_raw,
        "selected_count_after_length_filter": selected_count,
        "selected_over_length_count": selected_over_length_count,
        "infra_invalid_attempt_count": infra_invalid_attempt_count,
        "rejection_reason_counts": _count_rejection_reasons(rejected_rows),
    }


def _resolve_probe_parallelism(
    *,
    parser: argparse.ArgumentParser,
    task_batch_size: int,
    env_pool_size: int | None,
    max_in_flight_tasks: int | None,
) -> tuple[int, int, int]:
    if task_batch_size < 1:
        parser.error("--task-batch-size must be >= 1")

    resolved_env_pool_size = task_batch_size if env_pool_size is None else int(env_pool_size)
    if resolved_env_pool_size < 1:
        parser.error("--env-pool-size must be >= 1 when provided")
    resolved_env_pool_size = min(resolved_env_pool_size, task_batch_size)

    if max_in_flight_tasks is None:
        resolved_max_in_flight_tasks = resolved_env_pool_size
    else:
        resolved_max_in_flight_tasks = int(max_in_flight_tasks)
    if resolved_max_in_flight_tasks < 1:
        parser.error("--max-in-flight-tasks must be >= 1 when provided")
    resolved_max_in_flight_tasks = min(
        resolved_max_in_flight_tasks,
        task_batch_size,
        resolved_env_pool_size,
    )

    return task_batch_size, resolved_env_pool_size, resolved_max_in_flight_tasks


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _filter_rows_for_task(
    rows: Any,
    *,
    task_id: str,
    fallback_task_id: str = "",
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for row in _coerce_rows(rows):
        row_task_id = str(row.get("task_id", "")).strip()
        if row_task_id:
            if row_task_id == task_id:
                matched.append(row)
            continue
        if fallback_task_id and task_id == fallback_task_id:
            matched.append(row)
    return matched


def _extract_task_result(
    *,
    result: Mapping[str, Any],
    task_id: str,
    fallback_task_id: str = "",
) -> dict[str, Any]:
    return {
        "rollout_rows": _filter_rows_for_task(
            result.get("rollout_rows"),
            task_id=task_id,
            fallback_task_id=fallback_task_id,
        ),
        "selected_rows": _filter_rows_for_task(
            result.get("selected_rows"),
            task_id=task_id,
            fallback_task_id=fallback_task_id,
        ),
        "rejected_rows": _filter_rows_for_task(
            result.get("rejected_rows"),
            task_id=task_id,
            fallback_task_id=fallback_task_id,
        ),
    }


def _build_dataset_loader_for_tasks(
    tasks: Sequence[TaskSample],
) -> Callable[[str, str], Sequence[Mapping[str, Any]]]:
    rows = []
    for task in tasks:
        row = dict(task.raw)
        row.setdefault("task_id", task.task_id)
        row.setdefault("verifier_kind", task.verifier_kind)
        row.setdefault("task_family", task.task_family)
        row.setdefault("difficulty_band", task.difficulty_band)
        row.setdefault("difficulty_band_source", task.difficulty_band_source)
        rows.append(row)
    rows = tuple(rows)

    def _load_dataset(_dataset_id: str, _split: str) -> Sequence[Mapping[str, Any]]:
        return rows

    return _load_dataset


def _resolve_partition_eval_settings(
    *,
    parser: argparse.ArgumentParser,
    task_partition: str,
    eval_split_fraction: float | None,
    min_eval_rows: int | None,
) -> tuple[float, int]:
    if task_partition == "all":
        if eval_split_fraction is not None or min_eval_rows is not None:
            parser.error(
                "--eval-split-fraction and --min-eval-rows require --task-partition train or eval."
            )
        return 0.0, 0

    default_eval_split_fraction, default_min_eval_rows = _resolve_eval_split_defaults()
    resolved_eval_split_fraction = (
        float(eval_split_fraction)
        if eval_split_fraction is not None
        else default_eval_split_fraction
    )
    resolved_min_eval_rows = (
        int(min_eval_rows)
        if min_eval_rows is not None
        else default_min_eval_rows
    )
    if resolved_eval_split_fraction < 0.0 or resolved_eval_split_fraction >= 1.0:
        parser.error("--eval-split-fraction must be in [0.0, 1.0).")
    if resolved_min_eval_rows < 0:
        parser.error("--min-eval-rows must be >= 0.")
    return resolved_eval_split_fraction, resolved_min_eval_rows


def _resolve_eval_split_defaults() -> tuple[float, int]:
    fallback_fraction = 0.1
    fallback_min_rows = 1
    defaults = rft_runtime_defaults()
    loop = defaults.get("loop")
    if not isinstance(loop, Mapping):
        return fallback_fraction, fallback_min_rows

    fraction_value = loop.get("eval_split_fraction")
    if isinstance(fraction_value, bool) or not isinstance(fraction_value, (int, float)):
        resolved_fraction = fallback_fraction
    else:
        resolved_fraction = float(fraction_value)
    if not 0.0 <= resolved_fraction < 1.0:
        resolved_fraction = fallback_fraction

    min_rows_value = loop.get("eval_min_rows")
    if isinstance(min_rows_value, bool) or not isinstance(min_rows_value, int) or min_rows_value < 0:
        resolved_min_rows = fallback_min_rows
    else:
        resolved_min_rows = int(min_rows_value)
    return resolved_fraction, resolved_min_rows


def _resolve_probe_source_data_overrides(
    config: OnPolicyDataConfig,
) -> dict[str, Any] | None:
    strategy = str(config.difficulty_banding.strategy).strip().lower()
    if strategy != "rollout_probe":
        return None
    return {
        "difficulty_banding": {
            "strategy": "none",
            "default_band": "unbanded",
            "family_band_exact": {},
            "family_band_prefix": {},
            "rollout_probe_cache_path": "",
            "rollout_probe_required": False,
        }
    }


def _load_cache_payload(cache_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Difficulty-band cache contains invalid JSON: {cache_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Difficulty-band cache payload must be a mapping: {cache_path}")
    return payload


def _resolve_partial_cache_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".partial.json")


def _resolve_partial_records_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(PARTIAL_PROGRESS_RECORDS_SUFFIX)


def _cache_payload_is_complete(payload: Mapping[str, Any]) -> bool:
    status_text = str(payload.get("probe_status", "")).strip().lower()
    return not status_text or status_text == PROBE_STATUS_COMPLETE


def _records_by_task_id(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Difficulty-band cache payload must define a records list.")

    records_by_task_id: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Difficulty-band cache records must be mappings.")
        task_id = str(raw_record.get("task_id", "")).strip()
        difficulty_band = str(raw_record.get("difficulty_band", "")).strip()
        if not task_id or not difficulty_band:
            raise ValueError(
                "Difficulty-band cache records must include non-empty task_id and "
                "difficulty_band fields."
            )
        if task_id in records_by_task_id:
            raise ValueError(f"Difficulty-band cache contains duplicate task_id {task_id!r}.")
        records_by_task_id[task_id] = dict(raw_record)
    return records_by_task_id


def _ordered_records_for_tasks(
    *,
    tasks: Sequence[TaskSample],
    records_by_task_id: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_records: list[dict[str, Any]] = []
    for task in tasks:
        record = records_by_task_id.get(task.task_id)
        if record is not None:
            ordered_records.append(dict(record))
    return ordered_records


def _build_cache_payload(
    *,
    expected_cache_metadata: Mapping[str, Any],
    selected_task_count: int,
    records: Sequence[Mapping[str, Any]],
    probe_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "probe_status": str(probe_status).strip().lower() or PROBE_STATUS_COMPLETE,
        **dict(expected_cache_metadata),
        "task_count": len(records),
        "task_count_completed": len(records),
        "task_count_expected": int(selected_task_count),
        "records": [dict(record) for record in records],
    }


def _build_partial_progress_payload(
    *,
    expected_cache_metadata: Mapping[str, Any],
    selected_task_count: int,
    completed_count: int,
    partial_records_path: Path,
    latest_completed_task_id: str = "",
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "probe_status": PROBE_STATUS_INCOMPLETE,
        **dict(expected_cache_metadata),
        "task_count": int(completed_count),
        "task_count_completed": int(completed_count),
        "task_count_expected": int(selected_task_count),
        "partial_records_path": partial_records_path.name,
    }
    normalized_task_id = str(latest_completed_task_id).strip()
    if normalized_task_id:
        payload["latest_completed_task_id"] = normalized_task_id
    return payload


def _write_cache_payload(
    *,
    cache_path: Path,
    payload: Mapping[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)


def _append_partial_record(
    *,
    partial_records_path: Path,
    record: Mapping[str, Any],
) -> None:
    partial_records_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True) + "\n")


def _load_partial_records(
    partial_records_path: Path,
) -> dict[str, dict[str, Any]]:
    records_by_task_id: dict[str, dict[str, Any]] = {}
    with partial_records_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Difficulty-band partial progress records contain invalid JSON: "
                    f"{partial_records_path}:{line_number}"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise ValueError(
                    "Difficulty-band partial progress records must be mappings: "
                    f"{partial_records_path}:{line_number}"
                )
            task_id = str(parsed.get("task_id", "")).strip()
            difficulty_band = str(parsed.get("difficulty_band", "")).strip()
            if not task_id or not difficulty_band:
                raise ValueError(
                    "Difficulty-band partial progress records must include non-empty "
                    f"task_id and difficulty_band fields: {partial_records_path}:{line_number}"
                )
            if task_id in records_by_task_id:
                raise ValueError(
                    "Difficulty-band partial progress records contain duplicate task_id "
                    f"{task_id!r}: {partial_records_path}:{line_number}"
                )
            records_by_task_id[task_id] = dict(parsed)
    return records_by_task_id


def _load_resumable_records(
    *,
    payload: Mapping[str, Any],
    tasks: Sequence[TaskSample],
    partial_records_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if "records" in payload:
        records_by_task_id = _records_by_task_id(payload)
    else:
        resolved_partial_records_path = partial_records_path
        if resolved_partial_records_path is None:
            raw_path = str(payload.get("partial_records_path", "")).strip()
            if not raw_path:
                raise ValueError(
                    "Difficulty-band progress cache is missing both records and "
                    "partial_records_path."
                )
            resolved_partial_records_path = Path(raw_path)
        if not resolved_partial_records_path.is_absolute():
            if cache_path is None:
                raise ValueError(
                    "Difficulty-band progress cache partial_records_path must resolve to an "
                    "absolute path when loaded without an explicit cache path."
                )
            resolved_partial_records_path = cache_path.parent / resolved_partial_records_path
        if not resolved_partial_records_path.is_file():
            raise ValueError(
                "Difficulty-band progress cache is missing its partial records file: "
                f"{resolved_partial_records_path}"
            )
        records_by_task_id = _load_partial_records(resolved_partial_records_path)

    expected_task_ids = {task.task_id for task in tasks}
    unexpected_task_ids = sorted(set(records_by_task_id) - expected_task_ids)
    if unexpected_task_ids:
        raise ValueError(
            "Difficulty-band progress cache includes task_ids outside the selected task pool: "
            + ", ".join(repr(task_id) for task_id in unexpected_task_ids[:5])
        )

    expected_count = payload.get("task_count_expected")
    if expected_count is not None and int(expected_count) != len(tasks):
        raise ValueError(
            "Difficulty-band progress cache task_count_expected does not match the selected "
            f"task pool: expected {len(tasks)}, got {expected_count!r}."
        )
    return records_by_task_id


def _build_task_pool_fingerprint(tasks: Sequence[TaskSample]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(task.task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(task.task_family.encode("utf-8"))
        digest.update(b"\0")
        digest.update(task.image_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(task.problem_statement.encode("utf-8"))
        digest.update(b"\0")
        for target in task.fail_to_pass:
            digest.update(target.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\1")
        for target in task.pass_to_pass:
            digest.update(target.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\2")
    return digest.hexdigest()


def _build_expected_cache_metadata(
    *,
    settings: Any,
    data_config_name: str,
    probe_label: str,
    initial_model: str,
    turn_generator_mode: str,
    stage_name: str,
    task_partition: str,
    attempts_per_task: int,
    start_task_index: int,
    task_limit: int | None,
    eval_split_fraction: float,
    min_eval_rows: int,
    verify_submissions: bool,
    stage_handoff_overrides: Mapping[str, Any],
    stage_selection_contract: Mapping[str, Any],
    stage_correctness_contract: str,
    task_pool_size: int,
    task_pool_fingerprint: str,
) -> dict[str, Any]:
    return {
        "data_config_name": data_config_name,
        "dataset_id": settings.data.dataset_id,
        "dataset_split": settings.data.dataset_split,
        "patch_is_bug_introducing": bool(settings.data.patch_is_bug_introducing),
        "verifier_kind": str(settings.data.verifier_kind),
        "probe_label": probe_label,
        "initial_model": initial_model,
        "turn_generator_mode": turn_generator_mode,
        "stage_name": stage_name,
        "task_partition": task_partition,
        "attempts_per_task": attempts_per_task,
        "start_task_index": start_task_index,
        "task_limit": task_limit,
        "eval_split_fraction": eval_split_fraction,
        "min_eval_rows": min_eval_rows,
        "verify_submissions": bool(verify_submissions),
        "stage_handoff_overrides": dict(stage_handoff_overrides),
        "stage_selection_contract": dict(stage_selection_contract),
        "stage_correctness_contract": str(stage_correctness_contract),
        "task_pool_size": task_pool_size,
        "task_pool_fingerprint": task_pool_fingerprint,
    }


def _collect_probe_rollout_rows_for_task(
    *,
    data_config_name: str,
    turn_generator_mode: str,
    attempts_per_task: int,
    data_overrides: Mapping[str, Any] | None,
    verify_submissions: bool,
    stage_name: str,
    task: TaskSample,
    probe_step_index: int,
) -> list[dict[str, Any]]:
    collector = build_onpolicy_collector(
        data_config_name=data_config_name,
        turn_generator=_resolve_turn_generator(turn_generator_mode),
        runtime_overrides={
            "task_batch_size": 1,
            "attempts_per_task": int(attempts_per_task),
            "env_pool_size": 1,
            "max_in_flight_tasks": 1,
            "verify_submissions": bool(verify_submissions),
        },
        data_overrides=data_overrides,
        task_partition="all",
        task_eval_split_fraction=0.0,
        task_eval_min_rows=0,
        stage_name=stage_name,
        dataset_loader=_build_dataset_loader_for_tasks([task]),
    )
    rows = collector.collect_step(probe_step_index)
    return [dict(row) for row in rows]


def _probe_tasks_with_progress_writes(
    *,
    remaining_indexed_tasks: Sequence[tuple[int, TaskSample]],
    expected_cache_metadata: Mapping[str, Any],
    selected_task_count: int,
    partial_cache_path: Path,
    partial_records_path: Path,
    existing_records_by_task_id: dict[str, dict[str, Any]],
    submission_window: int,
    max_workers: int,
    data_config_name: str,
    turn_generator_mode: str,
    attempts_per_task: int,
    probe_source_data_overrides: Mapping[str, Any] | None,
    handoff_overrides: Mapping[str, Any],
    verify_submissions: bool,
    stage_name: str,
    max_tool_calls: int,
    tokenizer: Any,
) -> None:
    if not remaining_indexed_tasks:
        return

    task_iterator = iter(remaining_indexed_tasks)
    pending: dict[Future[list[dict[str, Any]]], tuple[int, TaskSample]] = {}
    normalized_submission_window = max(1, int(submission_window))
    normalized_worker_count = max(1, int(max_workers))

    _write_cache_payload(
        cache_path=partial_cache_path,
        payload=_build_partial_progress_payload(
            expected_cache_metadata=expected_cache_metadata,
            selected_task_count=selected_task_count,
            completed_count=len(existing_records_by_task_id),
            partial_records_path=partial_records_path,
        ),
    )

    def _submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            probe_step_index, task = next(task_iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _collect_probe_rollout_rows_for_task,
            data_config_name=data_config_name,
            turn_generator_mode=turn_generator_mode,
            attempts_per_task=attempts_per_task,
            data_overrides=probe_source_data_overrides,
            verify_submissions=verify_submissions,
            stage_name=stage_name,
            task=task,
            probe_step_index=probe_step_index,
        )
        pending[future] = (probe_step_index, task)
        return True

    executor = ThreadPoolExecutor(max_workers=normalized_worker_count)
    executor_closed = False
    try:
        while len(pending) < normalized_submission_window and _submit_next(executor):
            pass

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                probe_step_index, task = pending.pop(future)
                rollout_rows = future.result()
                task_result = build_rft_handoff_result_from_rollout_rows(
                    rollout_rows=rollout_rows,
                    max_tool_calls=max_tool_calls,
                    tokenizer=tokenizer,
                    handoff_overrides=handoff_overrides,
                )
                record = _build_band_record(
                    task=task,
                    probe_step_index=probe_step_index,
                    stage_name=stage_name,
                    result=task_result,
                )
                existing_records_by_task_id[task.task_id] = record
                _append_partial_record(
                    partial_records_path=partial_records_path,
                    record=record,
                )
                _write_cache_payload(
                    cache_path=partial_cache_path,
                    payload=_build_partial_progress_payload(
                        expected_cache_metadata=expected_cache_metadata,
                        selected_task_count=selected_task_count,
                        completed_count=len(existing_records_by_task_id),
                        partial_records_path=partial_records_path,
                        latest_completed_task_id=task.task_id,
                    ),
                )
                while len(pending) < normalized_submission_window and _submit_next(executor):
                    pass
    except Exception:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        executor_closed = True
        raise
    finally:
        if not executor_closed:
            executor.shutdown(wait=True)


def _cache_metadata_matches(
    *,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(payload.get(key) == value for key, value in expected.items())


def _cache_payload_covers_tasks(
    *,
    payload: Mapping[str, Any],
    tasks: Sequence[TaskSample],
) -> bool:
    if not _cache_payload_is_complete(payload):
        return False

    expected_task_ids = {task.task_id for task in tasks}
    try:
        records_by_task_id = _records_by_task_id(payload)
    except ValueError:
        return False
    return set(records_by_task_id) == expected_task_ids


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.attempts_per_task < 1:
        parser.error("--attempts-per-task must be >= 1")
    if args.start_task_index < 0:
        parser.error("--start-task-index must be >= 0")
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("--task-limit must be >= 1 when provided")
    (
        resolved_task_batch_size,
        resolved_env_pool_size,
        resolved_max_in_flight_tasks,
    ) = _resolve_probe_parallelism(
        parser=parser,
        task_batch_size=int(args.task_batch_size),
        env_pool_size=args.env_pool_size,
        max_in_flight_tasks=args.max_in_flight_tasks,
    )

    settings = resolve_on_policy_settings(data_config_name=args.data_config_name)
    resolved_eval_split_fraction, resolved_min_eval_rows = _resolve_partition_eval_settings(
        parser=parser,
        task_partition=args.task_partition,
        eval_split_fraction=args.eval_split_fraction,
        min_eval_rows=args.min_eval_rows,
    )
    cache_dir = _resolve_cache_dir(args.cache_dir)
    cache_path = resolve_on_policy_difficulty_band_cache_path(
        config=settings.data,
        cache_dir=cache_dir,
        probe_label=args.probe_label,
        task_partition=args.task_partition,
        start_task_index=int(args.start_task_index),
        task_limit=int(args.task_limit) if args.task_limit is not None else None,
        eval_split_fraction=resolved_eval_split_fraction,
        min_eval_rows=resolved_min_eval_rows,
    )
    resolved_stage_name = resolve_rft_stage_name(args.stage_name)
    verify_submissions = resolve_rft_stage_verify_submissions(resolved_stage_name)
    handoff_overrides = resolve_rft_stage_handoff_overrides(resolved_stage_name)
    stage_selection_contract = resolve_rft_stage_selection_contract(resolved_stage_name)
    stage_correctness_contract = resolve_rft_stage_correctness_contract(resolved_stage_name)
    probe_source_data_overrides = _resolve_probe_source_data_overrides(settings.data)
    expected_cache_metadata = _build_expected_cache_metadata(
        settings=settings,
        data_config_name=str(args.data_config_name).strip(),
        probe_label=str(args.probe_label).strip(),
        initial_model=str(args.initial_model).strip(),
        turn_generator_mode=str(args.turn_generator_mode).strip(),
        stage_name=resolved_stage_name,
        task_partition=str(args.task_partition).strip(),
        attempts_per_task=int(args.attempts_per_task),
        start_task_index=int(args.start_task_index),
        task_limit=int(args.task_limit) if args.task_limit is not None else None,
        eval_split_fraction=resolved_eval_split_fraction,
        min_eval_rows=resolved_min_eval_rows,
        verify_submissions=verify_submissions,
        stage_handoff_overrides=handoff_overrides,
        stage_selection_contract=stage_selection_contract,
        stage_correctness_contract=stage_correctness_contract,
        task_pool_size=0,
        task_pool_fingerprint="",
    )
    if args.print_path_only:
        print(cache_path)
        return 0

    tasks = load_task_samples(
        config=resolve_on_policy_settings(
            data_config_name=args.data_config_name,
            data_overrides=probe_source_data_overrides,
        ).data
        if probe_source_data_overrides is not None
        else settings.data,
        task_partition=args.task_partition,
        eval_split_fraction=resolved_eval_split_fraction,
        min_eval_rows=resolved_min_eval_rows,
    )
    expected_cache_metadata["task_pool_size"] = len(tasks)
    expected_cache_metadata["task_pool_fingerprint"] = _build_task_pool_fingerprint(tasks)
    start_index = int(args.start_task_index)
    if start_index >= len(tasks):
        raise ValueError(
            f"start_task_index={start_index} is outside the task pool of size {len(tasks)}."
        )

    sliced_tasks = tasks[start_index:]
    if args.task_limit is not None:
        sliced_tasks = sliced_tasks[: int(args.task_limit)]
    if not sliced_tasks:
        raise ValueError("No tasks selected for difficulty-band probing.")

    if cache_path.is_file() and not bool(args.force_refresh):
        payload = _load_cache_payload(cache_path)
        if _cache_metadata_matches(
            payload=payload,
            expected=expected_cache_metadata,
        ) and _cache_payload_covers_tasks(payload=payload, tasks=sliced_tasks):
            print(cache_path)
            return 0
    partial_cache_path = _resolve_partial_cache_path(cache_path)
    partial_records_path = _resolve_partial_records_path(cache_path)
    if bool(args.force_refresh):
        if partial_cache_path.exists():
            partial_cache_path.unlink()
        if partial_records_path.exists():
            partial_records_path.unlink()
    records_by_task_id: dict[str, dict[str, Any]] = {}
    resumed_from_partial = False
    if partial_cache_path.is_file() and not bool(args.force_refresh):
        partial_payload = _load_cache_payload(partial_cache_path)
        if _cache_metadata_matches(
            payload=partial_payload,
            expected=expected_cache_metadata,
        ):
            records_by_task_id = _load_resumable_records(
                payload=partial_payload,
                tasks=sliced_tasks,
                partial_records_path=partial_records_path,
            )
            resumed_from_partial = True
    if not resumed_from_partial:
        if partial_cache_path.exists():
            partial_cache_path.unlink()
        if partial_records_path.exists():
            partial_records_path.unlink()

    remaining_indexed_tasks = [
        (start_index + offset, task)
        for offset, task in enumerate(sliced_tasks)
        if task.task_id not in records_by_task_id
    ]
    if remaining_indexed_tasks:
        tokenizer = _load_tokenizer(args.initial_model)
        if records_by_task_id and not partial_records_path.exists():
            for record in _ordered_records_for_tasks(
                tasks=sliced_tasks,
                records_by_task_id=records_by_task_id,
            ):
                _append_partial_record(
                    partial_records_path=partial_records_path,
                    record=record,
                )
        max_workers = min(
            resolved_env_pool_size,
            resolved_max_in_flight_tasks,
            len(remaining_indexed_tasks),
        )
        _probe_tasks_with_progress_writes(
            remaining_indexed_tasks=remaining_indexed_tasks,
            expected_cache_metadata=expected_cache_metadata,
            selected_task_count=len(sliced_tasks),
            partial_cache_path=partial_cache_path,
            partial_records_path=partial_records_path,
            existing_records_by_task_id=records_by_task_id,
            submission_window=min(resolved_task_batch_size, len(remaining_indexed_tasks)),
            max_workers=max_workers,
            data_config_name=args.data_config_name,
            turn_generator_mode=args.turn_generator_mode,
            attempts_per_task=int(args.attempts_per_task),
            probe_source_data_overrides=probe_source_data_overrides,
            handoff_overrides=handoff_overrides,
            verify_submissions=verify_submissions,
            stage_name=resolved_stage_name,
            max_tool_calls=settings.runtime.max_tool_calls_per_turn,
            tokenizer=tokenizer,
        )

    ordered_records = _ordered_records_for_tasks(
        tasks=sliced_tasks,
        records_by_task_id=records_by_task_id,
    )
    if len(ordered_records) != len(sliced_tasks):
        raise ValueError(
            "Difficulty-band probe completed without records for every selected task: "
            f"expected {len(sliced_tasks)}, got {len(ordered_records)}."
        )

    payload = _build_cache_payload(
        expected_cache_metadata=expected_cache_metadata,
        selected_task_count=len(sliced_tasks),
        records=ordered_records,
        probe_status=PROBE_STATUS_COMPLETE,
    )
    _write_cache_payload(cache_path=cache_path, payload=payload)
    if partial_cache_path.exists():
        partial_cache_path.unlink()
    if partial_records_path.exists():
        partial_records_path.unlink()
    print(cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
