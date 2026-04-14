"""CLI helper that materializes rollout-backed on-policy difficulty bands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import (
    DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    resolve_on_policy_settings,
    rft_runtime_defaults,
)
from env.task_dataset import (
    TaskSample,
    load_task_samples,
    resolve_on_policy_difficulty_band_cache_path,
)
from runtime_paths import resolve_on_policy_difficulty_band_cache_dir
from trainer.rft_runtime import OnPolicyRFTRuntimeRequest, collect_onpolicy_rft_runtime_batch
from trainer.rft_runtime_loop import (
    _load_tokenizer,
    resolve_rft_stage_handoff_overrides,
    resolve_rft_stage_name,
    resolve_rft_stage_verify_submissions,
)

DEFAULT_PROBE_LABEL = "positive_rft_probe"
DEFAULT_ATTEMPTS_PER_TASK = 4


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
        help="Number of tasks to probe concurrently in each runtime batch (default: 1).",
    )
    parser.add_argument(
        "--env-pool-size",
        type=int,
        default=None,
        help=(
            "Container pool size used during probing. Defaults to --task-batch-size "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--max-in-flight-tasks",
        type=int,
        default=None,
        help=(
            "Maximum concurrent tasks dispatched inside each probe batch. Defaults to "
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
    rows = tuple(dict(task.raw) for task in tasks)

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


def _load_cache_payload(cache_path: Path) -> Mapping[str, Any]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Difficulty-band cache payload must be a mapping: {cache_path}")
    return payload


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
        "task_pool_size": task_pool_size,
        "task_pool_fingerprint": task_pool_fingerprint,
    }


def _cache_metadata_matches(
    *,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(payload.get(key) == value for key, value in expected.items())


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
    )
    resolved_stage_name = resolve_rft_stage_name(args.stage_name)
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
        task_pool_size=0,
        task_pool_fingerprint="",
    )
    if args.print_path_only:
        print(cache_path)
        return 0

    tasks = load_task_samples(
        config=settings.data,
        task_partition=args.task_partition,
        eval_split_fraction=resolved_eval_split_fraction,
        min_eval_rows=resolved_min_eval_rows,
    )
    expected_cache_metadata["task_pool_size"] = len(tasks)
    expected_cache_metadata["task_pool_fingerprint"] = _build_task_pool_fingerprint(tasks)
    if cache_path.is_file() and not bool(args.force_refresh):
        payload = _load_cache_payload(cache_path)
        if _cache_metadata_matches(payload=payload, expected=expected_cache_metadata):
            print(cache_path)
            return 0

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

    tokenizer = _load_tokenizer(args.initial_model)
    handoff_overrides = resolve_rft_stage_handoff_overrides(resolved_stage_name)
    verify_submissions = resolve_rft_stage_verify_submissions(resolved_stage_name)

    records: list[dict[str, Any]] = []
    if resolved_task_batch_size == 1:
        for offset, task in enumerate(sliced_tasks):
            probe_step_index = start_index + offset
            result = collect_onpolicy_rft_runtime_batch(
                request=OnPolicyRFTRuntimeRequest(
                    data_config_name=args.data_config_name,
                    turn_generator_mode=args.turn_generator_mode,
                    total_steps=1,
                    start_step_index=probe_step_index,
                    runtime_overrides={
                        "task_batch_size": 1,
                        "attempts_per_task": int(args.attempts_per_task),
                        "env_pool_size": 1,
                        "max_in_flight_tasks": 1,
                    },
                    handoff_overrides=handoff_overrides,
                    task_partition=args.task_partition,
                    task_eval_split_fraction=resolved_eval_split_fraction,
                    task_eval_min_rows=resolved_min_eval_rows,
                    verify_submissions=verify_submissions,
                    stage_name=resolved_stage_name,
                ),
                tokenizer=tokenizer,
            )
            records.append(
                _build_band_record(
                    task=task,
                    probe_step_index=probe_step_index,
                    stage_name=resolved_stage_name,
                    result=result,
                )
            )
    else:
        for chunk_start in range(0, len(sliced_tasks), resolved_task_batch_size):
            task_chunk = sliced_tasks[chunk_start : chunk_start + resolved_task_batch_size]
            chunk_probe_step_index = start_index + chunk_start
            result = collect_onpolicy_rft_runtime_batch(
                request=OnPolicyRFTRuntimeRequest(
                    data_config_name=args.data_config_name,
                    turn_generator_mode=args.turn_generator_mode,
                    total_steps=1,
                    start_step_index=chunk_probe_step_index,
                    runtime_overrides={
                        "task_batch_size": len(task_chunk),
                        "attempts_per_task": int(args.attempts_per_task),
                        "env_pool_size": min(resolved_env_pool_size, len(task_chunk)),
                        "max_in_flight_tasks": min(
                            resolved_max_in_flight_tasks,
                            len(task_chunk),
                        ),
                    },
                    handoff_overrides=handoff_overrides,
                    task_partition="all",
                    task_eval_split_fraction=0.0,
                    task_eval_min_rows=0,
                    verify_submissions=verify_submissions,
                    stage_name=resolved_stage_name,
                    dataset_loader=_build_dataset_loader_for_tasks(task_chunk),
                ),
                tokenizer=tokenizer,
            )

            fallback_task_id = task_chunk[0].task_id if len(task_chunk) == 1 else ""
            for offset, task in enumerate(task_chunk):
                probe_step_index = chunk_probe_step_index + offset
                task_result = _extract_task_result(
                    result=result,
                    task_id=task.task_id,
                    fallback_task_id=fallback_task_id,
                )
                records.append(
                    _build_band_record(
                        task=task,
                        probe_step_index=probe_step_index,
                        stage_name=resolved_stage_name,
                        result=task_result,
                    )
                )

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_config_name": args.data_config_name,
        "dataset_id": settings.data.dataset_id,
        "dataset_split": settings.data.dataset_split,
        "patch_is_bug_introducing": bool(settings.data.patch_is_bug_introducing),
        "verifier_kind": str(settings.data.verifier_kind),
        "probe_label": str(args.probe_label).strip(),
        "initial_model": str(args.initial_model).strip(),
        "turn_generator_mode": str(args.turn_generator_mode).strip(),
        "stage_name": resolved_stage_name,
        "task_partition": str(args.task_partition).strip(),
        "attempts_per_task": int(args.attempts_per_task),
        "start_task_index": start_index,
        "task_limit": int(args.task_limit) if args.task_limit is not None else None,
        "eval_split_fraction": resolved_eval_split_fraction,
        "min_eval_rows": resolved_min_eval_rows,
        "task_pool_size": expected_cache_metadata["task_pool_size"],
        "task_pool_fingerprint": expected_cache_metadata["task_pool_fingerprint"],
        "task_count": len(records),
        "records": records,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
