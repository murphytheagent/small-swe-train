"""Task sampling helpers for on-policy SWE dataset collection."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from config import OnPolicyDataConfig
from prompts.runtime_messages import build_onpolicy_initial_user_message
from runtime_paths import resolve_on_policy_bad_task_cache_dir


DatasetLoader = Callable[[str, str], Sequence[Mapping[str, Any]]]
SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS = 4000
SDPO_TASK_ROWS_SCHEMA_VERSION = 3
ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION = 1
_BAD_TASK_CACHE_PATH_ENV = "SMALL_SWE_BAD_TASK_CACHE_PATH"
_BAD_TASK_CACHE_DIR_ENV = "SMALL_SWE_BAD_TASK_CACHE_DIR"
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


@dataclass(frozen=True)
class TaskSample:
    task_id: str
    image_name: str
    problem_statement: str
    fail_to_pass: Any
    pass_to_pass: Any
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TaskPoolBuildResult:
    tasks: tuple[TaskSample, ...]
    scanned_rows: int
    last_error: str = ""


def _required_columns(config: OnPolicyDataConfig) -> tuple[str, str, str, str]:
    columns = config.columns
    return (
        columns.image_name,
        columns.problem_statement,
        columns.fail_to_pass,
        columns.pass_to_pass,
    )


def _dataset_column_names(dataset: Sequence[Mapping[str, Any]]) -> set[str]:
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, Sequence) and not isinstance(column_names, (str, bytes)):
        return {str(name) for name in column_names}
    if not dataset:
        return set()
    first_row = dataset[0]
    if isinstance(first_row, Mapping):
        return {str(key) for key in first_row}
    return set()


def _validate_required_columns(dataset: Sequence[Mapping[str, Any]], *, config: OnPolicyDataConfig) -> None:
    required = set(_required_columns(config))
    columns = _dataset_column_names(dataset)
    missing = sorted(required - columns)
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(
            f"Dataset {config.dataset_id!r} is missing required columns: {formatted}."
        )


@functools.lru_cache(maxsize=4)
def _load_hf_dataset_cached(dataset_id: str, split: str) -> Sequence[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "datasets package is required for HF task loading. Install train extras."
        ) from exc
    dataset = load_dataset(dataset_id, split=split)
    if not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
        raise ValueError(
            f"Dataset loader returned unsupported type for {dataset_id!r}:{split!r}."
        )
    return dataset  # type: ignore[return-value]


def load_hf_dataset(dataset_id: str, split: str) -> Sequence[Mapping[str, Any]]:
    return _load_hf_dataset_cached(dataset_id, split)


def resolve_on_policy_bad_task_cache_path(
    *,
    config: OnPolicyDataConfig,
    cache_dir: str | Path,
) -> Path:
    """Resolve deterministic bad-task cache path for one on-policy dataset config."""
    config_fingerprint = {
        "schema_version": ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "columns": {
            "image_name": config.columns.image_name,
            "problem_statement": config.columns.problem_statement,
            "fail_to_pass": config.columns.fail_to_pass,
            "pass_to_pass": config.columns.pass_to_pass,
        },
    }
    encoded = json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    dataset_slug = _slugify_for_filename(config.dataset_id)
    split_slug = _slugify_for_filename(config.dataset_split)
    return Path(cache_dir) / f"bad_tasks_{dataset_slug}_{split_slug}_{digest}.json"


def _coerce_task_row(
    row: Mapping[str, Any],
    *,
    config: OnPolicyDataConfig,
    row_index: int,
) -> TaskSample:
    image_name_raw = row.get(config.columns.image_name)
    prompt_raw = row.get(config.columns.problem_statement)
    if not isinstance(image_name_raw, str) or not image_name_raw.strip():
        raise ValueError(
            f"Row {row_index} has invalid image name column {config.columns.image_name!r}."
        )
    if not isinstance(prompt_raw, str) or not prompt_raw.strip():
        raise ValueError(
            f"Row {row_index} has invalid prompt column {config.columns.problem_statement!r}."
        )
    fail_to_pass = _normalize_test_targets(row.get(config.columns.fail_to_pass))
    if not fail_to_pass:
        raise ValueError(
            f"Row {row_index} has empty required test targets in column {config.columns.fail_to_pass!r}."
        )
    pass_to_pass = _normalize_test_targets(row.get(config.columns.pass_to_pass))
    if not pass_to_pass:
        raise ValueError(
            f"Row {row_index} has empty required test targets in column {config.columns.pass_to_pass!r}."
        )

    task_id_raw = row.get("instance_id") or row.get("task_id") or row.get("problem_id")
    if not isinstance(task_id_raw, str) or not task_id_raw.strip():
        task_id = f"{config.dataset_id}:{row_index}"
    else:
        task_id = task_id_raw.strip()

    return TaskSample(
        task_id=task_id,
        image_name=image_name_raw.strip(),
        problem_statement=prompt_raw,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        raw=dict(row),
    )


def _normalize_test_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [name for name in (str(key).strip() for key in value.keys()) if name]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        targets: list[str] = []
        for item in value:
            name = str(item).strip()
            if name:
                targets.append(name)
        return targets
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _normalize_test_targets(parsed)
        if "," in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.split(",")) if chunk]
        if "\n" in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.splitlines()) if chunk]
        return [stripped]
    return []


def _coerce_name_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, Mapping):
        return {
            name
            for name in (str(key).strip() for key in value.keys())
            if name
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            name
            for name in (str(item).strip() for item in value)
            if name
        }
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _coerce_name_set(parsed)
        return {stripped}
    return set()


def _resolve_bad_task_cache_path(config: OnPolicyDataConfig) -> Path | None:
    explicit_path = str(os.environ.get(_BAD_TASK_CACHE_PATH_ENV, "")).strip()
    if explicit_path:
        return Path(explicit_path)

    explicit_dir = str(os.environ.get(_BAD_TASK_CACHE_DIR_ENV, "")).strip()
    if explicit_dir:
        return resolve_on_policy_bad_task_cache_path(config=config, cache_dir=explicit_dir)

    project_root = Path(__file__).resolve().parents[2]
    default_cache_dir = resolve_on_policy_bad_task_cache_dir(project_root=project_root)
    return resolve_on_policy_bad_task_cache_path(config=config, cache_dir=default_cache_dir)


def _load_bad_task_filter(config: OnPolicyDataConfig) -> tuple[set[str], set[str]]:
    cache_path = _resolve_bad_task_cache_path(config)
    if cache_path is None or not cache_path.is_file():
        return set(), set()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Bad-task cache must be a mapping: {cache_path}")

    bad_task_ids = _coerce_name_set(payload.get("bad_task_ids"))
    bad_image_names = _coerce_name_set(payload.get("bad_image_names"))

    records = payload.get("records")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            status = str(record.get("status", "")).strip().lower()
            if status and status != "bad":
                continue
            task_id = str(record.get("task_id", "")).strip()
            image_name = str(record.get("image_name", "")).strip()
            if task_id:
                bad_task_ids.add(task_id)
            if image_name:
                bad_image_names.add(image_name)

    return bad_task_ids, bad_image_names


def _bad_task_filter_fingerprint(config: OnPolicyDataConfig) -> dict[str, Any]:
    bad_task_ids, bad_image_names = _load_bad_task_filter(config)
    return _bad_task_filter_fingerprint_from_sets(
        bad_task_ids=bad_task_ids,
        bad_image_names=bad_image_names,
    )


def _bad_task_filter_fingerprint_from_sets(
    *,
    bad_task_ids: set[str],
    bad_image_names: set[str],
) -> dict[str, Any]:
    payload = {
        "bad_task_ids": sorted(bad_task_ids),
        "bad_image_names": sorted(bad_image_names),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "present": bool(bad_task_ids or bad_image_names),
        "bad_task_count": len(bad_task_ids),
        "bad_image_count": len(bad_image_names),
        "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
    }


def _is_cached_bad_task(
    task: TaskSample,
    *,
    bad_task_ids: set[str],
    bad_image_names: set[str],
) -> bool:
    return task.task_id in bad_task_ids or task.image_name in bad_image_names


def _build_task_pool(
    dataset: Sequence[Mapping[str, Any]],
    *,
    config: OnPolicyDataConfig,
    bad_task_ids: set[str],
    bad_image_names: set[str],
) -> TaskPoolBuildResult:
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset {config.dataset_id!r}:{config.dataset_split!r} is empty."
        )

    _validate_required_columns(dataset, config=config)
    tasks: list[TaskSample] = []
    last_error: ValueError | None = None
    for row_index in range(len(dataset)):
        row = dataset[row_index]
        if not isinstance(row, Mapping):
            last_error = ValueError(f"Dataset row {row_index} is not a mapping.")
            continue
        try:
            task = _coerce_task_row(row, config=config, row_index=row_index)
        except ValueError as exc:
            last_error = exc
            continue
        if _is_cached_bad_task(
            task,
            bad_task_ids=bad_task_ids,
            bad_image_names=bad_image_names,
        ):
            continue
        tasks.append(task)

    return TaskPoolBuildResult(
        tasks=tuple(tasks),
        scanned_rows=len(dataset),
        last_error=str(last_error) if last_error is not None else "",
    )


@functools.lru_cache(maxsize=4)
def _load_hf_task_pool_cached(
    config: OnPolicyDataConfig,
    bad_task_filter_digest: str,
) -> TaskPoolBuildResult:
    del bad_task_filter_digest
    dataset = load_hf_dataset(config.dataset_id, config.dataset_split)
    bad_task_ids, bad_image_names = _load_bad_task_filter(config)
    return _build_task_pool(
        dataset,
        config=config,
        bad_task_ids=bad_task_ids,
        bad_image_names=bad_image_names,
    )


def _load_task_pool(
    *,
    config: OnPolicyDataConfig,
    dataset_loader: DatasetLoader | None = None,
) -> TaskPoolBuildResult:
    if dataset_loader is None or dataset_loader is load_hf_dataset:
        bad_task_ids, bad_image_names = _load_bad_task_filter(config)
        bad_filter_fingerprint = _bad_task_filter_fingerprint_from_sets(
            bad_task_ids=bad_task_ids,
            bad_image_names=bad_image_names,
        )
        return _load_hf_task_pool_cached(
            config,
            str(bad_filter_fingerprint["digest"]),
        )

    loader = dataset_loader
    dataset = loader(config.dataset_id, config.dataset_split)
    bad_task_ids, bad_image_names = _load_bad_task_filter(config)
    return _build_task_pool(
        dataset,
        config=config,
        bad_task_ids=bad_task_ids,
        bad_image_names=bad_image_names,
    )


def _resolve_sdpo_data_source(config: OnPolicyDataConfig) -> str:
    dataset_id = str(config.dataset_id).strip()
    if dataset_id:
        return dataset_id
    return "small_swe_phase_d"


def load_task_batch(
    *,
    step_index: int,
    batch_size: int,
    config: OnPolicyDataConfig,
    dataset_loader: DatasetLoader | None = None,
    task_partition: str = _TASK_PARTITION_ALL,
    eval_split_fraction: float = 0.0,
    min_eval_rows: int = 0,
) -> list[TaskSample]:
    """Load a deterministic on-policy task batch for a given global step."""
    if step_index < 0:
        raise ValueError("step_index must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    normalized_partition = _normalize_task_partition(task_partition)
    task_pool = _load_task_pool(config=config, dataset_loader=dataset_loader)
    tasks = list(task_pool.tasks)
    if not tasks:
        detail = task_pool.last_error if task_pool.last_error else "no valid rows found"
        raise ValueError(
            f"Unable to build task batch of size {batch_size} from "
            f"{config.dataset_id!r}:{config.dataset_split!r}. "
            f"Collected 0 valid rows after scanning {task_pool.scanned_rows}. "
            f"Last validation error: {detail}"
        )
    if normalized_partition != _TASK_PARTITION_ALL:
        train_tasks, eval_tasks = split_task_samples_for_eval(
            tasks,
            eval_split_fraction=eval_split_fraction,
            min_eval_rows=min_eval_rows,
        )
        tasks = train_tasks if normalized_partition == _TASK_PARTITION_TRAIN else eval_tasks
        if not tasks:
            return []

    if normalized_partition == _TASK_PARTITION_ALL and len(tasks) < batch_size:
        detail = task_pool.last_error if task_pool.last_error else "no valid rows found"
        partition_detail = (
            f" in task partition {normalized_partition!r}"
            if normalized_partition != _TASK_PARTITION_ALL
            else ""
        )
        raise ValueError(
            f"Unable to build task batch of size {batch_size} from "
            f"{config.dataset_id!r}:{config.dataset_split!r}{partition_detail}. "
            f"Collected {len(tasks)} valid rows after scanning {task_pool.scanned_rows}. "
            f"Last validation error: {detail}"
        )

    # Held-out train/eval partitions can be intentionally smaller than the collector width.
    # Keep them usable by reusing the deterministic partition with wraparound.
    start = (step_index * batch_size) % len(tasks)
    samples = [
        tasks[(start + offset) % len(tasks)]
        for offset in range(batch_size)
    ]
    return list(samples)


def build_sdpo_task_rows(
    *,
    config: OnPolicyDataConfig,
    dataset_loader: DatasetLoader | None = None,
    max_problem_statement_chars: int | None = SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
) -> list[dict[str, Any]]:
    """Build SDPO prompt rows from the full on-policy task dataset split."""
    prompt_char_limit = _coerce_sdpo_prompt_char_limit(max_problem_statement_chars)
    data_source = _resolve_sdpo_data_source(config)
    task_pool = _load_task_pool(config=config, dataset_loader=dataset_loader)
    rows: list[dict[str, Any]] = []
    skipped_for_prompt_length = 0
    for task in task_pool.tasks:
        if prompt_char_limit is not None and len(task.problem_statement) >= prompt_char_limit:
            skipped_for_prompt_length += 1
            continue

        fail_to_pass = list(task.fail_to_pass)
        pass_to_pass = list(task.pass_to_pass)
        initial_user_message = build_onpolicy_initial_user_message(
            problem_statement=task.problem_statement,
        )
        rows.append(
            {
                "prompt": [{"role": "user", "content": initial_user_message}],
                "task_id": task.task_id,
                "image_name": task.image_name,
                "data_source": data_source,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "reward_model": {
                    "ground_truth": {
                        "task_id": task.task_id,
                        "image_name": task.image_name,
                        "data_source": data_source,
                        "fail_to_pass": fail_to_pass,
                        "pass_to_pass": pass_to_pass,
                    }
                },
            }
        )

    if not rows:
        detail = task_pool.last_error if task_pool.last_error else "no valid rows found"
        if skipped_for_prompt_length > 0 and prompt_char_limit is not None:
            detail = (
                f"all candidate rows exceeded prompt-length filter "
                f"(problem_statement chars must be < {prompt_char_limit})"
            )
        raise ValueError(
            "Unable to build SDPO task rows from "
            f"{config.dataset_id!r}:{config.dataset_split!r}. "
            f"Last validation error: {detail}"
        )
    return rows


def resolve_sdpo_task_rows_cache_path(
    *,
    config: OnPolicyDataConfig,
    cache_dir: str | Path,
    max_problem_statement_chars: int | None = SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
) -> Path:
    """Resolve deterministic parquet cache path for SDPO prompt rows."""
    prompt_char_limit = _coerce_sdpo_prompt_char_limit(max_problem_statement_chars)
    bad_task_filter = _bad_task_filter_fingerprint(config)
    config_fingerprint = {
        "schema_version": SDPO_TASK_ROWS_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "columns": {
            "image_name": config.columns.image_name,
            "problem_statement": config.columns.problem_statement,
            "fail_to_pass": config.columns.fail_to_pass,
            "pass_to_pass": config.columns.pass_to_pass,
        },
        "max_problem_statement_chars": prompt_char_limit,
        "bad_task_filter": bad_task_filter,
    }
    encoded = json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    dataset_slug = _slugify_for_filename(config.dataset_id)
    split_slug = _slugify_for_filename(config.dataset_split)
    return Path(cache_dir) / f"sdpo_tasks_{dataset_slug}_{split_slug}_{digest}.parquet"


def preload_sdpo_task_rows_to_parquet(
    *,
    config: OnPolicyDataConfig,
    cache_dir: str | Path,
    force_refresh: bool = False,
    dataset_loader: DatasetLoader | None = None,
    max_problem_statement_chars: int | None = SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
) -> Path:
    """Materialize SDPO prompt rows to parquet and return the cache path."""
    prompt_char_limit = _coerce_sdpo_prompt_char_limit(max_problem_statement_chars)
    target_path = resolve_sdpo_task_rows_cache_path(
        config=config,
        cache_dir=cache_dir,
        max_problem_statement_chars=prompt_char_limit,
    )
    if target_path.is_file() and not force_refresh:
        return target_path

    rows = build_sdpo_task_rows(
        config=config,
        dataset_loader=dataset_loader,
        max_problem_statement_chars=prompt_char_limit,
    )
    _write_records_to_parquet(rows, target_path)
    return target_path


def split_sdpo_task_rows_for_eval(
    rows: Sequence[Mapping[str, Any]],
    *,
    eval_split_fraction: float,
    min_eval_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split SDPO task rows into deterministic train/eval partitions."""
    if eval_split_fraction < 0.0 or eval_split_fraction >= 1.0:
        raise ValueError("eval_split_fraction must be in [0.0, 1.0).")
    if min_eval_rows < 0:
        raise ValueError("min_eval_rows must be >= 0.")

    copied_rows = [dict(row) for row in rows]
    total = len(copied_rows)
    if total < 1:
        raise ValueError("rows must be non-empty.")

    max_eval_rows = total - 1
    if max_eval_rows < 1 or eval_split_fraction <= 0.0:
        return copied_rows, []

    eval_rows_target = int(total * eval_split_fraction)
    eval_rows_target = max(eval_rows_target, min_eval_rows)
    eval_rows_target = min(eval_rows_target, max_eval_rows)
    if eval_rows_target < 1:
        return copied_rows, []

    ranked_indexes = sorted(
        range(total),
        key=lambda row_index: _stable_split_rank(copied_rows[row_index], row_index=row_index),
    )
    eval_indexes = set(ranked_indexes[:eval_rows_target])

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(copied_rows):
        if row_index in eval_indexes:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        train_rows.append(eval_rows.pop())
    return train_rows, eval_rows


def resolve_sdpo_task_split_cache_paths(
    *,
    config: OnPolicyDataConfig,
    cache_dir: str | Path,
    eval_split_fraction: float,
    min_eval_rows: int,
    max_problem_statement_chars: int | None = SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
) -> tuple[Path, Path]:
    """Resolve deterministic train/eval parquet cache paths for SDPO task rows."""
    prompt_char_limit = _coerce_sdpo_prompt_char_limit(max_problem_statement_chars)
    bad_task_filter = _bad_task_filter_fingerprint(config)
    split_fingerprint = {
        "schema_version": SDPO_TASK_ROWS_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "columns": {
            "image_name": config.columns.image_name,
            "problem_statement": config.columns.problem_statement,
            "fail_to_pass": config.columns.fail_to_pass,
            "pass_to_pass": config.columns.pass_to_pass,
        },
        "eval_split_fraction": float(eval_split_fraction),
        "min_eval_rows": int(min_eval_rows),
        "max_problem_statement_chars": prompt_char_limit,
        "bad_task_filter": bad_task_filter,
    }
    encoded = json.dumps(split_fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    dataset_slug = _slugify_for_filename(config.dataset_id)
    split_slug = _slugify_for_filename(config.dataset_split)
    base = Path(cache_dir) / f"sdpo_tasks_{dataset_slug}_{split_slug}_{digest}"
    return base.with_name(base.name + "_train.parquet"), base.with_name(base.name + "_val.parquet")


def preload_sdpo_task_rows_split_to_parquet(
    *,
    config: OnPolicyDataConfig,
    cache_dir: str | Path,
    eval_split_fraction: float,
    min_eval_rows: int,
    force_refresh: bool = False,
    dataset_loader: DatasetLoader | None = None,
    max_problem_statement_chars: int | None = SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
) -> tuple[Path, Path]:
    """Materialize deterministic SDPO train/eval parquet caches."""
    prompt_char_limit = _coerce_sdpo_prompt_char_limit(max_problem_statement_chars)
    train_path, val_path = resolve_sdpo_task_split_cache_paths(
        config=config,
        cache_dir=cache_dir,
        eval_split_fraction=eval_split_fraction,
        min_eval_rows=min_eval_rows,
        max_problem_statement_chars=prompt_char_limit,
    )
    if train_path.is_file() and val_path.is_file() and not force_refresh:
        return train_path, val_path

    rows = build_sdpo_task_rows(
        config=config,
        dataset_loader=dataset_loader,
        max_problem_statement_chars=prompt_char_limit,
    )
    train_rows, eval_rows = split_sdpo_task_rows_for_eval(
        rows,
        eval_split_fraction=eval_split_fraction,
        min_eval_rows=min_eval_rows,
    )
    if not eval_rows:
        eval_rows = [dict(train_rows[0])]

    _write_records_to_parquet(train_rows, train_path)
    _write_records_to_parquet(eval_rows, val_path)
    return train_path, val_path


def _write_records_to_parquet(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    if not records:
        raise ValueError("Cannot write SDPO task parquet with zero records.")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on train extras
        raise RuntimeError(
            "Writing SDPO task parquet requires pyarrow. "
            "Install training extras (`pip install -e \".[train]\"`)."
        ) from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 2048
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(records), chunk_size):
            chunk_records = [dict(record) for record in records[start : start + chunk_size]]
            table = pa.Table.from_pylist(chunk_records)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _slugify_for_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if normalized:
        return normalized
    return "dataset"


def split_task_samples_for_eval(
    tasks: Sequence[TaskSample],
    *,
    eval_split_fraction: float,
    min_eval_rows: int,
) -> tuple[list[TaskSample], list[TaskSample]]:
    """Split valid task samples into deterministic train/eval partitions."""
    if eval_split_fraction < 0.0 or eval_split_fraction >= 1.0:
        raise ValueError("eval_split_fraction must be in [0.0, 1.0).")
    if min_eval_rows < 0:
        raise ValueError("min_eval_rows must be >= 0.")

    copied_tasks = list(tasks)
    total = len(copied_tasks)
    if total < 1:
        raise ValueError("tasks must be non-empty.")

    max_eval_rows = total - 1
    if max_eval_rows < 1 or eval_split_fraction <= 0.0:
        return copied_tasks, []

    eval_rows_target = int(total * eval_split_fraction)
    eval_rows_target = max(eval_rows_target, min_eval_rows)
    eval_rows_target = min(eval_rows_target, max_eval_rows)
    if eval_rows_target < 1:
        return copied_tasks, []

    ranked_indexes = sorted(
        range(total),
        key=lambda row_index: _stable_split_token(
            task_id=copied_tasks[row_index].task_id,
            image_name=copied_tasks[row_index].image_name,
            row_index=row_index,
        ),
    )
    eval_indexes = set(ranked_indexes[:eval_rows_target])

    train_tasks: list[TaskSample] = []
    eval_tasks: list[TaskSample] = []
    for row_index, task in enumerate(copied_tasks):
        if row_index in eval_indexes:
            eval_tasks.append(task)
        else:
            train_tasks.append(task)

    if not train_tasks:
        train_tasks.append(eval_tasks.pop())
    return train_tasks, eval_tasks


def _stable_split_rank(row: Mapping[str, Any], *, row_index: int) -> str:
    task_id = str(row.get("task_id", "")).strip()
    image_name = str(row.get("image_name", "")).strip()
    return _stable_split_token(task_id=task_id, image_name=image_name, row_index=row_index)


def _stable_split_token(*, task_id: str, image_name: str, row_index: int) -> str:
    token = f"{task_id}|{image_name}|{row_index}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _coerce_sdpo_prompt_char_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_problem_statement_chars must be an integer >= 1 or null.")
    if isinstance(value, int) and value >= 1:
        return int(value)
    raise ValueError("max_problem_statement_chars must be an integer >= 1 or null.")


def _normalize_task_partition(value: str) -> str:
    normalized = str(value).strip().lower()
    partition = _TASK_PARTITION_ALIASES.get(normalized)
    if partition is None:
        raise ValueError("task_partition must be one of: all, train, eval.")
    return partition
