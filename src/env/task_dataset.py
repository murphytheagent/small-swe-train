"""Task sampling helpers for on-policy SWE dataset collection."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from config import OnPolicyDataConfig


DatasetLoader = Callable[[str, str], Sequence[Mapping[str, Any]]]
SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS = 4000
SDPO_TASK_ROWS_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TaskSample:
    task_id: str
    image_name: str
    problem_statement: str
    fail_to_pass: Any
    pass_to_pass: Any
    raw: Mapping[str, Any]


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
) -> list[TaskSample]:
    """Load a deterministic on-policy task batch for a given global step."""
    if step_index < 0:
        raise ValueError("step_index must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    loader = dataset_loader or load_hf_dataset
    dataset = loader(config.dataset_id, config.dataset_split)

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset {config.dataset_id!r}:{config.dataset_split!r} is empty."
        )

    _validate_required_columns(dataset, config=config)

    start = (step_index * batch_size) % len(dataset)
    samples: list[TaskSample] = []
    last_error: ValueError | None = None
    scanned = 0

    # Real SWE datasets may contain occasional malformed rows. Keep deterministic
    # iteration but skip invalid samples until the requested batch is filled.
    while len(samples) < batch_size and scanned < len(dataset):
        row_index = (start + scanned) % len(dataset)
        scanned += 1
        row = dataset[row_index]
        if not isinstance(row, Mapping):
            last_error = ValueError(f"Dataset row {row_index} is not a mapping.")
            continue
        try:
            samples.append(_coerce_task_row(row, config=config, row_index=row_index))
        except ValueError as exc:
            last_error = exc

    if len(samples) < batch_size:
        detail = str(last_error) if last_error is not None else "no valid rows found"
        raise ValueError(
            f"Unable to build task batch of size {batch_size} from "
            f"{config.dataset_id!r}:{config.dataset_split!r}. "
            f"Collected {len(samples)} valid rows after scanning {scanned}. "
            f"Last validation error: {detail}"
        )

    return samples


def build_sdpo_task_rows(
    *,
    config: OnPolicyDataConfig,
    dataset_loader: DatasetLoader | None = None,
    max_problem_statement_chars: int | None = SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
) -> list[dict[str, Any]]:
    """Build SDPO prompt rows from the full on-policy task dataset split."""
    loader = dataset_loader or load_hf_dataset
    dataset = loader(config.dataset_id, config.dataset_split)
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset {config.dataset_id!r}:{config.dataset_split!r} is empty."
        )
    _validate_required_columns(dataset, config=config)

    prompt_char_limit = _coerce_sdpo_prompt_char_limit(max_problem_statement_chars)
    data_source = _resolve_sdpo_data_source(config)
    rows: list[dict[str, Any]] = []
    last_error: ValueError | None = None
    skipped_for_prompt_length = 0
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

        if prompt_char_limit is not None and len(task.problem_statement) >= prompt_char_limit:
            skipped_for_prompt_length += 1
            continue

        fail_to_pass = list(task.fail_to_pass)
        pass_to_pass = list(task.pass_to_pass)
        rows.append(
            {
                "prompt": [{"role": "user", "content": task.problem_statement}],
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
        detail = str(last_error) if last_error is not None else "no valid rows found"
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


def _stable_split_rank(row: Mapping[str, Any], *, row_index: int) -> str:
    task_id = str(row.get("task_id", "")).strip()
    image_name = str(row.get("image_name", "")).strip()
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
