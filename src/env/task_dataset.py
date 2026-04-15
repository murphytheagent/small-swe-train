"""Task sampling helpers for on-policy SWE dataset collection."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from config import OnPolicyDataConfig, OnPolicyDifficultyBandConfig
from prompts.runtime_messages import build_onpolicy_initial_user_message
from runtime_paths import (
    resolve_on_policy_bad_task_cache_dir,
)
from verifier_utils import (
    logical_task_identity_key,
    normalize_verifier_targets,
    resolve_problem_statement,
    validate_verifier_target_sets,
)


DatasetLoader = Callable[[str, str], Sequence[Mapping[str, Any]]]
SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS = 4000
SDPO_TASK_ROWS_SCHEMA_VERSION = 4
ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION = 1
ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION = 1
_BAD_TASK_CACHE_PATH_ENV = "SMALL_SWE_BAD_TASK_CACHE_PATH"
_BAD_TASK_CACHE_DIR_ENV = "SMALL_SWE_BAD_TASK_CACHE_DIR"
_DIFFICULTY_BAND_CACHE_PATH_ENV = "SMALL_SWE_DIFFICULTY_BAND_CACHE_PATH"
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
    verifier_kind: str = "pytest"
    task_family: str = ""
    difficulty_band: str = "unbanded"
    difficulty_band_source: str = "none"


@dataclass(frozen=True)
class TaskPoolBuildResult:
    tasks: tuple[TaskSample, ...]
    scanned_rows: int
    last_error: str = ""
    filtered_counts: Mapping[str, int] = field(default_factory=dict)
    filtered_task_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


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
        "patch_is_bug_introducing": bool(config.patch_is_bug_introducing),
        "verifier_kind": str(config.verifier_kind),
    }
    encoded = json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    dataset_slug = _slugify_for_filename(config.dataset_id)
    split_slug = _slugify_for_filename(config.dataset_split)
    return Path(cache_dir) / f"bad_tasks_{dataset_slug}_{split_slug}_{digest}.json"


def resolve_on_policy_difficulty_band_cache_path(
    *,
    config: OnPolicyDataConfig,
    cache_dir: str | Path,
    probe_label: str,
    task_partition: str = _TASK_PARTITION_ALL,
    start_task_index: int = 0,
    task_limit: int | None = None,
    eval_split_fraction: float | None = None,
    min_eval_rows: int | None = None,
) -> Path:
    """Resolve a descriptive difficulty-band cache path for one dataset config."""
    label_slug = _slugify_for_filename(probe_label, fallback="")
    if not label_slug:
        raise ValueError("probe_label must contain at least one filename-safe character.")
    dataset_slug = _slugify_for_filename(config.dataset_id)
    split_slug = _slugify_for_filename(config.dataset_split)
    normalized_partition = _normalize_task_partition(task_partition)
    scope_parts: list[str] = []
    if normalized_partition != _TASK_PARTITION_ALL:
        scope_parts.append(normalized_partition)
        if eval_split_fraction is not None:
            scope_parts.append(f"frac_{_slugify_for_filename(str(eval_split_fraction), fallback='0')}")
        if min_eval_rows is not None:
            scope_parts.append(f"mineval_{int(min_eval_rows)}")
    if int(start_task_index) != 0:
        scope_parts.append(f"start_{int(start_task_index)}")
    if task_limit is not None:
        scope_parts.append(f"limit_{int(task_limit)}")
    scope_suffix = ""
    if scope_parts:
        scope_slug = _slugify_for_filename("_".join(scope_parts), fallback="scope")
        scope_suffix = f"_{scope_slug}"
    return Path(cache_dir) / (
        f"difficulty_bands_{dataset_slug}_{split_slug}_{label_slug}{scope_suffix}.json"
    )


def _coerce_task_row(
    row: Mapping[str, Any],
    *,
    config: OnPolicyDataConfig,
    row_index: int,
) -> TaskSample:
    image_name_raw = row.get(config.columns.image_name)
    if not isinstance(image_name_raw, str) or not image_name_raw.strip():
        raise ValueError(
            f"Row {row_index} has invalid image name column {config.columns.image_name!r}."
        )
    fail_to_pass, pass_to_pass = validate_verifier_target_sets(
        fail_to_pass=row.get(config.columns.fail_to_pass),
        pass_to_pass=row.get(config.columns.pass_to_pass),
    )
    prompt_text, prompt_source = resolve_problem_statement(
        problem_statement=row.get(config.columns.problem_statement),
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        verifier_kind=config.verifier_kind,
    )

    task_id_raw = row.get("instance_id") or row.get("task_id") or row.get("problem_id")
    if not isinstance(task_id_raw, str) or not task_id_raw.strip():
        task_id = f"{config.dataset_id}:{row_index}"
    else:
        task_id = task_id_raw.strip()

    task_family, difficulty_band, difficulty_band_source = _resolve_task_difficulty_metadata(
        task_id=task_id,
        row=row,
        difficulty_banding=config.difficulty_banding,
    )

    raw_row = dict(row)
    raw_row.setdefault("task_id", task_id)
    raw_row.setdefault("prompt_source", prompt_source)
    raw_row.setdefault("verifier_kind", config.verifier_kind)
    raw_row.setdefault("task_family", task_family)
    raw_row.setdefault("difficulty_band", difficulty_band)
    raw_row.setdefault("difficulty_band_source", difficulty_band_source)
    return TaskSample(
        task_id=task_id,
        image_name=image_name_raw.strip(),
        problem_statement=prompt_text,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        verifier_kind=config.verifier_kind,
        raw=raw_row,
        task_family=task_family,
        difficulty_band=difficulty_band,
        difficulty_band_source=difficulty_band_source,
    )


def _normalize_test_targets(value: Any) -> list[str]:
    return normalize_verifier_targets(value)


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
    filtered_counts: dict[str, int] = {
        "invalid_row": 0,
        "cached_bad_task": 0,
        "duplicate_logical_task": 0,
    }
    filtered_task_ids: dict[str, list[str]] = {
        "cached_bad_task": [],
        "duplicate_logical_task": [],
    }
    seen_logical_keys: set[str] = set()
    for row_index in range(len(dataset)):
        row = dataset[row_index]
        if not isinstance(row, Mapping):
            last_error = ValueError(f"Dataset row {row_index} is not a mapping.")
            filtered_counts["invalid_row"] += 1
            continue
        try:
            task = _coerce_task_row(row, config=config, row_index=row_index)
        except ValueError as exc:
            if _should_reraise_task_row_error(config=config, error=exc):
                raise
            last_error = exc
            filtered_counts["invalid_row"] += 1
            continue
        if _is_cached_bad_task(
            task,
            bad_task_ids=bad_task_ids,
            bad_image_names=bad_image_names,
        ):
            filtered_counts["cached_bad_task"] += 1
            filtered_task_ids["cached_bad_task"].append(task.task_id)
            continue
        logical_task_key = logical_task_identity_key(
            problem_statement=task.problem_statement,
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            verifier_kind=task.verifier_kind,
        )
        if logical_task_key in seen_logical_keys:
            filtered_counts["duplicate_logical_task"] += 1
            filtered_task_ids["duplicate_logical_task"].append(task.task_id)
            continue
        seen_logical_keys.add(logical_task_key)
        tasks.append(task)

    return TaskPoolBuildResult(
        tasks=tuple(tasks),
        scanned_rows=len(dataset),
        last_error=str(last_error) if last_error is not None else "",
        filtered_counts=dict(filtered_counts),
        filtered_task_ids={
            key: tuple(value)
            for key, value in filtered_task_ids.items()
            if value
        },
    )


@functools.lru_cache(maxsize=4)
def _load_hf_task_pool_cached(
    config: OnPolicyDataConfig,
    bad_task_filter_digest: str,
    difficulty_banding_cache_token: str,
) -> TaskPoolBuildResult:
    del bad_task_filter_digest, difficulty_banding_cache_token
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
        difficulty_banding_cache_token = _difficulty_banding_cache_token(config)
        return _load_hf_task_pool_cached(
            config,
            str(bad_filter_fingerprint["digest"]),
            difficulty_banding_cache_token,
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


def _should_reraise_task_row_error(
    *,
    config: OnPolicyDataConfig,
    error: ValueError,
) -> bool:
    strategy = str(config.difficulty_banding.strategy).strip().lower()
    if strategy != "rollout_probe" or not bool(config.difficulty_banding.rollout_probe_required):
        return False
    error_text = str(error)
    return (
        "Difficulty-band cache" in error_text
        or "difficulty_banding.rollout_probe_required=true" in error_text
    )


def _resolve_sdpo_data_source(config: OnPolicyDataConfig) -> str:
    dataset_id = str(config.dataset_id).strip()
    if dataset_id:
        return dataset_id
    return "small_swe_phase_d"


def _difficulty_banding_fingerprint(config: OnPolicyDataConfig) -> dict[str, Any]:
    fingerprint = {
        "strategy": config.difficulty_banding.strategy,
        "default_band": config.difficulty_banding.default_band,
        "family_band_exact": [
            [family, band] for family, band in config.difficulty_banding.family_band_exact
        ],
        "family_band_prefix": [
            [prefix, band] for prefix, band in config.difficulty_banding.family_band_prefix
        ],
    }
    if str(config.difficulty_banding.strategy).strip().lower() == "rollout_probe":
        fingerprint["rollout_probe"] = _rollout_probe_cache_fingerprint(
            config.difficulty_banding
        )
    return fingerprint


def _difficulty_banding_cache_token(config: OnPolicyDataConfig) -> str:
    fingerprint = {
        "strategy": config.difficulty_banding.strategy,
        "default_band": config.difficulty_banding.default_band,
        "family_band_exact": [
            [family, band] for family, band in config.difficulty_banding.family_band_exact
        ],
        "family_band_prefix": [
            [prefix, band] for prefix, band in config.difficulty_banding.family_band_prefix
        ],
    }
    if str(config.difficulty_banding.strategy).strip().lower() == "rollout_probe":
        cache_path = _resolve_rollout_probe_cache_path(config.difficulty_banding)
        cache_fingerprint: dict[str, Any] = {
            "path": str(cache_path) if cache_path is not None else "",
            "required": bool(config.difficulty_banding.rollout_probe_required),
            "present": False,
            "modified_ns": 0,
            "file_size": 0,
        }
        if cache_path is not None and cache_path.is_file():
            stat = cache_path.stat()
            cache_fingerprint["present"] = True
            cache_fingerprint["modified_ns"] = int(stat.st_mtime_ns)
            cache_fingerprint["file_size"] = int(stat.st_size)
        fingerprint["rollout_probe_cache"] = cache_fingerprint
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_rollout_probe_cache_path(
    difficulty_banding: OnPolicyDifficultyBandConfig,
) -> Path | None:
    explicit_path = str(os.environ.get(_DIFFICULTY_BAND_CACHE_PATH_ENV, "")).strip()
    if explicit_path:
        return Path(explicit_path)

    configured_path = str(difficulty_banding.rollout_probe_cache_path).strip()
    if not configured_path:
        return None

    path = Path(configured_path)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


@functools.lru_cache(maxsize=8)
def _load_rollout_probe_cache_records_cached(
    cache_path_text: str,
    modified_ns: int,
    file_size: int,
) -> dict[str, dict[str, Any]]:
    del modified_ns, file_size
    cache_path = Path(cache_path_text)
    if not cache_path.is_file():
        return {}

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Difficulty-band cache must be a mapping: {cache_path}")

    schema_version = payload.get("schema_version")
    if schema_version is not None and int(schema_version) != ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Difficulty-band cache schema mismatch at "
            f"{cache_path}: expected {ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION}, "
            f"got {schema_version!r}."
        )

    probe_status = str(payload.get("probe_status", "")).strip().lower()
    if probe_status and probe_status != "complete":
        raise ValueError(
            "Difficulty-band cache is incomplete at "
            f"{cache_path}; resume the probe to materialize the final cache."
        )

    band_records = payload.get("records", payload.get("task_band_records"))
    records_by_task_id: dict[str, dict[str, Any]] = {}
    if isinstance(band_records, Sequence) and not isinstance(band_records, (str, bytes)):
        for index, raw_record in enumerate(band_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"Difficulty-band cache record {index} must be a mapping: {cache_path}"
                )
            task_id = str(raw_record.get("task_id", "")).strip()
            difficulty_band = str(raw_record.get("difficulty_band", "")).strip()
            if not task_id or not difficulty_band:
                raise ValueError(
                    "Difficulty-band cache records must include non-empty task_id and "
                    f"difficulty_band fields: {cache_path}"
                )
            records_by_task_id[task_id] = {
                "task_family": str(raw_record.get("task_family", "")).strip(),
                "difficulty_band": difficulty_band,
                "difficulty_band_source": str(
                    raw_record.get("difficulty_band_source", "")
                ).strip()
                or "rollout_probe:task_id",
            }
        return records_by_task_id

    band_mapping = payload.get("task_band_by_task_id", payload.get("task_bands"))
    if isinstance(band_mapping, Mapping):
        for raw_task_id, raw_value in band_mapping.items():
            task_id = str(raw_task_id).strip()
            if not task_id:
                continue
            if isinstance(raw_value, Mapping):
                difficulty_band = str(raw_value.get("difficulty_band", "")).strip()
                difficulty_band_source = str(
                    raw_value.get("difficulty_band_source", "")
                ).strip() or "rollout_probe:task_id"
                task_family = str(raw_value.get("task_family", "")).strip()
            else:
                difficulty_band = str(raw_value).strip()
                difficulty_band_source = "rollout_probe:task_id"
                task_family = ""
            if not difficulty_band:
                raise ValueError(
                    f"Difficulty-band cache task entry {task_id!r} is missing a band: {cache_path}"
                )
            records_by_task_id[task_id] = {
                "task_family": task_family,
                "difficulty_band": difficulty_band,
                "difficulty_band_source": difficulty_band_source,
            }
        return records_by_task_id

    raise ValueError(
        "Difficulty-band cache must define either `records` or `task_band_by_task_id`: "
        f"{cache_path}"
    )


def _load_rollout_probe_cache_records(
    difficulty_banding: OnPolicyDifficultyBandConfig,
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    cache_path = _resolve_rollout_probe_cache_path(difficulty_banding)
    if cache_path is None:
        if difficulty_banding.rollout_probe_required:
            raise ValueError(
                "difficulty_banding.rollout_probe_required=true but no rollout_probe_cache_path "
                "or SMALL_SWE_DIFFICULTY_BAND_CACHE_PATH was provided."
            )
        return None, {}

    if not cache_path.is_file():
        if difficulty_banding.rollout_probe_required:
            raise ValueError(f"Difficulty-band cache not found: {cache_path}")
        return cache_path, {}

    stat = cache_path.stat()
    records = _load_rollout_probe_cache_records_cached(
        str(cache_path),
        stat.st_mtime_ns,
        stat.st_size,
    )
    return cache_path, records


def _rollout_probe_cache_fingerprint(
    difficulty_banding: OnPolicyDifficultyBandConfig,
) -> dict[str, Any]:
    cache_path, records = _load_rollout_probe_cache_records(difficulty_banding)
    normalized_records = {
        task_id: {
            "difficulty_band": str(record.get("difficulty_band", "")).strip(),
            "difficulty_band_source": str(
                record.get("difficulty_band_source", "")
            ).strip(),
            "task_family": str(record.get("task_family", "")).strip(),
        }
        for task_id, record in sorted(records.items())
    }
    encoded = json.dumps(normalized_records, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(cache_path) if cache_path is not None else "",
        "required": bool(difficulty_banding.rollout_probe_required),
        "present": bool(records),
        "task_count": len(records),
        "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
    }


def _resolve_task_difficulty_metadata(
    *,
    task_id: str,
    row: Mapping[str, Any],
    difficulty_banding: OnPolicyDifficultyBandConfig,
) -> tuple[str, str, str]:
    task_family = _resolve_task_family(task_id=task_id, row=row)
    strategy = str(difficulty_banding.strategy).strip().lower()
    default_band = str(difficulty_banding.default_band).strip() or "unbanded"

    if strategy == "none":
        return task_family, default_band, "none"

    if strategy == "instance_id_family":
        if not task_family:
            return task_family, default_band, "instance_id_family:default"

        exact_mapping = dict(difficulty_banding.family_band_exact)
        if task_family in exact_mapping:
            return task_family, exact_mapping[task_family], "instance_id_family:exact"

        prefix_rules = sorted(
            difficulty_banding.family_band_prefix,
            key=lambda item: (-len(item[0]), item[0]),
        )
        for prefix, band in prefix_rules:
            if task_family.startswith(prefix):
                return task_family, band, "instance_id_family:prefix"

        return task_family, default_band, "instance_id_family:default"

    if strategy == "rollout_probe":
        _cache_path, records = _load_rollout_probe_cache_records(difficulty_banding)
        record = records.get(task_id)
        if record is None:
            if difficulty_banding.rollout_probe_required:
                raise ValueError(
                    f"Difficulty-band cache is missing task_id {task_id!r}."
                )
            return task_family, default_band, "rollout_probe:default"

        resolved_family = str(record.get("task_family", "")).strip() or task_family
        resolved_band = str(record.get("difficulty_band", "")).strip() or default_band
        resolved_source = str(record.get("difficulty_band_source", "")).strip()
        return resolved_family, resolved_band, resolved_source or "rollout_probe:task_id"

    return task_family, default_band, strategy


def _resolve_task_family(
    *,
    task_id: str,
    row: Mapping[str, Any],
) -> str:
    for key in ("task_family", "difficulty_family"):
        raw_value = row.get(key)
        if isinstance(raw_value, str):
            normalized = raw_value.strip()
            if normalized:
                return normalized

    normalized_task_id = task_id.strip()
    if not normalized_task_id:
        return ""

    suffix = normalized_task_id.rsplit(".", 1)[-1]
    if "__" not in suffix:
        return ""
    family = suffix.split("__", 1)[0].strip()
    return family


def load_task_samples(
    *,
    config: OnPolicyDataConfig,
    dataset_loader: DatasetLoader | None = None,
    task_partition: str = _TASK_PARTITION_ALL,
    eval_split_fraction: float = 0.0,
    min_eval_rows: int = 0,
) -> list[TaskSample]:
    """Load the deterministic task pool for one partition of the on-policy dataset."""
    normalized_partition = _normalize_task_partition(task_partition)
    task_pool = _load_task_pool(config=config, dataset_loader=dataset_loader)
    tasks = list(task_pool.tasks)
    if not tasks:
        detail = task_pool.last_error if task_pool.last_error else "no valid rows found"
        raise ValueError(
            f"Unable to load tasks from {config.dataset_id!r}:{config.dataset_split!r}. "
            f"Collected 0 valid rows after scanning {task_pool.scanned_rows}. "
            f"Last validation error: {detail}"
        )
    if normalized_partition == _TASK_PARTITION_ALL:
        return tasks

    train_tasks, eval_tasks = split_task_samples_for_eval(
        tasks,
        eval_split_fraction=eval_split_fraction,
        min_eval_rows=min_eval_rows,
    )
    return train_tasks if normalized_partition == _TASK_PARTITION_TRAIN else eval_tasks


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
    try:
        tasks = load_task_samples(
            config=config,
            dataset_loader=dataset_loader,
            task_partition=normalized_partition,
            eval_split_fraction=eval_split_fraction,
            min_eval_rows=min_eval_rows,
        )
    except ValueError as exc:
        if normalized_partition != _TASK_PARTITION_ALL:
            raise
        raise ValueError(
            f"Unable to build task batch of size {batch_size} from "
            f"{config.dataset_id!r}:{config.dataset_split!r}. {exc}"
        ) from exc
    if not tasks:
        return []

    if normalized_partition == _TASK_PARTITION_ALL and len(tasks) < batch_size:
        partition_detail = (
            f" in task partition {normalized_partition!r}"
            if normalized_partition != _TASK_PARTITION_ALL
            else ""
        )
        raise ValueError(
            f"Unable to build task batch of size {batch_size} from "
            f"{config.dataset_id!r}:{config.dataset_split!r}{partition_detail}. "
            f"Collected {len(tasks)} valid rows."
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
                "task_family": task.task_family,
                "difficulty_band": task.difficulty_band,
                "difficulty_band_source": task.difficulty_band_source,
                "data_source": data_source,
                "verifier_kind": task.verifier_kind,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "reward_model": {
                    "ground_truth": {
                        "task_id": task.task_id,
                        "image_name": task.image_name,
                        "task_family": task.task_family,
                        "difficulty_band": task.difficulty_band,
                        "difficulty_band_source": task.difficulty_band_source,
                        "data_source": data_source,
                        "verifier_kind": task.verifier_kind,
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
        "patch_is_bug_introducing": bool(config.patch_is_bug_introducing),
        "verifier_kind": str(config.verifier_kind),
        "difficulty_banding": _difficulty_banding_fingerprint(config),
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
        key=lambda row_index: _stable_split_rank(copied_rows[row_index]),
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
        "patch_is_bug_introducing": bool(config.patch_is_bug_introducing),
        "verifier_kind": str(config.verifier_kind),
        "difficulty_banding": _difficulty_banding_fingerprint(config),
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


def _slugify_for_filename(value: str, *, fallback: str = "dataset") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if normalized:
        return normalized
    return fallback


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
        key=lambda row_index: _stable_task_sample_split_rank(copied_tasks[row_index]),
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


def _stable_split_rank(row: Mapping[str, Any]) -> str:
    explicit_task_id = _resolve_explicit_task_id(row)
    image_name = str(row.get("image_name", "")).strip()
    prompt_text = _extract_task_prompt_text(row)
    fail_to_pass = _extract_task_targets(row, keys=("fail_to_pass", "FAIL_TO_PASS"))
    pass_to_pass = _extract_task_targets(row, keys=("pass_to_pass", "PASS_TO_PASS"))
    return _stable_split_token(
        explicit_task_id=explicit_task_id,
        image_name=image_name,
        prompt_text=prompt_text,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
    )


def _stable_task_sample_split_rank(task: TaskSample) -> str:
    return _stable_split_token(
        explicit_task_id=_resolve_explicit_task_id(task.raw),
        image_name=task.image_name,
        prompt_text=task.problem_statement,
        fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass,
    )


def _resolve_explicit_task_id(row: Mapping[str, Any]) -> str:
    for key in ("instance_id", "problem_id"):
        raw_value = row.get(key)
        if isinstance(raw_value, str):
            normalized = raw_value.strip()
            if normalized:
                return normalized
    return ""


def _extract_task_prompt_text(row: Mapping[str, Any]) -> str:
    problem_statement = row.get("problem_statement")
    if isinstance(problem_statement, str):
        return problem_statement.strip()

    prompt = row.get("prompt")
    if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        prompt_chunks: list[str] = []
        for item in prompt:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                prompt_chunks.append(content.strip())
        if prompt_chunks:
            return "\n".join(prompt_chunks)
    return ""


def _extract_task_targets(row: Mapping[str, Any], *, keys: Sequence[str]) -> list[str]:
    for key in keys:
        if key in row:
            return _normalize_test_targets(row.get(key))

    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            for key in keys:
                if key in ground_truth:
                    return _normalize_test_targets(ground_truth.get(key))
    return []


def _stable_split_token(
    *,
    explicit_task_id: str,
    image_name: str,
    prompt_text: str,
    fail_to_pass: Any,
    pass_to_pass: Any,
) -> str:
    normalized_image_name = image_name.strip()
    if explicit_task_id:
        identity_payload: dict[str, Any] = {
            "task_id": explicit_task_id,
            "image_name": normalized_image_name,
        }
    else:
        identity_payload = {
            "image_name": normalized_image_name,
            "problem_statement": prompt_text.strip(),
            "fail_to_pass": _normalize_test_targets(fail_to_pass),
            "pass_to_pass": _normalize_test_targets(pass_to_pass),
        }
    encoded = json.dumps(identity_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
