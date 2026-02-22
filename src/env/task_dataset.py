"""Task sampling helpers for on-policy SWE dataset collection."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from config import OnPolicyDataConfig


DatasetLoader = Callable[[str, str], Sequence[Mapping[str, Any]]]


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

    task_id_raw = row.get("instance_id") or row.get("task_id") or row.get("problem_id")
    if not isinstance(task_id_raw, str) or not task_id_raw.strip():
        task_id = f"{config.dataset_id}:{row_index}"
    else:
        task_id = task_id_raw.strip()

    return TaskSample(
        task_id=task_id,
        image_name=image_name_raw.strip(),
        problem_statement=prompt_raw,
        fail_to_pass=row.get(config.columns.fail_to_pass),
        pass_to_pass=row.get(config.columns.pass_to_pass),
        raw=dict(row),
    )


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
    for offset in range(batch_size):
        row_index = (start + offset) % len(dataset)
        row = dataset[row_index]
        if not isinstance(row, Mapping):
            raise ValueError(f"Dataset row {row_index} is not a mapping.")
        samples.append(_coerce_task_row(row, config=config, row_index=row_index))
    return samples
