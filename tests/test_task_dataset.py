from __future__ import annotations

import pytest

from config import OnPolicyDataConfig, OnPolicyDatasetColumns
from env.task_dataset import load_task_batch


def _config() -> OnPolicyDataConfig:
    return OnPolicyDataConfig(
        dataset_id="dummy/dataset",
        dataset_split="train",
        columns=OnPolicyDatasetColumns(
            image_name="image_name",
            problem_statement="problem_statement",
            fail_to_pass="FAIL_TO_PASS",
            pass_to_pass="PASS_TO_PASS",
        ),
    )


def test_load_task_batch_is_step_deterministic_with_wraparound() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "p0",
            "FAIL_TO_PASS": ["a"],
            "PASS_TO_PASS": ["b"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["c"],
            "PASS_TO_PASS": ["d"],
        },
        {
            "task_id": "task-2",
            "image_name": "img:2",
            "problem_statement": "p2",
            "FAIL_TO_PASS": ["e"],
            "PASS_TO_PASS": ["f"],
        },
    ]

    batch = load_task_batch(
        step_index=1,
        batch_size=2,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [sample.task_id for sample in batch] == ["task-2", "task-0"]
    assert batch[0].problem_statement == "p2"


def test_load_task_batch_rejects_missing_required_columns() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "p0",
            "FAIL_TO_PASS": ["a"],
        }
    ]

    with pytest.raises(ValueError, match="missing required columns"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_config(),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_batch_skips_invalid_rows_when_collecting_batch() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "p0",
            "FAIL_TO_PASS": ["a"],
            "PASS_TO_PASS": ["b"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "",
            "FAIL_TO_PASS": ["c"],
            "PASS_TO_PASS": ["d"],
        },
        {
            "task_id": "task-2",
            "image_name": "img:2",
            "problem_statement": "p2",
            "FAIL_TO_PASS": ["e"],
            "PASS_TO_PASS": ["f"],
        },
    ]

    batch = load_task_batch(
        step_index=0,
        batch_size=2,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [sample.task_id for sample in batch] == ["task-0", "task-2"]


def test_load_task_batch_errors_when_not_enough_valid_rows() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "",
            "FAIL_TO_PASS": ["a"],
            "PASS_TO_PASS": ["b"],
        }
    ]

    with pytest.raises(ValueError, match="Unable to build task batch"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_config(),
            dataset_loader=lambda _dataset_id, _split: rows,
        )
