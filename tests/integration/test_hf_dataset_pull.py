from __future__ import annotations

import os

import pytest

from config import resolve_on_policy_settings
from env.task_dataset import load_task_batch


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_HF_DATASET_INTEGRATION") != "1",
    reason="Set RUN_HF_DATASET_INTEGRATION=1 to validate pulling HF dataset.",
)


def test_pull_default_onpolicy_hf_dataset_and_required_columns() -> None:
    settings = resolve_on_policy_settings()
    samples = load_task_batch(
        step_index=0,
        batch_size=2,
        config=settings.data,
    )

    assert len(samples) == 2
    assert all(sample.task_id for sample in samples)
    assert all(sample.image_name for sample in samples)
    assert all(sample.problem_statement for sample in samples)
