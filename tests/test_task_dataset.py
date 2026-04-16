from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import OnPolicyDataConfig, OnPolicyDatasetColumns, OnPolicyDifficultyBandConfig
import env.task_dataset as dataset_module
from env.task_dataset import (
    TaskSample,
    build_sdpo_task_rows,
    load_task_batch,
    load_task_samples,
    preload_sdpo_task_rows_to_parquet,
    preload_sdpo_task_rows_split_to_parquet,
    resolve_on_policy_bad_task_cache_path,
    resolve_on_policy_difficulty_band_cache_path,
    resolve_sdpo_task_rows_cache_path,
    resolve_sdpo_task_split_cache_paths,
    split_task_samples_for_eval,
    split_sdpo_task_rows_for_eval,
)
from prompts.runtime_messages import build_onpolicy_initial_user_message


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


def _banded_config() -> OnPolicyDataConfig:
    return OnPolicyDataConfig(
        dataset_id="SWE-bench/SWE-smith-py",
        dataset_split="train",
        columns=OnPolicyDatasetColumns(
            image_name="image_name",
            problem_statement="problem_statement",
            fail_to_pass="FAIL_TO_PASS",
            pass_to_pass="PASS_TO_PASS",
        ),
        difficulty_banding=OnPolicyDifficultyBandConfig(
            strategy="instance_id_family",
            default_band="near_impossible",
            family_band_exact=(
                ("func_basic", "learnable"),
                ("combine_file", "near_impossible"),
            ),
            family_band_prefix=(("func_pm_", "near_impossible"),),
        ),
    )


def _rollout_probe_config(
    cache_path: str,
    *,
    rollout_probe_accept_partial: bool = False,
) -> OnPolicyDataConfig:
    return OnPolicyDataConfig(
        dataset_id="SWE-bench/SWE-smith-py",
        dataset_split="train",
        columns=OnPolicyDatasetColumns(
            image_name="image_name",
            problem_statement="problem_statement",
            fail_to_pass="FAIL_TO_PASS",
            pass_to_pass="PASS_TO_PASS",
        ),
        difficulty_banding=OnPolicyDifficultyBandConfig(
            strategy="rollout_probe",
            default_band="unbanded",
            rollout_probe_cache_path=cache_path,
            rollout_probe_required=True,
            rollout_probe_accept_partial=rollout_probe_accept_partial,
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


def test_split_task_samples_for_eval_is_deterministic() -> None:
    rows = [
        {
            "task_id": f"task-{index}",
            "image_name": f"img:{index}",
            "problem_statement": f"p{index}",
            "FAIL_TO_PASS": [f"f{index}"],
            "PASS_TO_PASS": [f"p{index}"],
        }
        for index in range(4)
    ]
    tasks = [
        load_task_batch(
            step_index=index,
            batch_size=1,
            config=_config(),
            dataset_loader=lambda _dataset_id, _split: rows,
        )[0]
        for index in range(4)
    ]

    train_a, eval_a = split_task_samples_for_eval(
        tasks,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )
    train_b, eval_b = split_task_samples_for_eval(
        tasks,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    assert [task.task_id for task in train_a] == [task.task_id for task in train_b]
    assert [task.task_id for task in eval_a] == [task.task_id for task in eval_b]
    assert train_a
    assert eval_a


def test_split_task_samples_for_eval_keeps_duplicate_logical_tasks_together() -> None:
    duplicate_a = TaskSample(
        task_id="synthetic:0",
        image_name="img:dup",
        problem_statement="Fix duplicated task",
        fail_to_pass=["tests/test_bug.py::test_fix"],
        pass_to_pass=["tests/test_ok.py::test_regression"],
        raw={},
    )
    duplicate_b = TaskSample(
        task_id="synthetic:1",
        image_name="img:dup",
        problem_statement="Fix duplicated task",
        fail_to_pass=["tests/test_bug.py::test_fix"],
        pass_to_pass=["tests/test_ok.py::test_regression"],
        raw={},
    )
    other_a = TaskSample(
        task_id="other:0",
        image_name="img:other-a",
        problem_statement="Other task A",
        fail_to_pass=["tests/test_bug.py::test_a"],
        pass_to_pass=["tests/test_ok.py::test_a"],
        raw={},
    )
    other_b = TaskSample(
        task_id="other:1",
        image_name="img:other-b",
        problem_statement="Other task B",
        fail_to_pass=["tests/test_bug.py::test_b"],
        pass_to_pass=["tests/test_ok.py::test_b"],
        raw={},
    )

    train_tasks, eval_tasks = split_task_samples_for_eval(
        [duplicate_a, duplicate_b, other_a, other_b],
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    train_ids = {task.task_id for task in train_tasks}
    eval_ids = {task.task_id for task in eval_tasks}
    duplicate_ids = {"synthetic:0", "synthetic:1"}

    assert duplicate_ids.issubset(train_ids) or duplicate_ids.issubset(eval_ids)


def test_load_task_batch_supports_deterministic_train_eval_partitions() -> None:
    rows = [
        {
            "task_id": f"task-{index}",
            "image_name": f"img:{index}",
            "problem_statement": f"p{index}",
            "FAIL_TO_PASS": [f"f{index}"],
            "PASS_TO_PASS": [f"p{index}"],
        }
        for index in range(4)
    ]

    train_batch = load_task_batch(
        step_index=0,
        batch_size=3,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
        task_partition="train",
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )
    eval_batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
        task_partition="eval",
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    assert len(train_batch) == 3
    assert len(eval_batch) == 1
    assert {sample.task_id for sample in train_batch}.isdisjoint(
        {sample.task_id for sample in eval_batch}
    )


def test_load_task_batch_attaches_difficulty_tags_from_instance_family_rules() -> None:
    rows = [
        {
            "instance_id": "repo.sha.func_basic__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        },
        {
            "instance_id": "repo.sha.func_pm_ctrl_shuffle__0002",
            "image_name": "img:2",
            "problem_statement": "p2",
            "FAIL_TO_PASS": ["f2"],
            "PASS_TO_PASS": ["p2"],
        },
        {
            "instance_id": "repo.sha.combine_module__0003",
            "image_name": "img:3",
            "problem_statement": "p3",
            "FAIL_TO_PASS": ["f3"],
            "PASS_TO_PASS": ["p3"],
        },
    ]

    batch = load_task_batch(
        step_index=0,
        batch_size=3,
        config=_banded_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [sample.task_family for sample in batch] == [
        "func_basic",
        "func_pm_ctrl_shuffle",
        "combine_module",
    ]
    assert [sample.difficulty_band for sample in batch] == [
        "learnable",
        "near_impossible",
        "near_impossible",
    ]
    assert [sample.difficulty_band_source for sample in batch] == [
        "instance_id_family:exact",
        "instance_id_family:prefix",
        "instance_id_family:default",
    ]
    assert batch[0].raw["difficulty_band"] == "learnable"
    assert batch[1].raw["task_family"] == "func_pm_ctrl_shuffle"


def test_load_task_batch_wraps_partitioned_batches_when_heldout_split_is_smaller_than_batch() -> None:
    rows = [
        {
            "task_id": f"task-{index}",
            "image_name": f"img:{index}",
            "problem_statement": f"p{index}",
            "FAIL_TO_PASS": [f"f{index}"],
            "PASS_TO_PASS": [f"p{index}"],
        }
        for index in range(4)
    ]

    eval_batch = load_task_batch(
        step_index=0,
        batch_size=3,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
        task_partition="eval",
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    assert len(eval_batch) == 3
    assert len({sample.task_id for sample in eval_batch}) == 1


def test_load_task_samples_returns_partitioned_pool() -> None:
    rows = [
        {
            "task_id": f"task-{index}",
            "image_name": f"img:{index}",
            "problem_statement": f"p{index}",
            "FAIL_TO_PASS": [f"f{index}"],
            "PASS_TO_PASS": [f"p{index}"],
        }
        for index in range(4)
    ]

    train_tasks = load_task_samples(
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
        task_partition="train",
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )
    eval_tasks = load_task_samples(
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
        task_partition="eval",
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    assert train_tasks
    assert eval_tasks
    assert {task.task_id for task in train_tasks}.isdisjoint(
        {task.task_id for task in eval_tasks}
    )


def test_build_sdpo_task_rows_carries_difficulty_tags() -> None:
    rows = [
        {
            "instance_id": "repo.sha.combine_file__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        }
    ]

    task_rows = build_sdpo_task_rows(
        config=_banded_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert task_rows[0]["task_family"] == "combine_file"
    assert task_rows[0]["difficulty_band"] == "near_impossible"
    assert task_rows[0]["difficulty_band_source"] == "instance_id_family:exact"
    assert task_rows[0]["reward_model"]["ground_truth"]["difficulty_band"] == "near_impossible"


def test_load_task_batch_uses_rollout_probe_cache_when_present(tmp_path: Path) -> None:
    cache_path = tmp_path / "difficulty_bands.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "records": [
                    {
                        "task_id": "repo.sha.func_basic__0001",
                        "task_family": "func_basic",
                        "difficulty_band": "easy",
                        "difficulty_band_source": "rollout_probe:selected_3_of_4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "repo.sha.func_basic__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        }
    ]

    batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=_rollout_probe_config(str(cache_path)),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert batch[0].task_family == "func_basic"
    assert batch[0].difficulty_band == "easy"
    assert batch[0].difficulty_band_source == "rollout_probe:selected_3_of_4"


def test_load_task_batch_requires_rollout_probe_entry_for_every_task(tmp_path: Path) -> None:
    cache_path = tmp_path / "difficulty_bands.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "repo.sha.func_basic__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        }
    ]

    with pytest.raises(ValueError, match="missing task_id"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_rollout_probe_config(str(cache_path)),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_batch_rejects_duplicate_rollout_probe_task_ids(tmp_path: Path) -> None:
    cache_path = tmp_path / "difficulty_bands.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "records": [
                    {
                        "task_id": "repo.sha.func_basic__0001",
                        "task_family": "func_basic",
                        "difficulty_band": "easy",
                    },
                    {
                        "task_id": "repo.sha.func_basic__0001",
                        "task_family": "func_basic",
                        "difficulty_band": "near_impossible",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "repo.sha.func_basic__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        }
    ]

    with pytest.raises(ValueError, match="duplicate task_id"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_rollout_probe_config(str(cache_path)),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_batch_reraises_partitioned_rollout_probe_errors(tmp_path: Path) -> None:
    cache_path = tmp_path / "difficulty_bands.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "repo.sha.func_basic__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        }
    ]

    with pytest.raises(ValueError, match="missing task_id"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_rollout_probe_config(str(cache_path)),
            dataset_loader=lambda _dataset_id, _split: rows,
            task_partition="train",
            eval_split_fraction=0.25,
            min_eval_rows=1,
        )


def test_load_task_batch_allows_empty_eval_partition_when_holdout_resolves_to_zero_rows() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "p0",
            "FAIL_TO_PASS": ["f0"],
            "PASS_TO_PASS": ["p0"],
        }
    ]

    eval_batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
        task_partition="eval",
        eval_split_fraction=0.25,
        min_eval_rows=0,
    )

    assert eval_batch == []


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

    assert [sample.task_id for sample in batch] == ["task-0", "task-1"]
    assert batch[1].raw["prompt_source"] == "target_preview_fallback"
    assert "Verifier target preview" in batch[1].problem_statement


def test_load_task_batch_errors_when_not_enough_valid_rows() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "broken row",
            "FAIL_TO_PASS": [],
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


def test_load_task_batch_skips_rows_with_empty_verifier_targets() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "p0",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": ["b"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["c"],
            "PASS_TO_PASS": ["d"],
        },
    ]

    batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [sample.task_id for sample in batch] == ["task-1"]


def test_load_task_batch_skips_rows_marked_bad_in_cache(tmp_path: Path, monkeypatch) -> None:
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
        {
            "task_id": "task-3",
            "image_name": "img:3",
            "problem_statement": "p3",
            "FAIL_TO_PASS": ["g"],
            "PASS_TO_PASS": ["h"],
        },
    ]
    cache_path = tmp_path / "bad_tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "bad_task_ids": ["task-1"],
                "bad_image_names": ["img:2"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SMALL_SWE_BAD_TASK_CACHE_PATH", str(cache_path))

    batch = load_task_batch(
        step_index=0,
        batch_size=2,
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [sample.task_id for sample in batch] == ["task-0", "task-3"]


def test_load_task_batch_steps_walk_filtered_task_pool(tmp_path: Path, monkeypatch) -> None:
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
    cache_path = tmp_path / "bad_tasks.json"
    cache_path.write_text(
        json.dumps({"bad_task_ids": ["task-1"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SMALL_SWE_BAD_TASK_CACHE_PATH", str(cache_path))

    observed = [
        load_task_batch(
            step_index=step_index,
            batch_size=1,
            config=_config(),
            dataset_loader=lambda _dataset_id, _split: rows,
        )[0].task_id
        for step_index in range(4)
    ]

    assert observed == ["task-0", "task-2", "task-0", "task-2"]


def test_build_sdpo_task_rows_uses_full_split_with_prompt_metadata() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "fix bug 0",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-2",
            "image_name": "img:2",
            "problem_statement": "fix bug 2",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix_2"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression_2"],
        },
    ]

    sdpo_rows = build_sdpo_task_rows(
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [row["task_id"] for row in sdpo_rows] == ["task-0", "task-1", "task-2"]
    expected_prompt = build_onpolicy_initial_user_message(
        problem_statement="fix bug 0",
    )
    assert sdpo_rows[0]["prompt"] == [{"role": "user", "content": expected_prompt}]
    assert sdpo_rows[0]["image_name"] == "img:0"
    assert sdpo_rows[0]["data_source"] == "dummy/dataset"
    assert sdpo_rows[0]["fail_to_pass"] == ["tests/test_bug.py::test_bugfix"]
    assert sdpo_rows[0]["pass_to_pass"] == ["tests/test_ok.py::test_regression"]
    assert sdpo_rows[0]["reward_model"]["ground_truth"]["data_source"] == "dummy/dataset"
    assert "Verifier target preview" in sdpo_rows[1]["prompt"][0]["content"]


def test_build_sdpo_task_rows_skips_rows_marked_bad_in_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "fix bug 0",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "fix bug 1",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix_1"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression_1"],
        },
        {
            "task_id": "task-2",
            "image_name": "img:2",
            "problem_statement": "fix bug 2",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix_2"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression_2"],
        },
    ]
    cache_path = tmp_path / "bad_tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "bad_task_ids": ["task-1"],
                "bad_image_names": ["img:2"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SMALL_SWE_BAD_TASK_CACHE_PATH", str(cache_path))

    sdpo_rows = build_sdpo_task_rows(
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [row["task_id"] for row in sdpo_rows] == ["task-0"]


def test_resolve_on_policy_bad_task_cache_path_is_deterministic(tmp_path: Path) -> None:
    resolved = resolve_on_policy_bad_task_cache_path(
        config=_config(),
        cache_dir=tmp_path,
    )

    assert resolved.parent == tmp_path
    assert resolved.name.startswith("bad_tasks_dummy_dataset_train_")
    assert resolved.suffix == ".json"


def test_resolve_on_policy_difficulty_band_cache_path_is_descriptive(tmp_path: Path) -> None:
    resolved = resolve_on_policy_difficulty_band_cache_path(
        config=_config(),
        cache_dir=tmp_path,
        probe_label="positive_rft_probe",
    )

    assert resolved.parent == tmp_path
    assert resolved.name == "difficulty_bands_dummy_dataset_train_positive_rft_probe.json"


def test_resolve_on_policy_difficulty_band_cache_path_scopes_partial_probe(tmp_path: Path) -> None:
    resolved = resolve_on_policy_difficulty_band_cache_path(
        config=_config(),
        cache_dir=tmp_path,
        probe_label="positive_rft_probe",
        task_partition="eval",
        start_task_index=32,
        task_limit=16,
        eval_split_fraction=0.25,
        min_eval_rows=2,
    )

    assert resolved.parent == tmp_path
    assert (
        resolved.name
        == "difficulty_bands_dummy_dataset_train_positive_rft_probe_eval_frac_0.25_mineval_2_start_32_limit_16.json"
    )


def test_resolve_on_policy_difficulty_band_cache_path_rejects_non_filename_probe_label(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="probe_label must contain at least one filename-safe character"):
        resolve_on_policy_difficulty_band_cache_path(
            config=_config(),
            cache_dir=tmp_path,
            probe_label="!!!",
        )


def test_load_task_batch_invalidates_cached_hf_pool_when_rollout_probe_source_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_a = tmp_path / "difficulty_bands_a.json"
    cache_b = tmp_path / "difficulty_bands_b.json"
    for cache_path, difficulty_band in (
        (cache_a, "easy"),
        (cache_b, "near_impossible"),
    ):
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                    "records": [
                        {
                            "task_id": "repo.sha.func_basic__0001",
                            "task_family": "func_basic",
                            "difficulty_band": difficulty_band,
                            "difficulty_band_source": f"rollout_probe:{difficulty_band}",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    rows = [
        {
            "task_id": "repo.sha.func_basic__0001",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        }
    ]
    config = OnPolicyDataConfig(
        dataset_id="dummy/dataset",
        dataset_split="train",
        columns=OnPolicyDatasetColumns(
            image_name="image_name",
            problem_statement="problem_statement",
            fail_to_pass="FAIL_TO_PASS",
            pass_to_pass="PASS_TO_PASS",
        ),
        difficulty_banding=OnPolicyDifficultyBandConfig(
            strategy="rollout_probe",
            default_band="unbanded",
            rollout_probe_required=True,
        ),
    )

    dataset_module._load_hf_task_pool_cached.cache_clear()
    dataset_module._load_rollout_probe_cache_records_cached.cache_clear()
    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda dataset_id, split: rows)

    monkeypatch.setenv("SMALL_SWE_DIFFICULTY_BAND_CACHE_PATH", str(cache_a))
    first_batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=config,
    )

    monkeypatch.setenv("SMALL_SWE_DIFFICULTY_BAND_CACHE_PATH", str(cache_b))
    second_batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=config,
    )

    assert first_batch[0].difficulty_band == "easy"
    assert second_batch[0].difficulty_band == "near_impossible"


def test_load_task_batch_fails_fast_when_required_rollout_probe_cache_omits_task(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "difficulty_bands_partial.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "records": [
                    {
                        "task_id": "task-a",
                        "task_family": "func_basic",
                        "difficulty_band": "easy",
                        "difficulty_band_source": "rollout_probe:easy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "task-a",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        },
        {
            "task_id": "task-b",
            "image_name": "img:2",
            "problem_statement": "p2",
            "FAIL_TO_PASS": ["f2"],
            "PASS_TO_PASS": ["p2"],
        },
    ]

    with pytest.raises(ValueError, match="Difficulty-band cache is missing task_id 'task-b'"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_rollout_probe_config(str(cache_path)),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_batch_rejects_incomplete_rollout_probe_cache(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "difficulty_bands_incomplete.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "probe_status": "incomplete",
                "records": [
                    {
                        "task_id": "task-a",
                        "task_family": "func_basic",
                        "difficulty_band": "easy",
                        "difficulty_band_source": "rollout_probe:easy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "task-a",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        },
    ]

    with pytest.raises(ValueError, match="Difficulty-band cache is incomplete"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_rollout_probe_config(str(cache_path)),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_samples_accepts_partial_rollout_probe_cache_as_labeled_subset(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "difficulty_bands_incomplete.partial.json"
    partial_records_path = tmp_path / "difficulty_bands_incomplete.partial.records.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": dataset_module.ON_POLICY_DIFFICULTY_BAND_CACHE_SCHEMA_VERSION,
                "probe_status": "incomplete",
                "task_count_expected": 2,
                "task_count_completed": 1,
                "partial_records_path": partial_records_path.name,
            }
        ),
        encoding="utf-8",
    )
    partial_records_path.write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "task_family": "func_basic",
                "difficulty_band": "learnable",
                "difficulty_band_source": "rollout_probe:selected_1_of_4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "task-a",
            "image_name": "img:1",
            "problem_statement": "p1",
            "FAIL_TO_PASS": ["f1"],
            "PASS_TO_PASS": ["p1"],
        },
        {
            "task_id": "task-b",
            "image_name": "img:2",
            "problem_statement": "p2",
            "FAIL_TO_PASS": ["f2"],
            "PASS_TO_PASS": ["p2"],
        },
    ]

    task_samples = load_task_samples(
        config=_rollout_probe_config(
            str(cache_path),
            rollout_probe_accept_partial=True,
        ),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [task.task_id for task in task_samples] == ["task-a"]
    assert task_samples[0].difficulty_band == "learnable"
    assert task_samples[0].difficulty_band_source == "rollout_probe:selected_1_of_4"


def test_build_sdpo_task_rows_filters_problem_statement_length_under_4k() -> None:
    rows = [
        {
            "task_id": "task-keep",
            "image_name": "img:0",
            "problem_statement": "a" * 3999,
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-drop",
            "image_name": "img:1",
            "problem_statement": "b" * 4000,
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
    ]

    sdpo_rows = build_sdpo_task_rows(
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [row["task_id"] for row in sdpo_rows] == ["task-keep"]


def test_preload_sdpo_task_rows_to_parquet_uses_existing_cache_file(tmp_path: Path) -> None:
    cache_path = resolve_sdpo_task_rows_cache_path(
        config=_config(),
        cache_dir=tmp_path,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("existing", encoding="utf-8")

    resolved = preload_sdpo_task_rows_to_parquet(
        config=_config(),
        cache_dir=tmp_path,
        dataset_loader=lambda _dataset_id, _split: (_ for _ in ()).throw(
            AssertionError("dataset loader should not be called for warm cache")
        ),
    )

    assert resolved == cache_path
    assert resolved.read_text(encoding="utf-8") == "existing"


def test_preload_sdpo_task_rows_to_parquet_builds_and_writes_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_write(records: object, output_path: str | Path) -> None:
        captured["records"] = records
        captured["output_path"] = Path(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("parquet", encoding="utf-8")

    monkeypatch.setattr(dataset_module, "_write_records_to_parquet", _stub_write)

    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "fix bug 0",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        }
    ]

    output_path = preload_sdpo_task_rows_to_parquet(
        config=_config(),
        cache_dir=tmp_path,
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert output_path == resolve_sdpo_task_rows_cache_path(config=_config(), cache_dir=tmp_path)
    assert captured["output_path"] == output_path
    assert output_path.read_text(encoding="utf-8") == "parquet"
    assert isinstance(captured["records"], list)
    assert captured["records"][0]["task_id"] == "task-0"


def test_resolve_sdpo_cache_paths_include_bad_task_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "bad_tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "bad_task_ids": ["task-1"],
                "bad_image_names": ["img:1"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SMALL_SWE_BAD_TASK_CACHE_PATH", str(cache_path))

    rows_path_a = resolve_sdpo_task_rows_cache_path(
        config=_config(),
        cache_dir=tmp_path,
    )
    train_path_a, val_path_a = resolve_sdpo_task_split_cache_paths(
        config=_config(),
        cache_dir=tmp_path,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    cache_path.write_text(
        json.dumps(
            {
                "bad_task_ids": ["task-2"],
                "bad_image_names": ["img:2"],
            }
        ),
        encoding="utf-8",
    )

    rows_path_b = resolve_sdpo_task_rows_cache_path(
        config=_config(),
        cache_dir=tmp_path,
    )
    train_path_b, val_path_b = resolve_sdpo_task_split_cache_paths(
        config=_config(),
        cache_dir=tmp_path,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    assert rows_path_a != rows_path_b
    assert train_path_a != train_path_b
    assert val_path_a != val_path_b


def test_split_sdpo_task_rows_for_eval_is_deterministic_and_non_empty() -> None:
    rows = [
        {"task_id": "task-0", "image_name": "img:0"},
        {"task_id": "task-1", "image_name": "img:1"},
        {"task_id": "task-2", "image_name": "img:2"},
        {"task_id": "task-3", "image_name": "img:3"},
        {"task_id": "task-4", "image_name": "img:4"},
    ]
    train_a, eval_a = split_sdpo_task_rows_for_eval(
        rows,
        eval_split_fraction=0.4,
        min_eval_rows=1,
    )
    train_b, eval_b = split_sdpo_task_rows_for_eval(
        rows,
        eval_split_fraction=0.4,
        min_eval_rows=1,
    )

    assert [row["task_id"] for row in train_a] == [row["task_id"] for row in train_b]
    assert [row["task_id"] for row in eval_a] == [row["task_id"] for row in eval_b]
    assert train_a
    assert eval_a


def test_split_sdpo_task_rows_for_eval_keeps_duplicate_logical_tasks_together() -> None:
    duplicate_prompt = [{"role": "user", "content": "Fix duplicated task"}]
    rows = [
        {
            "task_id": "synthetic:0",
            "image_name": "img:dup",
            "prompt": duplicate_prompt,
            "fail_to_pass": ["tests/test_bug.py::test_fix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "synthetic:1",
            "image_name": "img:dup",
            "prompt": duplicate_prompt,
            "fail_to_pass": ["tests/test_bug.py::test_fix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "other:0",
            "image_name": "img:other-a",
            "prompt": [{"role": "user", "content": "Other task A"}],
            "fail_to_pass": ["tests/test_bug.py::test_a"],
            "pass_to_pass": ["tests/test_ok.py::test_a"],
        },
        {
            "task_id": "other:1",
            "image_name": "img:other-b",
            "prompt": [{"role": "user", "content": "Other task B"}],
            "fail_to_pass": ["tests/test_bug.py::test_b"],
            "pass_to_pass": ["tests/test_ok.py::test_b"],
        },
    ]

    train_rows, eval_rows = split_sdpo_task_rows_for_eval(
        rows,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    train_ids = {str(row["task_id"]) for row in train_rows}
    eval_ids = {str(row["task_id"]) for row in eval_rows}
    duplicate_ids = {"synthetic:0", "synthetic:1"}

    assert duplicate_ids.issubset(train_ids) or duplicate_ids.issubset(eval_ids)


def test_preload_sdpo_task_rows_split_to_parquet_writes_train_and_val(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[list[dict[str, object]], Path]] = []

    def _stub_write(records: object, output_path: str | Path) -> None:
        assert isinstance(records, list)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("parquet", encoding="utf-8")
        writes.append((records, path))

    monkeypatch.setattr(dataset_module, "_write_records_to_parquet", _stub_write)

    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "fix bug 0",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "fix bug 1",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix_1"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression_1"],
        },
        {
            "task_id": "task-2",
            "image_name": "img:2",
            "problem_statement": "fix bug 2",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix_2"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression_2"],
        },
        {
            "task_id": "task-3",
            "image_name": "img:3",
            "problem_statement": "fix bug 3",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix_3"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression_3"],
        },
    ]

    train_path, val_path = preload_sdpo_task_rows_split_to_parquet(
        config=_config(),
        cache_dir=tmp_path,
        eval_split_fraction=0.25,
        min_eval_rows=1,
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    expected_train_path, expected_val_path = resolve_sdpo_task_split_cache_paths(
        config=_config(),
        cache_dir=tmp_path,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )
    assert train_path == expected_train_path
    assert val_path == expected_val_path
    assert train_path.read_text(encoding="utf-8") == "parquet"
    assert val_path.read_text(encoding="utf-8") == "parquet"
    assert len(writes) == 2
    assert writes[0][1] == train_path
    assert writes[1][1] == val_path
    assert writes[0][0]
    assert writes[1][0]


def test_resolve_sdpo_task_split_cache_paths_include_prompt_length_filter(tmp_path: Path) -> None:
    train_default, val_default = resolve_sdpo_task_split_cache_paths(
        config=_config(),
        cache_dir=tmp_path,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )
    train_relaxed, val_relaxed = resolve_sdpo_task_split_cache_paths(
        config=_config(),
        cache_dir=tmp_path,
        eval_split_fraction=0.25,
        min_eval_rows=1,
        max_problem_statement_chars=5000,
    )

    assert train_default != train_relaxed
    assert val_default != val_relaxed


def test_load_task_batch_rejects_duplicate_targets_within_group() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "Fix duplicates",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_fix", "tests/test_bug.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        }
    ]

    with pytest.raises(ValueError, match="Collected 0 valid rows"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_config(),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_batch_rejects_fail_pass_overlap() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "Fix overlap",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_bug.py::test_fix"],
        }
    ]

    with pytest.raises(ValueError, match="Collected 0 valid rows"):
        load_task_batch(
            step_index=0,
            batch_size=1,
            config=_config(),
            dataset_loader=lambda _dataset_id, _split: rows,
        )


def test_load_task_batch_builds_bounded_prompt_fallback_when_problem_statement_missing() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "",
            "FAIL_TO_PASS": ["TestBug"],
            "PASS_TO_PASS": [
                "TestRegressionA",
                "TestRegressionB",
                "TestRegressionC",
                "TestRegressionD",
                "TestRegressionE",
            ],
        }
    ]
    config = OnPolicyDataConfig(
        dataset_id="dummy/go",
        dataset_split="train",
        columns=OnPolicyDatasetColumns(
            image_name="image_name",
            problem_statement="problem_statement",
            fail_to_pass="FAIL_TO_PASS",
            pass_to_pass="PASS_TO_PASS",
        ),
        verifier_kind="go_test",
    )

    batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=config,
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert len(batch) == 1
    prompt = batch[0].problem_statement
    assert "Resolve the failing Go tests for this task." in prompt
    assert "FAIL_TO_PASS (1): TestBug" in prompt
    assert "PASS_TO_PASS (5): TestRegressionA, TestRegressionB, TestRegressionC, TestRegressionD, ... (+1 more)" in prompt


def test_load_task_pool_dedupes_duplicate_logical_tasks_with_accounting() -> None:
    rows = [
        {
            "task_id": "task-0",
            "image_name": "img:0",
            "problem_statement": "Fix duplicate",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "Fix duplicate",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
    ]

    pool = dataset_module._load_task_pool(
        config=_config(),
        dataset_loader=lambda _dataset_id, _split: rows,
    )

    assert [task.task_id for task in pool.tasks] == ["task-0"]
    assert pool.filtered_counts["duplicate_logical_task"] == 1
    assert pool.filtered_task_ids["duplicate_logical_task"] == ("task-1",)


def test_build_sdpo_task_rows_preserves_verifier_kind_in_reward_ground_truth() -> None:
    config = OnPolicyDataConfig(
        dataset_id="dummy/go",
        dataset_split="train",
        columns=OnPolicyDatasetColumns(
            image_name="image_name",
            problem_statement="problem_statement",
            fail_to_pass="FAIL_TO_PASS",
            pass_to_pass="PASS_TO_PASS",
        ),
        verifier_kind="go_test",
    )

    rows = build_sdpo_task_rows(
        config=config,
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-0",
                "image_name": "img:0",
                "problem_statement": "",
                "FAIL_TO_PASS": ["TestBug"],
                "PASS_TO_PASS": ["TestRegression"],
            }
        ],
    )

    assert rows[0]["verifier_kind"] == "go_test"
    assert rows[0]["reward_model"]["ground_truth"]["verifier_kind"] == "go_test"
