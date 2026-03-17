from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import OnPolicyDataConfig, OnPolicyDatasetColumns
import env.task_dataset as dataset_module
from env.task_dataset import (
    build_sdpo_task_rows,
    load_task_batch,
    preload_sdpo_task_rows_to_parquet,
    preload_sdpo_task_rows_split_to_parquet,
    resolve_on_policy_bad_task_cache_path,
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

    assert [row["task_id"] for row in sdpo_rows] == ["task-0", "task-2"]
    expected_prompt = build_onpolicy_initial_user_message(
        problem_statement="fix bug 0",
    )
    assert sdpo_rows[0]["prompt"] == [{"role": "user", "content": expected_prompt}]
    assert sdpo_rows[0]["image_name"] == "img:0"
    assert sdpo_rows[0]["data_source"] == "dummy/dataset"
    assert sdpo_rows[0]["fail_to_pass"] == ["tests/test_bug.py::test_bugfix"]
    assert sdpo_rows[0]["pass_to_pass"] == ["tests/test_ok.py::test_regression"]
    assert sdpo_rows[0]["reward_model"]["ground_truth"]["data_source"] == "dummy/dataset"


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
