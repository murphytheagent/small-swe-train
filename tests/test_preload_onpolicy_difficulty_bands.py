from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import (
    OnPolicyDataConfig,
    OnPolicyDatasetColumns,
    OnPolicyDifficultyBandConfig,
)
from env import preload_onpolicy_difficulty_bands as band_module
from env.task_dataset import TaskSample, load_task_batch


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


def _settings(
    *,
    eval_split_fraction: float = 0.1,
    eval_min_rows: int = 1,
    difficulty_banding: OnPolicyDifficultyBandConfig | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=OnPolicyDataConfig(
            dataset_id="dummy/dataset",
            dataset_split="train",
            columns=OnPolicyDatasetColumns(
                image_name="image_name",
                problem_statement="problem_statement",
                fail_to_pass="FAIL_TO_PASS",
                pass_to_pass="PASS_TO_PASS",
            ),
            difficulty_banding=difficulty_banding or OnPolicyDifficultyBandConfig(),
        ),
        runtime=SimpleNamespace(
            eval_split_fraction=eval_split_fraction,
            eval_min_rows=eval_min_rows,
            max_tool_calls_per_turn=4,
        ),
    )


def _stage_metadata(stage_name: str = "positive_rft") -> dict[str, object]:
    resolved_stage_name = band_module.resolve_rft_stage_name(stage_name)
    return {
        "verify_submissions": band_module.resolve_rft_stage_verify_submissions(
            resolved_stage_name
        ),
        "stage_handoff_overrides": band_module.resolve_rft_stage_handoff_overrides(
            resolved_stage_name
        ),
        "stage_selection_contract": band_module.resolve_rft_stage_selection_contract(
            resolved_stage_name
        ),
        "stage_correctness_contract": band_module.resolve_rft_stage_correctness_contract(
            resolved_stage_name
        ),
    }


def _stub_probe_collection(monkeypatch, collect_fn) -> None:
    def _fake_collect_probe(**kwargs):
        request = SimpleNamespace(
            data_config_name=kwargs["data_config_name"],
            turn_generator_mode=kwargs["turn_generator_mode"],
            start_step_index=kwargs["probe_step_index"],
            runtime_overrides={
                "task_batch_size": 1,
                "attempts_per_task": int(kwargs["attempts_per_task"]),
                "env_pool_size": 1,
                "max_in_flight_tasks": 1,
            },
            data_overrides=kwargs["data_overrides"],
            task_partition="all",
            task_eval_split_fraction=0.0,
            task_eval_min_rows=0,
            dataset_loader=object(),
            verify_submissions=bool(kwargs["verify_submissions"]),
            stage_name=str(kwargs["stage_name"]),
        )
        return collect_fn(request=request, tokenizer=None)

    monkeypatch.setattr(
        band_module,
        "_collect_probe_rollout_rows_for_task",
        _fake_collect_probe,
    )
    monkeypatch.setattr(
        band_module,
        "build_rft_handoff_result_from_rollout_rows",
        lambda *, rollout_rows, max_tool_calls, tokenizer, handoff_overrides=None: dict(rollout_rows),
    )


def test_main_print_path_only_uses_resolved_cache_path(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_difficulty_band_cache_dir",
        lambda *, project_root: Path("/tmp/difficulty-bands"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--print-path-only",
        ],
    )

    exit_code = band_module.main()
    output = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert output == "/tmp/difficulty-bands/difficulty_bands_dummy_dataset_train_positive_rft_probe.json"


def test_main_materializes_rollout_probe_cache(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        ),
        TaskSample(
            task_id="task-b",
            image_name="img:b",
            problem_statement="pb",
            fail_to_pass=["fb"],
            pass_to_pass=["pb"],
            raw={},
            task_family="combine_file",
        ),
    ]

    captured_requests = []

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        if request.start_step_index == 0:
            return {
                "rollout_rows": [
                    {"resolved": True},
                    {"resolved": False},
                    {"resolved": False},
                    {"resolved": False},
                ],
                "selected_rows": [{"task_id": "task-a"}],
                "rejected_rows": [
                    {"rft_rejection_reason": "unresolved"},
                    {"rft_rejection_reason": "unresolved"},
                    {"rft_rejection_reason": "unresolved"},
                ],
            }
        return {
            "rollout_rows": [
                {"resolved": True},
                {"resolved": True},
                {"resolved": True},
                {"resolved": False, "infra_invalid": True},
            ],
            "selected_rows": [
                {"task_id": "task-b"},
                {"task_id": "task-b"},
                {"task_id": "task-b"},
            ],
            "rejected_rows": [
                {"rft_rejection_reason": "infra_invalid", "infra_invalid": True},
            ],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "smoke",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path == tmp_path / "difficulty_bands_dummy_dataset_train_smoke.json"
    assert payload["patch_is_bug_introducing"] is True
    assert payload["verifier_kind"] == "pytest"
    assert payload["task_count"] == 2
    assert payload["records"][0]["task_id"] == "task-a"
    assert payload["records"][0]["difficulty_band"] == "learnable"
    assert payload["records"][0]["difficulty_band_source"] == "rollout_probe:selected_1_of_4"
    assert payload["records"][1]["task_id"] == "task-b"
    assert payload["records"][1]["difficulty_band"] == "easy"
    assert payload["records"][1]["infra_invalid_attempt_count"] == 1
    assert len(captured_requests) == 2
    assert captured_requests[0].runtime_overrides["task_batch_size"] == 1
    assert captured_requests[0].runtime_overrides["attempts_per_task"] == 4
    assert captured_requests[0].task_partition == "all"
    assert captured_requests[0].task_eval_split_fraction == 0.0
    assert captured_requests[0].task_eval_min_rows == 0
    assert captured_requests[0].dataset_loader is not None
    assert captured_requests[0].verify_submissions is True
    assert captured_requests[0].stage_name == "positive_rft"
    assert payload["stage_selection_contract"]["mode"] == "positive_rft"
    assert payload["stage_correctness_contract"] == "verifier"


def test_main_fails_when_probe_task_is_pure_infra_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-infra",
            image_name="img:infra",
            problem_statement="pinfra",
            fail_to_pass=["fi"],
            pass_to_pass=["pi"],
            raw={},
            task_family="func_basic",
        )
    ]

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    monkeypatch.setattr(
        band_module,
        "_collect_probe_rollout_rows_for_task",
        lambda **kwargs: {
            "rollout_rows": [
                {"resolved": False, "infra_invalid": True},
                {"resolved": False, "infra_invalid": True},
                {"resolved": False, "infra_invalid": True},
                {"resolved": False, "infra_invalid": True},
            ],
            "selected_rows": [],
            "rejected_rows": [
                {"rft_rejection_reason": "infra_invalid", "infra_invalid": True},
                {"rft_rejection_reason": "infra_invalid", "infra_invalid": True},
                {"rft_rejection_reason": "infra_invalid", "infra_invalid": True},
                {"rft_rejection_reason": "infra_invalid", "infra_invalid": True},
            ],
        },
    )
    monkeypatch.setattr(
        band_module,
        "build_rft_handoff_result_from_rollout_rows",
        lambda *, rollout_rows, max_tool_calls, tokenizer, handoff_overrides=None: dict(
            rollout_rows
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="only infra-invalid attempts"):
        band_module.main()


def test_main_batches_probe_tasks_when_requested(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        ),
        TaskSample(
            task_id="task-b",
            image_name="img:b",
            problem_statement="pb",
            fail_to_pass=["fb"],
            pass_to_pass=["pb"],
            raw={},
            task_family="combine_file",
        ),
        TaskSample(
            task_id="task-c",
            image_name="img:c",
            problem_statement="pc",
            fail_to_pass=["fc"],
            pass_to_pass=["pc"],
            raw={},
            task_family="func_basic",
        ),
    ]

    captured_requests = []

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        if request.start_step_index == 0:
            return {
                "rollout_rows": [
                    {"task_id": "task-a", "resolved": True},
                    {"task_id": "task-a", "resolved": True},
                    {"task_id": "task-a", "resolved": True},
                    {"task_id": "task-a", "resolved": False},
                ],
                "selected_rows": [
                    {"task_id": "task-a"},
                    {"task_id": "task-a"},
                    {"task_id": "task-a"},
                ],
                "rejected_rows": [
                    {"task_id": "task-a", "rft_rejection_reason": "unresolved"},
                ],
            }
        if request.start_step_index == 1:
            return {
                "rollout_rows": [
                    {"task_id": "task-b", "resolved": False},
                    {"task_id": "task-b", "resolved": False},
                    {"task_id": "task-b", "resolved": False},
                    {"task_id": "task-b", "resolved": False},
                ],
                "selected_rows": [],
                "rejected_rows": [
                    {"task_id": "task-b", "rft_rejection_reason": "unresolved"},
                    {"task_id": "task-b", "rft_rejection_reason": "unresolved"},
                    {"task_id": "task-b", "rft_rejection_reason": "unresolved"},
                    {"task_id": "task-b", "rft_rejection_reason": "unresolved"},
                ],
            }
        return {
            "rollout_rows": [
                {"task_id": "task-c", "resolved": True},
                {"task_id": "task-c", "resolved": False},
                {"task_id": "task-c", "resolved": False},
                {"task_id": "task-c", "resolved": False},
            ],
            "selected_rows": [{"task_id": "task-c"}],
            "rejected_rows": [
                {"task_id": "task-c", "rft_rejection_reason": "unresolved"},
                {"task_id": "task-c", "rft_rejection_reason": "unresolved"},
                {"task_id": "task-c", "rft_rejection_reason": "unresolved"},
            ],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "batched",
            "--task-batch-size",
            "2",
            "--env-pool-size",
            "3",
            "--max-in-flight-tasks",
            "5",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(captured_requests) == 3
    assert sorted(request.start_step_index for request in captured_requests) == [0, 1, 2]
    assert captured_requests[0].runtime_overrides["task_batch_size"] == 1
    assert captured_requests[0].runtime_overrides["env_pool_size"] == 1
    assert captured_requests[0].runtime_overrides["max_in_flight_tasks"] == 1
    assert captured_requests[0].task_partition == "all"
    assert captured_requests[0].task_eval_split_fraction == 0.0
    assert captured_requests[0].task_eval_min_rows == 0
    assert captured_requests[0].dataset_loader is not None
    assert [record["task_id"] for record in payload["records"]] == ["task-a", "task-b", "task-c"]
    assert payload["records"][0]["difficulty_band"] == "easy"
    assert payload["records"][1]["difficulty_band"] == "near_impossible"
    assert payload["records"][2]["difficulty_band"] == "learnable"


def test_main_resumes_from_partial_progress_cache(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        ),
        TaskSample(
            task_id="task-b",
            image_name="img:b",
            problem_statement="pb",
            fail_to_pass=["fb"],
            pass_to_pass=["pb"],
            raw={},
            task_family="combine_file",
        ),
        TaskSample(
            task_id="task-c",
            image_name="img:c",
            problem_statement="pc",
            fail_to_pass=["fc"],
            pass_to_pass=["pc"],
            raw={},
            task_family="func_basic",
        ),
    ]

    probe_calls: list[object] = []
    run_state = {"first_run": True}

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        probe_calls.append(request)
        if request.start_step_index == 0:
            return {
                "rollout_rows": [
                    {"task_id": "task-a", "resolved": True},
                ],
                "selected_rows": [
                    {"task_id": "task-a"},
                ],
                "rejected_rows": [],
            }
        if request.start_step_index == 1:
            return {
                "rollout_rows": [
                    {"task_id": "task-b", "resolved": True},
                ],
                "selected_rows": [{"task_id": "task-b"}],
                "rejected_rows": [],
            }
        if run_state["first_run"]:
            raise RuntimeError("probe interrupted after first chunk")
        return {
            "rollout_rows": [
                {"task_id": "task-c", "resolved": True},
            ],
            "selected_rows": [{"task_id": "task-c"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "resume",
            "--task-batch-size",
            "2",
        ],
    )

    with pytest.raises(RuntimeError, match="probe interrupted after first chunk"):
        band_module.main()

    output_path = tmp_path / "difficulty_bands_dummy_dataset_train_resume.json"
    partial_path = tmp_path / "difficulty_bands_dummy_dataset_train_resume.partial.json"
    assert not output_path.exists()
    partial_payload = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial_payload["probe_status"] == band_module.PROBE_STATUS_INCOMPLETE
    assert partial_payload["task_count_completed"] == 2
    partial_records_path = tmp_path / partial_payload["partial_records_path"]
    partial_records = [
        json.loads(line)
        for line in partial_records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["task_id"] for record in partial_records] == ["task-a", "task-b"]

    run_state["first_run"] = False
    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path == tmp_path / "difficulty_bands_dummy_dataset_train_resume.json"
    assert not partial_path.exists()
    assert not partial_records_path.exists()
    assert payload["probe_status"] == band_module.PROBE_STATUS_COMPLETE
    assert [record["task_id"] for record in payload["records"]] == ["task-a", "task-b", "task-c"]
    assert len(probe_calls) == 4
    assert probe_calls[-1].start_step_index == 2
    assert probe_calls[-1].runtime_overrides["task_batch_size"] == 1


def test_load_resumable_records_resolves_relative_partial_records_path_against_cache_path(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "difficulty_bands.partial.json"
    partial_records_path = tmp_path / "difficulty_bands.partial.records.jsonl"
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
    records = band_module._load_resumable_records(
        payload={
            "partial_records_path": partial_records_path.name,
            "task_count_expected": 1,
        },
        tasks=[
            TaskSample(
                task_id="task-a",
                image_name="img:a",
                problem_statement="pa",
                fail_to_pass=["fa"],
                pass_to_pass=["pa"],
                raw={},
                task_family="func_basic",
            )
        ],
        cache_path=cache_path,
    )

    assert records["task-a"]["difficulty_band"] == "learnable"


def test_main_cancels_queued_probe_futures_after_task_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id=f"task-{index}",
            image_name=f"img:{index}",
            problem_statement=f"p{index}",
            fail_to_pass=[f"f{index}"],
            pass_to_pass=[f"p{index}"],
            raw={},
            task_family="func_basic",
        )
        for index in range(3)
    ]
    observed_steps: list[int] = []

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        observed_steps.append(request.start_step_index)
        if request.start_step_index == 0:
            raise RuntimeError("probe boom")
        return {
            "rollout_rows": [{"resolved": True}],
            "selected_rows": [{"task_id": f"task-{request.start_step_index}"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--task-batch-size",
            "3",
            "--env-pool-size",
            "1",
            "--max-in-flight-tasks",
            "1",
        ],
    )

    with pytest.raises(RuntimeError, match="probe boom"):
        band_module.main()

    assert observed_steps == [0]


def test_build_dataset_loader_for_tasks_preserves_task_ids_without_source_id_fields() -> None:
    config = _config()
    loader = band_module._build_dataset_loader_for_tasks(
        [
            TaskSample(
                task_id="task-original",
                image_name="img:a",
                problem_statement="prompt-a",
                fail_to_pass=["fa"],
                pass_to_pass=["pa"],
                raw={
                    "image_name": "img:a",
                    "problem_statement": "prompt-a",
                    "FAIL_TO_PASS": ["fa"],
                    "PASS_TO_PASS": ["pa"],
                },
                task_family="func_basic",
            )
        ]
    )

    batch = load_task_batch(
        step_index=0,
        batch_size=1,
        config=config,
        dataset_loader=loader,
    )

    assert [task.task_id for task in batch] == ["task-original"]
    assert batch[0].raw["task_id"] == "task-original"


def test_main_uses_runtime_eval_split_defaults_for_partitioned_probe(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-eval",
            image_name="img:eval",
            problem_statement="pe",
            fail_to_pass=["fe"],
            pass_to_pass=["pe"],
            raw={},
            task_family="func_basic",
        )
    ]
    captured_load_kwargs = []
    captured_requests = []

    def _fake_load_task_samples(**kwargs):
        captured_load_kwargs.append(kwargs)
        return list(tasks)

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        return {
            "rollout_rows": [{"resolved": True}],
            "selected_rows": [{"task_id": "task-eval"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.25, "eval_min_rows": 2}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", _fake_load_task_samples)
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--task-partition",
            "eval",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path.name == (
        "difficulty_bands_dummy_dataset_train_positive_rft_probe_eval_frac_0.25_mineval_2.json"
    )
    assert captured_load_kwargs[0]["task_partition"] == "eval"
    assert captured_load_kwargs[0]["eval_split_fraction"] == 0.25
    assert captured_load_kwargs[0]["min_eval_rows"] == 2
    assert captured_requests[0].task_partition == "all"
    assert captured_requests[0].task_eval_split_fraction == 0.0
    assert captured_requests[0].task_eval_min_rows == 0
    assert captured_requests[0].dataset_loader is not None
    assert payload["task_partition"] == "eval"
    assert payload["eval_split_fraction"] == 0.25
    assert payload["min_eval_rows"] == 2


def test_main_disables_rollout_probe_dependency_while_building_probe(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]
    captured_load_configs = []
    captured_requests = []

    def _fake_load_task_samples(**kwargs):
        captured_load_configs.append(kwargs["config"])
        return list(tasks)

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        return {
            "rollout_rows": [{"resolved": True}],
            "selected_rows": [{"task_id": "task-a"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name, data_overrides=None: _settings(
            difficulty_banding=OnPolicyDifficultyBandConfig(
                strategy=(
                    "none"
                    if data_overrides
                    and data_overrides.get("difficulty_banding", {}).get("strategy") == "none"
                    else "rollout_probe"
                ),
                rollout_probe_cache_path="data/on_policy_difficulty_band_cache/missing.json",
                rollout_probe_required=not bool(data_overrides),
            )
        ),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", _fake_load_task_samples)
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
        ],
    )

    exit_code = band_module.main()
    _ = Path(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert captured_load_configs[0].difficulty_banding.strategy == "none"
    assert captured_requests[0].data_overrides == {
        "difficulty_banding": {
            "strategy": "none",
            "default_band": "unbanded",
            "family_band_exact": {},
            "family_band_prefix": {},
            "rollout_probe_cache_path": "",
            "rollout_probe_required": False,
        }
    }


def test_main_rebuilds_cache_when_existing_metadata_is_incompatible(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]
    captured_requests = []
    cache_path = tmp_path / "difficulty_bands_dummy_dataset_train_smoke.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_config_name": "on_policy_swe_smith",
                "dataset_id": "dummy/dataset",
                "dataset_split": "train",
                "patch_is_bug_introducing": True,
                "verifier_kind": "pytest",
                "probe_label": "smoke",
                "initial_model": "/tmp/old-model",
                "turn_generator_mode": "default",
                "stage_name": "positive_rft",
                "task_partition": "all",
                "attempts_per_task": 4,
                "start_task_index": 0,
                "task_limit": None,
                "eval_split_fraction": 0.0,
                "min_eval_rows": 0,
                **_stage_metadata(),
                "task_count": 1,
                "records": [
                    {
                        "task_id": "task-a",
                        "difficulty_band": "near_impossible",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        return {
            "rollout_rows": [{"resolved": True}],
            "selected_rows": [{"task_id": "task-a"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/new-model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "smoke",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(captured_requests) == 1
    assert payload["initial_model"] == "/tmp/new-model"


def test_main_reuses_freshly_materialized_cache(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]
    captured_requests = []

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        return {
            "rollout_rows": [{"resolved": True}],
            "selected_rows": [{"task_id": "task-a"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "smoke",
        ],
    )

    first_exit_code = band_module.main()
    first_output_path = Path(capsys.readouterr().out.strip())

    monkeypatch.setattr(
        band_module,
        "_collect_probe_rollout_rows_for_task",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("probe should not rerun")),
    )

    second_exit_code = band_module.main()
    second_output_path = Path(capsys.readouterr().out.strip())

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert len(captured_requests) == 1
    assert second_output_path == first_output_path


def test_main_reuses_cache_when_task_pool_fingerprint_matches(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]
    cache_path = tmp_path / "difficulty_bands_dummy_dataset_train_smoke.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_config_name": "on_policy_swe_smith",
                "dataset_id": "dummy/dataset",
                "dataset_split": "train",
                "patch_is_bug_introducing": True,
                "verifier_kind": "pytest",
                "probe_label": "smoke",
                "initial_model": "/tmp/model",
                "turn_generator_mode": "default",
                "stage_name": "positive_rft",
                "task_partition": "all",
                "attempts_per_task": 4,
                "start_task_index": 0,
                "task_limit": None,
                "eval_split_fraction": 0.0,
                "min_eval_rows": 0,
                **_stage_metadata(),
                "task_pool_size": 1,
                "task_pool_fingerprint": band_module._build_task_pool_fingerprint(tasks),
                "task_count": 1,
                "records": [
                    {
                        "task_id": "task-a",
                        "difficulty_band": "easy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(
        band_module,
        "_collect_probe_rollout_rows_for_task",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "smoke",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path == cache_path
    assert payload["records"][0]["difficulty_band"] == "easy"


def test_main_rebuilds_cache_when_matching_metadata_cache_omits_selected_task(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        ),
        TaskSample(
            task_id="task-b",
            image_name="img:b",
            problem_statement="pb",
            fail_to_pass=["fb"],
            pass_to_pass=["pb"],
            raw={},
            task_family="combine_file",
        ),
    ]
    captured_requests = []
    cache_path = tmp_path / "difficulty_bands_dummy_dataset_train_smoke.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_config_name": "on_policy_swe_smith",
                "dataset_id": "dummy/dataset",
                "dataset_split": "train",
                "patch_is_bug_introducing": True,
                "verifier_kind": "pytest",
                "probe_label": "smoke",
                "initial_model": "/tmp/model",
                "turn_generator_mode": "default",
                "stage_name": "positive_rft",
                "task_partition": "all",
                "attempts_per_task": 4,
                "start_task_index": 0,
                "task_limit": None,
                "eval_split_fraction": 0.0,
                "min_eval_rows": 0,
                **_stage_metadata(),
                "task_pool_size": 2,
                "task_pool_fingerprint": band_module._build_task_pool_fingerprint(tasks),
                "task_count": 1,
                "records": [
                    {
                        "task_id": "task-a",
                        "difficulty_band": "easy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        task_id = "task-a" if request.start_step_index == 0 else "task-b"
        return {
            "rollout_rows": [{"resolved": True, "task_id": task_id}],
            "selected_rows": [{"task_id": task_id}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "smoke",
            "--task-batch-size",
            "2",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(captured_requests) == 2
    assert output_path == cache_path
    assert sorted(record["task_id"] for record in payload["records"]) == ["task-a", "task-b"]


def test_main_rebuilds_cache_when_task_pool_fingerprint_changes(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    tasks = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="pa",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]
    captured_requests = []
    cache_path = tmp_path / "difficulty_bands_dummy_dataset_train_smoke.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_config_name": "on_policy_swe_smith",
                "dataset_id": "dummy/dataset",
                "dataset_split": "train",
                "patch_is_bug_introducing": True,
                "verifier_kind": "pytest",
                "probe_label": "smoke",
                "initial_model": "/tmp/model",
                "turn_generator_mode": "default",
                "stage_name": "positive_rft",
                "task_partition": "all",
                "attempts_per_task": 4,
                "start_task_index": 0,
                "task_limit": None,
                "eval_split_fraction": 0.0,
                "min_eval_rows": 0,
                **_stage_metadata(),
                "task_pool_size": 1,
                "task_pool_fingerprint": "stale-task-pool",
                "task_count": 1,
                "records": [
                    {
                        "task_id": "task-a",
                        "difficulty_band": "near_impossible",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        return {
            "rollout_rows": [{"resolved": True}],
            "selected_rows": [{"task_id": "task-a"}],
            "rejected_rows": [],
        }

    monkeypatch.setattr(
        band_module,
        "resolve_on_policy_settings",
        lambda data_config_name: _settings(),
    )
    monkeypatch.setattr(
        band_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.1, "eval_min_rows": 1}},
    )
    monkeypatch.setattr(band_module, "load_task_samples", lambda **kwargs: list(tasks))
    monkeypatch.setattr(band_module, "_load_tokenizer", lambda model_path: object())
    _stub_probe_collection(monkeypatch, _fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_onpolicy_difficulty_bands.py",
            "--initial-model",
            "/tmp/model",
            "--cache-dir",
            str(tmp_path),
            "--probe-label",
            "smoke",
        ],
    )

    exit_code = band_module.main()
    output_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(captured_requests) == 1
    assert payload["task_pool_fingerprint"] == band_module._build_task_pool_fingerprint(tasks)
    assert payload["records"][0]["difficulty_band"] == "easy"


def test_task_pool_fingerprint_changes_when_task_content_changes() -> None:
    original = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="original prompt",
            fail_to_pass=["fa"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]
    updated = [
        TaskSample(
            task_id="task-a",
            image_name="img:a",
            problem_statement="updated prompt",
            fail_to_pass=["fa", "fb"],
            pass_to_pass=["pa"],
            raw={},
            task_family="func_basic",
        )
    ]

    assert band_module._build_task_pool_fingerprint(original) != band_module._build_task_pool_fingerprint(updated)
