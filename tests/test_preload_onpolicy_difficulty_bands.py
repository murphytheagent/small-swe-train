from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from config import OnPolicyDataConfig, OnPolicyDatasetColumns
from env import preload_onpolicy_difficulty_bands as band_module
from env.task_dataset import TaskSample


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
) -> SimpleNamespace:
    return SimpleNamespace(
        data=_config(),
        runtime=SimpleNamespace(
            eval_split_fraction=eval_split_fraction,
            eval_min_rows=eval_min_rows,
        ),
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
    monkeypatch.setattr(band_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)
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
    assert captured_requests[0].verify_submissions is True
    assert captured_requests[0].stage_name == "positive_rft"


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
    monkeypatch.setattr(band_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)
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
    assert captured_load_kwargs[0]["task_partition"] == "eval"
    assert captured_load_kwargs[0]["eval_split_fraction"] == 0.25
    assert captured_load_kwargs[0]["min_eval_rows"] == 2
    assert captured_requests[0].task_eval_split_fraction == 0.25
    assert captured_requests[0].task_eval_min_rows == 2
    assert payload["task_partition"] == "eval"
    assert payload["eval_split_fraction"] == 0.25
    assert payload["min_eval_rows"] == 2


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
    monkeypatch.setattr(band_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)
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
        "collect_onpolicy_rft_runtime_batch",
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
    monkeypatch.setattr(band_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)
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
