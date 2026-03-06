from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from config import OnPolicyDataConfig, OnPolicyDatasetColumns
import env.preflight_onpolicy_dataset as preflight_module
from env.shell_helpers import build_python_interpreter_resolver_shell


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


def test_main_prints_resolved_path_only(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        preflight_module,
        "resolve_on_policy_settings",
        lambda data_config_name: SimpleNamespace(data=_config()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_onpolicy_dataset.py",
            "--cache-dir",
            str(tmp_path),
            "--print-path-only",
        ],
    )

    exit_code = preflight_module.main()
    output = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert output.endswith(".json")
    assert "bad_tasks_dummy_dataset_train_" in output


def test_probe_command_falls_back_from_python3_to_python() -> None:
    command = preflight_module._build_probe_command()

    assert build_python_interpreter_resolver_shell(var_name="pybin") in command
    assert '"${pybin}" - <<' in command


def test_resolve_cache_path_scopes_noncanonical_probe_settings(tmp_path: Path) -> None:
    settings = SimpleNamespace(data=_config())

    canonical = preflight_module._resolve_cache_path(
        settings=settings,
        cache_dir=str(tmp_path),
        max_images=None,
        probe_timeout_sec=preflight_module._DEFAULT_PROBE_TIMEOUT_SEC,
    )
    partial = preflight_module._resolve_cache_path(
        settings=settings,
        cache_dir=str(tmp_path),
        max_images=5,
        probe_timeout_sec=preflight_module._DEFAULT_PROBE_TIMEOUT_SEC,
    )
    slow = preflight_module._resolve_cache_path(
        settings=settings,
        cache_dir=str(tmp_path),
        max_images=None,
        probe_timeout_sec=240,
    )

    assert canonical != partial
    assert canonical != slow
    assert "__max_images_5" in partial.name
    assert "__probe_timeout_240s" in slow.name


def test_main_writes_bad_task_cache(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        preflight_module,
        "resolve_on_policy_settings",
        lambda data_config_name: SimpleNamespace(data=_config()),
    )
    monkeypatch.setattr(
        preflight_module,
        "scan_dataset_for_bad_verifier_tasks",
        lambda **kwargs: {
            "schema_version": preflight_module.ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION,
            "dataset_id": kwargs["config"].dataset_id,
            "dataset_split": kwargs["config"].dataset_split,
            "generated_at_utc": "2026-03-06T00:00:00+00:00",
            "probe_timeout_sec": kwargs["probe_timeout_sec"],
            "scanned_task_count": 3,
            "probed_image_count": 2,
            "bad_task_ids": ["task-1", "task-2"],
            "bad_image_names": ["img:bad"],
            "records": [
                {
                    "task_id": "task-1",
                    "image_name": "img:bad",
                    "status": "bad",
                    "reason": "pytest_unavailable",
                }
            ],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_onpolicy_dataset.py",
            "--cache-dir",
            str(tmp_path),
            "--probe-timeout-sec",
            "77",
        ],
    )

    exit_code = preflight_module.main()
    output_lines = capsys.readouterr().out.strip().splitlines()
    cache_path = Path(output_lines[0])
    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert cache_path.is_file()
    assert payload["bad_task_ids"] == ["task-1", "task-2"]
    assert payload["bad_image_names"] == ["img:bad"]
    assert output_lines[1:] == ["bad_images=1", "bad_tasks=2"]
