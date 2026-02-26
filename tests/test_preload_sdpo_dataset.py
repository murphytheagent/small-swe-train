from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from config import OnPolicyDataConfig, OnPolicyDatasetColumns
from env import preload_sdpo_dataset as preload_module


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


def test_resolve_eval_split_defaults_reads_runtime_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        preload_module,
        "rft_runtime_defaults",
        lambda: {"loop": {"eval_split_fraction": 0.25, "eval_min_rows": 3}},
    )

    fraction, min_rows = preload_module._resolve_eval_split_defaults()

    assert fraction == 0.25
    assert min_rows == 3


def test_main_uses_config_defaults_for_split_args_when_unset(
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def _stub_resolve_split_paths(
        *,
        config: OnPolicyDataConfig,
        cache_dir: str | Path,
        eval_split_fraction: float,
        min_eval_rows: int,
    ) -> tuple[Path, Path]:
        captured["dataset_id"] = config.dataset_id
        captured["cache_dir"] = str(cache_dir)
        captured["eval_split_fraction"] = eval_split_fraction
        captured["min_eval_rows"] = min_eval_rows
        return Path("/tmp/train.parquet"), Path("/tmp/val.parquet")

    monkeypatch.setattr(preload_module, "_resolve_eval_split_defaults", lambda: (0.33, 7))
    monkeypatch.setattr(preload_module, "resolve_sdpo_task_split_cache_paths", _stub_resolve_split_paths)
    monkeypatch.setattr(
        preload_module,
        "resolve_on_policy_settings",
        lambda data_config_name: SimpleNamespace(data=_config()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preload_sdpo_dataset.py",
            "--cache-dir",
            "/tmp/sdpo-cache",
            "--emit-split",
            "--emit-hydra-overrides",
            "--print-path-only",
        ],
    )

    exit_code = preload_module.main()
    output_lines = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0
    assert captured["dataset_id"] == "dummy/dataset"
    assert captured["cache_dir"] == "/tmp/sdpo-cache"
    assert captured["eval_split_fraction"] == 0.33
    assert captured["min_eval_rows"] == 7
    assert output_lines == [
        "data.train_files=/tmp/train.parquet",
        "data.val_files=/tmp/val.parquet",
    ]
