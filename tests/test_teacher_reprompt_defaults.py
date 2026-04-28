from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from verl_integration.reprompt_adapter import DEFAULT_MAX_REPROMPT_LEN


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_teacher_reprompt_default_max_reprompt_len_is_16384() -> None:
    assert DEFAULT_MAX_REPROMPT_LEN == 16384


def test_sdpo_config_updates_only_reprompt_budget_paths() -> None:
    config_text = (REPO_ROOT / "configs" / "verl" / "sdpo_swe.yaml").read_text(encoding="utf-8")

    assert "max_reprompt_len: 16384" in config_text
    assert "include_teacher_memory_blocks: true" in config_text
    assert "max_response_length: 12288" in config_text


def test_teacher_reprompt_pilot_slurm_default_budget_is_16384() -> None:
    script_text = (
        REPO_ROOT / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"
    ).read_text(encoding="utf-8")

    assert 'PILOT_MAX_REPROMPT_LEN:-16384' in script_text
    assert 'PILOT_MAX_REPROMPT_LEN:-12288' not in script_text
    assert 'PILOT_PARTIAL_FLUSH_EVERY_ROWS:-1' in script_text
    assert 'PILOT_PARTIAL_SUMMARY_EVERY_ROWS:-1' in script_text


def test_teacher_reprompt_pilot_slurm_uses_configured_vllm_port() -> None:
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    env["SMALL_SWE_VLLM_BASE_URL"] = "http://127.0.0.1:18123/v1"
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--host 127.0.0.1 --port 18123" in result.stdout


def test_teacher_reprompt_pilot_slurm_parses_typed_gpu_gres_count() -> None:
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    env["SLURM_GPUS_ON_NODE"] = "gpu:a100:4"
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--tensor-parallel-size 4" in result.stdout
    assert "--tensor-parallel-size 100" not in result.stdout
