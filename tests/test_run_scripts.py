from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script_path = _repo_root() / "scripts" / script_name
    return subprocess.run(
        ["bash", str(script_path), "--dry-run", *args],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )


def test_run_rft_script_dry_run_prints_verl_command() -> None:
    result = _run_script("run_rft.sh", "trainer.total_training_steps=1")
    assert "python -m verl.trainer.main_ppo" in result.stdout
    assert "--config-name rft_swe" in result.stdout
    assert "trainer.total_training_steps=1" in result.stdout


def test_run_sdft_script_dry_run_includes_loss_mode_override() -> None:
    result = _run_script("run_sdft.sh")
    assert "--config-name sdpo_swe" in result.stdout
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=sdft" in result.stdout


def test_run_sdpo_script_dry_run_prints_sdpo_config() -> None:
    result = _run_script("run_sdpo.sh", "data.train_batch_size=4")
    assert "--config-name sdpo_swe" in result.stdout
    assert "data.train_batch_size=4" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_sets_onpolicy_overrides() -> None:
    result = _run_script("run_rft_onpolicy_rollout_proof.sh")
    assert "--config-name rft_swe" in result.stdout
    assert "on_policy.enabled=true" in result.stdout
    assert "on_policy.rollout_only=true" in result.stdout
