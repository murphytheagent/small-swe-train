from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_script(
    script_name: str,
    *args: str,
    env_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script_path = _repo_root() / "scripts" / script_name
    env = os.environ.copy()
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(script_path), "--dry-run", *args],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def test_run_rft_script_dry_run_prints_verl_command() -> None:
    result = _run_script("run_rft.sh", "trainer.total_training_steps=1")
    assert "-m torch.distributed.run" in result.stdout
    assert "-m verl_integration.fsdp_sft_trainer_entry" in result.stdout
    assert "--config-name rft_swe" in result.stdout
    assert "trainer.total_training_steps=1" in result.stdout


def test_run_rft_script_dry_run_defaults_vllm_tensor_parallel_to_nproc() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={"NPROC_PER_NODE": "8"},
    )
    assert "--tensor-parallel-size 8" in result.stdout


def test_run_rft_script_dry_run_respects_explicit_vllm_extra_args() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_VLLM_EXTRA_ARGS": "--tensor-parallel-size 2 --max-num-seqs 16",
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--max-num-seqs 16" in result.stdout


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
    assert "-m verl_integration.fsdp_sft_trainer_entry" in result.stdout
    assert "trainer.logger=\\[console\\,wandb\\]" in result.stdout
    assert "trainer.default_local_dir=" in result.stdout
    assert "data.on_policy.enabled=true" in result.stdout
    assert "data.on_policy.turn_generator_mode=proof_tool_chain" in result.stdout
    assert "data.on_policy.total_steps=1" in result.stdout
    assert "+data.on_policy.runtime_overrides.task_batch_size=" in result.stdout
    assert "model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_honors_model_override() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={"ON_POLICY_PROOF_MODEL_PATH": "Qwen/Qwen2.5-1.5B-Instruct"},
    )
    assert "model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_propagates_steps() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={"ON_POLICY_PROOF_STEPS": "3"},
    )
    assert "trainer.total_training_steps=3" in result.stdout
    assert "data.on_policy.total_steps=3" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_defaults_train_batch_to_world_size() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={"ON_POLICY_PROOF_NPROC_PER_NODE": "8"},
    )
    assert "trainer.n_gpus_per_node=8" in result.stdout
    assert "data.train_batch_size=8" in result.stdout
    assert "data.micro_batch_size_per_gpu=1" in result.stdout
    assert "+data.on_policy.runtime_overrides.task_batch_size=8" in result.stdout
    assert "+data.on_policy.runtime_overrides.env_pool_size=8" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_honors_explicit_batch_overrides() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={
            "ON_POLICY_PROOF_NPROC_PER_NODE": "8",
            "ON_POLICY_TRAIN_BATCH_SIZE": "16",
            "ON_POLICY_MICRO_BATCH_SIZE_PER_GPU": "2",
        },
    )
    assert "data.train_batch_size=16" in result.stdout
    assert "data.micro_batch_size_per_gpu=2" in result.stdout
