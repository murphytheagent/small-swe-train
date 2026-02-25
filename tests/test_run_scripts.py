from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping

import config

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


def _write_python_defaults_stub(tmp_path: Path, defaults_line: str) -> Path:
    stub_path = tmp_path / "python-defaults-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        "  payload=\"$(cat)\"\n"
        "  if [[ \"${payload}\" == *\"torch.cuda.device_count\"* ]]; then\n"
        "    printf '%s\\n' \"${STUB_GPU_COUNT:-0}\"\n"
        "  else\n"
        f"    printf '%s\\n' '{defaults_line}'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exec python3 \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def test_run_rft_script_dry_run_prints_verl_command() -> None:
    result = _run_script("run_rft.sh", "trainer.total_training_steps=1")
    assert "-m torch.distributed.run" in result.stdout
    assert "-m verl_integration.fsdp_sft_trainer_entry" in result.stdout
    assert "--config-name rft_swe" in result.stdout
    assert "trainer.total_training_steps=1" in result.stdout


def test_run_rft_script_dry_run_defaults_vllm_tp_dp_for_eight_gpus() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={"NPROC_PER_NODE": "8"},
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--data-parallel-size 4" in result.stdout


def test_run_rft_script_dry_run_honors_centralized_default_dp_for_divisible_topology(
    tmp_path: Path,
) -> None:
    fake_python = _write_python_defaults_stub(
        tmp_path,
        (
            "100 8 64 32 1 1 512 0.1 1 2 2 "
            "http://127.0.0.1:8000/v1 "
            "Qwen/Qwen3-4B-Instruct-2507 90 1024 0.0 1.0 8 12288 "
            "lora bf16 16 32 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    )
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "PYTHON_BIN": str(fake_python),
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--data-parallel-size 2" in result.stdout


def test_run_rft_script_dry_run_defaults_nproc_to_detected_gpu_count(
    tmp_path: Path,
) -> None:
    fake_python = _write_python_defaults_stub(
        tmp_path,
        (
            "100 8 64 32 1 1 512 0.1 1 2 4 "
            "http://127.0.0.1:8000/v1 "
            "Qwen/Qwen3-4B-Instruct-2507 90 1024 0.0 1.0 8 12288 "
            "lora bf16 16 32 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    )
    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "PYTHON_BIN": str(fake_python),
            "STUB_GPU_COUNT": "8",
            "NPROC_PER_NODE": "",
        },
    )
    assert "--nproc_per_node 8" in result.stdout


def test_run_rft_script_dry_run_uses_centralized_collector_in_flight_default() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
    )
    assert "collector_max_in_flight_tasks=32" in result.stdout


def test_run_rft_script_dry_run_allows_explicit_tp_override() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_VLLM_TP_SIZE": "8",
        },
    )
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--data-parallel-size" not in result.stdout


def test_run_rft_script_dry_run_nondivisible_tp_override_falls_back_to_dp_one() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_VLLM_TP_SIZE": "3",
        },
    )
    assert "--tensor-parallel-size 3" in result.stdout
    assert "--data-parallel-size" not in result.stdout


def test_run_rft_script_dry_run_propagates_collector_in_flight_override() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS": "6",
        },
    )
    assert "collector_max_in_flight_tasks=6" in result.stdout


def test_run_rft_script_dry_run_propagates_collector_max_turns_override() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT": "16",
        },
    )
    assert "collector_max_turns_per_attempt=16" in result.stdout


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


def test_run_rft_script_dry_run_uses_centralized_sequence_length_overrides() -> None:
    expected_max_length = config.resolve_rft_handoff_settings().max_sequence_length
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
    )
    assert f"max_model_len={expected_max_length}" in result.stdout


def test_run_rft_script_dry_run_direct_mode_uses_centralized_runtime_overrides() -> None:
    on_policy_defaults = config.on_policy_runtime_defaults()
    expected_max_turns = on_policy_defaults["max_turns_per_attempt"]
    expected_max_length = config.resolve_rft_handoff_settings().max_sequence_length
    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_RUNTIME_MODE": "direct",
            "RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT": "",
        },
    )
    assert (
        "actor_rollout_ref.model.lora.target_modules="
        "\\[q_proj\\,k_proj\\,v_proj\\,o_proj\\,gate_proj\\,up_proj\\,down_proj\\]"
    ) in result.stdout
    assert f"+data.on_policy.runtime_overrides.max_turns_per_attempt={expected_max_turns}" in result.stdout
    assert f"max_model_len={expected_max_length}" in result.stdout


def test_run_sdft_script_dry_run_includes_loss_mode_override() -> None:
    result = _run_script("run_sdft.sh")
    assert "--config-name sdpo_swe" in result.stdout
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=sdft" in result.stdout


def test_run_sdpo_script_dry_run_prints_sdpo_config() -> None:
    result = _run_script("run_sdpo.sh", "data.train_batch_size=4")
    assert "-m verl_integration.main_ppo_entry" in result.stdout
    assert "--config-name sdpo_swe" in result.stdout
    assert "data.train_batch_size=4" in result.stdout


def test_run_sdpo_script_dry_run_allows_entrypoint_override() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"SDPO_TRAINER_MODULE": "verl.trainer.main_ppo"},
    )
    assert "-m verl.trainer.main_ppo" in result.stdout


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


def test_run_flash_attn_rebuild_script_dry_run_uses_safe_defaults() -> None:
    result = _run_script("run_flash_attn_rebuild.sh")
    assert "--partition gpu" in result.stdout
    assert "--gres" not in result.stdout
    assert "--cpus-per-task 8" in result.stdout
    assert "--mem 128G" in result.stdout
    assert "rebuild-flash-attn" in result.stdout
    assert "CORES=8" in result.stdout
    assert "FLASH_ATTN_CUDA_ARCHS=120" in result.stdout
    assert "pipefail" not in result.stdout


def test_run_flash_attn_rebuild_script_dry_run_has_no_log_dir_side_effect(tmp_path: Path) -> None:
    log_dir = tmp_path / "flash-attn-rebuild-logs"
    result = _run_script(
        "run_flash_attn_rebuild.sh",
        env_overrides={"FLASH_ATTN_BUILD_LOG_DIR": str(log_dir)},
    )
    assert str(log_dir / "slurm-%j.out") in result.stdout
    assert str(log_dir / "slurm-%j.err") in result.stdout
    assert not log_dir.exists()


def test_run_flash_attn_rebuild_script_honors_env_overrides() -> None:
    result = _run_script(
        "run_flash_attn_rebuild.sh",
        env_overrides={
            "FLASH_ATTN_BUILD_PARTITION": "gpu",
            "FLASH_ATTN_BUILD_GRES": "gpu:2",
            "FLASH_ATTN_BUILD_CPUS_PER_TASK": "12",
            "FLASH_ATTN_BUILD_MEM": "96G",
            "FLASH_ATTN_BUILD_TIME": "02:30:00",
            "FLASH_ATTN_BUILD_MAX_JOBS": "4",
            "FLASH_ATTN_CUDA_ARCHS": "90",
        },
    )
    assert "--partition gpu" in result.stdout
    assert "--gres gpu:2" in result.stdout
    assert "--cpus-per-task 12" in result.stdout
    assert "--mem 96G" in result.stdout
    assert "--time 02:30:00" in result.stdout
    assert "rebuild-flash-attn" in result.stdout
    assert "CORES=4" in result.stdout
    assert "FLASH_ATTN_CUDA_ARCHS=90" in result.stdout
