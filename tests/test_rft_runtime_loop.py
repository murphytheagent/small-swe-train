from __future__ import annotations

from pathlib import Path

import pytest

from trainer.rft_runtime_loop import (
    build_trainer_step_command,
    build_vllm_server_command,
    prune_old_global_step_checkpoints,
    prune_old_step_checkpoints,
    prune_old_step_payloads,
    resolve_latest_hf_checkpoint,
)


def test_build_trainer_step_command_includes_required_dataset_and_checkpoint_overrides(
    tmp_path: Path,
) -> None:
    command = build_trainer_step_command(
        nnodes=1,
        nproc_per_node=8,
        trainer_module="verl.trainer.fsdp_sft_trainer",
        config_name="rft_swe",
        config_dir=tmp_path / "configs",
        model_path="Qwen/Qwen3-4B-Instruct-2507",
        train_parquet_path=tmp_path / "accepted.parquet",
        val_parquet_path=tmp_path / "accepted.parquet",
        trainer_output_dir=tmp_path / "checkpoints",
        train_batch_size=32,
        sft_num_epoch_per_batch=1,
        trainer_overrides=("trainer.total_training_steps=1",),
    )

    command_text = " ".join(command)
    assert "torchrun" in command_text
    assert "-m verl.trainer.fsdp_sft_trainer" in command_text
    assert "trainer.total_training_steps=1" in command_text
    assert "trainer.checkpoint.save_contents=[model,hf_model,extra]" in command_text
    assert "data.multiturn.enable=true" in command_text
    assert "data.custom_cls.path=null" in command_text
    assert f"data.train_files={tmp_path / 'accepted.parquet'}" in command_text
    assert "model.partial_pretrain=Qwen/Qwen3-4B-Instruct-2507" in command_text


def test_build_vllm_server_command_uses_host_and_port_from_base_url() -> None:
    command = build_vllm_server_command(
        python_bin="python3",
        launch_module="vllm.entrypoints.openai.api_server",
        base_url="http://127.0.0.1:8000/v1",
        model_path="/tmp/model",
        served_model_name="Qwen/Qwen3-4B-Instruct-2507",
        extra_args=("--dtype", "bfloat16"),
    )

    assert command[:7] == [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert "--model" in command
    assert "/tmp/model" in command
    assert "--served-model-name" in command
    assert "Qwen/Qwen3-4B-Instruct-2507" in command
    assert command[-2:] == ["--dtype", "bfloat16"]


def test_resolve_latest_hf_checkpoint_returns_highest_global_step(tmp_path: Path) -> None:
    older = tmp_path / "global_step_8" / "huggingface"
    newer = tmp_path / "global_step_12" / "huggingface"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    resolved = resolve_latest_hf_checkpoint(tmp_path)

    assert resolved == newer


def test_resolve_latest_hf_checkpoint_requires_huggingface_export(tmp_path: Path) -> None:
    (tmp_path / "global_step_3").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="huggingface"):
        resolve_latest_hf_checkpoint(tmp_path)


def test_prune_old_step_checkpoints_keeps_latest_roots(tmp_path: Path) -> None:
    output_dir = tmp_path / "rft_runtime"
    for step in range(4):
        checkpoint_root = output_dir / f"rft_step_{step:05d}" / "trainer_checkpoints"
        (checkpoint_root / "global_step_1" / "huggingface").mkdir(parents=True)

    pruned = prune_old_step_checkpoints(output_dir=output_dir, keep_last=2)

    assert [path.parent.name for path in pruned] == ["rft_step_00000", "rft_step_00001"]
    assert not (output_dir / "rft_step_00000" / "trainer_checkpoints").exists()
    assert not (output_dir / "rft_step_00001" / "trainer_checkpoints").exists()
    assert (output_dir / "rft_step_00002" / "trainer_checkpoints").is_dir()
    assert (output_dir / "rft_step_00003" / "trainer_checkpoints").is_dir()


def test_prune_old_step_checkpoints_requires_positive_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_old_step_checkpoints(output_dir=tmp_path, keep_last=0)


def test_prune_old_global_step_checkpoints_keeps_latest_steps(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "trainer_checkpoints"
    for step in (1, 3, 5):
        (checkpoint_root / f"global_step_{step}" / "huggingface").mkdir(parents=True)

    pruned = prune_old_global_step_checkpoints(checkpoint_root=checkpoint_root, keep_last=1)

    assert [path.name for path in pruned] == ["global_step_1", "global_step_3"]
    assert not (checkpoint_root / "global_step_1").exists()
    assert not (checkpoint_root / "global_step_3").exists()
    assert (checkpoint_root / "global_step_5").is_dir()


def test_prune_old_global_step_checkpoints_requires_positive_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_old_global_step_checkpoints(checkpoint_root=tmp_path, keep_last=0)


def test_prune_old_step_payloads_keeps_latest_step_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "rft_runtime"
    for step in range(4):
        step_dir = output_dir / f"rft_step_{step:05d}"
        (step_dir / "collector_artifacts").mkdir(parents=True)
        (step_dir / "collector_artifacts" / "rollout_rows.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (step_dir / "accepted_trajectories.parquet").write_text("stub", encoding="utf-8")

    pruned = prune_old_step_payloads(output_dir=output_dir, keep_last=2)

    pruned_names = {str(path.relative_to(output_dir)) for path in pruned}
    assert "rft_step_00000/collector_artifacts" in pruned_names
    assert "rft_step_00000/accepted_trajectories.parquet" in pruned_names
    assert "rft_step_00001/collector_artifacts" in pruned_names
    assert "rft_step_00001/accepted_trajectories.parquet" in pruned_names

    assert not (output_dir / "rft_step_00000" / "collector_artifacts").exists()
    assert not (output_dir / "rft_step_00000" / "accepted_trajectories.parquet").exists()
    assert not (output_dir / "rft_step_00001" / "collector_artifacts").exists()
    assert not (output_dir / "rft_step_00001" / "accepted_trajectories.parquet").exists()
    assert (output_dir / "rft_step_00002" / "collector_artifacts").is_dir()
    assert (output_dir / "rft_step_00002" / "accepted_trajectories.parquet").is_file()
    assert (output_dir / "rft_step_00003" / "collector_artifacts").is_dir()
    assert (output_dir / "rft_step_00003" / "accepted_trajectories.parquet").is_file()


def test_prune_old_step_payloads_requires_positive_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_old_step_payloads(output_dir=tmp_path, keep_last=0)
