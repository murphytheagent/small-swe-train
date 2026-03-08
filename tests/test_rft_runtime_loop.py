from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from urllib.error import HTTPError

import pytest

import config
import trainer.rft_runtime_loop as rft_runtime_loop
from trainer.rft_runtime_loop import (
    RFTLoopConfig,
    _is_http_endpoint_ready,
    build_trainer_step_command,
    build_vllm_server_command,
    filter_selected_rows_by_token_length,
    prune_old_global_step_checkpoints,
    prune_old_step_checkpoints,
    prune_old_step_payloads,
    reset_step_artifacts,
    resolve_data_max_length,
    resolve_effective_train_batch_size,
    resolve_micro_batch_size_per_gpu,
    resolve_latest_hf_checkpoint,
    split_selected_rows_for_eval,
    upsample_selected_rows_to_batch_multiple,
)


class _StubTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
    ):
        del add_generation_prompt, tokenize
        token_count = sum(len(str(message.get("content", ""))) for message in messages)
        input_ids = list(range(token_count))
        if return_dict:
            return {"input_ids": input_ids}
        return input_ids


def test_load_tokenizer_sets_fix_mistral_regex_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            calls.append((model_path, dict(kwargs)))
            return {"tokenizer": "ok"}

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    tokenizer = rft_runtime_loop._load_tokenizer("/tmp/model")
    assert tokenizer == {"tokenizer": "ok"}
    assert calls == [
        (
            "/tmp/model",
            {
                "trust_remote_code": False,
                "fix_mistral_regex": True,
            },
        )
    ]


def test_load_tokenizer_retries_without_fix_mistral_regex_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            _ = model_path
            calls.append(dict(kwargs))
            if "fix_mistral_regex" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'fix_mistral_regex'")
            return {"tokenizer": "fallback"}

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    tokenizer = rft_runtime_loop._load_tokenizer("/tmp/model")
    assert tokenizer == {"tokenizer": "fallback"}
    assert calls == [
        {"trust_remote_code": False, "fix_mistral_regex": True},
        {"trust_remote_code": False},
    ]


def test_build_trainer_step_command_includes_required_dataset_and_checkpoint_overrides(
    tmp_path: Path,
) -> None:
    command = build_trainer_step_command(
        python_bin="python3",
        nnodes=1,
        nproc_per_node=8,
        trainer_module="verl.trainer.fsdp_sft_trainer",
        config_name="rft_swe",
        config_dir=tmp_path / "configs",
        model_path=config.DEFAULT_TRAINING_MODEL_NAME,
        train_parquet_path=tmp_path / "accepted.parquet",
        val_parquet_path=tmp_path / "accepted_eval.parquet",
        trainer_output_dir=tmp_path / "checkpoints",
        train_batch_size=32,
        sft_num_epoch_per_batch=1,
        trainer_overrides=("trainer.total_training_steps=1",),
    )

    command_text = " ".join(command)
    assert "python3 -m torch.distributed.run" in command_text
    assert "-m verl.trainer.fsdp_sft_trainer" in command_text
    assert "trainer.total_training_steps=1" in command_text
    assert "trainer.save_freq=2147483647" in command_text
    assert "trainer.logger=[console]" in command_text
    assert "trainer.checkpoint.save_contents=[hf_model]" in command_text
    assert "trainer.checkpoint.load_contents=[hf_model]" in command_text
    assert "data.multiturn.enable=true" in command_text
    assert "data.custom_cls.path=null" in command_text
    assert f"data.train_files={tmp_path / 'accepted.parquet'}" in command_text
    assert f"data.val_files={tmp_path / 'accepted_eval.parquet'}" in command_text
    assert f"model.partial_pretrain={config.DEFAULT_TRAINING_MODEL_NAME}" in command_text


def test_build_trainer_step_command_allows_inner_wandb_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMALL_SWE_RFT_INNER_TRAINER_WANDB_ENABLE", "1")

    command = build_trainer_step_command(
        python_bin="python3",
        nnodes=1,
        nproc_per_node=8,
        trainer_module="verl.trainer.fsdp_sft_trainer",
        config_name="rft_swe",
        config_dir=tmp_path / "configs",
        model_path=config.DEFAULT_TRAINING_MODEL_NAME,
        train_parquet_path=tmp_path / "accepted.parquet",
        val_parquet_path=tmp_path / "accepted_eval.parquet",
        trainer_output_dir=tmp_path / "checkpoints",
        train_batch_size=32,
        sft_num_epoch_per_batch=1,
        trainer_overrides=(),
    )

    command_text = " ".join(command)
    assert "trainer.logger=[console,wandb]" in command_text


def test_build_vllm_server_command_uses_host_and_port_from_base_url() -> None:
    command = build_vllm_server_command(
        python_bin="python3",
        launch_module="vllm.entrypoints.openai.api_server",
        base_url="http://127.0.0.1:8000/v1",
        model_path="/tmp/model",
        served_model_name=config.DEFAULT_TRAINING_MODEL_NAME,
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
    assert config.DEFAULT_TRAINING_MODEL_NAME in command
    assert command[-2:] == ["--dtype", "bfloat16"]


def test_resolve_latest_hf_checkpoint_prefers_most_recent_checkpoint(tmp_path: Path) -> None:
    older = tmp_path / "global_step_100" / "huggingface"
    newer = tmp_path / "global_step_3" / "huggingface"
    older.mkdir(parents=True)
    time.sleep(0.01)
    newer.mkdir(parents=True)

    resolved = resolve_latest_hf_checkpoint(tmp_path)

    assert resolved == newer


def test_resolve_latest_hf_checkpoint_requires_huggingface_export(tmp_path: Path) -> None:
    (tmp_path / "global_step_3").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="huggingface"):
        resolve_latest_hf_checkpoint(tmp_path)


def test_prune_old_step_checkpoints_keeps_latest_roots(tmp_path: Path) -> None:
    output_dir = tmp_path / "rft_runtime"
    step_dirs: list[Path] = []
    for step in range(4):
        step_dir = output_dir / f"rft_step_{step:05d}"
        checkpoint_root = step_dir / "trainer_checkpoints"
        (checkpoint_root / "global_step_1" / "huggingface").mkdir(parents=True)
        step_dirs.append(step_dir)

    pruned = prune_old_step_checkpoints(step_dirs=step_dirs, keep_last=2)

    assert [path.parent.name for path in pruned] == ["rft_step_00000", "rft_step_00001"]
    assert not (output_dir / "rft_step_00000" / "trainer_checkpoints").exists()
    assert not (output_dir / "rft_step_00001" / "trainer_checkpoints").exists()
    assert (output_dir / "rft_step_00002" / "trainer_checkpoints").is_dir()
    assert (output_dir / "rft_step_00003" / "trainer_checkpoints").is_dir()


def test_prune_old_step_checkpoints_requires_positive_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_old_step_checkpoints(step_dirs=[tmp_path], keep_last=0)


def test_prune_old_step_checkpoints_scopes_to_current_run_step_dirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "rft_runtime"
    stale_dir = output_dir / "rft_step_00050" / "trainer_checkpoints"
    (stale_dir / "global_step_1" / "huggingface").mkdir(parents=True)

    current_steps: list[Path] = []
    for step in range(2):
        step_dir = output_dir / f"rft_step_{step:05d}"
        checkpoint_root = step_dir / "trainer_checkpoints"
        (checkpoint_root / "global_step_1" / "huggingface").mkdir(parents=True)
        current_steps.append(step_dir)

    pruned = prune_old_step_checkpoints(step_dirs=current_steps, keep_last=1)

    assert [path.parent.name for path in pruned] == ["rft_step_00000"]
    assert (output_dir / "rft_step_00050" / "trainer_checkpoints").is_dir()


def test_run_loop_requires_checkpoint_when_trainer_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        assert request.start_step_index == 0
        return {
            "selected_rows": [
                {
                    "task_id": "task-1",
                    "attempt_index": 0,
                    "step_index": 0,
                    "turn_index": 0,
                    "resolved": False,
                    "format_valid": True,
                    "final_turn_has_submit": True,
                    "final_submit_format_valid": True,
                    "prompt": "Fix bug",
                    "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
                    "trajectory_history": [
                        "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
                    ],
                }
            ],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(_rows, parquet_path: Path):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del command, cwd
        # Intentionally do not materialize a checkpoint.

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=1,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=False,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.0,
    )

    with pytest.raises(RuntimeError, match="produced no checkpoint"):
        rft_runtime_loop.run_rft_runtime_loop(config)


def test_run_loop_skips_checkpoint_root_prune_when_trainer_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state = {"collect_calls": 0, "prune_calls": 0}

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        step = call_state["collect_calls"]
        call_state["collect_calls"] += 1
        assert request.start_step_index == step
        if step == 0:
            return {
                "selected_rows": [
                    {
                        "task_id": "task-1",
                        "attempt_index": 0,
                        "step_index": 0,
                        "turn_index": 0,
                        "resolved": False,
                        "format_valid": True,
                        "final_turn_has_submit": True,
                        "final_submit_format_valid": True,
                        "prompt": "Fix bug",
                        "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
                        "trajectory_history": [
                            "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
                        ],
                    }
                ],
                "rejected_rows": [],
            }
        return {
            "selected_rows": [],
            "rejected_rows": [{"task_id": "task-2", "rft_rejection_reason": "non_terminal"}],
        }

    def _fake_write_selected_rows(_rows, parquet_path: Path):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _fake_prune_old_global_step_checkpoints(*, checkpoint_root, keep_last):
        del checkpoint_root, keep_last
        return []

    def _fake_prune_old_step_checkpoints(*, step_dirs, keep_last):
        del step_dirs, keep_last
        call_state["prune_calls"] += 1
        return []

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        _fake_prune_old_global_step_checkpoints,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        _fake_prune_old_step_checkpoints,
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=2,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=False,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert call_state["collect_calls"] == 2
    assert call_state["prune_calls"] == 1
    summary_path = config.output_dir / "rft_step_00001" / "rft_step_summary.json"
    assert summary_path.is_file()
    assert "trainer_skipped" in summary_path.read_text(encoding="utf-8")


def test_run_loop_checkpoint_pruning_tracks_only_checkpoint_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state = {"collect_calls": 0, "prune_calls": 0, "step_dir_args": []}

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _selected_row(step_index: int) -> dict[str, object]:
        return {
            "task_id": f"task-{step_index}",
            "attempt_index": 0,
            "step_index": step_index,
            "turn_index": 0,
            "resolved": False,
            "format_valid": True,
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
            "prompt": "Fix bug",
            "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            "trajectory_history": [
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ],
        }

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        step = call_state["collect_calls"]
        call_state["collect_calls"] += 1
        assert request.start_step_index == step
        if step in {0, 2}:
            return {"selected_rows": [_selected_row(step)], "rejected_rows": []}
        return {
            "selected_rows": [],
            "rejected_rows": [{"task_id": "task-skip", "rft_rejection_reason": "non_terminal"}],
        }

    def _fake_write_selected_rows(_rows, parquet_path: Path):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _fake_prune_old_global_step_checkpoints(*, checkpoint_root, keep_last):
        del checkpoint_root, keep_last
        return []

    def _fake_prune_old_step_checkpoints(*, step_dirs, keep_last):
        del keep_last
        call_state["prune_calls"] += 1
        call_state["step_dir_args"].append([Path(item).name for item in step_dirs])
        return []

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        _fake_prune_old_global_step_checkpoints,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        _fake_prune_old_step_checkpoints,
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=3,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=2,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=False,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert call_state["collect_calls"] == 3
    assert call_state["prune_calls"] == 2
    assert call_state["step_dir_args"][0] == ["rft_step_00000"]
    assert call_state["step_dir_args"][1] == ["rft_step_00000", "rft_step_00002"]


def test_run_loop_does_not_restart_vllm_after_final_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state: dict[str, object] = {"controllers": []}

    class _FakeVLLMController:
        def __init__(self, *, config, log_path: Path) -> None:
            del config, log_path
            self.start_calls: list[str] = []
            self.stop_calls = 0
            self._active = False
            controllers = call_state["controllers"]
            assert isinstance(controllers, list)
            controllers.append(self)

        def start(self, *, model_path: str) -> None:
            self.start_calls.append(model_path)
            self._active = True

        def stop(self) -> None:
            if self._active:
                self.stop_calls += 1
                self._active = False

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        assert request.start_step_index == 0
        return {
            "selected_rows": [
                {
                    "task_id": "task-1",
                    "attempt_index": 0,
                    "step_index": 0,
                    "turn_index": 0,
                    "resolved": False,
                    "format_valid": True,
                    "final_turn_has_submit": True,
                    "final_submit_format_valid": True,
                    "prompt": "Fix bug",
                    "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
                    "trajectory_history": [
                        "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
                    ],
                }
            ],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(_rows, parquet_path: Path):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(rft_runtime_loop, "VLLMServerController", _FakeVLLMController)
    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_payloads",
        lambda **_kwargs: [],
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=1,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=True,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    controllers = call_state["controllers"]
    assert isinstance(controllers, list)
    assert len(controllers) == 1
    controller = controllers[0]
    assert controller.start_calls == ["Qwen/Qwen3-0.6B"]
    assert controller.stop_calls == 1


def test_run_loop_uses_vllm_compatible_checkpoint_for_followup_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_state: dict[str, object] = {
        "controllers": [],
        "collect_calls": 0,
        "trainer_model_paths": [],
    }

    class _FakeVLLMController:
        def __init__(self, *, config, log_path: Path) -> None:
            del config, log_path
            self.start_calls: list[str] = []
            self.stop_calls = 0
            self._active = False
            controllers = call_state["controllers"]
            assert isinstance(controllers, list)
            controllers.append(self)

        def start(self, *, model_path: str) -> None:
            self.start_calls.append(model_path)
            self._active = True

        def stop(self) -> None:
            if self._active:
                self.stop_calls += 1
                self._active = False

    def _selected_row(step_index: int) -> dict[str, object]:
        return {
            "task_id": f"task-{step_index}",
            "attempt_index": 0,
            "step_index": step_index,
            "turn_index": 0,
            "resolved": False,
            "format_valid": True,
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
            "prompt": "Fix bug",
            "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            "trajectory_history": [
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ],
        }

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        step = call_state["collect_calls"]
        assert isinstance(step, int)
        call_state["collect_calls"] = step + 1
        assert request.start_step_index == step
        return {"selected_rows": [_selected_row(step)], "rejected_rows": []}

    def _fake_write_selected_rows(_rows, parquet_path: Path):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_model_paths = call_state["trainer_model_paths"]
        assert isinstance(trainer_model_paths, list)
        trainer_model_paths.append(str(kwargs["model_path"]))
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _fake_materialize_vllm_compatible_checkpoint(*, checkpoint_dir: Path, trainer_overrides):
        del trainer_overrides
        merged = Path(checkpoint_dir).parent / "huggingface_vllm_merged"
        merged.mkdir(parents=True, exist_ok=True)
        return merged

    monkeypatch.setattr(rft_runtime_loop, "VLLMServerController", _FakeVLLMController)
    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "materialize_vllm_compatible_checkpoint",
        _fake_materialize_vllm_compatible_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_payloads",
        lambda **_kwargs: [],
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=2,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=True,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    trainer_model_paths = call_state["trainer_model_paths"]
    assert isinstance(trainer_model_paths, list)
    assert len(trainer_model_paths) == 2
    assert trainer_model_paths[0] == "Qwen/Qwen3-0.6B"
    assert trainer_model_paths[1].endswith("huggingface_vllm_merged")

    controllers = call_state["controllers"]
    assert isinstance(controllers, list)
    assert len(controllers) == 1
    controller = controllers[0]
    assert controller.start_calls[0] == "Qwen/Qwen3-0.6B"
    assert controller.start_calls[1].endswith("huggingface_vllm_merged")


def test_run_loop_upsamples_selected_rows_to_effective_batch_multiple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_counts: list[int] = []

    def _selected_row(task_id: str) -> dict[str, object]:
        return {
            "task_id": task_id,
            "attempt_index": 0,
            "step_index": 0,
            "turn_index": 0,
            "resolved": False,
            "format_valid": True,
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
            "prompt": "Fix bug",
            "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            "trajectory_history": [
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ],
        }

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        assert request.start_step_index == 0
        return {
            "selected_rows": [
                _selected_row("task-1"),
                _selected_row("task-2"),
                _selected_row("task-3"),
            ],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path):
        write_counts.append(len(rows))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

    def _fake_build_trainer_step_command(**kwargs):
        assert kwargs["train_batch_size"] == 2
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_payloads",
        lambda **_kwargs: [],
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=2,
        rft_steps=1,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=2,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=False,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert write_counts == [3, 4]
    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count_raw"] == 3
    assert summary["selected_count_for_train"] == 4
    assert summary["selected_rows_upsampled"] == 1
    assert summary["effective_train_batch_size"] == 2
    assert summary["avg_generation_length_raw"] > 0.0
    assert summary["avg_generation_length"] > 0.0


def test_run_loop_writes_eval_parquet_and_uses_eval_val_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_calls: list[tuple[str, int]] = []

    def _selected_row(task_id: str) -> dict[str, object]:
        return {
            "task_id": task_id,
            "attempt_index": 0,
            "step_index": 0,
            "turn_index": 0,
            "resolved": False,
            "format_valid": True,
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
            "prompt": "Fix bug",
            "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            "trajectory_history": [
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ],
        }

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        assert request.start_step_index == 0
        return {
            "selected_rows": [
                _selected_row("task-1"),
                _selected_row("task-2"),
                _selected_row("task-3"),
                _selected_row("task-4"),
            ],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path):
        write_calls.append((parquet_path.name, len(rows)))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

    def _fake_build_trainer_step_command(**kwargs):
        assert Path(kwargs["train_parquet_path"]).name == "accepted_trajectories.parquet"
        assert Path(kwargs["val_parquet_path"]).name == "accepted_trajectories_eval.parquet"
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_payloads",
        lambda **_kwargs: [],
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=1,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=False,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.25,
        eval_min_rows=1,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert write_calls == [
        ("accepted_trajectories.parquet", 3),
        ("accepted_trajectories_eval.parquet", 1),
    ]
    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count_raw"] == 4
    assert summary["selected_count_for_train_raw"] == 3
    assert summary["selected_count_for_train"] == 3
    assert summary["selected_count_for_eval"] == 1
    assert summary["eval_split_fallback_to_train"] is False
    assert summary["eval_parquet"].endswith("accepted_trajectories_eval.parquet")


def test_run_loop_upsamples_eval_rows_to_effective_batch_multiple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_calls: list[tuple[str, int]] = []

    def _selected_row(task_id: str) -> dict[str, object]:
        return {
            "task_id": task_id,
            "attempt_index": 0,
            "step_index": 0,
            "turn_index": 0,
            "resolved": False,
            "format_valid": True,
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
            "prompt": "Fix bug",
            "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            "trajectory_history": [
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ],
        }

    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        assert request.start_step_index == 0
        return {
            "selected_rows": [
                _selected_row("task-1"),
                _selected_row("task-2"),
                _selected_row("task-3"),
                _selected_row("task-4"),
                _selected_row("task-5"),
            ],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path):
        write_calls.append((parquet_path.name, len(rows)))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

    def _fake_build_trainer_step_command(**kwargs):
        assert kwargs["train_batch_size"] == 2
        assert Path(kwargs["train_parquet_path"]).name == "accepted_trajectories.parquet"
        assert Path(kwargs["val_parquet_path"]).name == "accepted_trajectories_eval.parquet"
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_1" / "huggingface").mkdir(parents=True, exist_ok=True)

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_1" / "huggingface"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(rft_runtime_loop, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    monkeypatch.setattr(
        rft_runtime_loop,
        "write_selected_rows_to_multiturn_parquet",
        _fake_write_selected_rows,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "build_trainer_step_command",
        _fake_build_trainer_step_command,
    )
    monkeypatch.setattr(rft_runtime_loop, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_latest_hf_checkpoint",
        _fake_resolve_latest_hf_checkpoint,
    )
    monkeypatch.setattr(rft_runtime_loop, "resolve_micro_batch_size_per_gpu", lambda **_kwargs: 2)
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_global_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_checkpoints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "prune_old_step_payloads",
        lambda **_kwargs: [],
    )

    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=1,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=2,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=False,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
        eval_split_fraction=0.2,
        eval_min_rows=1,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert write_calls == [
        ("accepted_trajectories.parquet", 4),
        ("accepted_trajectories_eval.parquet", 2),
    ]
    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count_raw"] == 5
    assert summary["selected_count_for_train_raw"] == 4
    assert summary["selected_count_for_train"] == 4
    assert summary["selected_rows_upsampled"] == 0
    assert summary["selected_count_for_eval_raw"] == 1
    assert summary["selected_count_for_eval"] == 2
    assert summary["selected_rows_eval_upsampled"] == 1
    assert summary["effective_eval_batch_size"] == 2
    assert summary["eval_split_fallback_to_train"] is False


def test_prune_old_global_step_checkpoints_keeps_latest_steps(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "trainer_checkpoints"
    newest = checkpoint_root / "global_step_1" / "huggingface"
    older = checkpoint_root / "global_step_5" / "huggingface"
    older.mkdir(parents=True)
    time.sleep(0.01)
    newest.mkdir(parents=True)

    pruned = prune_old_global_step_checkpoints(checkpoint_root=checkpoint_root, keep_last=1)

    assert [path.name for path in pruned] == ["global_step_5"]
    assert not (checkpoint_root / "global_step_5").exists()
    assert (checkpoint_root / "global_step_1").is_dir()


def test_prune_old_global_step_checkpoints_requires_positive_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_old_global_step_checkpoints(checkpoint_root=tmp_path, keep_last=0)


def test_prune_old_step_payloads_keeps_latest_step_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "rft_runtime"
    step_dirs: list[Path] = []
    for step in range(4):
        step_dir = output_dir / f"rft_step_{step:05d}"
        (step_dir / "collector_artifacts").mkdir(parents=True)
        (step_dir / "collector_artifacts" / "rollout_rows.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (step_dir / "accepted_trajectories.parquet").write_text("stub", encoding="utf-8")
        (step_dir / "accepted_trajectories_eval.parquet").write_text("stub", encoding="utf-8")
        step_dirs.append(step_dir)

    pruned = prune_old_step_payloads(step_dirs=step_dirs, keep_last=2)

    pruned_names = {str(path.relative_to(output_dir)) for path in pruned}
    assert "rft_step_00000/collector_artifacts" in pruned_names
    assert "rft_step_00000/accepted_trajectories.parquet" in pruned_names
    assert "rft_step_00000/accepted_trajectories_eval.parquet" in pruned_names
    assert "rft_step_00001/collector_artifacts" in pruned_names
    assert "rft_step_00001/accepted_trajectories.parquet" in pruned_names
    assert "rft_step_00001/accepted_trajectories_eval.parquet" in pruned_names

    assert not (output_dir / "rft_step_00000" / "collector_artifacts").exists()
    assert not (output_dir / "rft_step_00000" / "accepted_trajectories.parquet").exists()
    assert not (output_dir / "rft_step_00000" / "accepted_trajectories_eval.parquet").exists()
    assert not (output_dir / "rft_step_00001" / "collector_artifacts").exists()
    assert not (output_dir / "rft_step_00001" / "accepted_trajectories.parquet").exists()
    assert not (output_dir / "rft_step_00001" / "accepted_trajectories_eval.parquet").exists()
    assert (output_dir / "rft_step_00002" / "collector_artifacts").is_dir()
    assert (output_dir / "rft_step_00002" / "accepted_trajectories.parquet").is_file()
    assert (output_dir / "rft_step_00002" / "accepted_trajectories_eval.parquet").is_file()
    assert (output_dir / "rft_step_00003" / "collector_artifacts").is_dir()
    assert (output_dir / "rft_step_00003" / "accepted_trajectories.parquet").is_file()
    assert (output_dir / "rft_step_00003" / "accepted_trajectories_eval.parquet").is_file()


def test_prune_old_step_payloads_requires_positive_keep_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_old_step_payloads(step_dirs=[tmp_path], keep_last=0)


def test_resolve_effective_train_batch_size_clamps_to_global_selected() -> None:
    resolved = resolve_effective_train_batch_size(
        requested=1024,
        selected_count=48,
        world_size=8,
    )
    assert resolved == 48


def test_resolve_effective_train_batch_size_enforces_world_size_divisibility_when_possible() -> None:
    resolved = resolve_effective_train_batch_size(
        requested=65,
        selected_count=500,
        world_size=8,
    )
    assert resolved == 64


def test_resolve_effective_train_batch_size_returns_none_when_below_world_size() -> None:
    resolved = resolve_effective_train_batch_size(
        requested=64,
        selected_count=3,
        world_size=8,
    )
    assert resolved is None


def test_resolve_effective_train_batch_size_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="requested"):
        resolve_effective_train_batch_size(requested=0, selected_count=8, world_size=8)
    with pytest.raises(ValueError, match="selected_count"):
        resolve_effective_train_batch_size(requested=1, selected_count=0, world_size=8)
    with pytest.raises(ValueError, match="world_size"):
        resolve_effective_train_batch_size(requested=1, selected_count=8, world_size=0)
    with pytest.raises(ValueError, match="micro_batch_size_per_gpu"):
        resolve_effective_train_batch_size(
            requested=1,
            selected_count=8,
            world_size=8,
            micro_batch_size_per_gpu=0,
        )


def test_resolve_effective_train_batch_size_enforces_micro_batch_divisibility() -> None:
    resolved = resolve_effective_train_batch_size(
        requested=47,
        selected_count=47,
        world_size=8,
        micro_batch_size_per_gpu=4,
    )
    assert resolved == 32


def test_upsample_selected_rows_to_batch_multiple_repeats_rows_to_fit_batch() -> None:
    rows = [
        {"task_id": "task-1"},
        {"task_id": "task-2"},
        {"task_id": "task-3"},
    ]
    upsampled_rows, upsampled_count = upsample_selected_rows_to_batch_multiple(
        rows,
        global_batch_size=2,
    )
    assert upsampled_count == 1
    assert len(upsampled_rows) == 4
    assert upsampled_rows[:3] == rows
    assert upsampled_rows[-1]["task_id"] == "task-1"


def test_upsample_selected_rows_to_batch_multiple_is_noop_when_already_divisible() -> None:
    rows = [
        {"task_id": "task-1"},
        {"task_id": "task-2"},
        {"task_id": "task-3"},
        {"task_id": "task-4"},
    ]
    upsampled_rows, upsampled_count = upsample_selected_rows_to_batch_multiple(
        rows,
        global_batch_size=2,
    )
    assert upsampled_count == 0
    assert upsampled_rows == rows


def test_upsample_selected_rows_to_batch_multiple_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="global_batch_size"):
        upsample_selected_rows_to_batch_multiple(
            [{"task_id": "task-1"}],
            global_batch_size=0,
        )
    with pytest.raises(ValueError, match="selected_rows"):
        upsample_selected_rows_to_batch_multiple([], global_batch_size=2)


def test_split_selected_rows_for_eval_holds_out_rows_deterministically() -> None:
    selected_rows = [
        {"task_id": "task-1", "step_index": 0, "attempt_index": 0, "turn_index": 0},
        {"task_id": "task-2", "step_index": 0, "attempt_index": 0, "turn_index": 0},
        {"task_id": "task-3", "step_index": 0, "attempt_index": 0, "turn_index": 0},
        {"task_id": "task-4", "step_index": 0, "attempt_index": 0, "turn_index": 0},
    ]

    train_rows_a, eval_rows_a = split_selected_rows_for_eval(
        selected_rows,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )
    train_rows_b, eval_rows_b = split_selected_rows_for_eval(
        selected_rows,
        eval_split_fraction=0.25,
        min_eval_rows=1,
    )

    assert len(train_rows_a) == 3
    assert len(eval_rows_a) == 1
    assert train_rows_a == train_rows_b
    assert eval_rows_a == eval_rows_b
    assert {row["task_id"] for row in train_rows_a}.isdisjoint({row["task_id"] for row in eval_rows_a})


def test_split_selected_rows_for_eval_disables_holdout_when_fraction_is_zero() -> None:
    selected_rows = [
        {"task_id": "task-1"},
        {"task_id": "task-2"},
    ]
    train_rows, eval_rows = split_selected_rows_for_eval(
        selected_rows,
        eval_split_fraction=0.0,
        min_eval_rows=1,
    )
    assert len(train_rows) == 2
    assert eval_rows == []


def test_resolve_micro_batch_size_per_gpu_prefers_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "rft_swe.yaml").write_text(
        "data:\n  micro_batch_size_per_gpu: 4\n",
        encoding="utf-8",
    )

    resolved_default = resolve_micro_batch_size_per_gpu(
        config_dir=config_dir,
        config_name="rft_swe",
        trainer_overrides=(),
    )
    assert resolved_default == 4

    resolved_override = resolve_micro_batch_size_per_gpu(
        config_dir=config_dir,
        config_name="rft_swe",
        trainer_overrides=("+data.micro_batch_size_per_gpu=2",),
    )
    assert resolved_override == 2


def test_resolve_data_max_length_prefers_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "rft_swe.yaml").write_text(
        "data:\n  max_length: 256\n",
        encoding="utf-8",
    )

    resolved_default = resolve_data_max_length(
        config_dir=config_dir,
        config_name="rft_swe",
        trainer_overrides=(),
    )
    assert resolved_default == 256

    resolved_override = resolve_data_max_length(
        config_dir=config_dir,
        config_name="rft_swe",
        trainer_overrides=("+data.max_length=128",),
    )
    assert resolved_override == 128


def test_filter_selected_rows_by_token_length_drops_overlength_rows() -> None:
    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            add_generation_prompt: bool = False,
            tokenize: bool = True,
            return_dict: bool = True,
        ):
            del add_generation_prompt, tokenize
            token_count = sum(len(str(message.get("content", ""))) for message in messages)
            input_ids = list(range(token_count))
            if return_dict:
                return {"input_ids": input_ids}
            return input_ids

    selected_rows = [
        {
            "task_id": "task-short",
            "prompt": "Fix bug",
            "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
        },
        {
            "task_id": "task-long",
            "prompt": "Fix bug",
            "assistant_response": "x" * 500,
        },
    ]

    kept_rows, dropped_count = filter_selected_rows_by_token_length(
        selected_rows=selected_rows,
        tokenizer=_FakeTokenizer(),
        max_sequence_length=400,
    )

    assert dropped_count == 1
    assert len(kept_rows) == 1
    assert kept_rows[0]["task_id"] == "task-short"


def test_filter_selected_rows_by_token_length_requires_chat_template_tokenizer() -> None:
    with pytest.raises(ValueError, match="apply_chat_template"):
        filter_selected_rows_by_token_length(
            selected_rows=[
                {
                    "task_id": "task-1",
                    "prompt": "Fix bug",
                    "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
                }
            ],
            tokenizer=object(),
            max_sequence_length=128,
        )


def test_compute_average_generation_length_prefers_action_mask() -> None:
    selected_rows = [
        {"action_mask_rft": [0, 1, 1, 0]},
        {"action_mask_rft": [1, 0]},
    ]

    value = rft_runtime_loop.compute_average_generation_length(
        selected_rows=selected_rows,
        tokenizer=_StubTokenizer(),
    )

    assert value == pytest.approx(1.5)


def test_compute_average_generation_length_falls_back_to_token_labels() -> None:
    selected_rows = [
        {"token_labels": [-100, -100, 7, 9]},
        {"token_labels": [-100, 1]},
    ]

    value = rft_runtime_loop.compute_average_generation_length(
        selected_rows=selected_rows,
        tokenizer=_StubTokenizer(),
    )

    assert value == pytest.approx(1.5)


def test_compute_average_generation_length_falls_back_to_assistant_text() -> None:
    selected_rows = [
        {"assistant_response": "abc"},
        {
            "trajectory_history": [
                "<tool_response>stderr: warning",
                "abcd",
            ]
        },
    ]

    value = rft_runtime_loop.compute_average_generation_length(
        selected_rows=selected_rows,
        tokenizer=_StubTokenizer(),
    )

    assert value == pytest.approx(3.5)


def test_compute_average_generation_length_returns_none_without_signal() -> None:
    selected_rows = [{"task_id": "task-1"}]

    value = rft_runtime_loop.compute_average_generation_length(
        selected_rows=selected_rows,
        tokenizer=_StubTokenizer(),
    )

    assert value is None


def test_run_command_extracts_inner_loss_metrics(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "print('step:1 - train/loss:0.45 - train/lr(1e-3):0.1')\n"
            "print('step:2 - train/loss:0.40 - train/lr(1e-3):0.0')\n"
            "print('step:2 - val/loss:0.48')\n"
        ),
    ]

    metrics = rft_runtime_loop._run_command(command, cwd=tmp_path)

    assert metrics["train_step_last"] == 2
    assert metrics["train_loss_last"] == pytest.approx(0.40)
    assert metrics["val_step_last"] == 2
    assert metrics["val_loss_last"] == pytest.approx(0.48)


def test_reset_step_artifacts_removes_mutable_outputs(tmp_path: Path) -> None:
    step_dir = tmp_path / "rft_step_00000"
    (step_dir / "collector_artifacts").mkdir(parents=True)
    (step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface").mkdir(parents=True)
    (step_dir / "accepted_trajectories.parquet").write_text("stub", encoding="utf-8")
    (step_dir / "accepted_trajectories_eval.parquet").write_text("stub", encoding="utf-8")
    (step_dir / "rft_step_summary.json").write_text("{}", encoding="utf-8")

    reset_step_artifacts(step_dir)

    assert not (step_dir / "collector_artifacts").exists()
    assert not (step_dir / "trainer_checkpoints").exists()
    assert not (step_dir / "accepted_trajectories.parquet").exists()
    assert not (step_dir / "accepted_trajectories_eval.parquet").exists()
    assert not (step_dir / "rft_step_summary.json").exists()


def test_http_readiness_requires_2xx(monkeypatch) -> None:
    class _Response:
        def __init__(self, status: int, payload: str = '{"data":[{"id":"model"}]}') -> None:
            self.status = status
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self._payload.encode("utf-8")

    monkeypatch.setattr(rft_runtime_loop, "urlopen", lambda request, timeout: _Response(200))
    assert _is_http_endpoint_ready("http://127.0.0.1:8000/v1/models") is True

    monkeypatch.setattr(rft_runtime_loop, "urlopen", lambda request, timeout: _Response(404))
    assert _is_http_endpoint_ready("http://127.0.0.1:8000/v1/models") is False


def test_http_readiness_rejects_http_error(monkeypatch) -> None:
    def _raise_http_error(request, timeout):
        raise HTTPError(
            url="http://127.0.0.1:8000/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(rft_runtime_loop, "urlopen", _raise_http_error)
    assert _is_http_endpoint_ready("http://127.0.0.1:8000/v1/models") is False


def test_http_readiness_includes_authorization_header_when_api_key_present(monkeypatch) -> None:
    captured = {"authorization": None}

    class _Response:
        def __init__(self, status: int, payload: str = '{"data":[{"id":"model"}]}') -> None:
            self.status = status
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self._payload.encode("utf-8")

    def _fake_urlopen(request, timeout):
        del timeout
        captured["authorization"] = request.get_header("Authorization")
        return _Response(200)

    monkeypatch.setattr(rft_runtime_loop, "urlopen", _fake_urlopen)

    assert (
        _is_http_endpoint_ready(
            "http://127.0.0.1:8000/v1/models",
            api_key="api-test-key",
        )
        is True
    )
    assert captured["authorization"] == "Bearer api-test-key"


def test_http_readiness_requires_expected_model_when_provided(monkeypatch) -> None:
    class _Response:
        def __init__(self, payload: str) -> None:
            self.status = 200
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self._payload.encode("utf-8")

    monkeypatch.setattr(
        rft_runtime_loop,
        "urlopen",
        lambda request, timeout: _Response('{"data":[{"id":"other-model"}]}'),
    )
    assert (
        _is_http_endpoint_ready(
            "http://127.0.0.1:8000/v1/models",
            expected_model_name="expected-model",
        )
        is False
    )

    monkeypatch.setattr(
        rft_runtime_loop,
        "urlopen",
        lambda request, timeout: _Response('{"data":[{"id":"expected-model"}]}'),
    )
    assert (
        _is_http_endpoint_ready(
            "http://127.0.0.1:8000/v1/models",
            expected_model_name="expected-model",
        )
        is True
    )


def test_vllm_controller_rejects_occupied_endpoint_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RFTLoopConfig(
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        config_name="rft_swe",
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        python_bin="python3",
        nnodes=1,
        nproc_per_node=1,
        rft_steps=1,
        samples_per_task=1,
        task_batch_size=1,
        sft_num_epoch_per_batch=1,
        checkpoint_keep_last=1,
        train_batch_size=1,
        output_dir=tmp_path / "runtime",
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="default",
        initial_model="Qwen/Qwen3-0.6B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_served_model="Qwen/Qwen3-0.6B",
        manage_vllm=True,
        vllm_launch_module="trainer.vllm_api_server_entry",
        vllm_ready_timeout_sec=1,
        vllm_stop_timeout_sec=1,
        vllm_extra_args=(),
        trainer_overrides=(),
        dry_run=False,
    )
    controller = rft_runtime_loop.VLLMServerController(
        config=config,
        log_path=tmp_path / "vllm_server.log",
    )

    monkeypatch.setattr(
        rft_runtime_loop,
        "_is_http_endpoint_ready",
        lambda url, *, api_key=None, expected_model_name=None: expected_model_name is None,
    )

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("subprocess.Popen should not be called when the endpoint is occupied")

    monkeypatch.setattr(rft_runtime_loop.subprocess, "Popen", _unexpected_popen)

    with pytest.raises(RuntimeError, match="already has a ready endpoint"):
        controller.start(model_path="/tmp/model")


def test_resolve_vllm_api_key_prefers_small_swe_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-value")
    monkeypatch.setenv("SMALL_SWE_VLLM_API_KEY", "small-swe-value")

    assert rft_runtime_loop._resolve_vllm_api_key() == "small-swe-value"


def test_load_model_from_config_uses_dtype_when_supported() -> None:
    captured: dict[str, object] = {}

    class _AutoModel:
        @staticmethod
        def from_config(model_config, trust_remote_code=False, **kwargs):
            del model_config
            assert trust_remote_code is False
            captured.update(kwargs)
            return kwargs

    payload = rft_runtime_loop._load_model_from_config_with_dtype_fallback(
        auto_model_cls=_AutoModel,
        model_config=object(),
        model_kwargs={"dtype": "bfloat16"},
    )

    assert payload["dtype"] == "bfloat16"
    assert captured["dtype"] == "bfloat16"


def test_load_model_from_config_falls_back_to_torch_dtype_for_legacy_api() -> None:
    captured: dict[str, object] = {}

    class _AutoModel:
        @staticmethod
        def from_config(model_config, trust_remote_code=False, **kwargs):
            del model_config
            assert trust_remote_code is False
            if "dtype" in kwargs:
                raise TypeError("got an unexpected keyword argument 'dtype'")
            captured.update(kwargs)
            return kwargs

    payload = rft_runtime_loop._load_model_from_config_with_dtype_fallback(
        auto_model_cls=_AutoModel,
        model_config=object(),
        model_kwargs={"dtype": "bfloat16"},
    )

    assert "dtype" not in payload
    assert payload["torch_dtype"] == "bfloat16"
    assert captured["torch_dtype"] == "bfloat16"
