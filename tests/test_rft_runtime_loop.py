from __future__ import annotations

import io
import json
import signal
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
    resolve_apply_chat_template_kwargs,
    reset_step_artifacts,
    resolve_data_max_length,
    resolve_effective_train_batch_size,
    resolve_micro_batch_size_per_gpu,
    resolve_latest_hf_checkpoint,
    split_selected_rows_for_eval,
    upsample_selected_rows_to_batch_multiple,
)


class _StubTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return list(range(len(text)))

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
        **_kwargs,
    ):
        text = self._render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        input_ids = [ord(char) for char in text]
        if return_dict:
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        return input_ids

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        del add_special_tokens
        input_ids = [ord(char) for char in text]
        payload = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload

    def _render(self, messages, *, add_generation_prompt: bool) -> str:
        chunks = ["SYS"]
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            chunks.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "".join(chunks)


def test_load_tokenizer_sets_fix_mistral_regex_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_hf_tokenizer(model_path: str, **kwargs):
        calls.append((model_path, dict(kwargs)))
        return {"tokenizer": "ok"}

    fake_verl = types.ModuleType("verl")
    fake_verl_utils = types.ModuleType("verl.utils")
    fake_verl_utils.hf_tokenizer = _fake_hf_tokenizer
    fake_verl.utils = fake_verl_utils
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_verl_utils)

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

    def _fake_hf_tokenizer(model_path: str, **kwargs):
        _ = model_path
        calls.append(dict(kwargs))
        if "fix_mistral_regex" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'fix_mistral_regex'")
        return {"tokenizer": "fallback"}

    fake_verl = types.ModuleType("verl")
    fake_verl_utils = types.ModuleType("verl.utils")
    fake_verl_utils.hf_tokenizer = _fake_hf_tokenizer
    fake_verl.utils = fake_verl_utils
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_verl_utils)

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
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        config_name="rft_swe",
        config_dir=tmp_path / "configs",
        model_path=config.DEFAULT_TRAINING_MODEL_NAME,
        train_parquet_path=tmp_path / "accepted.parquet",
        trainer_output_dir=tmp_path / "checkpoints",
        train_batch_size=32,
        train_min_rows=32,
        sft_num_epoch_per_batch=1,
        token_cache_fingerprint="abc123",
        trainer_overrides=("trainer.total_training_steps=1",),
    )

    command_text = " ".join(command)
    assert "python3 -m torch.distributed.run" in command_text
    assert "-m verl_integration.fsdp_sft_trainer_entry" in command_text
    assert "trainer.total_training_steps=1" in command_text
    assert "trainer.save_freq=2147483647" in command_text
    assert "trainer.test_freq=0" in command_text
    assert "trainer.logger=[console]" in command_text
    assert "trainer.checkpoint.save_contents=[hf_model]" in command_text
    assert "trainer.checkpoint.load_contents=[hf_model]" in command_text
    assert "trainer.test_freq=0" in command_text
    assert "data.train_min_rows=32" in command_text
    assert "data.multiturn.enable=false" in command_text
    assert "data.custom_cls.path=pkg://trainer.rft_token_cache" in command_text
    assert "data.custom_cls.name=CachedRFTSFTDataset" in command_text
    assert "data.token_cache.schema_version=1" in command_text
    assert "data.token_cache.expected_fingerprint=abc123" in command_text
    assert f"data.train_files={tmp_path / 'accepted.parquet'}" in command_text
    assert "data.val_files=[]" in command_text
    assert "accepted_eval.parquet" not in command_text
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
        trainer_module="verl_integration.fsdp_sft_trainer_entry",
        config_name="rft_swe",
        config_dir=tmp_path / "configs",
        model_path=config.DEFAULT_TRAINING_MODEL_NAME,
        train_parquet_path=tmp_path / "accepted.parquet",
        trainer_output_dir=tmp_path / "checkpoints",
        train_batch_size=32,
        train_min_rows=32,
        sft_num_epoch_per_batch=1,
        token_cache_fingerprint="abc123",
        trainer_overrides=(),
    )

    command_text = " ".join(command)
    assert "trainer.logger=[console,wandb]" in command_text


def test_build_trainer_step_command_rejects_unpatched_trainer_module(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="local entrypoint skips verl validation"):
        build_trainer_step_command(
            python_bin="python3",
            nnodes=1,
            nproc_per_node=8,
            trainer_module="verl.trainer.fsdp_sft_trainer",
            config_name="rft_swe",
            config_dir=tmp_path / "configs",
            model_path=config.DEFAULT_TRAINING_MODEL_NAME,
            train_parquet_path=tmp_path / "accepted.parquet",
            trainer_output_dir=tmp_path / "checkpoints",
            train_batch_size=32,
            train_min_rows=32,
            sft_num_epoch_per_batch=1,
            token_cache_fingerprint="0" * 64,
            trainer_overrides=(),
        )


def test_parse_args_defaults_to_fixed_eval_task_count(tmp_path: Path) -> None:
    parsed = rft_runtime_loop._parse_args(
        [
            "--project-root",
            str(tmp_path),
            "--config-dir",
            str(tmp_path / "configs"),
            "--rft-steps",
            "1",
            "--samples-per-task",
            "1",
            "--task-batch-size",
            "1",
            "--sft-num-epoch-per-batch",
            "1",
            "--train-batch-size",
            "1",
            "--output-dir",
            str(tmp_path / "runtime"),
            "--initial-model",
            "Qwen/Qwen3-0.6B",
            "--vllm-base-url",
            "http://127.0.0.1:8000/v1",
            "--vllm-served-model",
            "Qwen/Qwen3-0.6B",
        ]
    )

    assert parsed.eval_task_count == 50


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


def test_prune_old_step_payloads_preserves_protected_committed_step_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "rft_runtime"
    step_dirs: list[Path] = []
    for step in range(3):
        step_dir = output_dir / f"rft_step_{step:05d}"
        (step_dir / "collector_artifacts").mkdir(parents=True)
        (step_dir / "accepted_trajectories.parquet").write_text("train", encoding="utf-8")
        (step_dir / "accepted_trajectories_eval.parquet").write_text("eval", encoding="utf-8")
        step_dirs.append(step_dir)

    pruned = prune_old_step_payloads(
        step_dirs=step_dirs,
        keep_last=1,
        protected_step_dirs=[step_dirs[1]],
    )

    assert [path.parent.name for path in pruned] == [
        "rft_step_00000",
        "rft_step_00000",
        "rft_step_00000",
    ]
    assert not (step_dirs[0] / "collector_artifacts").exists()
    assert (step_dirs[1] / "collector_artifacts").is_dir()
    assert (step_dirs[2] / "collector_artifacts").is_dir()


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

    def _fake_write_selected_rows(_rows, parquet_path: Path, **_kwargs):
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
        eval_task_count=0,
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

    def _fake_write_selected_rows(_rows, parquet_path: Path, **_kwargs):
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

    def _fake_materialize_vllm_compatible_checkpoint(*, checkpoint_dir: Path, trainer_overrides):
        del trainer_overrides
        merged = Path(checkpoint_dir).parent / "huggingface_vllm_merged"
        merged.mkdir(parents=True, exist_ok=True)
        return merged

    def _fake_prune_old_global_step_checkpoints(*, checkpoint_root, keep_last):
        del checkpoint_root, keep_last
        return []

    def _fake_prune_old_step_checkpoints(*, step_dirs, keep_last, protected_step_dirs=()):
        del step_dirs, keep_last, protected_step_dirs
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
        "materialize_vllm_compatible_checkpoint",
        _fake_materialize_vllm_compatible_checkpoint,
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
        eval_task_count=0,
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

    def _fake_write_selected_rows(_rows, parquet_path: Path, **_kwargs):
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

    def _fake_prune_old_step_checkpoints(*, step_dirs, keep_last, protected_step_dirs=()):
        del keep_last, protected_step_dirs
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
        eval_task_count=0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert call_state["collect_calls"] == 3
    assert call_state["prune_calls"] == 2
    assert call_state["step_dir_args"][0] == ["rft_step_00000"]
    assert call_state["step_dir_args"][1] == ["rft_step_00000", "rft_step_00002"]


def test_run_loop_resumes_from_latest_committed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "runtime"
    previous_step_dir = output_dir / "rft_step_00000"
    previous_hf = previous_step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface"
    previous_vllm = previous_step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface_vllm_merged"
    previous_hf.mkdir(parents=True)
    previous_vllm.mkdir(parents=True)
    (output_dir / rft_runtime_loop._RFT_RUNTIME_LOOP_MANIFEST_FILE_NAME).write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-16 00:00 UTC",
                "config": {},
                "steps": [{"step_index": 0}],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / rft_runtime_loop._RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME).write_text(
        json.dumps(
            {
                "stage": "format_rft",
                "committed_step_index": 0,
                "latest_hf_checkpoint": str(previous_hf),
                "latest_vllm_checkpoint": str(previous_vllm),
                "resume_model_path": str(previous_vllm),
                "selection_contract": {"mode": "format_first_rft"},
                "correctness_contract": "heuristic",
                "committed_utc": "2026-03-16 00:00 UTC",
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {
        "model_paths": [],
        "step_indexes": [],
        "tokenizer_paths": [],
    }

    def _fake_load_tokenizer(model_path: str):
        tokenizer_paths = captured["tokenizer_paths"]
        assert isinstance(tokenizer_paths, list)
        tokenizer_paths.append(str(model_path))
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        step_indexes = captured["step_indexes"]
        assert isinstance(step_indexes, list)
        step_indexes.append(request.start_step_index)
        return {
            "selected_rows": [
                {
                    "task_id": "task-1",
                    "attempt_index": 0,
                    "step_index": request.start_step_index,
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

    def _fake_write_selected_rows(_rows, parquet_path: Path, **_kwargs):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        model_paths = captured["model_paths"]
        assert isinstance(model_paths, list)
        model_paths.append(str(kwargs["model_path"]))
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
        output_dir=output_dir,
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
        eval_task_count=0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    model_paths = captured["model_paths"]
    assert isinstance(model_paths, list)
    assert model_paths == [str(previous_vllm)]
    tokenizer_paths = captured["tokenizer_paths"]
    assert isinstance(tokenizer_paths, list)
    assert tokenizer_paths == [str(previous_vllm)]
    step_indexes = captured["step_indexes"]
    assert isinstance(step_indexes, list)
    assert step_indexes == [1]
    latest_commit = json.loads(
        (output_dir / rft_runtime_loop._RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert latest_commit["committed_step_index"] == 1
    manifest = json.loads(
        (output_dir / rft_runtime_loop._RFT_RUNTIME_LOOP_MANIFEST_FILE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert [step["step_index"] for step in manifest["steps"]] == [0, 1]
    assert manifest["steps"][1]["token_cache_model_path"] == str(previous_vllm)


def test_run_loop_resume_rejects_changed_fixed_eval_task_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "runtime"
    previous_step_dir = output_dir / "rft_step_00000"
    previous_hf = previous_step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface"
    previous_vllm = previous_step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface_vllm_merged"
    previous_hf.mkdir(parents=True)
    previous_vllm.mkdir(parents=True)
    (output_dir / rft_runtime_loop._RFT_RUNTIME_LOOP_MANIFEST_FILE_NAME).write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-16 00:00 UTC",
                "config": {},
                "fixed_eval_task_ids": ["task-old"],
                "steps": [{"step_index": 0}],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / rft_runtime_loop._RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME).write_text(
        json.dumps(
            {
                "stage": "format_rft",
                "committed_step_index": 0,
                "latest_hf_checkpoint": str(previous_hf),
                "latest_vllm_checkpoint": str(previous_vllm),
                "resume_model_path": str(previous_vllm),
                "selection_contract": {"mode": "format_first_rft"},
                "correctness_contract": "heuristic",
                "committed_utc": "2026-03-16 00:00 UTC",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rft_runtime_loop,
        "validate_fixed_eval_task_pool",
        lambda **_kwargs: ("task-new",),
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "_load_tokenizer",
        lambda _model_path: pytest.fail("tokenizer should not load after eval-pool mismatch"),
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
        output_dir=output_dir,
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
        eval_task_count=50,
    )

    with pytest.raises(RuntimeError, match="eval task pool changed across resume"):
        rft_runtime_loop.run_rft_runtime_loop(config)


def test_run_loop_rejects_stage_switch_when_resuming_latest_committed_checkpoint(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runtime"
    previous_step_dir = output_dir / "rft_step_00000"
    previous_hf = previous_step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface"
    previous_vllm = previous_step_dir / "trainer_checkpoints" / "global_step_1" / "huggingface_vllm_merged"
    previous_hf.mkdir(parents=True)
    previous_vllm.mkdir(parents=True)
    (output_dir / rft_runtime_loop._RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME).write_text(
        json.dumps(
            {
                "stage": "format_rft",
                "committed_step_index": 0,
                "latest_hf_checkpoint": str(previous_hf),
                "latest_vllm_checkpoint": str(previous_vllm),
                "resume_model_path": str(previous_vllm),
                "selection_contract": {"mode": "format_first_rft"},
                "correctness_contract": "heuristic",
                "committed_utc": "2026-03-16 00:00 UTC",
            }
        ),
        encoding="utf-8",
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
        output_dir=output_dir,
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
        eval_task_count=0,
        stage_name="positive_rft",
    )

    with pytest.raises(ValueError, match="different stage_name"):
        rft_runtime_loop.run_rft_runtime_loop(config)


def test_run_loop_resume_fails_closed_when_latest_commit_is_incomplete(tmp_path: Path) -> None:
    output_dir = tmp_path / "runtime"
    output_dir.mkdir(parents=True)
    missing_path = output_dir / "missing-model"
    (output_dir / rft_runtime_loop._RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME).write_text(
        json.dumps(
            {
                "stage": "format_rft",
                "committed_step_index": 0,
                "latest_hf_checkpoint": str(missing_path),
                "latest_vllm_checkpoint": str(missing_path),
                "resume_model_path": str(missing_path),
                "selection_contract": {"mode": "format_first_rft"},
                "correctness_contract": "heuristic",
                "committed_utc": "2026-03-16 00:00 UTC",
            }
        ),
        encoding="utf-8",
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
        output_dir=output_dir,
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
        eval_task_count=0,
    )

    with pytest.raises(RuntimeError, match="Latest committed checkpoint is incomplete"):
        rft_runtime_loop.run_rft_runtime_loop(config)


def test_load_existing_runtime_manifest_recovers_missing_committed_step_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runtime"
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / rft_runtime_loop._RFT_RUNTIME_LOOP_MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-16 00:00 UTC",
                "config": {"rft_steps": 2},
                "steps": [{"step_index": 0, "selected_count": 1}],
            }
        ),
        encoding="utf-8",
    )
    recovered_step_dir = output_dir / "rft_step_00001"
    recovered_step_dir.mkdir(parents=True)
    (recovered_step_dir / "rft_step_summary.json").write_text(
        json.dumps({"step_index": 1, "selected_count": 2}),
        encoding="utf-8",
    )

    manifest = rft_runtime_loop._load_existing_runtime_manifest(
        output_dir=output_dir,
        default_config={"rft_steps": 2},
        committed_step_index=1,
    )

    assert [step["step_index"] for step in manifest["steps"]] == [0, 1]


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

    def _fake_write_selected_rows(_rows, parquet_path: Path, **_kwargs):
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
        eval_task_count=0,
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
        "tokenizer_model_paths": [],
        "collect_tokenizer_paths": [],
        "cache_write_model_paths": [],
        "cache_write_fingerprints": [],
        "trainer_cache_fingerprints": [],
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

    class _NamedStubTokenizer(_StubTokenizer):
        def __init__(self, model_path: str) -> None:
            self.name_or_path = model_path

    def _fake_load_tokenizer(model_path: str):
        tokenizer_model_paths = call_state["tokenizer_model_paths"]
        assert isinstance(tokenizer_model_paths, list)
        tokenizer_model_paths.append(str(model_path))
        return _NamedStubTokenizer(str(model_path))

    def _fake_collect(*, request, tokenizer):
        collect_tokenizer_paths = call_state["collect_tokenizer_paths"]
        assert isinstance(collect_tokenizer_paths, list)
        collect_tokenizer_paths.append(str(tokenizer.name_or_path))
        step = call_state["collect_calls"]
        assert isinstance(step, int)
        call_state["collect_calls"] = step + 1
        assert request.start_step_index == step
        return {"selected_rows": [_selected_row(step)], "rejected_rows": []}

    def _fake_write_selected_rows(_rows, parquet_path: Path, **kwargs):
        cache_write_model_paths = call_state["cache_write_model_paths"]
        assert isinstance(cache_write_model_paths, list)
        cache_write_model_paths.append(str(kwargs["tokenizer"].name_or_path))
        cache_write_fingerprints = call_state["cache_write_fingerprints"]
        assert isinstance(cache_write_fingerprints, list)
        cache_write_fingerprints.append(str(kwargs["cache_fingerprint"]))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_model_paths = call_state["trainer_model_paths"]
        assert isinstance(trainer_model_paths, list)
        trainer_model_paths.append(str(kwargs["model_path"]))
        trainer_cache_fingerprints = call_state["trainer_cache_fingerprints"]
        assert isinstance(trainer_cache_fingerprints, list)
        trainer_cache_fingerprints.append(str(kwargs["token_cache_fingerprint"]))
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
        eval_task_count=0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    trainer_model_paths = call_state["trainer_model_paths"]
    assert isinstance(trainer_model_paths, list)
    assert len(trainer_model_paths) == 2
    assert trainer_model_paths[0] == "Qwen/Qwen3-0.6B"
    assert trainer_model_paths[1].endswith("huggingface_vllm_merged")
    tokenizer_model_paths = call_state["tokenizer_model_paths"]
    assert isinstance(tokenizer_model_paths, list)
    assert tokenizer_model_paths == trainer_model_paths
    collect_tokenizer_paths = call_state["collect_tokenizer_paths"]
    assert isinstance(collect_tokenizer_paths, list)
    assert collect_tokenizer_paths == trainer_model_paths
    cache_write_model_paths = call_state["cache_write_model_paths"]
    assert isinstance(cache_write_model_paths, list)
    assert cache_write_model_paths == trainer_model_paths
    cache_write_fingerprints = call_state["cache_write_fingerprints"]
    trainer_cache_fingerprints = call_state["trainer_cache_fingerprints"]
    assert isinstance(cache_write_fingerprints, list)
    assert isinstance(trainer_cache_fingerprints, list)
    assert trainer_cache_fingerprints == cache_write_fingerprints
    assert len(set(cache_write_fingerprints)) == 2

    controllers = call_state["controllers"]
    assert isinstance(controllers, list)
    assert len(controllers) == 1
    controller = controllers[0]
    assert controller.start_calls[0] == "Qwen/Qwen3-0.6B"
    assert controller.start_calls[1].endswith("huggingface_vllm_merged")
    summary_0 = json.loads(
        (config.output_dir / "rft_step_00000" / "rft_step_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary_1 = json.loads(
        (config.output_dir / "rft_step_00001" / "rft_step_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_0["token_cache_model_path"] == trainer_model_paths[0]
    assert summary_1["token_cache_model_path"] == trainer_model_paths[1]
    assert summary_0["token_cache_fingerprint"] == cache_write_fingerprints[0]
    assert summary_1["token_cache_fingerprint"] == cache_write_fingerprints[1]


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

    def _fake_write_selected_rows(rows, parquet_path: Path, **_kwargs):
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
        eval_task_count=0,
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


def test_run_loop_collects_outer_eval_without_inner_sft_val_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_calls: list[tuple[str, int]] = []
    filter_kwargs_calls: list[dict[str, object]] = []
    request_partitions: list[str] = []

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
        request_partitions.append(request.task_partition)
        assert request.task_eval_split_fraction == 0.25
        assert request.task_eval_min_rows == 1
        assert request.task_eval_task_count == 1
        if request.task_partition == "train":
            return {
                "selected_rows": [
                    _selected_row("task-1"),
                    _selected_row("task-2"),
                    _selected_row("task-3"),
                ],
                "rejected_rows": [],
            }
        assert request.task_partition == "eval"
        assert request.runtime_overrides["task_batch_size"] == 1
        assert request.runtime_overrides["attempts_per_task"] == 1
        return {
            "selected_rows": [_selected_row("task-4")],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path, **_kwargs):
        write_calls.append((parquet_path.name, len(rows)))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

    def _fake_build_trainer_step_command(**kwargs):
        assert Path(kwargs["train_parquet_path"]).name == "accepted_trajectories.parquet"
        assert "val_parquet_path" not in kwargs
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
    monkeypatch.setattr(
        rft_runtime_loop,
        "validate_fixed_eval_task_pool",
        lambda **_kwargs: ("task-4",),
    )
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
    monkeypatch.setattr(
        rft_runtime_loop,
        "filter_selected_rows_by_token_length",
        lambda *, selected_rows, tokenizer, max_sequence_length, **kwargs: (
            filter_kwargs_calls.append(dict(kwargs)) or list(selected_rows),
            0,
        ),
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
        trainer_overrides=("++data.apply_chat_template_kwargs.enable_thinking=false",),
        dry_run=False,
        eval_split_fraction=0.25,
        eval_min_rows=1,
        eval_task_count=1,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert request_partitions == ["train", "eval"]
    assert write_calls == [("accepted_trajectories.parquet", 3)]
    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count_raw"] == 3
    assert summary["selected_count_for_train_raw"] == 3
    assert summary["selected_count_for_train"] == 3
    assert summary["selected_count_for_eval"] == 1
    assert summary["eval_selected_count_raw"] == 1
    assert summary["eval_task_count"] == 1
    assert summary["eval_split_fallback_to_train"] is False
    assert summary["eval_parquet"] is None
    assert summary["outer_eval_artifact_dir"].endswith("collector_artifacts/eval")
    assert filter_kwargs_calls == [
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ]


def test_run_loop_validates_fixed_eval_task_pool_before_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def _fake_validate_fixed_eval_task_pool(**kwargs):
        events.append("validate")
        assert kwargs["data_config_name"] == "on_policy_swe_smith"
        assert kwargs["eval_task_count"] == 50
        return tuple(f"task-{index}" for index in range(50))

    def _fake_load_tokenizer(_model_path: str):
        events.append("tokenizer")
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(
        rft_runtime_loop,
        "validate_fixed_eval_task_pool",
        _fake_validate_fixed_eval_task_pool,
    )
    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)

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
        eval_min_rows=0,
        eval_task_count=50,
    )

    with pytest.raises(RuntimeError, match="stop after validation"):
        rft_runtime_loop.run_rft_runtime_loop(config)

    assert events == ["validate", "tokenizer"]


def test_run_loop_keeps_outer_eval_out_of_inner_sft_batching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_calls: list[tuple[str, int]] = []
    request_partitions: list[str] = []

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
        request_partitions.append(request.task_partition)
        assert request.task_eval_split_fraction == 0.2
        assert request.task_eval_min_rows == 1
        if request.task_partition == "train":
            return {
                "selected_rows": [
                    _selected_row("task-1"),
                    _selected_row("task-2"),
                    _selected_row("task-3"),
                    _selected_row("task-4"),
                ],
                "rejected_rows": [],
            }
        assert request.task_partition == "eval"
        return {
            "selected_rows": [_selected_row("task-5")],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path, **_kwargs):
        write_calls.append((parquet_path.name, len(rows)))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

    def _fake_build_trainer_step_command(**kwargs):
        assert kwargs["train_batch_size"] == 2
        assert Path(kwargs["train_parquet_path"]).name == "accepted_trajectories.parquet"
        assert "val_parquet_path" not in kwargs
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

    assert request_partitions == ["train", "eval"]
    assert write_calls == [("accepted_trajectories.parquet", 4)]
    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count_raw"] == 4
    assert summary["selected_count_for_train_raw"] == 4
    assert summary["selected_count_for_train"] == 4
    assert summary["selected_rows_upsampled"] == 0
    assert summary["selected_count_for_eval_raw"] == 1
    assert summary["selected_count_for_eval"] == 1
    assert summary["eval_selected_count_raw"] == 1
    assert summary["selected_rows_eval_upsampled"] == 0
    assert summary["effective_eval_batch_size"] is None
    assert summary["eval_split_fallback_to_train"] is False


def test_run_loop_rejects_empty_eval_selection_without_train_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_calls: list[tuple[str, int]] = []
    request_partitions: list[str] = []

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
        request_partitions.append(request.task_partition)
        if request.task_partition == "train":
            return {
                "selected_rows": [
                    _selected_row("task-1"),
                    _selected_row("task-2"),
                ],
                "rejected_rows": [],
            }
        assert request.task_partition == "eval"
        return {
            "selected_rows": [],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path, **_kwargs):
        write_calls.append((parquet_path.name, len(rows)))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

    def _fake_build_trainer_step_command(**kwargs):
        raise AssertionError("trainer command should not be built when held-out eval is empty")

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
        "resolve_micro_batch_size_per_gpu",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "resolve_data_max_length",
        lambda **_kwargs: 4096,
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "filter_selected_rows_by_token_length",
        lambda *, selected_rows, tokenizer, max_sequence_length, **_kwargs: (
            list(selected_rows),
            0,
        ),
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "compute_average_generation_length",
        lambda *, selected_rows, tokenizer: 0.0,
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

    with pytest.raises(RuntimeError, match="outer-step eval telemetry"):
        rft_runtime_loop.run_rft_runtime_loop(config)

    assert request_partitions == ["train", "eval"]
    assert write_calls == [("accepted_trajectories.parquet", 2)]


def test_run_loop_positive_stage_requests_resolved_only_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[tuple[str, bool, dict[str, object] | None]] = []

    def _selected_row(task_id: str) -> dict[str, object]:
        return {
            "task_id": task_id,
            "attempt_index": 0,
            "step_index": 0,
            "turn_index": 0,
            "resolved": True,
            "format_valid": False,
            "final_turn_has_submit": True,
            "final_submit_format_valid": False,
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
        handoff = request.handoff_overrides
        assert isinstance(handoff, dict)
        assert request.task_eval_task_count == 1
        if request.task_partition == "eval":
            assert request.runtime_overrides["task_batch_size"] == 1
            assert request.runtime_overrides["attempts_per_task"] == 1
        captured_requests.append(
            (request.task_partition, request.verify_submissions, request.stage_name, handoff)
        )
        return {
            "selected_rows": [_selected_row(f"{request.task_partition}-task")],
            "rejected_rows": [],
        }

    def _fake_write_selected_rows(rows, parquet_path: Path, **_kwargs):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return len(rows)

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

    monkeypatch.setattr(rft_runtime_loop, "_load_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(
        rft_runtime_loop,
        "validate_fixed_eval_task_pool",
        lambda **_kwargs: ("eval-task",),
    )
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
        eval_task_count=1,
        stage_name="positive_rft",
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    assert captured_requests == [
        (
            "train",
            True,
            "positive_rft",
            {
                "selection": {
                    "require_terminal": False,
                    "require_format_valid": False,
                    "require_resolved": True,
                    "reject_on_invalid_final_submit": False,
                }
            },
        ),
        (
            "eval",
            True,
            "positive_rft",
            {
                "selection": {
                    "require_terminal": False,
                    "require_format_valid": False,
                    "require_resolved": True,
                    "reject_on_invalid_final_submit": False,
                }
            },
        ),
    ]
    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["stage"] == "positive_rft"
    assert summary["selection_contract"] == {
        "mode": "positive_rft",
        "require_terminal": False,
        "require_format_valid": False,
        "require_resolved": True,
        "reject_on_invalid_final_submit": False,
    }
    assert summary["correctness_contract"] == "verifier"


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


def test_resolve_rft_stage_helpers_cover_format_and_positive_modes() -> None:
    assert rft_runtime_loop.resolve_rft_stage_name("format") == "format_rft"
    assert rft_runtime_loop.resolve_rft_stage_name("positive") == "positive_rft"
    assert rft_runtime_loop.resolve_rft_stage_verify_submissions("format_rft") is False
    assert rft_runtime_loop.resolve_rft_stage_verify_submissions("positive_rft") is True
    assert rft_runtime_loop.resolve_rft_stage_selection_contract("format_rft") == {
        "mode": "format_first_rft",
        "require_terminal": True,
        "require_format_valid": True,
    }
    assert rft_runtime_loop.resolve_rft_stage_selection_contract("positive_rft") == {
        "mode": "positive_rft",
        "require_terminal": False,
        "require_format_valid": False,
        "require_resolved": True,
        "reject_on_invalid_final_submit": False,
    }
    assert rft_runtime_loop.resolve_rft_stage_correctness_contract("format_rft") == "heuristic"
    assert rft_runtime_loop.resolve_rft_stage_correctness_contract("positive_rft") == "verifier"
    assert rft_runtime_loop.resolve_rft_stage_handoff_overrides("format_rft") == {}
    assert rft_runtime_loop.resolve_rft_stage_handoff_overrides("positive_rft") == {
        "selection": {
            "require_terminal": False,
            "require_format_valid": False,
            "require_resolved": True,
            "reject_on_invalid_final_submit": False,
        }
    }


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


def test_resolve_apply_chat_template_kwargs_prefers_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "rft_swe.yaml").write_text(
        "data:\n"
        "  apply_chat_template_kwargs:\n"
        "    enable_thinking: true\n"
        "    custom_flag: keep\n",
        encoding="utf-8",
    )

    resolved = resolve_apply_chat_template_kwargs(
        config_dir=config_dir,
        config_name="rft_swe",
        trainer_overrides=(
            "++data.apply_chat_template_kwargs.enable_thinking=false",
            "+data.apply_chat_template_kwargs.extra=7",
        ),
    )

    assert resolved == {
        "enable_thinking": False,
        "custom_flag": "keep",
        "extra": 7,
    }


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
    assert kept_rows[0]["selected_token_count"] > 0
    assert kept_rows[0]["selected_over_budget"] is False


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


def test_filter_selected_rows_by_token_length_uses_chat_template_kwargs() -> None:
    captured_kwargs: list[dict[str, object]] = []

    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            add_generation_prompt: bool = False,
            tokenize: bool = True,
            return_dict: bool = True,
            **kwargs,
        ):
            del add_generation_prompt, tokenize
            captured_kwargs.append(dict(kwargs))
            token_count = sum(len(str(message.get("content", ""))) for message in messages)
            input_ids = list(range(token_count))
            if return_dict:
                return {"input_ids": input_ids}
            return input_ids

    kept_rows, dropped_count = filter_selected_rows_by_token_length(
        selected_rows=[
            {
                "task_id": "task-1",
                "prompt": "Fix bug",
                "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
            }
        ],
        tokenizer=_FakeTokenizer(),
        max_sequence_length=512,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert len(kept_rows) == 1
    assert dropped_count == 0
    assert captured_kwargs == [{"enable_thinking": False}]


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
            "print('step:2 - train/loss:0.40 - train/lr(1e-3):0.05')\n"
            "print('step:3 - train/loss:0.42 - train/lr(1e-3):0.0')\n"
            "print('step:2 - val/loss:0.48')\n"
            "print('step:3 - val/loss:0.44')\n"
        ),
    ]

    metrics = rft_runtime_loop._run_command(command, cwd=tmp_path)

    assert metrics["train_step_first"] == 1
    assert metrics["train_loss_first"] == pytest.approx(0.45)
    assert metrics["train_step_last"] == 3
    assert metrics["train_loss_last"] == pytest.approx(0.42)
    assert metrics["train_loss_min"] == pytest.approx(0.40)
    assert metrics["train_loss_min_step"] == 2
    assert metrics["train_loss_delta"] == pytest.approx(-0.03)
    assert metrics["val_step_first"] == 2
    assert metrics["val_loss_first"] == pytest.approx(0.48)
    assert metrics["val_step_last"] == 3
    assert metrics["val_loss_last"] == pytest.approx(0.44)
    assert metrics["val_loss_min"] == pytest.approx(0.44)
    assert metrics["val_loss_min_step"] == 3
    assert metrics["val_loss_delta"] == pytest.approx(-0.04)


def test_run_loop_writes_inner_loss_delta_metrics_to_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_load_tokenizer(_model_path: str):
        return _StubTokenizer()

    def _fake_collect(*, request, tokenizer):
        del request, tokenizer
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

    def _fake_write_selected_rows(_rows, parquet_path: Path, **_kwargs):
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("stub", encoding="utf-8")
        return 1

    def _fake_build_trainer_step_command(**kwargs):
        trainer_output_dir = Path(kwargs["trainer_output_dir"])
        return ["fake-trainer", str(trainer_output_dir)]

    def _fake_run_command(command, *, cwd: Path):
        del cwd
        trainer_output_dir = Path(command[1])
        (trainer_output_dir / "global_step_2" / "huggingface").mkdir(parents=True, exist_ok=True)
        return {
            "train_step_first": 1,
            "train_loss_first": 0.45,
            "train_step_last": 2,
            "train_loss_last": 0.40,
            "train_loss_min": 0.40,
            "train_loss_min_step": 2,
            "train_loss_delta": -0.05,
            "val_step_first": 1,
            "val_loss_first": 0.50,
            "val_step_last": 2,
            "val_loss_last": 0.46,
            "val_loss_min": 0.46,
            "val_loss_min_step": 2,
            "val_loss_delta": -0.04,
        }

    def _fake_resolve_latest_hf_checkpoint(checkpoint_root: Path):
        target = Path(checkpoint_root) / "global_step_2" / "huggingface"
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
        eval_split_fraction=0.0,
        eval_task_count=0,
    )

    rft_runtime_loop.run_rft_runtime_loop(config)

    summary_path = config.output_dir / "rft_step_00000" / "rft_step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["inner_train_loss_first"] == pytest.approx(0.45)
    assert summary["inner_train_loss_last"] == pytest.approx(0.40)
    assert summary["inner_train_loss_delta"] == pytest.approx(-0.05)
    assert summary["inner_train_loss_min"] == pytest.approx(0.40)
    assert summary["inner_train_loss_min_step"] == 2
    assert summary["inner_val_loss_first"] == pytest.approx(0.50)
    assert summary["inner_val_loss_last"] == pytest.approx(0.46)
    assert summary["inner_val_loss_delta"] == pytest.approx(-0.04)
    assert summary["inner_val_loss_min"] == pytest.approx(0.46)
    assert summary["inner_val_loss_min_step"] == 2


def test_run_command_starts_new_session_and_cleans_up_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_popen_kwargs: dict[str, object] = {}
    cleanup_calls: list[tuple[int, float]] = []

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 4321
            self.stdout = io.StringIO("step:1 - train/loss:0.45 - train/lr(1e-3):0.1\n")

        def wait(self) -> int:
            return 0

    def _fake_popen(*args, **kwargs):
        del args
        captured_popen_kwargs.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(rft_runtime_loop.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        rft_runtime_loop,
        "_cleanup_process_group",
        lambda process_group_id, *, timeout_sec: cleanup_calls.append((process_group_id, timeout_sec)),
    )

    metrics = rft_runtime_loop._run_command(["python3", "-m", "trainer"], cwd=tmp_path)

    assert captured_popen_kwargs["start_new_session"] is True
    assert cleanup_calls == [(4321, rft_runtime_loop._DEFAULT_PROCESS_GROUP_CLEANUP_TIMEOUT_SEC)]
    assert metrics["train_step_last"] == 1
    assert metrics["train_loss_last"] == pytest.approx(0.45)


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


def test_build_models_url_normalizes_bare_host_and_existing_api_paths() -> None:
    assert rft_runtime_loop._build_models_url("http://127.0.0.1:8000") == (
        "http://127.0.0.1:8000/v1/models"
    )
    assert rft_runtime_loop._build_models_url("http://127.0.0.1:8000/v1") == (
        "http://127.0.0.1:8000/v1/models"
    )
    assert rft_runtime_loop._build_models_url(
        "http://127.0.0.1:8000/v1/chat/completions"
    ) == "http://127.0.0.1:8000/v1/models"


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


def test_cleanup_process_group_escalates_from_sigterm_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals_sent: list[tuple[int, signal.Signals]] = []
    wait_responses = iter((False, False, True))

    monkeypatch.setattr(
        rft_runtime_loop,
        "_wait_for_process_group_exit",
        lambda process_group_id, *, timeout_sec: next(wait_responses),
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "_signal_process_group",
        lambda process_group_id, sig: signals_sent.append((process_group_id, sig)),
    )

    rft_runtime_loop._cleanup_process_group(9876, timeout_sec=0.5)

    assert signals_sent == [
        (9876, signal.SIGTERM),
        (9876, signal.SIGKILL),
    ]


def test_vllm_controller_uses_new_session_and_process_group_cleanup(
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
    captured_popen_kwargs: dict[str, object] = {}
    cleanup_calls: list[tuple[int, int]] = []
    snapshot_labels: list[str] = []

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 6543
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    def _fake_popen(*args, **kwargs):
        del args
        captured_popen_kwargs.update(kwargs)
        return _FakeProcess()

    def _fake_ready(url, *, api_key=None, expected_model_name=None):
        del url, api_key
        return expected_model_name is not None

    monkeypatch.setattr(rft_runtime_loop.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(rft_runtime_loop, "_is_http_endpoint_ready", _fake_ready)
    monkeypatch.setattr(
        rft_runtime_loop,
        "_cleanup_process_group",
        lambda process_group_id, *, timeout_sec: cleanup_calls.append((process_group_id, timeout_sec)),
    )
    monkeypatch.setattr(
        rft_runtime_loop,
        "_append_vllm_debug_snapshot",
        lambda *, log_path, label: snapshot_labels.append(label),
    )

    controller.start(model_path="/tmp/model")
    controller.stop()

    assert captured_popen_kwargs["start_new_session"] is True
    assert cleanup_calls == [(6543, 1)]
    assert snapshot_labels == ["pre-launch GPU snapshot for model=/tmp/model"]


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
