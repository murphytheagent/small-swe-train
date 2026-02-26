from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from config import DEFAULT_TRAINING_MODEL_NAME, resolve_on_policy_settings
from env.runtime_protocol import ToolRequest, ToolResponse
from rollout.onpolicy_collector import OnPolicyRolloutCollector
from trainer.sdpo_trainer import SDPOTrainerConfig, SDPOTrainerScaffold


@dataclass
class FakeExecutor:
    requests: list[ToolRequest]

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


class _CharTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        if isinstance(text, list):
            input_ids_batch = []
            offsets_batch = []
            for item in text:
                encoded = self(item, return_offsets_mapping=return_offsets_mapping)
                input_ids_batch.append(encoded["input_ids"])
                offsets_batch.append(encoded["offset_mapping"])
            return {"input_ids": input_ids_batch, "offset_mapping": offsets_batch}

        input_ids = [index + 1 for index, _char in enumerate(text)]
        offsets = [(index, index + 1) for index, _char in enumerate(text)]
        return {"input_ids": input_ids, "offset_mapping": offsets}


class _FakePool:
    def acquire(self, tasks):
        from env.container_pool import ContainerHandle

        return (
            ContainerHandle(
                task_id=tasks[0].task_id,
                image_name=tasks[0].image_name,
                container_id="cid-1",
                container_name="cname-1",
            ),
        )

    def release_all(self) -> None:
        return None


class _FakeCollectorExecutor:
    def run(self, request):
        from env.runtime_protocol import ToolResponse

        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


def _build_test_onpolicy_rft_collector() -> tuple[OnPolicyRolloutCollector, _CharTokenizer]:
    settings = resolve_on_policy_settings(
        data_config_name="on_policy_swe_smith",
        runtime_overrides={
            "enabled": True,
            "rollout_only": True,
            "task_batch_size": 1,
            "attempts_per_task": 2,
            "max_turns_per_attempt": 2,
            "env_pool_size": 1,
            "tool_timeout_sec": 1,
            "container_start_timeout_sec": 1,
            "attempt_timeout_sec": 10,
            "max_tool_calls_per_turn": 3,
        },
    )

    def turn_generator(**kwargs: object) -> str:
        attempt_index = int(kwargs["attempt_index"])
        turn_index = int(kwargs["turn_index"])
        if attempt_index == 0:
            if turn_index == 0:
                return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
            return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix bug",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
        ],
        pool_factory=lambda _runtime: _FakePool(),
        executor_factory=lambda _handle, _runtime: _FakeCollectorExecutor(),
    )
    return collector, _CharTokenizer()


def test_run_rft_epoch_computes_mask_based_stats() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))
    batch = [
        {"token_labels": ["think", "tool_call", "other"], "format_valid": True},
        {"token_labels": ["tool_call", "tool_call"], "format_valid": False},
    ]

    stats = trainer.run_rft_epoch(batch)

    assert 0.0 <= stats.loss <= 1.0
    assert stats.teacher_student_kl == 0.0
    assert stats.format_valid_rate == 0.5


def test_run_sdpo_step_uses_reward_fn_metrics() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))
    batch = [
        {
            "response_text": (
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ),
            "resolved": True,
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
                }
            },
        }
    ]

    stats = trainer.run_sdpo_step(batch)

    assert stats.loss == 0.0
    assert stats.teacher_student_kl == 0.0
    assert stats.format_valid_rate == 1.0


def test_run_end_to_end_global_step_exposes_reprompt_and_ema_artifacts() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME, ema_beta=0.5))
    executor = FakeExecutor(requests=[])
    batch = [
        {
            "prompt": "Fix test failure",
            "response_text": (
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
            ),
            "resolved": True,
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
                }
            },
        }
    ]

    artifacts = trainer.run_end_to_end_global_step(batch, executor=executor)

    assert artifacts.training_stats.loss == 0.0
    assert artifacts.rewards == (1.0,)
    assert artifacts.self_distillation_mask == (True,)
    assert artifacts.teacher_ema_proxy == 0.5
    assert artifacts.loss_history == (0.0,)
    assert artifacts.rollout_tool_response_blocks
    assert executor.requests == []


def test_evaluate_format_gates_requires_all_thresholds() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))
    assert trainer.evaluate_format_gates(
        {
            "parse_valid_rate": 0.99,
            "allowed_tool_rate": 1.0,
            "required_arg_presence": 1.0,
            "tool_call_block_presence_rate": 1.0,
            "tool_call_count_valid_rate": 1.0,
            "submit_singleton_rule_rate": 1.0,
            "thinking_delimiter_balance_rate": 1.0,
            "terminal_submission_rate": 1.0,
        }
    )

    assert not trainer.evaluate_format_gates(
        {
            "parse_valid_rate": 0.9,
            "allowed_tool_rate": 1.0,
            "required_arg_presence": 1.0,
            "tool_call_block_presence_rate": 1.0,
            "tool_call_count_valid_rate": 1.0,
            "submit_singleton_rule_rate": 1.0,
            "thinking_delimiter_balance_rate": 1.0,
            "terminal_submission_rate": 1.0,
        }
    )

    assert not trainer.evaluate_format_gates(
        {
            "parse_valid_rate": 1.0,
            "allowed_tool_rate": 1.0,
            "required_arg_presence": 1.0,
            "tool_call_block_presence_rate": 1.0,
            "tool_call_count_valid_rate": 1.0,
            "submit_singleton_rule_rate": 1.0,
            "thinking_delimiter_balance_rate": 1.0,
            "terminal_submission_rate": 0.5,
        }
    )


def test_run_onpolicy_rft_step_uses_centralized_handoff_filter() -> None:
    collector, tokenizer = _build_test_onpolicy_rft_collector()

    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))
    artifacts = trainer.run_onpolicy_rft_step(
        total_steps=1,
        collector=collector,
        tokenizer=tokenizer,
    )

    assert artifacts.selected_count == 1
    assert artifacts.rejected_count == 1
    assert artifacts.training_stats.format_valid_rate == 1.0
    assert artifacts.dataproto_payload["meta_info"]["selected_count"] == 1
    assert artifacts.checkpoint_exists is False
    assert artifacts.checkpoint_dir is None


def test_run_onpolicy_rft_step_writes_checkpoint_manifest(tmp_path: Path) -> None:
    collector, tokenizer = _build_test_onpolicy_rft_collector()
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))

    artifacts = trainer.run_onpolicy_rft_step(
        total_steps=1,
        collector=collector,
        tokenizer=tokenizer,
        checkpoint_dir=tmp_path / "checkpoints",
        global_step=10,
    )

    assert artifacts.checkpoint_exists is True
    assert artifacts.checkpoint_dir is not None

    checkpoint_dir = Path(artifacts.checkpoint_dir)
    manifest_path = checkpoint_dir / "rft_step_manifest.json"
    latest_pointer = checkpoint_dir.parent / "latest_checkpoint.txt"

    assert checkpoint_dir.name == "global_step_10"
    assert manifest_path.exists()
    assert latest_pointer.exists()
    assert latest_pointer.read_text(encoding="utf-8") == str(checkpoint_dir)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["global_step"] == 10
    assert payload["selected_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["training_stats"]["format_valid_rate"] == 1.0


def test_run_onpolicy_rft_step_checkpoint_requires_explicit_global_step(tmp_path: Path) -> None:
    collector, tokenizer = _build_test_onpolicy_rft_collector()
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))

    with pytest.raises(
        ValueError,
        match="global_step is required when writing RFT checkpoint artifacts",
    ):
        trainer.run_onpolicy_rft_step(
            total_steps=1,
            collector=collector,
            tokenizer=tokenizer,
            checkpoint_dir=tmp_path / "checkpoints",
        )


def test_run_onpolicy_rft_step_checkpoint_validation_fails_before_rollout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))

    def _unexpected_collect(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("collect_rft_sft_batch_for_steps should not run")

    monkeypatch.setattr("trainer.rft_trainer.collect_rft_sft_batch_for_steps", _unexpected_collect)

    with pytest.raises(
        ValueError,
        match="global_step is required when writing RFT checkpoint artifacts",
    ):
        trainer.run_onpolicy_rft_step(
            total_steps=1,
            collector=object(),
            tokenizer=object(),
            checkpoint_dir=tmp_path / "checkpoints",
        )
