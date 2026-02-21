from __future__ import annotations

from dataclasses import dataclass

from env.runtime_protocol import ToolRequest, ToolResponse
from trainer.sdpo_trainer import SDPOTrainerConfig, SDPOTrainerScaffold


@dataclass
class FakeExecutor:
    requests: list[ToolRequest]

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


def test_run_rft_epoch_computes_mask_based_stats() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name="Qwen/Qwen3-4B"))
    batch = [
        {"token_labels": ["think", "tool_call", "other"], "format_valid": True},
        {"token_labels": ["tool_call", "tool_call"], "format_valid": False},
    ]

    stats = trainer.run_rft_epoch(batch)

    assert 0.0 <= stats.loss <= 1.0
    assert stats.teacher_student_kl == 0.0
    assert stats.format_valid_rate == 0.5


def test_run_sdpo_step_uses_reward_fn_metrics() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name="Qwen/Qwen3-4B"))
    batch = [
        {
            "response_text": (
                "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"foo\"}}</tool_call>"
            ),
            "resolved": True,
        }
    ]

    stats = trainer.run_sdpo_step(batch)

    assert stats.loss == 0.0
    assert stats.teacher_student_kl == 0.0
    assert stats.format_valid_rate == 1.0


def test_run_end_to_end_global_step_exposes_reprompt_and_ema_artifacts() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name="Qwen/Qwen3-4B", ema_beta=0.5))
    executor = FakeExecutor(requests=[])
    batch = [
        {
            "prompt": "Fix test failure",
            "response_text": (
                "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"tests/test_math.py::test_add\"}}</tool_call>"
            ),
            "resolved": True,
        }
    ]

    artifacts = trainer.run_end_to_end_global_step(batch, executor=executor)

    assert artifacts.training_stats.loss == 0.0
    assert artifacts.rewards == (1.0,)
    assert artifacts.self_distillation_mask == (True,)
    assert artifacts.teacher_ema_proxy == 0.5
    assert artifacts.loss_history == (0.0,)
    assert artifacts.rollout_tool_response_blocks
    assert executor.requests[0].tool == "search"


def test_evaluate_format_gates_requires_all_thresholds() -> None:
    trainer = SDPOTrainerScaffold(SDPOTrainerConfig(model_name="Qwen/Qwen3-4B"))
    assert trainer.evaluate_format_gates(
        {
            "parse_valid_rate": 0.99,
            "allowed_tool_rate": 1.0,
            "required_arg_presence": 1.0,
            "tool_call_block_presence_rate": 1.0,
            "tool_call_count_valid_rate": 1.0,
            "submit_singleton_rule_rate": 1.0,
            "thinking_delimiter_balance_rate": 1.0,
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
        }
    )
