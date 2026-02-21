from __future__ import annotations

from trainer.sdpo_trainer import SDPOTrainerConfig, SDPOTrainerScaffold


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
