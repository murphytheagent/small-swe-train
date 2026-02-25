"""Deterministic SDPO trainer scaffold with delegated RFT utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.tokenization import SupportsOffsetsTokenizer
from rollout.onpolicy_collector import OnPolicyRolloutCollector
from trainer.common import SDPOTrainerConfig, TrainingStepStats
from trainer.rft_trainer import OnPolicyRFTStepArtifacts, RFTTrainerScaffold
from verl_integration.env_bridge import (
    ToolExecutor,
    build_tool_response_payload,
    run_env_bridge_step,
)
from verl_integration.reprompt_adapter import build_self_distillation_batch
from verl_integration.reward_function import reward_fn


@dataclass(frozen=True)
class EndToEndStepArtifacts:
    training_stats: TrainingStepStats
    rewards: tuple[float, ...]
    feedback: tuple[str, ...]
    teacher_prompts: tuple[str, ...]
    self_distillation_mask: tuple[bool, ...]
    format_metrics: Mapping[str, float]
    prompt_truncated: tuple[bool, ...]
    rollout_tool_response_blocks: tuple[tuple[str, ...], ...]
    teacher_ema_proxy: float
    loss_history: tuple[float, ...]


class SDPOTrainerScaffold:
    """Deterministic SDPO facade until full verl runtime wiring lands."""

    def __init__(self, config: SDPOTrainerConfig) -> None:
        self._config = config
        self._rft_trainer = RFTTrainerScaffold(config=config)
        self._teacher_ema_proxy = 0.0
        self._loss_history: list[float] = []

    @property
    def config(self) -> SDPOTrainerConfig:
        return self._config

    @property
    def rft_trainer(self) -> RFTTrainerScaffold:
        """Dedicated RFT trainer scaffold sharing the same model/runtime config."""
        return self._rft_trainer

    def run_rft_epoch(self, batch: Sequence[Mapping[str, Any]]) -> TrainingStepStats:
        """Compatibility shim that delegates deterministic RFT stats to RFTTrainerScaffold."""
        return self._rft_trainer.run_rft_epoch(batch)

    def run_sdpo_step(self, batch: Sequence[Mapping[str, Any]]) -> TrainingStepStats:
        """Run a deterministic SDPO-style step with reward-function diagnostics."""
        if not batch:
            return TrainingStepStats(loss=0.0, teacher_student_kl=0.0, format_valid_rate=0.0)

        rewards, info = reward_fn(
            batch,
            max_tool_calls=self._config.max_tool_calls_per_turn,
        )

        mean_reward = sum(rewards) / len(rewards)
        metrics_payload = info.get("format_metrics", [{}])[0]
        format_valid_rate = float(metrics_payload.get("parse_valid_rate", 0.0))

        # Placeholder scalar until full logits/teacher model wiring exists.
        teacher_student_kl = 1.0 - mean_reward

        return TrainingStepStats(
            loss=1.0 - mean_reward,
            teacher_student_kl=teacher_student_kl,
            format_valid_rate=format_valid_rate,
        )

    def run_end_to_end_global_step(
        self,
        batch: Sequence[Mapping[str, Any]],
        *,
        executor: ToolExecutor | None = None,
    ) -> EndToEndStepArtifacts:
        """Run deterministic rollout -> reward -> reprompt -> train -> EMA update."""
        if not batch:
            empty_stats = TrainingStepStats(loss=0.0, teacher_student_kl=0.0, format_valid_rate=0.0)
            return EndToEndStepArtifacts(
                training_stats=empty_stats,
                rewards=(),
                feedback=(),
                teacher_prompts=(),
                self_distillation_mask=(),
                format_metrics={},
                prompt_truncated=(),
                rollout_tool_response_blocks=(),
                teacher_ema_proxy=self._teacher_ema_proxy,
                loss_history=tuple(self._loss_history),
            )

        rollout_rows: list[dict[str, Any]] = []
        rollout_tool_response_blocks: list[tuple[str, ...]] = []

        for row_index, sample in enumerate(batch):
            row = dict(sample)
            response_text = str(row.get("response_text") or row.get("assistant_response") or "")
            row.setdefault("response_text", response_text)
            row.setdefault("assistant_response", response_text)

            tool_blocks: tuple[str, ...] = ()
            if executor is not None and response_text:
                try:
                    bridge_result = run_env_bridge_step(
                        response_text,
                        executor=executor,
                        max_tool_calls=self._config.max_tool_calls_per_turn,
                        step_index_start=row_index * self._config.max_tool_calls_per_turn,
                    )
                    tool_blocks = bridge_result.tool_response_blocks
                    if "tool_output" not in row and bridge_result.steps:
                        row["tool_output"] = build_tool_response_payload(bridge_result.steps[0].response)
                except ValueError as exc:
                    row["bridge_error"] = str(exc)

            rollout_tool_response_blocks.append(tool_blocks)
            rollout_rows.append(row)

        rewards, info = reward_fn(
            rollout_rows,
            max_tool_calls=self._config.max_tool_calls_per_turn,
        )
        reprompt_batch = build_self_distillation_batch(
            rollout_rows,
            include_student_attempt_for_teacher=self._config.include_student_attempt_for_teacher,
        )
        training_stats = self.run_sdpo_step(rollout_rows)

        mean_reward = sum(rewards) / len(rewards)
        self._teacher_ema_proxy = (
            (1.0 - self._config.ema_beta) * self._teacher_ema_proxy
            + self._config.ema_beta * mean_reward
        )
        self._loss_history.append(training_stats.loss)

        metrics = info.get("format_metrics", [{}])[0]
        format_metrics: Mapping[str, float] = {
            key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
        }

        return EndToEndStepArtifacts(
            training_stats=training_stats,
            rewards=tuple(rewards),
            feedback=tuple(str(item) for item in info.get("feedback", [])),
            teacher_prompts=tuple(str(item) for item in reprompt_batch["teacher_prompts"]),
            self_distillation_mask=tuple(bool(item) for item in reprompt_batch["self_distillation_mask"]),
            format_metrics=format_metrics,
            prompt_truncated=tuple(bool(item) for item in reprompt_batch["prompt_truncated"]),
            rollout_tool_response_blocks=tuple(rollout_tool_response_blocks),
            teacher_ema_proxy=self._teacher_ema_proxy,
            loss_history=tuple(self._loss_history),
        )

    def evaluate_format_gates(self, rollout_stats: Mapping[str, float]) -> bool:
        """Check whether rollout stats pass entry gates for main SDPO stage."""
        required = {
            "parse_valid_rate": 0.985,
            "allowed_tool_rate": 0.995,
            "required_arg_presence": 0.985,
            "tool_call_block_presence_rate": 0.98,
            "tool_call_count_valid_rate": 0.98,
            "submit_singleton_rule_rate": 0.995,
            "thinking_delimiter_balance_rate": 0.995,
            "terminal_submission_rate": 0.98,
        }
        for metric_name, threshold in required.items():
            if rollout_stats.get(metric_name, 0.0) < threshold:
                return False
        return True

    def run_onpolicy_rft_step(
        self,
        *,
        total_steps: int,
        collector: OnPolicyRolloutCollector,
        tokenizer: SupportsOffsetsTokenizer,
        handoff_overrides: Mapping[str, Any] | None = None,
        output_dir: str | None = None,
        checkpoint_dir: str | Path | None = None,
        global_step: int | None = None,
    ) -> OnPolicyRFTStepArtifacts:
        """Compatibility shim delegating RFT handoff + checkpoint logic to RFTTrainerScaffold."""
        return self._rft_trainer.run_onpolicy_rft_step(
            total_steps=total_steps,
            collector=collector,
            tokenizer=tokenizer,
            handoff_overrides=handoff_overrides,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
        )
