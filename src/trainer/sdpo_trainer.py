"""Deterministic trainer scaffolds delegating to integration-layer adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from verl_integration.mask_injector import inject_response_mask
from verl_integration.reward_function import reward_fn


@dataclass(frozen=True)
class SDPOTrainerConfig:
    model_name: str
    max_tool_calls_per_turn: int = 3
    include_student_attempt_for_teacher: bool = True
    top_k_distillation: int = 100
    ema_beta: float = 0.005


@dataclass(frozen=True)
class TrainingStepStats:
    loss: float
    teacher_student_kl: float
    format_valid_rate: float


class SDPOTrainerScaffold:
    """Deterministic trainer facade until full verl runtime wiring lands."""

    def __init__(self, config: SDPOTrainerConfig) -> None:
        self._config = config

    @property
    def config(self) -> SDPOTrainerConfig:
        return self._config

    def run_rft_epoch(self, batch: Sequence[Mapping[str, Any]]) -> TrainingStepStats:
        """Run a deterministic RFT pass over pre-labeled records.

        This method computes stage masks and returns aggregate statistics only;
        it does not execute model optimization.
        """
        if not batch:
            return TrainingStepStats(loss=0.0, teacher_student_kl=0.0, format_valid_rate=0.0)

        masked_batch = inject_response_mask(batch, stage="rft")
        token_totals = [len(sample["response_mask"]) for sample in masked_batch]
        trained_totals = [sum(1 for flag in sample["response_mask"] if flag) for sample in masked_batch]

        average_train_fraction = sum(
            trained / total for trained, total in zip(trained_totals, token_totals) if total > 0
        ) / len(masked_batch)
        format_valid_rate = sum(1.0 for sample in batch if sample.get("format_valid", True)) / len(batch)

        return TrainingStepStats(
            loss=1.0 - average_train_fraction,
            teacher_student_kl=0.0,
            format_valid_rate=format_valid_rate,
        )

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
        }
        for metric_name, threshold in required.items():
            if rollout_stats.get(metric_name, 0.0) < threshold:
                return False
        return True
