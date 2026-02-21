"""SDPO trainer scaffold with stable interfaces for future integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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
    """Interface-only scaffold for RFT/SDFT/SDPO stages."""

    def __init__(self, config: SDPOTrainerConfig) -> None:
        self._config = config

    @property
    def config(self) -> SDPOTrainerConfig:
        return self._config

    def run_rft_epoch(self, batch: Sequence[Mapping[str, Any]]) -> TrainingStepStats:
        """Run one supervised RFT epoch over a prepared batch."""
        raise NotImplementedError("RFT trainer implementation is pending.")

    def run_sdpo_step(self, batch: Sequence[Mapping[str, Any]]) -> TrainingStepStats:
        """Run one step-SDPO update over on-policy trajectories."""
        raise NotImplementedError("SDPO trainer implementation is pending.")

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
