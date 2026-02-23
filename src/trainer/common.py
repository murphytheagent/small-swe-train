"""Shared trainer dataclasses used by RFT and SDPO scaffolds."""

from __future__ import annotations

from dataclasses import dataclass

from config import MAX_TOOL_CALLS_PER_TURN


@dataclass(frozen=True)
class SDPOTrainerConfig:
    model_name: str
    max_tool_calls_per_turn: int = MAX_TOOL_CALLS_PER_TURN
    include_student_attempt_for_teacher: bool = True
    top_k_distillation: int = 100
    ema_beta: float = 0.005


@dataclass(frozen=True)
class TrainingStepStats:
    loss: float
    teacher_student_kl: float
    format_valid_rate: float
