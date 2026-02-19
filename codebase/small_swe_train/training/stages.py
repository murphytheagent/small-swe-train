from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Protocol

from small_swe_train.config import (
    FormatBootstrapConfig,
    SdftConfig,
    StepSdpoConfig,
    TerminalHindsightConfig,
)
from small_swe_train.training.metrics import MetricSink
from small_swe_train.types import DistillationTargets, ToolAction, Trajectory


class PolicyModel(Protocol):
    """Minimal model API expected by the staged trainer."""

    def sample_action(self, context: str) -> ToolAction:
        ...

    def train_format_step(self, context: str, target_action: ToolAction) -> float:
        ...

    def train_reverse_kl_step(self, targets: DistillationTargets) -> float:
        ...

    def train_sdpo_step(self, targets: DistillationTargets) -> float:
        ...


class SWEEnvironment(Protocol):
    """Minimal runtime API for trajectory collection."""

    def reset(self, task_id: str) -> str:
        ...

    def step(self, action: ToolAction) -> tuple[str, float, bool]:
        ...


def run_format_bootstrap(
    model: PolicyModel,
    examples: Iterable[tuple[str, ToolAction]],
    cfg: FormatBootstrapConfig,
    metrics: MetricSink,
) -> None:
    """Stage 0: teach schema-compliant tool actions."""
    for idx, (context, target_action) in enumerate(examples):
        if idx >= cfg.max_examples:
            break
        loss = model.train_format_step(context, target_action)
        metrics.log("format.loss", loss)

        # TODO: replace with parser-based validity checks on model samples.
        metrics.log("format.valid_action_rate", 1.0)
        metrics.log("format.invalid_schema_rate", 0.0)


def run_sdft(
    model: PolicyModel,
    rollout_targets: Iterable[DistillationTargets],
    cfg: SdftConfig,
    metrics: MetricSink,
) -> None:
    """Stage 1: on-policy reverse-KL against demo-conditioned self-teacher."""
    for idx, targets in enumerate(rollout_targets):
        if idx >= cfg.max_examples:
            break
        loss = model.train_reverse_kl_step(targets)
        metrics.log("sdft.reverse_kl", loss)

        if idx % max(1, cfg.demo_uplift_eval_every) == 0:
            # TODO: replace with held-out eval comparing with/without demos.
            metrics.log("sdft.demo_uplift", 0.0)


def run_step_sdpo(
    model: PolicyModel,
    rollout_targets: Iterable[DistillationTargets],
    cfg: StepSdpoConfig,
    metrics: MetricSink,
) -> None:
    """Stage 2: feedback-conditioned step-level SDPO distillation."""
    for idx, targets in enumerate(rollout_targets):
        if idx >= cfg.max_episodes:
            break
        loss = model.train_sdpo_step(targets)
        metrics.log("sdpo.teacher_student_kl", loss)

        # TODO: replace with real counters from runtime improvements.
        metrics.log("sdpo.step_fix_rate", 0.0)


def run_terminal_hindsight(
    model: PolicyModel,
    trajectories: Iterable[Trajectory],
    cfg: TerminalHindsightConfig,
    metrics: MetricSink,
) -> None:
    """Stage 3: optional hindsight distillation for delayed error signals."""
    if not cfg.enabled:
        return

    for trajectory in trajectories:
        if not trajectory.steps:
            continue
        _ = model
        _ = cfg.last_k_steps
        # TODO: build DistillationTargets from terminal feedback and key steps.
        metrics.log("sdpo.terminal_hindsight_gain", 0.0)


def log_stage_config(stage_name: str, cfg: object) -> dict[str, object]:
    """Return a serialized config snapshot for reporting."""
    data = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else {"value": str(cfg)}
    return {"stage": stage_name, "config": data}
