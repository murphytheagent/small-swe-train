"""Deterministic trainer scaffolds delegating to integration-layer adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import MAX_TOOL_CALLS_PER_TURN
from data.tokenization import SupportsOffsetsTokenizer
from rollout.onpolicy_collector import OnPolicyRolloutCollector
from verl_integration.mask_injector import inject_response_mask
from verl_integration.reprompt_adapter import build_self_distillation_batch
from verl_integration.reward_function import reward_fn
from verl_integration.env_bridge import ToolExecutor, run_env_bridge_step
from verl_integration.onpolicy_rollout_adapter import collect_rft_sft_batch_for_steps


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


@dataclass(frozen=True)
class OnPolicyRFTStepArtifacts:
    training_stats: TrainingStepStats
    selected_count: int
    rejected_count: int
    dataproto_payload: Mapping[str, Any]
    selection_reasons: tuple[str, ...]
    checkpoint_dir: str | None
    checkpoint_exists: bool


class SDPOTrainerScaffold:
    """Deterministic trainer facade until full verl runtime wiring lands."""

    def __init__(self, config: SDPOTrainerConfig) -> None:
        self._config = config
        self._teacher_ema_proxy = 0.0
        self._loss_history: list[float] = []

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
                        first_response = bridge_result.steps[0].response
                        row["tool_output"] = {
                            "stdout": first_response.stdout,
                            "stderr": first_response.stderr,
                            "exit_code": first_response.exit_code,
                            "metadata": dict(first_response.metadata),
                        }
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
        """Run rollout -> preprocess -> centralized RFT selection -> RFT train stats."""
        handoff_result = collect_rft_sft_batch_for_steps(
            total_steps=total_steps,
            collector=collector,
            tokenizer=tokenizer,
            handoff_overrides=handoff_overrides,
            output_dir=output_dir,
        )

        selected_rows = handoff_result["selected_rows"]
        rejected_rows = handoff_result["rejected_rows"]
        training_stats = self.run_rft_epoch(selected_rows)
        reasons = tuple(
            str(row.get("rft_rejection_reason", ""))
            for row in rejected_rows
            if row.get("rft_rejection_reason")
        )
        checkpoint_path: Path | None = None
        if checkpoint_dir is not None:
            resolved_global_step = self._resolve_global_step(global_step, fallback=total_steps)
            checkpoint_path = self._write_rft_checkpoint(
                checkpoint_dir=Path(checkpoint_dir),
                global_step=resolved_global_step,
                training_stats=training_stats,
                dataproto_payload=handoff_result["dataproto_payload"],
                selected_count=len(selected_rows),
                rejected_count=len(rejected_rows),
                selection_reasons=reasons,
            )

        return OnPolicyRFTStepArtifacts(
            training_stats=training_stats,
            selected_count=len(selected_rows),
            rejected_count=len(rejected_rows),
            dataproto_payload=handoff_result["dataproto_payload"],
            selection_reasons=reasons,
            checkpoint_dir=str(checkpoint_path) if checkpoint_path is not None else None,
            checkpoint_exists=checkpoint_path is not None,
        )

    @staticmethod
    def _resolve_global_step(global_step: int | None, *, fallback: int) -> int:
        if global_step is None:
            resolved = fallback
        else:
            resolved = global_step
        if resolved < 0:
            raise ValueError("global_step must be >= 0 when writing RFT checkpoint artifacts.")
        return resolved

    def _write_rft_checkpoint(
        self,
        *,
        checkpoint_dir: Path,
        global_step: int,
        training_stats: TrainingStepStats,
        dataproto_payload: Mapping[str, Any],
        selected_count: int,
        rejected_count: int,
        selection_reasons: Sequence[str],
    ) -> Path:
        step_dir = checkpoint_dir / f"global_step_{global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": self._config.model_name,
            "global_step": int(global_step),
            "selected_count": int(selected_count),
            "rejected_count": int(rejected_count),
            "training_stats": {
                "loss": float(training_stats.loss),
                "teacher_student_kl": float(training_stats.teacher_student_kl),
                "format_valid_rate": float(training_stats.format_valid_rate),
            },
            "selection_reasons": [str(reason) for reason in selection_reasons],
            "dataproto_meta_info": dict(dataproto_payload.get("meta_info", {})),
        }

        manifest_path = step_dir / "rft_step_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")

        latest_pointer = checkpoint_dir / "latest_checkpoint.txt"
        latest_pointer.parent.mkdir(parents=True, exist_ok=True)
        latest_pointer.write_text(str(step_dir), encoding="utf-8")

        return step_dir
