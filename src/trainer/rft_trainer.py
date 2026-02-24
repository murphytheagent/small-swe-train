"""Deterministic RFT trainer scaffold and on-policy rollout handoff utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME
from data.tokenization import SupportsOffsetsTokenizer
from env.task_dataset import DatasetLoader
from rollout.onpolicy_collector import (
    AssistantTurnGenerator,
    AttemptResolver,
    ExecutorFactory,
    OnPolicyRolloutCollector,
    PoolFactory,
)
from trainer.common import SDPOTrainerConfig, TrainingStepStats
from trainer.rft_handoff import (
    build_onpolicy_collector,
    collect_rft_sft_batch_for_steps,
)
from verl_integration.mask_injector import inject_response_mask


@dataclass(frozen=True)
class OnPolicyRFTStepArtifacts:
    training_stats: TrainingStepStats
    selected_count: int
    rejected_count: int
    dataproto_payload: Mapping[str, Any]
    selection_reasons: tuple[str, ...]
    checkpoint_dir: str | None
    checkpoint_exists: bool


class RFTTrainerScaffold:
    """Deterministic RFT facade until full verl runtime wiring lands."""

    def __init__(self, config: SDPOTrainerConfig) -> None:
        self._config = config

    @property
    def config(self) -> SDPOTrainerConfig:
        return self._config

    def run_rft_epoch(self, batch: Sequence[Mapping[str, Any]]) -> TrainingStepStats:
        """Run a deterministic RFT pass over pre-labeled records."""
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
        resolved_global_step: int | None = None
        if checkpoint_dir is not None:
            resolved_global_step = self._resolve_global_step(global_step)

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
            assert resolved_global_step is not None
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

    def run_onpolicy_rft_step_from_config(
        self,
        *,
        total_steps: int,
        tokenizer: SupportsOffsetsTokenizer,
        turn_generator: AssistantTurnGenerator | None = None,
        data_config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
        runtime_overrides: Mapping[str, Any] | None = None,
        data_overrides: Mapping[str, Any] | None = None,
        dataset_loader: DatasetLoader | None = None,
        pool_factory: PoolFactory | None = None,
        executor_factory: ExecutorFactory | None = None,
        attempt_resolver: AttemptResolver | None = None,
        handoff_overrides: Mapping[str, Any] | None = None,
        output_dir: str | None = None,
        checkpoint_dir: str | Path | None = None,
        global_step: int | None = None,
    ) -> OnPolicyRFTStepArtifacts:
        """Build the real on-policy collector from config, then run one RFT handoff step."""
        collector = build_onpolicy_collector(
            turn_generator=turn_generator,
            data_config_name=data_config_name,
            runtime_overrides=runtime_overrides,
            data_overrides=data_overrides,
            dataset_loader=dataset_loader,
            pool_factory=pool_factory,
            executor_factory=executor_factory,
            attempt_resolver=attempt_resolver,
        )
        return self.run_onpolicy_rft_step(
            total_steps=total_steps,
            collector=collector,
            tokenizer=tokenizer,
            handoff_overrides=handoff_overrides,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
        )

    @staticmethod
    def _resolve_global_step(global_step: int | None) -> int:
        if global_step is None:
            raise ValueError("global_step is required when writing RFT checkpoint artifacts.")
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
