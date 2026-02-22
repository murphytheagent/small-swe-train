"""Thin adapter wiring centralized config into on-policy rollout collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME, resolve_on_policy_settings
from rollout.onpolicy_collector import AssistantTurnGenerator, OnPolicyRolloutCollector
from schemas import RolloutRow


def build_onpolicy_collector(
    *,
    turn_generator: AssistantTurnGenerator | None = None,
    data_config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    runtime_overrides: Mapping[str, Any] | None = None,
    data_overrides: Mapping[str, Any] | None = None,
) -> OnPolicyRolloutCollector:
    """Build a collector using only centralized config authority."""
    settings = resolve_on_policy_settings(
        data_config_name=data_config_name,
        runtime_overrides=runtime_overrides,
        data_overrides=data_overrides,
    )
    return OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
    )


def collect_rollouts_for_steps(
    *,
    total_steps: int,
    collector: OnPolicyRolloutCollector,
    output_dir: str | Path | None = None,
) -> list[list[RolloutRow]]:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")

    all_rows: list[list[RolloutRow]] = []
    base_dir = Path(output_dir) if output_dir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for step_index in range(total_steps):
        rows = collector.collect_step(step_index)
        all_rows.append(rows)
        if base_dir is not None:
            output_path = base_dir / f"step_{step_index:05d}.jsonl"
            write_rollout_rows_jsonl(output_path, rows)
    return all_rows


def write_rollout_rows_jsonl(path: str | Path, rows: Sequence[RolloutRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
