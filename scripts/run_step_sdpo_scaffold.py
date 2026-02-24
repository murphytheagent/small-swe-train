#!/usr/bin/env python3
"""Run one deterministic Step-SDPO scaffold step with explicit file I/O contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from config import DEFAULT_TRAINING_MODEL_NAME, MAX_TOOL_CALLS_PER_TURN
    from trainer.sdpo_trainer import SDPOTrainerConfig, SDPOTrainerScaffold
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from config import DEFAULT_TRAINING_MODEL_NAME, MAX_TOOL_CALLS_PER_TURN
    from trainer.sdpo_trainer import SDPOTrainerConfig, SDPOTrainerScaffold


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Line {line_no} in {path} must be a JSON object.")
            rows.append(dict(payload))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"{path} contains a non-object entry at index {index}.")
            rows.append(dict(item))
        return rows
    raise ValueError(f"{path} must be JSON object/array-of-objects or JSONL.")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input rollout rows JSON/JSONL.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for runner artifacts.")
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=MAX_TOOL_CALLS_PER_TURN,
        help=f"Max tool calls per turn (default: {MAX_TOOL_CALLS_PER_TURN}).",
    )
    parser.add_argument("--ema-beta", type=float, default=0.005, help="EMA beta for trainer proxy metric.")
    parser.add_argument(
        "--exclude-student-attempt-for-teacher",
        action="store_true",
        help="Disable student-attempt block inclusion in teacher prompts.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise ValueError("Input must contain at least one rollout row.")

    trainer = SDPOTrainerScaffold(
        SDPOTrainerConfig(
            model_name=DEFAULT_TRAINING_MODEL_NAME,
            max_tool_calls_per_turn=args.max_tool_calls,
            include_student_attempt_for_teacher=not args.exclude_student_attempt_for_teacher,
            ema_beta=args.ema_beta,
        )
    )
    artifacts = trainer.run_end_to_end_global_step(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_rows_path = args.output_dir / "rollout_rows.jsonl"
    teacher_prompts_path = args.output_dir / "teacher_prompts.jsonl"
    summary_path = args.output_dir / "sdpo_step_summary.json"

    _write_jsonl(rollout_rows_path, rows)

    teacher_rows: list[dict[str, Any]] = []
    for index, teacher_prompt in enumerate(artifacts.teacher_prompts):
        teacher_rows.append(
            {
                "row_index": index,
                "teacher_prompt": teacher_prompt,
                "self_distillation_mask": bool(artifacts.self_distillation_mask[index]),
                "prompt_truncated": bool(artifacts.prompt_truncated[index]),
                "reward": float(artifacts.rewards[index]),
                "feedback": str(artifacts.feedback[index]),
            }
        )
    _write_jsonl(teacher_prompts_path, teacher_rows)

    rewards = [float(value) for value in artifacts.rewards]
    summary_payload = {
        "input_path": str(args.input),
        "row_count": len(rows),
        "training_stats": {
            "loss": float(artifacts.training_stats.loss),
            "teacher_student_kl": float(artifacts.training_stats.teacher_student_kl),
            "format_valid_rate": float(artifacts.training_stats.format_valid_rate),
        },
        "format_metrics": {key: float(value) for key, value in artifacts.format_metrics.items()},
        "reward_summary": {
            "count": len(rewards),
            "mean": _mean(rewards),
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
        },
        "self_distillation_enabled_count": int(sum(1 for flag in artifacts.self_distillation_mask if flag)),
        "prompt_truncated_count": int(sum(1 for flag in artifacts.prompt_truncated if flag)),
        "teacher_ema_proxy": float(artifacts.teacher_ema_proxy),
        "loss_history": [float(value) for value in artifacts.loss_history],
    }

    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result_payload = {
        "output_dir": str(args.output_dir),
        "summary_path": str(summary_path),
        "rollout_rows_path": str(rollout_rows_path),
        "teacher_prompts_path": str(teacher_prompts_path),
        "row_count": len(rows),
    }
    print(json.dumps(result_payload, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
