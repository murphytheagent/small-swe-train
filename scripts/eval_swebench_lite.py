#!/usr/bin/env python3
"""Evaluate SWE-bench Lite predictions and optional baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from eval.swebench_lite import (
        compare_resolve_rates,
        evaluate_swebench_lite,
        summarize_episode_results,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from eval.swebench_lite import (
        compare_resolve_rates,
        evaluate_swebench_lite,
        summarize_episode_results,
    )


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Line {line_no} in {path} must be a JSON object.")
            rows.append(payload)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [payload]
    if isinstance(payload, Sequence):
        rows = [item for item in payload if isinstance(item, Mapping)]
        if len(rows) != len(payload):
            raise ValueError(f"{path} contains non-object entries.")
        return rows
    raise ValueError(f"{path} must be a JSON object, array of objects, or JSONL file.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path, help="SWE-bench episode metadata JSON/JSONL.")
    parser.add_argument("--predictions", required=True, type=Path, help="Candidate predictions JSON/JSONL.")
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        help="Optional baseline predictions JSON/JSONL for resolve-rate delta.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    episodes = _load_rows(args.episodes)
    candidate_predictions = _load_rows(args.predictions)
    candidate_results = evaluate_swebench_lite(episodes=episodes, predictions=candidate_predictions)
    candidate_summary = summarize_episode_results(candidate_results)

    payload: dict[str, Any] = {
        "candidate_summary": {
            "total_episodes": candidate_summary.total_episodes,
            "resolved_episodes": candidate_summary.resolved_episodes,
            "unresolved_episodes": candidate_summary.unresolved_episodes,
            "resolve_rate": candidate_summary.resolve_rate,
        }
    }

    if args.baseline_predictions:
        baseline_predictions = _load_rows(args.baseline_predictions)
        baseline_results = evaluate_swebench_lite(episodes=episodes, predictions=baseline_predictions)
        payload["comparison"] = compare_resolve_rates(
            baseline_results=baseline_results,
            candidate_results=candidate_results,
        )

    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")

    print(serialized.rstrip())


if __name__ == "__main__":
    main()
