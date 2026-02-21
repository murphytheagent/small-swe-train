#!/usr/bin/env python3
"""Prepare deterministic RFT-ready records from SWE trajectory samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from verl_integration.data_preprocessor import preprocess_trajectories
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from verl_integration.data_preprocessor import preprocess_trajectories


def _load_json_rows(path: Path) -> list[Mapping[str, Any]]:
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


def _write_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = len(rows)
    format_valid_rows = sum(1 for row in rows if bool(row.get("format_valid")))
    parse_error_rows = sum(1 for row in rows if bool(row.get("parse_error")))
    validation_error_rows = sum(1 for row in rows if bool(row.get("validation_errors")))
    invalid_rows = total_rows - format_valid_rows
    format_valid_rate = (format_valid_rows / total_rows) if total_rows else 0.0

    return {
        "total_rows": total_rows,
        "format_valid_rows": format_valid_rows,
        "invalid_rows": invalid_rows,
        "parse_error_rows": parse_error_rows,
        "validation_error_rows": validation_error_rows,
        "format_valid_rate": format_valid_rate,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to input JSON/JSONL trajectories.")
    parser.add_argument("--output", required=True, type=Path, help="Path for output JSONL rows.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional path to write a JSON summary report.",
    )
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional input row limit (0 means no limit).")
    parser.add_argument(
        "--validate-min-rows",
        type=int,
        default=100,
        help="Require at least this many rows after preprocessing (0 disables check).",
    )
    parser.add_argument(
        "--min-format-valid-rate",
        type=float,
        default=0.0,
        help="Require format_valid_rate >= threshold (0 disables check).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    samples = _load_json_rows(args.input)
    if args.limit > 0:
        samples = samples[: args.limit]

    rows = preprocess_trajectories(samples, max_tool_calls=args.max_tool_calls)
    summary = _summarize_rows(rows)

    if args.validate_min_rows > 0 and summary["total_rows"] < args.validate_min_rows:
        raise SystemExit(
            f"Row count check failed: expected >= {args.validate_min_rows}, got {summary['total_rows']}."
        )

    if args.min_format_valid_rate > 0 and summary["format_valid_rate"] < args.min_format_valid_rate:
        raise SystemExit(
            "Format validity check failed: "
            f"expected >= {args.min_format_valid_rate:.3f}, got {summary['format_valid_rate']:.3f}."
        )

    _write_jsonl_rows(args.output, rows)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
