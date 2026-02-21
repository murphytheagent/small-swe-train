#!/usr/bin/env python3
"""Prepare RFT-ready records from SWE trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw trajectory JSON/JSONL input (or directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for prepared records (.jsonl, .parquet, or .arrow).",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3-4B",
        help="Hugging Face tokenizer model name (default: Qwen/Qwen3-4B).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes to ingest.",
    )
    return parser


def main() -> None:
    _ensure_src_on_path()
    from data.trajectory_ingestion import run_ingestion

    parser = _build_arg_parser()
    args = parser.parse_args()
    stats = run_ingestion(
        input_path=args.input,
        output_path=args.output,
        tokenizer_model=args.tokenizer_model,
        max_episodes=args.max_episodes,
    )
    print(
        f"Wrote {stats['records_written']} records "
        f"({stats['episodes_ingested']} episodes from {stats['raw_records']} raw records) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
