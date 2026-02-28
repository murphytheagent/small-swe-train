"""CLI helper for SDPO prompt-row parquet cache materialization."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from config import (
    DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    rft_runtime_defaults,
    resolve_on_policy_settings,
)
from env.task_dataset import (
    SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
    preload_sdpo_task_rows_to_parquet,
    preload_sdpo_task_rows_split_to_parquet,
    resolve_sdpo_task_rows_cache_path,
    resolve_sdpo_task_split_cache_paths,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve or materialize SDPO task-row parquet cache.",
    )
    parser.add_argument(
        "--data-config-name",
        default=DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
        help="Named config from configs/data/<name>.yaml (default: on_policy_swe_smith).",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory that stores the deterministic parquet cache file.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild cache even when the resolved parquet file already exists.",
    )
    parser.add_argument(
        "--print-path-only",
        action="store_true",
        help="Print resolved cache path without loading dataset or writing parquet.",
    )
    parser.add_argument(
        "--preloaded-task-parquet",
        default="",
        help="Use this existing parquet path for both train/val overrides.",
    )
    parser.add_argument(
        "--emit-hydra-overrides",
        action="store_true",
        help="Emit `data.train_files=...` and `data.val_files=...` lines.",
    )
    parser.add_argument(
        "--emit-split",
        action="store_true",
        help="Emit split train/val paths and use split materialization logic.",
    )
    parser.add_argument(
        "--eval-split-fraction",
        type=float,
        default=None,
        help="Evaluation split fraction in [0.0, 1.0).",
    )
    parser.add_argument(
        "--eval-min-rows",
        type=int,
        default=None,
        help="Minimum eval rows when split fraction is non-zero.",
    )
    parser.add_argument(
        "--max-problem-statement-chars",
        type=int,
        default=SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS,
        help=(
            "Keep only rows where problem_statement length is < this character count "
            f"(default: {SDPO_DEFAULT_MAX_PROBLEM_STATEMENT_CHARS})."
        ),
    )
    return parser


def _resolve_eval_split_defaults() -> tuple[float, int]:
    fallback_fraction = 0.1
    fallback_min_rows = 1
    defaults = rft_runtime_defaults()
    loop = defaults.get("loop")
    if not isinstance(loop, Mapping):
        return fallback_fraction, fallback_min_rows

    fraction = _coerce_eval_split_fraction(loop.get("eval_split_fraction"), fallback=fallback_fraction)
    min_rows = _coerce_eval_min_rows(loop.get("eval_min_rows"), fallback=fallback_min_rows)
    return fraction, min_rows


def _coerce_eval_split_fraction(value: Any, *, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        candidate = float(value)
        if 0.0 <= candidate < 1.0:
            return candidate
    return fallback


def _coerce_eval_min_rows(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 0:
        return int(value)
    return fallback


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    settings = resolve_on_policy_settings(data_config_name=args.data_config_name)
    eval_split_fraction_default, eval_min_rows_default = _resolve_eval_split_defaults()
    eval_split_fraction = (
        float(args.eval_split_fraction)
        if args.eval_split_fraction is not None
        else eval_split_fraction_default
    )
    eval_min_rows = (
        int(args.eval_min_rows)
        if args.eval_min_rows is not None
        else eval_min_rows_default
    )

    preloaded_task_parquet = str(args.preloaded_task_parquet or "").strip()
    if preloaded_task_parquet:
        train_path = preloaded_task_parquet
        val_path = preloaded_task_parquet
    elif args.emit_split:
        if args.print_path_only:
            train_path, val_path = resolve_sdpo_task_split_cache_paths(
                config=settings.data,
                cache_dir=args.cache_dir,
                eval_split_fraction=eval_split_fraction,
                min_eval_rows=eval_min_rows,
                max_problem_statement_chars=args.max_problem_statement_chars,
            )
        else:
            train_path, val_path = preload_sdpo_task_rows_split_to_parquet(
                config=settings.data,
                cache_dir=args.cache_dir,
                eval_split_fraction=eval_split_fraction,
                min_eval_rows=eval_min_rows,
                force_refresh=bool(args.force_refresh),
                max_problem_statement_chars=args.max_problem_statement_chars,
            )
    else:
        if args.print_path_only:
            path = resolve_sdpo_task_rows_cache_path(
                config=settings.data,
                cache_dir=args.cache_dir,
                max_problem_statement_chars=args.max_problem_statement_chars,
            )
        else:
            path = preload_sdpo_task_rows_to_parquet(
                config=settings.data,
                cache_dir=args.cache_dir,
                force_refresh=bool(args.force_refresh),
                max_problem_statement_chars=args.max_problem_statement_chars,
            )
        train_path = path
        val_path = path

    if args.emit_hydra_overrides:
        print(f"data.train_files={train_path}")
        print(f"data.val_files={val_path}")
    elif args.emit_split or preloaded_task_parquet:
        print(f"train={train_path}")
        print(f"val={val_path}")
    else:
        print(train_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
