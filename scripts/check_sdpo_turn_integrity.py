#!/usr/bin/env python3
"""Offline integrity checks for SDPO turn-level reprompt artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from verl_integration.reprompt_adapter import build_self_distillation_batch


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL rows.")
    parser.add_argument(
        "--turn-supervision-mode",
        default="current_turn",
        choices=("next_turn", "current_turn"),
        help="Turn supervision mode used for reprompt construction.",
    )
    parser.add_argument(
        "--max-truncation-rate",
        type=float,
        default=0.05,
        help="Maximum allowed truncated prompt fraction before failure.",
    )
    parser.add_argument(
        "--include-student-attempt-for-teacher",
        action="store_true",
        help="Include current student attempt block when building reprompts.",
    )
    return parser.parse_args(argv)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(raw_text)
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of row objects.")
        iterable = payload
    else:
        iterable = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    for item in iterable:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _coerce_binary_mask(value: Any, *, width: int | None = None) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        data: list[int] = []
    else:
        data = [1 if bool(item) else 0 for item in value]

    if width is not None:
        if len(data) < width:
            data.extend([0] * (width - len(data)))
        elif len(data) > width:
            data = data[:width]
    return data


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _validate_mask_subset(
    *,
    response_mask: Sequence[int],
    turn_mask: Sequence[int],
    row_index: int,
    turn_index: int,
    violations: list[str],
) -> None:
    for token_index, flag in enumerate(turn_mask):
        if not flag:
            continue
        if token_index >= len(response_mask) or int(response_mask[token_index]) == 0:
            violations.append(
                f"row={row_index} turn={turn_index}: turn_response_mask is not a subset of _response_mask "
                f"(token_index={token_index})."
            )


def _validate_response_mask_presence(
    *,
    row: Mapping[str, Any],
    row_index: int,
    violations: list[str],
) -> list[int]:
    response_mask = _coerce_binary_mask(row.get("_response_mask"))
    if response_mask:
        return response_mask
    violations.append(f"row={row_index}: missing or empty _response_mask.")
    return []


def _validate_explicit_masks(rows: Sequence[Mapping[str, Any]], violations: list[str]) -> None:
    for row_index, row in enumerate(rows):
        response_mask = _coerce_binary_mask(row.get("_response_mask"))
        if not response_mask:
            continue
        explicit_masks = row.get("turn_response_masks")
        if not isinstance(explicit_masks, Sequence) or isinstance(explicit_masks, (str, bytes)):
            continue
        for turn_index, raw_turn_mask in enumerate(explicit_masks):
            turn_mask = _coerce_binary_mask(raw_turn_mask)
            if len(turn_mask) > len(response_mask):
                violations.append(
                    "row={row} turn={turn}: turn_response_mask length ({turn_len}) exceeds _response_mask "
                    "length ({resp_len}).".format(
                        row=row_index,
                        turn=turn_index,
                        turn_len=len(turn_mask),
                        resp_len=len(response_mask),
                    )
                )
                continue
            if len(turn_mask) < len(response_mask):
                turn_mask.extend([0] * (len(response_mask) - len(turn_mask)))
            _validate_mask_subset(
                response_mask=response_mask,
                turn_mask=turn_mask,
                row_index=row_index,
                turn_index=turn_index,
                violations=violations,
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _load_rows(Path(args.input))
    if not rows:
        print("No rows found in input.", file=sys.stderr)
        return 1

    violations: list[str] = []
    _validate_explicit_masks(rows, violations)

    batch = build_self_distillation_batch(
        rows,
        include_student_attempt_for_teacher=args.include_student_attempt_for_teacher,
        turn_supervision_mode=args.turn_supervision_mode,
    )

    prompt_truncated = [bool(item) for item in batch.get("prompt_truncated", [])]
    turn_prompt_truncated = batch.get("turn_prompt_truncated", [])
    total_prompt_count = len(prompt_truncated)
    truncated_count = sum(1 for flag in prompt_truncated if flag)
    if isinstance(turn_prompt_truncated, Sequence):
        for row_flags in turn_prompt_truncated:
            if isinstance(row_flags, Sequence) and not isinstance(row_flags, (str, bytes)):
                total_prompt_count += len(row_flags)
                truncated_count += sum(1 for flag in row_flags if bool(flag))

    truncation_rate = (truncated_count / total_prompt_count) if total_prompt_count > 0 else 0.0
    if truncation_rate > float(args.max_truncation_rate):
        violations.append(
            "prompt truncation rate exceeded threshold: "
            f"rate={truncation_rate:.4f} threshold={float(args.max_truncation_rate):.4f}"
        )

    turn_teacher_prompts = batch.get("turn_teacher_prompts", [])
    turn_response_masks = batch.get("turn_response_masks", [])
    turn_distillation_mask = batch.get("turn_distillation_mask", [])

    for row_index, row in enumerate(rows):
        prompts = turn_teacher_prompts[row_index] if row_index < len(turn_teacher_prompts) else []
        masks = turn_response_masks[row_index] if row_index < len(turn_response_masks) else []
        active = turn_distillation_mask[row_index] if row_index < len(turn_distillation_mask) else []

        if not (len(prompts) == len(masks) == len(active)):
            violations.append(
                "row={row}: cardinality mismatch for turn_teacher_prompts/turn_response_masks/"
                "turn_distillation_mask ({p}/{m}/{a}).".format(
                    row=row_index,
                    p=len(prompts),
                    m=len(masks),
                    a=len(active),
                )
            )
            continue

        response_mask = _validate_response_mask_presence(
            row=row,
            row_index=row_index,
            violations=violations,
        )
        if not response_mask:
            continue
        for turn_index, turn_mask_raw in enumerate(masks):
            turn_mask = _coerce_binary_mask(turn_mask_raw)
            if len(turn_mask) > len(response_mask):
                violations.append(
                    "row={row} turn={turn}: generated turn_response_mask length ({turn_len}) exceeds "
                    "_response_mask length ({resp_len}).".format(
                        row=row_index,
                        turn=turn_index,
                        turn_len=len(turn_mask),
                        resp_len=len(response_mask),
                    )
                )
                continue
            if len(turn_mask) < len(response_mask):
                turn_mask.extend([0] * (len(response_mask) - len(turn_mask)))
            _validate_mask_subset(
                response_mask=response_mask,
                turn_mask=turn_mask,
                row_index=row_index,
                turn_index=turn_index,
                violations=violations,
            )

        if args.turn_supervision_mode == "current_turn":
            assistant_turns = _coerce_text_list(row.get("trajectory_assistant_turns"))
            for turn_index, prompt in enumerate(prompts):
                if turn_index >= len(assistant_turns) or turn_index >= len(active):
                    continue
                if not bool(active[turn_index]):
                    continue
                target_turn_text = assistant_turns[turn_index].strip()
                if target_turn_text and target_turn_text in str(prompt):
                    violations.append(
                        f"row={row_index} turn={turn_index}: target-turn leakage detected in teacher prompt."
                    )

    print(
        "Checked rows={rows} prompts={prompts} truncation_rate={rate:.4f}".format(
            rows=len(rows),
            prompts=total_prompt_count,
            rate=truncation_rate,
        )
    )
    if violations:
        print("Integrity check failed:")
        for item in violations:
            print(f"- {item}")
        return 1

    print("Integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
