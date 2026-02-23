"""Helpers for serializing selected RFT trajectories into MultiTurnSFT parquet shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

_TOOL_RESPONSE_PREFIX = "<tool_response>"


def build_multiturn_dataset_records(
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert selected attempt rows into verl MultiTurnSFTDataset parquet records."""
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        messages = build_multiturn_messages(row, row_index=index)
        records.append(
            {
                "messages": messages,
                "task_id": _as_text(row.get("task_id")),
                "attempt_index": _coerce_int(row.get("attempt_index"), fallback=0),
                "step_index": _coerce_int(row.get("step_index"), fallback=0),
                "turn_index": _coerce_int(row.get("turn_index"), fallback=0),
                "resolved": _coerce_bool(row.get("resolved"), fallback=False),
                "format_valid": _coerce_bool(row.get("format_valid"), fallback=False),
                "final_turn_has_submit": _coerce_bool(
                    row.get("final_turn_has_submit"),
                    fallback=False,
                ),
                "final_submit_format_valid": _coerce_bool(
                    row.get("final_submit_format_valid"),
                    fallback=False,
                ),
            }
        )
    return records


def build_multiturn_messages(
    row: Mapping[str, Any],
    *,
    row_index: int,
) -> list[dict[str, str]]:
    """Build a multi-turn chat transcript from one selected RFT row."""
    prompt = _as_text(row.get("prompt")).strip()
    messages: list[dict[str, str]] = []
    if prompt:
        messages.append({"role": "user", "content": prompt})

    history_values = _coerce_text_sequence(row.get("trajectory_history"))
    if history_values:
        for entry in history_values:
            stripped = entry.strip()
            if not stripped:
                continue
            role = "user" if stripped.startswith(_TOOL_RESPONSE_PREFIX) else "assistant"
            messages.append({"role": role, "content": stripped})
    else:
        assistant_response = _as_text(row.get("assistant_response")).strip()
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})

    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError(
            f"selected_rows[{row_index}] cannot be serialized: no assistant turns available."
        )
    return messages


def write_selected_rows_to_multiturn_parquet(
    selected_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> int:
    """Write selected rows to a parquet shard consumable by verl MultiTurnSFTDataset."""
    records = build_multiturn_dataset_records(selected_rows)
    _write_records_to_parquet(records, output_path)
    return len(records)


def _write_records_to_parquet(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    if not records:
        raise ValueError("Cannot write MultiTurnSFT parquet with zero records.")

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on train extras
        raise RuntimeError(
            "Writing MultiTurnSFT parquet requires pandas/pyarrow. "
            "Install training extras (`pip install -e \".[train]\"`)."
        ) from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame.from_records(records)
    dataframe.to_parquet(target, index=False)


def _coerce_text_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    text_values: list[str] = []
    for item in value:
        text = _as_text(item)
        if text:
            text_values.append(text)
    return text_values


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return fallback
    return fallback


def _coerce_bool(value: Any, *, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return fallback
