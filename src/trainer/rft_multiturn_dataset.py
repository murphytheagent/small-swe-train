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
        row_label = f"selected_rows[{index}]"
        task_id = _require_non_empty_text(row.get("task_id"), label=f"{row_label}.task_id")
        image_name = _require_non_empty_text(row.get("image_name"), label=f"{row_label}.image_name")
        attempt_index = _require_int(row.get("attempt_index"), label=f"{row_label}.attempt_index")
        step_index = _require_int(row.get("step_index"), label=f"{row_label}.step_index")
        turn_index = _require_int(row.get("turn_index"), label=f"{row_label}.turn_index")
        messages = build_multiturn_messages(row, row_index=index)
        records.append(
            {
                # Keep `messages` for existing MultiTurnSFT readers and add `prompt`
                # so RLHFDataset-based SDPO runs can consume the same handoff parquet.
                "messages": messages,
                "prompt": messages,
                "task_id": task_id,
                "image_name": image_name,
                "attempt_index": attempt_index,
                "step_index": step_index,
                "turn_index": turn_index,
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


def _require_non_empty_text(value: Any, *, label: str) -> str:
    text = _as_text(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string.")
    return text


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


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= 0.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label} must be an integer >= 0.")
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer >= 0.") from exc
    else:
        raise ValueError(f"{label} must be an integer >= 0.")
    if parsed < 0:
        raise ValueError(f"{label} must be an integer >= 0.")
    return parsed


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
