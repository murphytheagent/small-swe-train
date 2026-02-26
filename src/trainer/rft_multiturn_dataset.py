"""Helpers for serializing selected RFT trajectories into MultiTurnSFT parquet shards."""

from __future__ import annotations

import json
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
        resolved = _coerce_bool(row.get("resolved"), fallback=False)
        format_valid = _coerce_bool(row.get("format_valid"), fallback=False)
        final_turn_has_submit = _coerce_bool(
            row.get("final_turn_has_submit"),
            fallback=False,
        )
        final_submit_format_valid = _coerce_bool(
            row.get("final_submit_format_valid"),
            fallback=False,
        )
        fail_to_pass = _resolve_test_targets_from_row(row, key="fail_to_pass")
        pass_to_pass = _resolve_test_targets_from_row(row, key="pass_to_pass")
        messages = build_multiturn_messages(row, row_index=index)
        prompt_messages = build_rollout_prompt_messages(messages, row_index=index)
        reward_ground_truth = {
            "task_id": task_id,
            "image_name": image_name,
            "attempt_index": attempt_index,
            "step_index": step_index,
            "turn_index": turn_index,
            "resolved": resolved,
            "format_valid": format_valid,
            "final_turn_has_submit": final_turn_has_submit,
            "final_submit_format_valid": final_submit_format_valid,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
        }
        data_source = _as_text(row.get("data_source")).strip() or "small_swe_phase_d"
        records.append(
            {
                # Keep `messages` as the full trajectory, and store `prompt` as the
                # rollout context that should precede a fresh assistant generation.
                "messages": messages,
                "prompt": prompt_messages,
                "data_source": data_source,
                "reward_model": {"ground_truth": reward_ground_truth},
                "task_id": task_id,
                "image_name": image_name,
                "attempt_index": attempt_index,
                "step_index": step_index,
                "turn_index": turn_index,
                "resolved": resolved,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "format_valid": format_valid,
                "final_turn_has_submit": final_turn_has_submit,
                "final_submit_format_valid": final_submit_format_valid,
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


def build_rollout_prompt_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    row_index: int,
) -> list[dict[str, str]]:
    """Build the SDPO rollout prompt context by trimming trailing assistant turns."""
    prompt_messages: list[dict[str, str]] = []
    for item in messages:
        role = _as_text(item.get("role")).strip()
        content = _as_text(item.get("content"))
        if not role or not content:
            continue
        prompt_messages.append({"role": role, "content": content})

    while prompt_messages and prompt_messages[-1]["role"] == "assistant":
        prompt_messages.pop()

    if not prompt_messages:
        raise ValueError(
            f"selected_rows[{row_index}] cannot be serialized: rollout prompt context is empty."
        )
    if prompt_messages[-1]["role"] == "assistant":
        raise ValueError(
            f"selected_rows[{row_index}] cannot be serialized: rollout prompt must not end on assistant."
        )
    return prompt_messages


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


def _extract_reward_ground_truth_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            return ground_truth
    return {}


def _resolve_test_targets_from_row(row: Mapping[str, Any], *, key: str) -> list[str]:
    ground_truth = _extract_reward_ground_truth_from_row(row)
    for source in (row, ground_truth):
        for candidate_key in (key, key.upper()):
            if candidate_key in source:
                return _coerce_test_targets(source.get(candidate_key))
    return []


def _coerce_test_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [name for name in (str(key).strip() for key in value.keys()) if name]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        targets: list[str] = []
        for item in value:
            name = str(item).strip()
            if name:
                targets.append(name)
        return targets
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _coerce_test_targets(parsed)
        if "," in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.split(",")) if chunk]
        if "\n" in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.splitlines()) if chunk]
        return [stripped]
    return []
