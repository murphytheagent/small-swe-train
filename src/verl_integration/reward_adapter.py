"""DataProto adapters for SWE-bridge reward computation in PPO runtime."""

from __future__ import annotations

import numbers
from typing import Any, Mapping, Sequence

from config import MAX_TOOL_CALLS_PER_TURN
from verl_integration.reward_function import reward_fn

try:  # pragma: no cover - exercised in train runtime
    import torch
except ModuleNotFoundError:  # pragma: no cover - unit-test environments without train deps
    torch = None  # type: ignore[assignment]


def dataproto_to_rows(batch: Any, tokenizer: Any) -> list[dict[str, Any]]:
    """Convert a verl DataProto batch into row dictionaries used by local reward logic."""
    non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
    responses = getattr(batch, "batch", {}).get("responses")
    response_mask = getattr(batch, "batch", {}).get("response_mask")

    batch_size = _resolve_batch_size(responses=responses, non_tensor_batch=non_tensor_batch)
    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        response_ids = _coerce_int_list(_select_index(responses, index))
        response_text = _decode_response(tokenizer=tokenizer, token_ids=response_ids)

        raw_prompt_messages = _normalize_messages(_select_non_tensor(non_tensor_batch, "raw_prompt", index))
        prompt_text = _extract_prompt_text(
            messages=raw_prompt_messages,
            fallback=_as_text(_select_non_tensor(non_tensor_batch, "prompt", index)),
        )

        trajectory_steps = _coerce_mapping_list(
            _select_non_tensor(non_tensor_batch, "trajectory_steps", index)
        )
        tool_response_blocks = _coerce_text_list(
            _select_non_tensor(non_tensor_batch, "tool_response_blocks", index)
        )
        trajectory_assistant_turns = _coerce_text_list(
            _select_non_tensor(non_tensor_batch, "trajectory_assistant_turns", index)
        )
        trajectory_assistant_turn_token_lengths = _coerce_int_list(
            _select_non_tensor(non_tensor_batch, "trajectory_assistant_turn_token_lengths", index)
        )
        trajectory_turn_tool_response_blocks = _coerce_nested_text_list(
            _select_non_tensor(non_tensor_batch, "trajectory_turn_tool_response_blocks", index)
        )
        tool_output = _extract_last_tool_output(trajectory_steps)
        reward_ground_truth = _extract_reward_ground_truth(
            _select_non_tensor(non_tensor_batch, "reward_model", index)
        )

        row = {
            "prompt": prompt_text,
            "task_block": prompt_text,
            "assistant_response": response_text,
            "response_text": response_text,
            "task_id": _coerce_non_empty_text(
                _select_non_tensor(non_tensor_batch, "task_id", index),
                fallback=f"sample-{index}",
            ),
            "image_name": _coerce_non_empty_text(
                _select_non_tensor(non_tensor_batch, "image_name", index),
                fallback="swe.unknown",
            ),
            "step_index": _coerce_non_negative_int(
                _select_non_tensor(non_tensor_batch, "step_index", index),
                fallback=index,
            ),
            "attempt_index": _coerce_non_negative_int(
                _select_non_tensor(non_tensor_batch, "attempt_index", index),
                fallback=0,
            ),
            "turn_index": _coerce_non_negative_int(
                _select_non_tensor(non_tensor_batch, "turn_index", index),
                fallback=0,
            ),
            "resolved": _coerce_bool(
                reward_ground_truth.get("resolved")
                if "resolved" in reward_ground_truth
                else _select_non_tensor(non_tensor_batch, "resolved", index),
                fallback=False,
            ),
            "tool_output": tool_output,
            "trajectory_steps": trajectory_steps,
            "tool_response_blocks": tool_response_blocks,
            "trajectory_assistant_turns": trajectory_assistant_turns,
            "trajectory_assistant_turn_token_lengths": trajectory_assistant_turn_token_lengths,
            "trajectory_turn_tool_response_blocks": trajectory_turn_tool_response_blocks,
            "loop_exit_reason": _as_text(_select_non_tensor(non_tensor_batch, "loop_exit_reason", index)),
            "bridge_error": _as_text(_select_non_tensor(non_tensor_batch, "bridge_error", index)),
            "timeout_error": _as_text(_select_non_tensor(non_tensor_batch, "timeout_error", index)),
            "executor_error": _as_text(_select_non_tensor(non_tensor_batch, "executor_error", index)),
            "final_turn_has_submit": _coerce_bool(
                _select_non_tensor(non_tensor_batch, "final_turn_has_submit", index),
                fallback=False,
            ),
            "final_submit_format_valid": _coerce_bool(
                _select_non_tensor(non_tensor_batch, "final_submit_format_valid", index),
                fallback=False,
            ),
            "include_student_attempt_for_teacher": _coerce_bool(
                _select_non_tensor(non_tensor_batch, "include_student_attempt_for_teacher", index),
                fallback=True,
            ),
            "_raw_prompt_messages": raw_prompt_messages,
            "_response_mask": _resolve_response_mask(
                raw_mask_value=_select_index(response_mask, index),
                fallback_length=max(len(response_ids), 1),
            ),
        }

        for key in (
            "fail_to_pass",
            "pass_to_pass",
            "data_source",
            "fail_to_pass_results",
            "pass_to_pass_results",
            "fail_to_pass_all_passed",
            "pass_to_pass_all_passed",
            "fail_to_pass_verified",
            "pass_to_pass_verified",
            "verification_missing",
            "verification_error",
            "verification_feedback",
            "submission_final_response",
            "resolved",
        ):
            value = _select_non_tensor(non_tensor_batch, key, index)
            if value is None and key in reward_ground_truth:
                value = reward_ground_truth.get(key)
            if value is not None:
                row[key] = value
        rows.append(row)

    return rows


def rows_to_reward_tensor(
    rows: Sequence[Mapping[str, Any]],
    *,
    response_width: int | None = None,
    device: Any = None,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
) -> tuple[Any, dict[str, list[Any]]]:
    """Compute token-level reward tensor for SDPO PPO from row-wise rewards."""
    if torch is None:
        raise RuntimeError("rows_to_reward_tensor requires torch. Install training extras first.")

    if not rows:
        width = int(response_width or 1)
        return torch.zeros((0, width), dtype=torch.float32, device=device), {}

    rewards, reward_info = reward_fn(rows, max_tool_calls=max_tool_calls)
    if response_width is not None and response_width >= 1:
        width = int(response_width)
    else:
        width = max(len(_coerce_binary_mask(row.get("_response_mask"))) for row in rows)
        width = max(width, 1)

    reward_tensor = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    for index, reward_value in enumerate(rewards):
        mask = _coerce_binary_mask(rows[index].get("_response_mask"), length_hint=width)
        target_index = _last_true_index(mask)
        reward_tensor[index, target_index] = float(reward_value)

    reward_extra_infos_dict: dict[str, list[Any]] = {}
    for key, value in reward_info.items():
        if not isinstance(value, list):
            continue
        if len(value) != len(rows):
            continue
        reward_extra_infos_dict[str(key)] = list(value)

    if "feedback" not in reward_extra_infos_dict:
        reward_extra_infos_dict["feedback"] = ["" for _ in range(len(rows))]

    return reward_tensor, reward_extra_infos_dict


def _resolve_batch_size(*, responses: Any, non_tensor_batch: Mapping[str, Any]) -> int:
    if responses is not None:
        return _sequence_length(responses)
    for value in non_tensor_batch.values():
        length = _sequence_length(value)
        if length > 0:
            return length
    return 0


def _sequence_length(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "shape"):
        shape = getattr(value, "shape")
        if isinstance(shape, Sequence) and shape:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                pass
    try:
        return int(len(value))
    except TypeError:
        return 0


def _select_non_tensor(non_tensor_batch: Mapping[str, Any], key: str, index: int) -> Any:
    if key not in non_tensor_batch:
        return None
    return _select_index(non_tensor_batch.get(key), index)


def _select_index(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    try:
        return value[index]
    except Exception:
        return value


def _decode_response(*, tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return " ".join(str(token) for token in token_ids)
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        decoded = tokenizer.decode(token_ids)
    return _as_text(decoded)


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, Mapping):
            continue
        role = _as_text(item.get("role")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        content = _coerce_message_content(item.get("content"))
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _coerce_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text_piece = _as_text(item.get("text")).strip()
                if text_piece:
                    chunks.append(text_piece)
            else:
                text_piece = _as_text(item).strip()
                if text_piece:
                    chunks.append(text_piece)
        return "\n".join(chunks).strip()
    return _as_text(value).strip()


def _extract_prompt_text(*, messages: Sequence[Mapping[str, Any]], fallback: str) -> str:
    for item in messages:
        if _as_text(item.get("role")).strip().lower() != "user":
            continue
        content = _as_text(item.get("content")).strip()
        if content:
            return content
    fallback_text = fallback.strip()
    if fallback_text:
        return fallback_text
    return "SWE task prompt unavailable."


def _coerce_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append({str(key): item[key] for key in item})
    return rows


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[str] = []
    for item in value:
        text = _as_text(item).strip()
        if text:
            rows.append(text)
    return rows


def _coerce_nested_text_list(value: Any) -> list[list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[list[str]] = []
    for item in value:
        rows.append(_coerce_text_list(item))
    return rows


def _extract_last_tool_output(trajectory_steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not trajectory_steps:
        return {}
    last_step = trajectory_steps[-1]
    metadata_raw = last_step.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    return {
        "stdout": _as_text(last_step.get("stdout")),
        "stderr": _as_text(last_step.get("stderr")),
        "exit_code": _coerce_non_negative_int(last_step.get("exit_code"), fallback=0),
        "metadata": metadata,
    }


def _extract_reward_ground_truth(reward_model_value: Any) -> Mapping[str, Any]:
    if isinstance(reward_model_value, Mapping):
        ground_truth = reward_model_value.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            return ground_truth
    return {}


def _resolve_response_mask(*, raw_mask_value: Any, fallback_length: int) -> list[int]:
    mask = _coerce_binary_mask(raw_mask_value)
    if mask:
        return mask
    return [1] * max(fallback_length, 1)


def _coerce_binary_mask(value: Any, *, length_hint: int | None = None) -> list[int]:
    values = _coerce_int_list(value)
    if length_hint is not None and length_hint >= 1:
        if len(values) < length_hint:
            values = values + [0] * (length_hint - len(values))
        elif len(values) > length_hint:
            values = values[:length_hint]
    return [1 if item else 0 for item in values]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    detached = value
    if hasattr(detached, "detach"):
        detached = detached.detach()
    if hasattr(detached, "cpu"):
        detached = detached.cpu()
    if hasattr(detached, "tolist"):
        tolist_value = detached.tolist()
        if isinstance(tolist_value, Sequence) and not isinstance(tolist_value, (str, bytes)):
            return [_coerce_int_scalar(item) for item in tolist_value]
    if isinstance(detached, Sequence) and not isinstance(detached, (str, bytes)):
        return [_coerce_int_scalar(item) for item in detached]
    return []


def _coerce_int_scalar(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _last_true_index(mask: Sequence[int]) -> int:
    for index in range(len(mask) - 1, -1, -1):
        if int(mask[index]) != 0:
            return index
    return max(len(mask) - 1, 0)


def _coerce_non_empty_text(value: Any, *, fallback: str) -> str:
    text = _as_text(value).strip()
    if text:
        return text
    return fallback


def _coerce_non_negative_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, numbers.Integral):
        parsed = int(value)
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            parsed = int(stripped)
        except ValueError:
            return fallback
    else:
        return fallback
    if parsed < 0:
        return fallback
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


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
