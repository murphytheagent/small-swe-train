"""Deterministic conversion from SWE-style traces to verl-ready training rows."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from data.feedback_canonicalizer import build_feedback_packet
from data.tool_schema_adapter import adapt_external_tool_call
from losses.action_masking import build_action_token_mask
from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn
from schemas import ActionEnvelope, ToolCall, validate_tool_call


def _parse_assistant_response(assistant_response: str, *, max_tool_calls: int) -> ActionEnvelope:
    stripped = assistant_response.strip()
    if stripped.startswith("<|im_start|>assistant"):
        return parse_chatml_assistant_turn(stripped, max_tool_calls=max_tool_calls)
    return parse_assistant_turn_payload(stripped, max_tool_calls=max_tool_calls)


def _adapt_external_calls(external_calls: Sequence[Any]) -> tuple[ToolCall, ...]:
    adapted_calls: list[ToolCall] = []
    for call_index, raw_call in enumerate(external_calls):
        if not isinstance(raw_call, Mapping):
            raise ValueError(
                f"external_tool_calls[{call_index}] must be a mapping with 'tool' and 'args'."
            )
        tool_name = raw_call.get("tool")
        args = raw_call.get("args", {})
        if not isinstance(tool_name, str):
            raise ValueError("Each external tool call requires string 'tool'.")
        if not isinstance(args, Mapping):
            raise ValueError("Each external tool call requires mapping 'args'.")
        adapted_calls.append(adapt_external_tool_call(tool_name, args))
    if not adapted_calls:
        raise ValueError("At least one external tool call is required when assistant_response is absent.")
    return tuple(adapted_calls)


def _labels_from_envelope(envelope: ActionEnvelope) -> list[str]:
    labels: list[str] = []
    if envelope.thinking:
        think_tokens = max(1, len(envelope.thinking.split()))
        labels.extend(["think"] * think_tokens)

    for call in envelope.tool_calls:
        serialized = json.dumps(call.to_dict(), sort_keys=True, ensure_ascii=True)
        tool_tokens = max(1, len(serialized.split()))
        labels.extend(["tool_call"] * tool_tokens)

    if not labels:
        labels.append("other")
    return labels


def _coerce_step_index(value: Any, *, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise ValueError("step_index must be an integer >= 0")
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError("step_index must be an integer >= 0")
        coerced = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            coerced = int(stripped)
        except ValueError as exc:
            raise ValueError("step_index must be an integer >= 0") from exc
    else:
        raise ValueError("step_index must be an integer >= 0")

    if coerced < 0:
        raise ValueError("step_index must be an integer >= 0")
    return coerced


def preprocess_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    max_tool_calls: int = 3,
) -> list[dict[str, Any]]:
    """Convert raw trajectory examples into deterministic, verl-style row dicts.

    Per input sample, one output row is emitted with parser/validation diagnostics,
    stage masks, and canonicalized feedback packet metadata.
    """
    rows: list[dict[str, Any]] = []

    for index, sample in enumerate(trajectories):
        prompt = str(sample.get("prompt", ""))
        assistant_response_raw = sample.get("assistant_response", "")
        if assistant_response_raw is None:
            assistant_response = ""
        elif isinstance(assistant_response_raw, str):
            assistant_response = assistant_response_raw
        else:
            assistant_response = str(assistant_response_raw)
        include_student_attempt_for_teacher = bool(
            sample.get("include_student_attempt_for_teacher", True)
        )

        tool_calls: tuple[ToolCall, ...] = ()
        parse_error: str | None = None

        try:
            step_index = _coerce_step_index(sample.get("step_index"), fallback=index)
            if assistant_response:
                envelope = _parse_assistant_response(
                    assistant_response,
                    max_tool_calls=max_tool_calls,
                )
                tool_calls = envelope.tool_calls
            else:
                external_calls = sample.get("external_tool_calls", [])
                if isinstance(external_calls, (str, bytes)) or not isinstance(
                    external_calls, Sequence
                ):
                    raise ValueError("external_tool_calls must be a sequence of call objects")
                tool_calls = _adapt_external_calls(external_calls)
                envelope = ActionEnvelope(tool_calls=tool_calls, thinking=sample.get("thinking"))
        except (TurnParseError, ValueError) as exc:
            parse_error = str(exc)
            envelope = None

        validation_errors: list[str] = []
        if envelope is not None:
            for call_index, call in enumerate(envelope.tool_calls):
                errors = validate_tool_call(call)
                validation_errors.extend(
                    f"tool_call[{call_index}]: {error}" for error in errors
                )

        token_labels = _labels_from_envelope(envelope) if envelope is not None else []
        action_mask_rft = build_action_token_mask(token_labels, stage="rft") if token_labels else []
        action_mask_step_sdpo = (
            build_action_token_mask(token_labels, stage="step_sdpo") if token_labels else []
        )

        tool_output = sample.get("tool_output", {})
        if not isinstance(tool_output, Mapping):
            tool_output = {}

        if envelope is not None:
            first_tool = envelope.tool_calls[0]
            feedback_packet = build_feedback_packet(
                step_index=step_index,
                tool=first_tool.tool,
                tool_input=first_tool.args,
                tool_output=tool_output,
                include_student_attempt_for_teacher=include_student_attempt_for_teacher,
            )
            feedback_payload = feedback_packet.to_dict()
            tool_calls_payload = [call.to_dict() for call in envelope.tool_calls]
        else:
            feedback_payload = None
            tool_calls_payload = []

        rows.append(
            {
                "prompt": prompt,
                "assistant_response": assistant_response,
                "tool_calls": tool_calls_payload,
                "token_labels": token_labels,
                "action_mask_rft": action_mask_rft,
                "action_mask_step_sdpo": action_mask_step_sdpo,
                "format_valid": envelope is not None and not validation_errors,
                "parse_error": parse_error,
                "validation_errors": validation_errors,
                "feedback_packet": feedback_payload,
            }
        )

    return rows
