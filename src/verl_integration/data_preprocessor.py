"""Deterministic conversion from SWE-style traces to verl-ready training rows."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from data.feedback_canonicalizer import build_feedback_packet
from data.tokenization import (
    LabeledSpan,
    SupportsOffsetsTokenizer,
    build_labeled_spans,
    tokenize_batch_with_labels,
)
from data.tool_schema_adapter import adapt_external_tool_call
from losses.action_masking import build_action_token_mask
from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn
from config import MAX_TOOL_CALLS_PER_TURN
from schemas import ActionEnvelope, ToolCall, validate_tool_call

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


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


def _label_blocks_from_envelope(envelope: ActionEnvelope) -> list[dict[str, str]]:
    """Return structured block metadata for tokenizer-aligned mask generation."""
    blocks: list[dict[str, str]] = []
    if envelope.thinking:
        blocks.append({"type": "think", "text": envelope.thinking})
    for call in envelope.tool_calls:
        serialized = json.dumps(call.to_dict(), sort_keys=True, ensure_ascii=True)
        blocks.append({"type": "tool_call", "text": serialized})
    return blocks


def _approx_labels_from_envelope(envelope: ActionEnvelope) -> list[str]:
    """Approximate per-token labels using whitespace word counts.

    WARNING: These counts will NOT match subword tokenizer output.  Use
    ``label_blocks`` together with the real tokenizer to generate masks
    that are aligned with actual token IDs.
    """
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


def _coerce_bool_flag(value: Any, *, fallback: bool) -> bool:
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
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return fallback


def preprocess_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    tokenizer: SupportsOffsetsTokenizer | None = None,
) -> list[dict[str, Any]]:
    """Convert raw trajectory examples into deterministic, verl-style row dicts.

    Per input sample, one output row is emitted with parser/validation diagnostics,
    stage masks, and canonicalized feedback packet metadata.

    When *tokenizer* is provided, ``input_ids``, ``token_labels``, and stage
    masks are derived from real subword tokenization with offset-aligned labels.
    Otherwise approximate whitespace-based labels are emitted.
    """
    rows: list[dict[str, Any]] = []
    tokenization_row_indexes: list[int] = []
    tokenization_texts: list[str] = []
    tokenization_spans: list[list[LabeledSpan]] = []

    for index, sample in enumerate(trajectories):
        prompt = str(sample.get("prompt", ""))
        assistant_response_raw = sample.get("assistant_response", "")
        if assistant_response_raw is None:
            assistant_response = ""
        elif isinstance(assistant_response_raw, str):
            assistant_response = assistant_response_raw
        else:
            assistant_response = str(assistant_response_raw)
        include_student_attempt_for_teacher = _coerce_bool_flag(
            sample.get("include_student_attempt_for_teacher"),
            fallback=True,
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
                thinking_raw = sample.get("thinking")
                thinking = str(thinking_raw) if thinking_raw is not None else None
                envelope = ActionEnvelope(tool_calls=tool_calls, thinking=thinking)
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

        label_blocks = _label_blocks_from_envelope(envelope) if envelope is not None else []
        canonical_text: str | None = None

        if envelope is not None and tokenizer is not None:
            canonical_text, labeled_spans = build_labeled_spans(envelope)
            token_labels = []
            tokenization_row_indexes.append(len(rows))
            tokenization_texts.append(canonical_text)
            tokenization_spans.append(labeled_spans)
        else:
            token_labels = _approx_labels_from_envelope(envelope) if envelope is not None else []

        action_mask_format_rft = (
            build_action_token_mask(token_labels, stage="format_rft") if token_labels else []
        )
        action_mask_positive_rft = (
            build_action_token_mask(token_labels, stage="positive_rft") if token_labels else []
        )
        action_mask_turn_sdpo = (
            build_action_token_mask(token_labels, stage="turn_sdpo") if token_labels else []
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

        row: dict[str, Any] = {
            "prompt": prompt,
            "assistant_response": assistant_response,
            "tool_calls": tool_calls_payload,
            "label_blocks": label_blocks,
            "token_labels": token_labels,
            "stage": "format_rft",
            "action_mask_format_rft": action_mask_format_rft,
            "action_mask_positive_rft": action_mask_positive_rft,
            "action_mask_turn_sdpo": action_mask_turn_sdpo,
            "action_mask_rft": action_mask_format_rft,
            "action_mask_step_sdpo": action_mask_turn_sdpo,
            "assistant_action_token_count": sum(1 for flag in action_mask_format_rft if flag),
            "format_valid": envelope is not None and not validation_errors,
            "parse_error": parse_error,
            "validation_errors": validation_errors,
            "feedback_packet": feedback_payload,
        }
        if canonical_text is not None:
            row["canonical_text"] = canonical_text
        rows.append(row)

    if tokenizer is not None and tokenization_row_indexes:
        batch_input_ids, batch_labels = tokenize_batch_with_labels(
            tokenization_texts,
            tokenization_spans,
            tokenizer,
        )
        for row_index, input_ids, token_labels in zip(
            tokenization_row_indexes,
            batch_input_ids,
            batch_labels,
        ):
            row = rows[row_index]
            row["input_ids"] = input_ids
            row["token_labels"] = token_labels
            row["action_mask_format_rft"] = (
                build_action_token_mask(token_labels, stage="format_rft") if token_labels else []
            )
            row["action_mask_positive_rft"] = (
                build_action_token_mask(token_labels, stage="positive_rft") if token_labels else []
            )
            row["action_mask_turn_sdpo"] = (
                build_action_token_mask(token_labels, stage="turn_sdpo") if token_labels else []
            )
            row["action_mask_rft"] = row["action_mask_format_rft"]
            row["action_mask_step_sdpo"] = row["action_mask_turn_sdpo"]
            row["assistant_action_token_count"] = sum(
                1 for flag in row["action_mask_format_rft"] if flag
            )

    return rows
