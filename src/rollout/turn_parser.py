"""ChatML assistant-turn parser for optional thinking + ordered tool calls."""

from __future__ import annotations

import json
import re
from typing import Iterable

from schemas import ActionEnvelope, ToolCall, make_tool_call

_ASSISTANT_PREFIX = "<|im_start|>assistant"
_ASSISTANT_END = "<|im_end|>"
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class TurnParseError(ValueError):
    """Raised when assistant-turn payload cannot be parsed safely."""


def extract_chatml_assistant_payload(turn_text: str) -> str:
    """Extract assistant payload between ChatML assistant delimiters."""
    stripped = turn_text.strip()
    if not stripped.startswith(_ASSISTANT_PREFIX):
        raise TurnParseError("Turn does not start with '<|im_start|>assistant'.")
    end_index = stripped.rfind(_ASSISTANT_END)
    if end_index < 0:
        raise TurnParseError("Missing '<|im_end|>' terminator.")
    tail = stripped[end_index + len(_ASSISTANT_END) :].strip()
    if tail:
        raise TurnParseError("Unexpected text after ChatML end delimiter.")
    payload = stripped[len(_ASSISTANT_PREFIX) : end_index]
    return payload.lstrip("\n").strip()


def _strip_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    cursor = 0
    chunks: list[str] = []
    for start, end in spans:
        chunks.append(text[cursor:start])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def parse_assistant_turn_payload(payload: str, max_tool_calls: int = 3) -> ActionEnvelope:
    """Parse assistant payload into canonical action envelope."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be >= 1")

    think_opens = payload.count("<think>")
    think_closes = payload.count("</think>")
    if think_opens != think_closes:
        raise TurnParseError("Unbalanced <think> delimiters.")

    think_matches = list(_THINK_PATTERN.finditer(payload))
    if len(think_matches) > 1:
        raise TurnParseError("At most one <think> block is allowed per assistant turn.")

    thinking: str | None = None
    spans_to_strip: list[tuple[int, int]] = []
    if think_matches:
        match = think_matches[0]
        thinking = match.group(1).strip() or None
        spans_to_strip.append((match.start(), match.end()))

    tool_matches = list(_TOOL_CALL_PATTERN.finditer(payload))
    if not tool_matches:
        raise TurnParseError("At least one <tool_call> block is required.")
    if len(tool_matches) > max_tool_calls:
        raise TurnParseError(
            f"Too many tool calls: got {len(tool_matches)}, max is {max_tool_calls}."
        )

    tool_calls: list[ToolCall] = []
    for match in tool_matches:
        spans_to_strip.append((match.start(), match.end()))
        raw_json = match.group(1).strip()
        try:
            payload_obj = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise TurnParseError(f"Invalid tool_call JSON: {exc.msg}") from exc
        if not isinstance(payload_obj, dict):
            raise TurnParseError("Each <tool_call> payload must decode to a JSON object.")
        tool_calls.append(make_tool_call(payload_obj))

    leftover = _strip_spans(payload, sorted(spans_to_strip, key=lambda span: span[0]))
    if leftover.strip():
        raise TurnParseError(
            "Assistant payload contains text outside <think>/<tool_call> blocks."
        )

    try:
        return ActionEnvelope(tool_calls=tuple(tool_calls), thinking=thinking)
    except ValueError as exc:
        raise TurnParseError(str(exc)) from exc


def parse_chatml_assistant_turn(turn_text: str, max_tool_calls: int = 3) -> ActionEnvelope:
    """Parse a full ChatML assistant turn string into an ActionEnvelope."""
    payload = extract_chatml_assistant_payload(turn_text)
    return parse_assistant_turn_payload(payload, max_tool_calls=max_tool_calls)
