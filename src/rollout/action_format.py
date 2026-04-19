"""Shared parse/render helpers for assistant-action payloads.

This module centralizes the assistant-action surface so future payload-format
migrations can move one entrypoint at a time instead of updating each consumer
independently.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from config import MAX_TOOL_CALLS_PER_TURN
from prompts.model_delimiters import ModelDelimiters, default_delimiters
from schemas import ActionEnvelope, ToolCall

from .turn_parser import parse_assistant_turn_payload, parse_chatml_assistant_turn


def is_chatml_assistant_turn(
    assistant_text: str,
    *,
    delimiters: ModelDelimiters | None = None,
) -> bool:
    """Return True when *assistant_text* looks like a full ChatML assistant turn."""
    d = delimiters or default_delimiters()
    return assistant_text.strip().startswith(f"{d.role_start}assistant")


def parse_assistant_text(
    assistant_text: str,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
) -> ActionEnvelope:
    """Parse raw assistant text as either ChatML turn text or bare payload."""
    stripped = assistant_text.strip()
    if is_chatml_assistant_turn(stripped):
        return parse_chatml_assistant_turn(stripped, max_tool_calls=max_tool_calls)
    return parse_assistant_turn_payload(stripped, max_tool_calls=max_tool_calls)


def serialize_tool_call_payload(
    call: ToolCall | Mapping[str, Any],
    *,
    compact: bool = False,
) -> str:
    """Serialize one tool-call payload deterministically."""
    payload = call.to_dict() if isinstance(call, ToolCall) else dict(call)
    dump_kwargs: dict[str, Any] = {
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if compact:
        dump_kwargs["separators"] = (",", ":")
    return json.dumps(payload, **dump_kwargs)


def render_tool_call_block(
    call: ToolCall | Mapping[str, Any],
    *,
    delimiters: ModelDelimiters | None = None,
    compact: bool = False,
) -> str:
    """Render one tool-call block using the current delimiter contract."""
    d = delimiters or default_delimiters()
    payload = serialize_tool_call_payload(call, compact=compact)
    return f"{d.tool_call_start}{payload}{d.tool_call_end}"


def render_assistant_action_text(
    envelope: ActionEnvelope,
    *,
    delimiters: ModelDelimiters | None = None,
    compact: bool = False,
) -> str:
    """Render one assistant action envelope using the current delimiter contract."""
    d = delimiters or default_delimiters()
    chunks: list[str] = []
    if envelope.thinking:
        chunks.append(f"{d.think_start}{envelope.thinking}{d.think_end}")
    chunks.extend(
        render_tool_call_block(call, delimiters=d, compact=compact)
        for call in envelope.tool_calls
    )
    return "".join(chunks)
