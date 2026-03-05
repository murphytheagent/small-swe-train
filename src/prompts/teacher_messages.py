"""Teacher-side prompt blocks for SDPO reprompting."""

from __future__ import annotations

from config import MAX_TOOL_CALLS_PER_TURN, TERMINAL_TOOL_NAME

from .model_delimiters import ModelDelimiters
from .runtime_messages import build_assistant_contract_prompt

_TURN_SUPERVISION_NEXT = "next_turn"
_TURN_SUPERVISION_CURRENT = "current_turn"


def _normalize_supervision_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return _TURN_SUPERVISION_NEXT
    if normalized in {_TURN_SUPERVISION_NEXT, _TURN_SUPERVISION_CURRENT}:
        return normalized
    return _TURN_SUPERVISION_NEXT


def build_teacher_output_contract_block(
    *,
    delimiters: ModelDelimiters | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    terminal_tool: str = TERMINAL_TOOL_NAME,
    supervision_mode: str = _TURN_SUPERVISION_NEXT,
) -> str:
    """Build OUTPUT_CONTRACT_BLOCK text that steers teacher policy improvement."""
    normalized_mode = _normalize_supervision_mode(supervision_mode)
    base_contract = build_assistant_contract_prompt(
        delimiters=delimiters,
        max_tool_calls=max_tool_calls,
        terminal_tool=terminal_tool,
        include_tool_schema=False,
        include_examples=False,
        include_repeat_warning=False,
    )
    if normalized_mode == _TURN_SUPERVISION_CURRENT:
        return (
            "Now that you have seen the student's attempt, adhere to following contracts in your revised attempt:\n"
            f"{base_contract}\n"
            "Produce the best corrected action for the current turn.\n"
        )
    return (
        "Now that you have seen the student's attempt, adhere to following contracts in your revised attempt:\n"
        f"{base_contract}\n"
        "Now correctly solve the original issue, focus only on what to do best in the next turn.\n"
    )
