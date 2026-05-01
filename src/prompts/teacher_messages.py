"""Teacher-side prompt blocks for SDPO reprompting."""

from __future__ import annotations

from config import MAX_TOOL_CALLS_PER_TURN, TERMINAL_TOOL_NAME

from .model_delimiters import ModelDelimiters
from .runtime_messages import build_assistant_contract_prompt

_TURN_SUPERVISION_NEXT = "next_turn"
_TURN_SUPERVISION_CURRENT = "current_turn"
_TEACHER_TOOL_USAGE_GUIDANCE = (
    "Teacher-specific tool guidance:\n"
    "- Normal tool flow: use file_search to locate likely files, use text_search to locate exact strings or symbols inside a known scope, use read to inspect contents, and use apply_patch to edit; for apply_patch always include both args.path and args.patch.\n"
    "- You may reuse an exact repo-relative path the student already found for your own read, text_search, or apply_patch calls; do not guess new prefixes or repeat file_search unless needed.\n"
    "- If your corrected action completes the task, use the terminal submit tool in this turn; do not avoid submit just because this is a revised student turn.\n"
)


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
    action_payload_format: str | None = None,
    supervision_mode: str = _TURN_SUPERVISION_NEXT,
) -> str:
    """Build OUTPUT_CONTRACT_BLOCK text that steers teacher policy improvement."""
    normalized_mode = _normalize_supervision_mode(supervision_mode)
    base_contract = build_assistant_contract_prompt(
        delimiters=delimiters,
        max_tool_calls=max_tool_calls,
        terminal_tool=terminal_tool,
        action_payload_format=action_payload_format,
        include_tool_schema=False,
        include_examples=False,
        include_repeat_warning=False,
    )
    if normalized_mode == _TURN_SUPERVISION_CURRENT:
        return (
            "Now that you have seen the student's attempt, adhere to following contracts in your revised attempt:\n"
            f"{base_contract}\n"
            f"{_TEACHER_TOOL_USAGE_GUIDANCE}"
            "Produce the best corrected action for the current turn.\n"
        )
    return (
        "Now that you have seen the student's attempt, adhere to following contracts in your revised attempt:\n"
        f"{base_contract}\n"
        f"{_TEACHER_TOOL_USAGE_GUIDANCE}"
        "Now correctly solve the original issue, focus only on what to do best in the next turn.\n"
    )
