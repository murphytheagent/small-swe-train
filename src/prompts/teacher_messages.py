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
    )
    if normalized_mode == _TURN_SUPERVISION_CURRENT:
        return (
            "Teacher objective (turn-level SDPO, current-turn reflection):\n"
            "1) The conversation above is a previous attempt at fixing the task.\n"
            "2) Use the compacted trajectory evidence (including tool/verifier feedback) to reflect on what the teacher should have done differently for this current turn.\n"
            "3) Keep the revision grounded in observed failures and improve only the current turn decision quality.\n"
            "4) All tools in this attempt have already executed, so assume a potentially modified repository state.\n"
            f"{base_contract}\n"
            "Now produce the best corrected action for the current turn.\n"
        )
    return (
        "Teacher objective (turn-level SDPO):\n"
        "1) The multi-turn conversation above is a previous attempt at fixing the issue stated in the task statement.\n"
        "2) This attempt is not complete, it can be on the right track or completely wrong.\n"
        "3) Learn from the interactions from this attempt, correct mistakes, and take correct actions in the next turn.\n"
        "4) All tools in this attempt have been executed already, so assume you are working potentially modified repo.\n"
        f"{base_contract}\n"
        "Now correctly solve the original issue, focus only on what to do best in the next turn.\n"
    )
