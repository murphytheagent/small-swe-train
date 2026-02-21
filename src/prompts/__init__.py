"""Prompt package."""

from .chat_contract import (
    CHATML_END,
    CHATML_START,
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    TOOL_RESPONSE_END,
    TOOL_RESPONSE_START,
    build_assistant_contract_prompt,
)

__all__ = [
    "CHATML_END",
    "CHATML_START",
    "THINK_END",
    "THINK_START",
    "TOOL_CALL_END",
    "TOOL_CALL_START",
    "TOOL_RESPONSE_END",
    "TOOL_RESPONSE_START",
    "build_assistant_contract_prompt",
]
