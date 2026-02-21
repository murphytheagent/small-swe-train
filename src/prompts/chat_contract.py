"""Prompt fragments for ChatML + delimited tool-call contract."""

from __future__ import annotations

CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"


def build_assistant_contract_prompt(max_tool_calls: int = 3, terminal_tool: str = "submit") -> str:
    """Return an instruction block that matches the v1.6 action contract."""
    return (
        "Assistant output contract:\n"
        f"1) Optional reasoning span: {THINK_START}...{THINK_END}\n"
        f"2) 1..{max_tool_calls} ordered tool calls: "
        f"{TOOL_CALL_START}{{\"tool\":\"...\",\"args\":{{...}}}}{TOOL_CALL_END}\n"
        f"3) Terminal tool is '{terminal_tool}', and if present it must be the only tool call."
    )
