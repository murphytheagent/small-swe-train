"""Prompt fragments for delimited tool-call contract.

Delimiter strings are loaded from the model-family YAML config via
``default_delimiters()``.  Legacy module-level constants (``CHATML_START``,
etc.) are derived from the default (Qwen3) config for backward compatibility.
"""

from __future__ import annotations

from prompts.model_delimiters import ModelDelimiters, default_delimiters
from runtime_config import DEFAULT_MAX_TOOL_CALLS_PER_TURN

_d = default_delimiters()
CHATML_START: str = _d.role_start
CHATML_END: str = _d.role_end
THINK_START: str = _d.think_start
THINK_END: str = _d.think_end
TOOL_CALL_START: str = _d.tool_call_start
TOOL_CALL_END: str = _d.tool_call_end
TOOL_RESPONSE_START: str = _d.tool_response_start
TOOL_RESPONSE_END: str = _d.tool_response_end
del _d


def build_assistant_contract_prompt(
    *,
    delimiters: ModelDelimiters | None = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN,
    terminal_tool: str = "submit",
) -> str:
    """Return an instruction block that matches the v1.6 action contract."""
    d = delimiters or default_delimiters()
    return (
        "Assistant output contract:\n"
        f"1) Optional reasoning span: {d.think_start}...{d.think_end}\n"
        f"2) 1..{max_tool_calls} ordered tool calls: "
        f"{d.tool_call_start}{{\"tool\":\"...\",\"args\":{{...}}}}{d.tool_call_end}\n"
        f"3) Terminal tool is '{terminal_tool}', and if present it must be the only tool call."
    )
