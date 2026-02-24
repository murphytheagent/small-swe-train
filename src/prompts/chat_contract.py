"""Prompt fragments for delimited tool-call contract.

Delimiter strings are loaded from the model-family YAML config via
``default_delimiters()``.  Legacy module-level constants (``CHATML_START``,
etc.) are derived from the default (Qwen3) config for backward compatibility.
"""

from __future__ import annotations

import types
from typing import Any, Mapping, get_args, get_origin, get_type_hints

from prompts.model_delimiters import ModelDelimiters, default_delimiters
from config import MAX_TOOL_CALLS_PER_TURN, TERMINAL_TOOL_NAME
from schemas import ALLOWED_TOOLS, TOOL_SCHEMAS

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


def _type_label(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            if annotation is float:
                return "number"
            return annotation.__name__
        return str(annotation)
    if origin in (list, tuple, set):
        args = get_args(annotation)
        inner = _type_label(args[0]) if args else "Any"
        return f"{origin.__name__}[{inner}]"
    if origin is dict:
        args = get_args(annotation)
        if len(args) == 2:
            return f"dict[{_type_label(args[0])}, {_type_label(args[1])}]"
        return "dict"
    if origin is types.UnionType or str(origin) == "typing.Union":
        args = [arg for arg in get_args(annotation) if arg is not type(None)]  # noqa: E721
        return " | ".join(_type_label(arg) for arg in args) or "Any"
    return str(annotation)


def _field_label(name: str, annotation: Any, constraints: Mapping[str, Any]) -> str:
    qualifiers: list[str] = []
    min_length = constraints.get("min_length")
    minimum = constraints.get("minimum")
    maximum = constraints.get("maximum")
    if isinstance(min_length, int):
        qualifiers.append(f"min_len={min_length}")
    if minimum is not None and maximum is not None:
        qualifiers.append(f"{minimum}..{maximum}")
    elif minimum is not None:
        qualifiers.append(f">={minimum}")
    elif maximum is not None:
        qualifiers.append(f"<={maximum}")
    base = f"{name}:{_type_label(annotation)}"
    if not qualifiers:
        return base
    return f"{base}({', '.join(qualifiers)})"


def _tool_schema_line(tool_name: str, schema: Mapping[str, Any]) -> str:
    source = schema.get("source")
    if not isinstance(source, type):
        return f"   - {tool_name} args: <schema unavailable>"

    hints = get_type_hints(source)
    constraints_raw = schema.get("constraints")
    constraints = dict(constraints_raw) if isinstance(constraints_raw, Mapping) else {}
    required_raw = schema.get("required")
    required_set: set[str] = set()
    if isinstance(required_raw, (list, tuple, set)):
        required_set = {str(name) for name in required_raw if isinstance(name, str)}
    required_fields: list[str] = []
    optional_fields: list[str] = []
    for field_name, annotation in hints.items():
        field_constraints_raw = constraints.get(field_name)
        field_constraints = (
            dict(field_constraints_raw) if isinstance(field_constraints_raw, Mapping) else {}
        )
        rendered = _field_label(field_name, annotation, field_constraints)
        if field_name in required_set:
            required_fields.append(rendered)
        else:
            optional_fields.append(rendered)

    required_block = ", ".join(required_fields) if required_fields else "-"
    optional_block = ", ".join(optional_fields) if optional_fields else "-"
    return f"   - {tool_name} args: required {{{required_block}}}; optional {{{optional_block}}}"


def _build_tool_schema_prompt() -> str:
    lines = ["8) Tool arg schema (from TOOL_SCHEMAS):"]
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if isinstance(schema, Mapping):
            lines.append(_tool_schema_line(tool_name, schema))
    return "\n".join(lines)


def build_assistant_contract_prompt(
    *,
    delimiters: ModelDelimiters | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    terminal_tool: str = TERMINAL_TOOL_NAME,
) -> str:
    """Return an instruction block that matches the v1.6 action contract."""
    d = delimiters or default_delimiters()
    allowed_tools_text = ", ".join(ALLOWED_TOOLS)
    return (
        "Assistant output contract:\n"
        "1) Output only contract blocks; no plain prose outside delimiters.\n"
        f"2) Optional reasoning span: {d.think_start}...{d.think_end}\n"
        f"3) 1..{max_tool_calls} ordered tool calls: "
        f"{d.tool_call_start}{{\"tool\":\"...\",\"args\":{{...}}}}{d.tool_call_end}\n"
        f"4) Allowed tools are exactly: {allowed_tools_text}.\n"
        "5) Required args by tool: "
        "bash.command, search.query, edit.path+edit.patch, submit.final_response.\n"
        "6) Do not invent tool names or wrapper labels.\n"
        f"7) Terminal tool is '{terminal_tool}', and if present it must be the only tool call.\n"
        f"{_build_tool_schema_prompt()}"
    )
