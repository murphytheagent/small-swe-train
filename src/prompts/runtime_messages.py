"""Runtime prompt templates and tool-contract text for on-policy rollouts."""

from __future__ import annotations

import json
import types
from typing import Any, Mapping, get_args, get_origin, get_type_hints

from config import MAX_TOOL_CALLS_PER_TURN, TERMINAL_TOOL_NAME
from schemas import ALLOWED_TOOLS, TOOL_SCHEMAS

from .model_delimiters import ModelDelimiters, default_delimiters

_d = default_delimiters()
CHATML_START: str = _d.role_start
CHATML_END: str = _d.role_end
TOOL_CALL_START: str = _d.tool_call_start
TOOL_CALL_END: str = _d.tool_call_end
TOOL_RESPONSE_START: str = _d.tool_response_start
TOOL_RESPONSE_END: str = _d.tool_response_end
del _d

_DEFAULT_SYSTEM_PROMPT_PREFIX = (
    "You are a software engineering agent working in a real repository.\n"
    "Inspect code, run tools, apply targeted patches, and validate behavior with tests.\n"
    "Return one assistant turn at a time and follow the tool-output contract exactly.\n"
)
_SDPO_ROLLOUT_FOLLOWUP_USER_MESSAGE = (
    "Return the next assistant turn now. Use bash/search/edit while still working. "
    "If solved, return one submit tool call with a concise final_response."
)


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


def _build_required_args_prompt() -> str:
    required_fields: list[str] = []
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not isinstance(schema, Mapping):
            continue
        required_raw = schema.get("required")
        if not isinstance(required_raw, (list, tuple, set)):
            continue
        for field_name in required_raw:
            if isinstance(field_name, str):
                required_fields.append(f"{tool_name}.{field_name}")
    rendered = ", ".join(required_fields) if required_fields else "-"
    return f"5) Required args by tool: {rendered}.\n"


def _build_tool_examples_prompt() -> str:
    examples: list[str] = []
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not isinstance(schema, Mapping):
            continue
        example_raw = schema.get("prompt_example")
        if not isinstance(example_raw, Mapping):
            continue
        serialized = json.dumps(dict(example_raw), ensure_ascii=True, sort_keys=True)
        examples.append(f"   - {tool_name}: {serialized}")
    if not examples:
        return ""
    return "9) Realistic examples (one tool call each):\n" + "\n".join(examples)


def build_assistant_contract_prompt(
    *,
    delimiters: ModelDelimiters | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    terminal_tool: str = TERMINAL_TOOL_NAME,
) -> str:
    """Return an instruction block that matches the v1.6 action contract."""
    d = delimiters or default_delimiters()
    allowed_tools_text = ", ".join(ALLOWED_TOOLS)
    tool_schema_block = _build_tool_schema_prompt()
    tool_examples_block = _build_tool_examples_prompt()
    suffix = (
        f"{tool_schema_block}\n{tool_examples_block}"
        if tool_examples_block
        else tool_schema_block
    )
    return (
        "Assistant output contract:\n"
        "1) Surround each tool action with a tool-call delimiter block.\n"
        f"2) Emit 1..{max_tool_calls} ordered tool calls: "
        f"{d.tool_call_start}{{\"tool\":\"...\",\"args\":{{...}}}}{d.tool_call_end}\n"
        "3) Every tool-call JSON object MUST include both keys: 'tool' and 'args'.\n"
        "   'args' MUST be a JSON object (never put command/query/path at top level).\n"
        f"4) Allowed tools: {allowed_tools_text}.\n"
        f"{_build_required_args_prompt()}"
        "6) Do not invent tool names or wrapper labels.\n"
        f"7) Terminal tool is '{terminal_tool}', you must end conversation with this tool, and if present it must be the only tool call.\n"
        f"{suffix}"
    )


def build_onpolicy_system_prompt() -> str:
    """Build the default system prompt for on-policy runtime rollouts."""
    return _DEFAULT_SYSTEM_PROMPT_PREFIX + build_assistant_contract_prompt()


def build_sdpo_rollout_followup_user_message() -> str:
    """Build the continuation nudge for bridge-loop SDPO rollouts."""
    return _SDPO_ROLLOUT_FOLLOWUP_USER_MESSAGE


def build_onpolicy_initial_user_message(
    *,
    problem_statement: str,
    fail_to_pass: Any,
    pass_to_pass: Any,
) -> str:
    """Build the initial user message for one on-policy task attempt."""
    fail_to_pass_text = _stable_json(fail_to_pass)
    pass_to_pass_text = _stable_json(pass_to_pass)
    return (
        "You are solving one software engineering task.\n"
        "Task objective:\n"
        f"{problem_statement}\n\n"
        "Test targets:\n"
        "- FAIL_TO_PASS: tests currently failing that should pass after your fix.\n"
        f"{fail_to_pass_text}\n\n"
        "- PASS_TO_PASS: tests currently passing that must keep passing (regression guard).\n"
        f"{pass_to_pass_text}\n\n"
        "Execution guidance:\n"
        "- Use tool calls to inspect code, apply patches, and run validation commands.\n"
        "- Submit only when you are ready to end the attempt."
    )


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)
