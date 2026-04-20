"""Runtime prompt templates and tool-contract text for on-policy rollouts."""

from __future__ import annotations

import json
import types
from typing import Any, Mapping, get_args, get_origin, get_type_hints

from config import (
    ACTION_PAYLOAD_FORMAT,
    MAX_TOOL_CALLS_PER_TURN,
    SUPPORTED_ACTION_PAYLOAD_FORMATS,
    TERMINAL_TOOL_NAME,
)
from rollout.action_format import render_tool_call_block
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
_DEFAULT_SYSTEM_PROMPT_WORKFLOW = (
    "First inspect surrounding code, related tests, and project configuration so you understand how "
    "this repository is built and validated.\n"
    "When practical, understand the root cause and broader context before editing, but do not wait "
    "for complete understanding before making progress.\n"
    "Make focused, minimal changes that address the issue instead of speculative churn.\n"
    "Use the bash tool to run repository-specific validation commands. Do not assume a standard "
    "test command; infer the correct commands from this codebase.\n"
    "If applicable and feasible, verify your changes with the repository's own tests, build, lint, "
    "or other validation procedures.\n"
    "Start with the most specific relevant validation you can identify from the changed code, "
    "nearby tests, stack traces, failing outputs, or naming conventions, then broaden to more "
    "general checks as confidence grows.\n"
    "If the exact targeted command is unclear or too expensive to find quickly, run a lightweight "
    "related validation instead of skipping validation entirely.\n"
    "If no relevant test exists, add one when appropriate and feasible while following existing "
    "test patterns.\n"
    "If the environment prevents validation, state that limitation briefly and make the best "
    "supported patch you can.\n"
)
_SDPO_ROLLOUT_FOLLOWUP_USER_MESSAGE = (
    "Return the next assistant turn now. Use bash/read/file_search/text_search/apply_patch while still working. "
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
    lines = ["Tool arg schema:"]
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
    return f"Required args by tool: {rendered}."


def _normalize_action_payload_format(value: str | None) -> str:
    normalized = str(value or ACTION_PAYLOAD_FORMAT).strip().lower()
    if normalized not in SUPPORTED_ACTION_PAYLOAD_FORMATS:
        allowed = ", ".join(SUPPORTED_ACTION_PAYLOAD_FORMATS)
        raise ValueError(f"Unsupported action payload format {normalized!r}. Expected one of: {allowed}.")
    return normalized


def _render_contract_signature(*, delimiters: ModelDelimiters, action_payload_format: str) -> str:
    if action_payload_format == "json":
        return f'{delimiters.tool_call_start}{{"tool":"...","args":{{...}}}}{delimiters.tool_call_end}'
    return '<tool_call name="tool_name"><arg_name><![CDATA[value]]></arg_name></tool_call>'


def _build_tool_examples_prompt(*, action_payload_format: str) -> str:
    examples: list[str] = []
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not isinstance(schema, Mapping):
            continue
        example_raw = schema.get("prompt_example")
        if not isinstance(example_raw, Mapping):
            continue
        if action_payload_format == "json":
            serialized = json.dumps(dict(example_raw), ensure_ascii=True, sort_keys=True)
        else:
            serialized = render_tool_call_block(
                dict(example_raw),
                payload_format=action_payload_format,
            )
        examples.append(f"   - {tool_name}: {serialized}")
    if not examples:
        return ""
    return "Realistic examples (one tool call each):\n" + "\n".join(examples)


def _build_tool_usage_guardrails_prompt() -> str:
    lines = [
        "Tool usage guardrails:",
        "   - Use 'read' to inspect file contents and line ranges. Prefer it over bash commands like cat/sed/head for reading code.",
        "   - Use 'file_search' to discover likely repo-relative file paths under the repository root.",
        "   - Use 'text_search' for exact fixed-string matches inside a known file or directory scope.",
        "   - If 'read' says the output was truncated, call 'read' again with narrower start_line/end_line before editing.",
        "   - If 'read' says path not found, use 'file_search' or a lightweight directory listing to find the real repo-relative path. Do not guess the path.",
        "   - For 'apply_patch', always include both 'path' and 'patch' in normal use.",
        "   - Prefer using 'apply_patch' for file edits instead of bash heredocs, cat >, or sed -i.",
        "   - If an apply_patch attempt fails, read the file again and retry with a smaller patch and exact surrounding context.",
    ]
    return "\n".join(lines)


def build_assistant_contract_prompt(
    *,
    delimiters: ModelDelimiters | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    terminal_tool: str = TERMINAL_TOOL_NAME,
    action_payload_format: str | None = None,
    include_tool_schema: bool = True,
    include_examples: bool = True,
    include_repeat_warning: bool = True,
) -> str:
    """Return an instruction block that matches the v1.6 action contract."""
    d = delimiters or default_delimiters()
    resolved_payload_format = _normalize_action_payload_format(action_payload_format)
    allowed_tools_text = ", ".join(ALLOWED_TOOLS)
    tool_schema_block = _build_tool_schema_prompt() if include_tool_schema else ""
    tool_examples_block = (
        _build_tool_examples_prompt(action_payload_format=resolved_payload_format)
        if include_examples
        else ""
    )
    tool_usage_guardrails_block = _build_tool_usage_guardrails_prompt()

    lines: list[str] = []
    lines.append("Surround each tool action with a tool-call delimiter block.")
    lines.append(
        f"Emit ordered tool calls (max {max_tool_calls}): "
        f"{_render_contract_signature(delimiters=d, action_payload_format=resolved_payload_format)}"
    )
    if resolved_payload_format == "json":
        lines.append(
            "Every tool-call JSON object MUST include both keys: 'tool' and 'args'. "
            "'args' MUST be a JSON object (never put command/query/path at top level)."
        )
    else:
        lines.append(
            "Every XML tool call MUST use a 'name' attribute for the tool and direct child elements for args. "
            "Do not add an <args> wrapper."
        )
        lines.append(
            "Use CDATA for string-valued args so multiline command, patch, and final_response round-trip cleanly. "
            "XML still normalizes line endings to LF."
        )
        lines.append(
            "For list-valued args, wrap repeated children in one container element, for example "
            "<changed_paths><path><![CDATA[src/app.py]]></path></changed_paths>."
        )
        lines.append(
            "Do not emit namespaces, DTDs, processing instructions, extra attributes, or mixed JSON/XML payloads."
        )
    lines.append("Begin with a tool-call block. Do not emit prose before the first tool call.")
    lines.append(f"Allowed tools: {allowed_tools_text}.")
    lines.append(_build_required_args_prompt())
    lines.append("Do not invent tool names or wrapper labels.")
    lines.append(
        f"Terminal tool is '{terminal_tool}', you must end conversation with this tool, "
        "and if present it must be the only tool call."
    )
    if tool_schema_block:
        lines.append(tool_schema_block)
    if tool_examples_block:
        lines.append(tool_examples_block)
    if tool_usage_guardrails_block:
        lines.append(tool_usage_guardrails_block)

    numbered_lines: list[str] = []
    step_index = 1
    for line in lines:
        if not line:
            continue
        prefix = f"{step_index}) "
        numbered_lines.append(prefix + line)
        step_index += 1

    return "Assistant output contract:\n" + "\n".join(numbered_lines)


def build_onpolicy_system_prompt(*, action_payload_format: str | None = None) -> str:
    """Build the default system prompt for on-policy runtime rollouts."""
    return (
        _DEFAULT_SYSTEM_PROMPT_PREFIX
        + _DEFAULT_SYSTEM_PROMPT_WORKFLOW
        + build_assistant_contract_prompt(action_payload_format=action_payload_format or ACTION_PAYLOAD_FORMAT)
    )


def build_sdpo_rollout_followup_user_message() -> str:
    """Build the continuation nudge for bridge-loop SDPO rollouts."""
    return _SDPO_ROLLOUT_FOLLOWUP_USER_MESSAGE


def build_onpolicy_initial_user_message(
    *,
    problem_statement: str,
) -> str:
    """Build the initial user message for one on-policy task attempt."""
    return (
        "You are solving one software engineering task.\n"
        "Task objective:\n"
        f"{problem_statement}\n\n"
        "Execution guidance:\n"
        "- Use tool calls to inspect code, apply patches, and run validation commands.\n"
        "- Submit only when you are ready to end the attempt."
    )


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)
