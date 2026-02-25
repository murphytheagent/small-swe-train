"""Typed protocol contracts shared across parser, adapters, and trainer scaffolds.

This module is the **single source of truth** for tool names, argument shapes,
and validation constraints.  ``TOOL_SCHEMAS`` is the registry;
``validate_tool_call()`` checks a parsed ``ToolCall`` against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict, cast, get_type_hints

AllowedTool = Literal["bash", "search", "apply_patch", "submit"]
# NOTE: AllowedTool above must remain a Literal for static type narrowing.
BASH_TOOL_NAME: str = "bash"
SEARCH_TOOL_NAME: str = "search"
APPLY_PATCH_TOOL_NAME: str = "apply_patch"
TERMINAL_TOOL_NAME: str = "submit"
LEGACY_TERMINAL_TOOL_ALIAS: str = "answer"
LEGACY_EDIT_TOOL_ALIAS: str = "edit"
# Backward-compatibility constant for older imports.
EDIT_TOOL_NAME: str = APPLY_PATCH_TOOL_NAME

ALLOWED_TOOLS: tuple[str, ...] = (
    BASH_TOOL_NAME,
    SEARCH_TOOL_NAME,
    APPLY_PATCH_TOOL_NAME,
    TERMINAL_TOOL_NAME,
)
_ALLOWED_TOOLS = set(ALLOWED_TOOLS)

class BashArgs(TypedDict, total=False):
    command: str
    cwd: str
    timeout_sec: int


class SearchArgs(TypedDict, total=False):
    query: str
    path_hint: str
    top_k: int


class ApplyPatchArgs(TypedDict, total=False):
    patch: str
    path: str
    description: str


class SubmitArgs(TypedDict, total=False):
    final_response: str
    changed_paths: list[str]


class ToolOutput(TypedDict, total=False):
    stdout: str
    stderr: str
    exit_code: int


# ---------------------------------------------------------------------------
# Tool schema registry + validator
# ---------------------------------------------------------------------------
# Each entry maps a canonical tool name to:
#   source      – TypedDict defining the allowed fields and their types
#   required    – which fields must be present
#   constraints – per-field validation rules (min_length, minimum, maximum)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    BASH_TOOL_NAME: {
        "source": BashArgs,
        "required": ["command"],
        "prompt_example": {
            "tool": BASH_TOOL_NAME,
            "args": {
                "command": "python -m pytest -q",
                "cwd": "/workspace/project",
                "timeout_sec": 120,
            },
        },
        "constraints": {
            "command": {"min_length": 1},
            "cwd": {"min_length": 1},
            "timeout_sec": {"minimum": 1, "maximum": 600},
        },
    },
    SEARCH_TOOL_NAME: {
        "source": SearchArgs,
        "required": ["query"],
        "prompt_example": {
            "tool": SEARCH_TOOL_NAME,
            "args": {
                "query": "load_config",
                "path_hint": "src",
                "top_k": 5,
            },
        },
        "constraints": {
            "query": {"min_length": 1},
            "top_k": {"minimum": 1, "maximum": 50},
        },
    },
    APPLY_PATCH_TOOL_NAME: {
        "source": ApplyPatchArgs,
        "required": ["patch"],
        "prompt_example": {
            "tool": APPLY_PATCH_TOOL_NAME,
            "args": {
                "path": "src/app.py",
                "patch": "@@ -12,1 +12,1 @@\n-return False\n+return True",
            },
        },
        "constraints": {
            "patch": {"min_length": 1},
            "path": {"min_length": 1},
        },
    },
    TERMINAL_TOOL_NAME: {
        "source": SubmitArgs,
        "required": ["final_response"],
        "prompt_example": {
            "tool": TERMINAL_TOOL_NAME,
            "args": {
                "final_response": "Implemented the fix and verified tests pass.",
                "changed_paths": [
                    "src/app.py",
                    "tests/test_app.py",
                ],
            },
        },
        "constraints": {
            "final_response": {"min_length": 1},
        },
    },
}

_TYPE_MAP: dict[type, tuple[type, ...]] = {
    str: (str,),
    int: (int,),
    float: (int, float),
    bool: (bool,),
}


def canonical_tool_name(tool: str) -> AllowedTool:
    """Normalize legacy aliases and enforce canonical tool names."""
    normalized = tool.strip().lower()
    if normalized == LEGACY_TERMINAL_TOOL_ALIAS:
        normalized = TERMINAL_TOOL_NAME
    if normalized == LEGACY_EDIT_TOOL_ALIAS:
        normalized = APPLY_PATCH_TOOL_NAME
    if normalized not in _ALLOWED_TOOLS:
        raise ValueError(f"Unsupported tool: {tool!r}")
    return cast(AllowedTool, normalized)


@dataclass(frozen=True)
class ToolCall:
    tool: AllowedTool
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = canonical_tool_name(self.tool)
        object.__setattr__(self, "tool", canonical)
        object.__setattr__(self, "args", dict(self.args))

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass(frozen=True)
class ActionEnvelope:
    tool_calls: tuple[ToolCall, ...]
    thinking: str | None = None

    def __post_init__(self) -> None:
        if not self.tool_calls:
            raise ValueError("At least one tool call is required.")
        has_submit = any(call.tool == TERMINAL_TOOL_NAME for call in self.tool_calls)
        if has_submit and len(self.tool_calls) != 1:
            raise ValueError("'submit' must be the only tool call in the final turn.")
        if self.thinking is not None and not self.thinking.strip():
            object.__setattr__(self, "thinking", None)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"tool_calls": [call.to_dict() for call in self.tool_calls]}
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        return payload


@dataclass(frozen=True)
class CanonicalFeedback:
    normalization_version: str
    normalized_text: str
    truncated: bool
    raw_sha256: str
    artifact_identities: tuple[str, ...]
    actionable_error_text: str | None
    localization_hints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalization_version": self.normalization_version,
            "normalized_text": self.normalized_text,
            "truncated": self.truncated,
            "raw_sha256": self.raw_sha256,
            "artifact_identities": list(self.artifact_identities),
            "actionable_error_text": self.actionable_error_text,
            "localization_hints": list(self.localization_hints),
        }


@dataclass(frozen=True)
class SelfContainmentChecks:
    has_failing_artifact_identity: bool
    has_actionable_error_text: bool
    has_localization_hint: bool

    @property
    def is_self_contained(self) -> bool:
        return (
            self.has_failing_artifact_identity
            and self.has_actionable_error_text
            and self.has_localization_hint
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "has_failing_artifact_identity": self.has_failing_artifact_identity,
            "has_actionable_error_text": self.has_actionable_error_text,
            "has_localization_hint": self.has_localization_hint,
        }


@dataclass(frozen=True)
class FeedbackPacket:
    step_index: int
    tool: AllowedTool
    tool_input: dict[str, Any]
    tool_output: dict[str, Any]
    canonical_feedback: CanonicalFeedback
    self_containment_checks: SelfContainmentChecks
    is_self_contained: bool
    include_student_attempt_for_teacher: bool = True

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        expected = self.self_containment_checks.is_self_contained
        if self.is_self_contained != expected:
            raise ValueError(
                "is_self_contained must equal conjunction of self_containment_checks fields"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool": self.tool,
            "tool_input": dict(self.tool_input),
            "tool_output": dict(self.tool_output),
            "canonical_feedback": self.canonical_feedback.to_dict(),
            "self_containment_checks": self.self_containment_checks.to_dict(),
            "is_self_contained": self.is_self_contained,
            "include_student_attempt_for_teacher": self.include_student_attempt_for_teacher,
        }


def make_tool_call(payload: Mapping[str, Any]) -> ToolCall:
    """Build a ToolCall from parsed JSON payload."""
    if "tool" not in payload:
        raise ValueError("Missing 'tool' in tool call payload.")
    if "args" not in payload:
        raise ValueError("Missing 'args' in tool call payload.")
    args = payload["args"]
    if not isinstance(args, dict):
        raise ValueError("'args' must be a JSON object.")
    tool_raw = payload["tool"]
    if not isinstance(tool_raw, str):
        raise ValueError("'tool' must be a string.")
    return ToolCall(tool=canonical_tool_name(tool_raw), args=dict(args))


def validate_tool_call(tool_call: ToolCall) -> list[str]:
    """Validate *tool_call* against ``TOOL_SCHEMAS``.

    Returns a list of human-readable error strings.  Empty list means valid.
    """
    schema = TOOL_SCHEMAS.get(tool_call.tool)
    if schema is None:
        return [f"Unknown tool: {tool_call.tool!r}"]

    errors: list[str] = []
    hints = get_type_hints(schema["source"])
    allowed_fields = set(hints)
    required = set(schema.get("required", ()))
    constraints: dict[str, dict[str, Any]] = schema.get("constraints", {})

    for key in tool_call.args:
        if key not in allowed_fields:
            errors.append(f"Unknown arg '{key}' for tool '{tool_call.tool}'")

    for key in required:
        if key not in tool_call.args:
            errors.append(f"Missing required arg '{key}' for tool '{tool_call.tool}'")

    for key, value in tool_call.args.items():
        if key not in allowed_fields:
            continue

        expected = hints[key]
        origin = getattr(expected, "__origin__", None)
        acceptable = _TYPE_MAP.get(expected if origin is None else origin)
        if acceptable and not isinstance(value, acceptable):
            errors.append(
                f"Arg '{key}': expected {expected.__name__}, "
                f"got {type(value).__name__}"
            )

        c = constraints.get(key, {})
        if isinstance(value, str):
            ml = c.get("min_length")
            if ml is not None and len(value) < ml:
                errors.append(f"Arg '{key}': length must be >= {ml}")
        if isinstance(value, (int, float)):
            lo = c.get("minimum")
            hi = c.get("maximum")
            if lo is not None and value < lo:
                errors.append(f"Arg '{key}': must be >= {lo}")
            if hi is not None and value > hi:
                errors.append(f"Arg '{key}': must be <= {hi}")

    return errors
