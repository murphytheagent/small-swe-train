"""Typed protocol contracts shared across parser, adapters, and trainer scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict, cast

AllowedTool = Literal["bash", "search", "edit", "submit"]
_ALLOWED_TOOLS = {"bash", "search", "edit", "submit"}


class BashArgs(TypedDict, total=False):
    command: str
    cwd: str
    timeout_sec: int


class SearchArgs(TypedDict, total=False):
    query: str
    path_hint: str
    top_k: int


class EditArgs(TypedDict, total=False):
    path: str
    patch: str
    description: str


class SubmitArgs(TypedDict, total=False):
    final_response: str
    changed_paths: list[str]
    confidence: float


class ToolOutput(TypedDict, total=False):
    stdout: str
    stderr: str
    exit_code: int


def canonical_tool_name(tool: str) -> AllowedTool:
    """Normalize legacy aliases and enforce canonical tool names."""
    normalized = tool.strip().lower()
    if normalized == "answer":
        normalized = "submit"
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
        has_submit = any(call.tool == "submit" for call in self.tool_calls)
        if has_submit and len(self.tool_calls) != 1:
            raise ValueError("If submit appears, it must be the only tool call in the turn.")
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
