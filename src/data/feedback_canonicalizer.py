"""Feedback canonicalization and self-containment diagnostics for turn-SDPO."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from config import (
    resolve_feedback_deterministic_truncation_settings,
    resolve_feedback_self_containment_signals_enabled,
)
from schemas import (
    CanonicalFeedback,
    FeedbackPacket,
    SelfContainmentChecks,
    canonical_tool_name,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_FILE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?[\\/\w.-]+\.(?:py|md|txt|json|yaml|yml|toml|ini|cfg|sh|js|ts|tsx|jsx)(?::\d+)?"
)
_TEST_SELECTOR_RE = re.compile(r"\b[\w./-]+::[\w./-]+\b")
_ERROR_LINE_RE = re.compile(r"(?im)^.*(?:error|exception|failed|traceback).*$")
_LINE_HINT_RE = re.compile(r"(?:[A-Za-z]:)?[\\/\w.-]+\.\w+:\d+")
_SYMBOL_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)")


def collect_raw_env_payload(tool_output: Mapping[str, Any]) -> str:
    """Collect a deterministic text payload from tool output fields."""
    stdout = str(tool_output.get("stdout", ""))
    stderr = str(tool_output.get("stderr", ""))

    extras = {k: v for k, v in tool_output.items() if k not in {"stdout", "stderr"}}
    chunks: list[str] = []
    if stdout:
        chunks.append(f"STDOUT:\n{stdout}")
    if stderr:
        chunks.append(f"STDERR:\n{stderr}")
    if extras:
        chunks.append("META:\n" + json.dumps(extras, sort_keys=True, ensure_ascii=True))

    if not chunks:
        return json.dumps(dict(tool_output), sort_keys=True, ensure_ascii=True)
    return "\n\n".join(chunks)


def normalize_text(raw_text: str) -> str:
    """Apply deterministic text normalization rules from v1.6 design."""
    cleaned = _ANSI_ESCAPE_RE.sub("", raw_text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def truncate_head_tail_tokens(
    text: str,
    head_tokens: int = 768,
    tail_tokens: int = 768,
) -> tuple[str, bool]:
    """Apply deterministic head+tail truncation over whitespace tokens."""
    tokens = text.split()
    if not tokens:
        return "", False
    if len(tokens) <= head_tokens + tail_tokens:
        return text, False
    truncated_tokens = tokens[:head_tokens] + ["<...truncated...>"] + tokens[-tail_tokens:]
    return " ".join(truncated_tokens), True


def truncate_tool_output_payload(
    tool_output: Mapping[str, Any],
    *,
    head_tokens: int = 768,
    tail_tokens: int = 768,
) -> tuple[dict[str, Any], bool]:
    """Apply deterministic truncation to string fields in a tool-output payload."""
    truncated = False

    def _truncate_value(value: Any) -> Any:
        nonlocal truncated
        if isinstance(value, str):
            text, did_truncate = truncate_head_tail_tokens(
                value,
                head_tokens=head_tokens,
                tail_tokens=tail_tokens,
            )
            if did_truncate:
                truncated = True
            return text
        if isinstance(value, Mapping):
            return {key: _truncate_value(nested) for key, nested in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_truncate_value(item) for item in value]
        return value

    normalized = {key: _truncate_value(value) for key, value in tool_output.items()}
    return normalized, truncated


def extract_artifact_identities(text: str) -> tuple[str, ...]:
    """Extract deterministic artifact identities from canonicalized text."""
    identities: set[str] = set()
    identities.update(_TEST_SELECTOR_RE.findall(text))
    identities.update(_FILE_PATH_RE.findall(text))
    return tuple(sorted(identities))


def extract_actionable_error_text(text: str) -> str | None:
    """Return first actionable error line if available; otherwise None."""
    for match in _ERROR_LINE_RE.finditer(text):
        line = match.group(0).strip(" -:\t")
        if line:
            return line
    return None


def extract_localization_hints(text: str) -> tuple[str, ...]:
    """Extract file:line and symbol-like hints for repair localization."""
    hints: set[str] = set()
    hints.update(_LINE_HINT_RE.findall(text))
    hints.update(_TEST_SELECTOR_RE.findall(text))
    hints.update(_SYMBOL_RE.findall(text))
    return tuple(sorted(hints))


def canonicalize_tool_feedback(
    tool_output: Mapping[str, Any],
    *,
    normalization_version: str = "v1",
    head_tokens: int = 768,
    tail_tokens: int = 768,
) -> CanonicalFeedback:
    """Build canonical feedback payload from raw tool output."""
    raw_payload = collect_raw_env_payload(tool_output)
    normalized = normalize_text(raw_payload)
    truncated_text, truncated = truncate_head_tail_tokens(
        normalized,
        head_tokens=head_tokens,
        tail_tokens=tail_tokens,
    )
    include_self_containment_signals = resolve_feedback_self_containment_signals_enabled()
    if include_self_containment_signals:
        artifact_identities = extract_artifact_identities(truncated_text)
        actionable_error_text = extract_actionable_error_text(truncated_text)
        localization_hints = extract_localization_hints(truncated_text)
    else:
        artifact_identities = ()
        actionable_error_text = None
        localization_hints = ()

    return CanonicalFeedback(
        normalization_version=normalization_version,
        normalized_text=truncated_text,
        truncated=truncated,
        raw_sha256=hashlib.sha256(raw_payload.encode("utf-8")).hexdigest(),
        artifact_identities=artifact_identities,
        actionable_error_text=actionable_error_text,
        localization_hints=localization_hints,
    )


def build_feedback_packet(
    *,
    step_index: int,
    tool: str,
    tool_input: Mapping[str, Any],
    tool_output: Mapping[str, Any],
    include_student_attempt_for_teacher: bool = True,
    head_tokens: int | None = None,
    tail_tokens: int | None = None,
) -> FeedbackPacket:
    """Build schema-aligned feedback packet with derived diagnostics."""
    truncation_settings = resolve_feedback_deterministic_truncation_settings()
    if head_tokens is None:
        head_tokens = truncation_settings.head_tokens
    if tail_tokens is None:
        tail_tokens = truncation_settings.tail_tokens

    canonical_feedback = canonicalize_tool_feedback(
        tool_output,
        head_tokens=head_tokens,
        tail_tokens=tail_tokens,
    )
    checks = SelfContainmentChecks(
        has_failing_artifact_identity=bool(canonical_feedback.artifact_identities),
        has_actionable_error_text=bool(
            canonical_feedback.actionable_error_text
            and canonical_feedback.actionable_error_text.strip()
        ),
        has_localization_hint=bool(canonical_feedback.localization_hints),
    )

    return FeedbackPacket(
        step_index=step_index,
        tool=canonical_tool_name(tool),
        tool_input=dict(tool_input),
        tool_output=dict(tool_output),
        canonical_feedback=canonical_feedback,
        self_containment_checks=checks,
        is_self_contained=checks.is_self_contained,
        include_student_attempt_for_teacher=include_student_attempt_for_teacher,
    )
