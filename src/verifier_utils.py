"""Shared verifier-kind, selector, and prompt-preview helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

VALID_VERIFIER_KINDS = ("pytest", "go_test", "node_test", "command")
_MAX_TARGETS_PER_GROUP = 4096
_MAX_SELECTOR_TEXT_LENGTH = 512
_TARGET_PREVIEW_LIMIT = 4


def normalize_verifier_kind(value: Any, *, label: str = "verifier_kind") -> str:
    text = str(value).strip().lower()
    if text not in VALID_VERIFIER_KINDS:
        allowed = ", ".join(VALID_VERIFIER_KINDS)
        raise ValueError(f"{label} must be one of: {allowed}.")
    return text


def normalize_verifier_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [name for name in (str(key).strip() for key in value.keys()) if name]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        targets: list[str] = []
        for item in value:
            name = str(item).strip()
            if name:
                targets.append(name)
        return targets
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return normalize_verifier_targets(parsed)
        if "," in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.split(",")) if chunk]
        if "\n" in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.splitlines()) if chunk]
        return [stripped]
    return []


def validate_verifier_target_sets(
    *,
    fail_to_pass: Any,
    pass_to_pass: Any,
) -> tuple[list[str], list[str]]:
    fail_targets = normalize_verifier_targets(fail_to_pass)
    pass_targets = normalize_verifier_targets(pass_to_pass)
    if not fail_targets:
        raise ValueError("FAIL_TO_PASS must contain at least one target.")
    if not pass_targets:
        raise ValueError("PASS_TO_PASS must contain at least one target.")

    fail_duplicates = _duplicate_targets(fail_targets)
    if fail_duplicates:
        rendered = ", ".join(fail_duplicates[:5])
        raise ValueError(f"FAIL_TO_PASS has duplicate targets: {rendered}")

    pass_duplicates = _duplicate_targets(pass_targets)
    if pass_duplicates:
        rendered = ", ".join(pass_duplicates[:5])
        raise ValueError(f"PASS_TO_PASS has duplicate targets: {rendered}")

    overlap = sorted(set(fail_targets).intersection(pass_targets))
    if overlap:
        rendered = ", ".join(overlap[:5])
        raise ValueError(f"FAIL_TO_PASS and PASS_TO_PASS overlap: {rendered}")

    _validate_target_payload_size("FAIL_TO_PASS", fail_targets)
    _validate_target_payload_size("PASS_TO_PASS", pass_targets)
    return fail_targets, pass_targets


def logical_task_identity_key(
    *,
    problem_statement: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    verifier_kind: str,
) -> str:
    payload = {
        "problem_statement": _normalize_problem_statement(problem_statement),
        "fail_to_pass": sorted(str(value).strip() for value in fail_to_pass if str(value).strip()),
        "pass_to_pass": sorted(str(value).strip() for value in pass_to_pass if str(value).strip()),
        "verifier_kind": normalize_verifier_kind(verifier_kind),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def resolve_problem_statement(
    *,
    problem_statement: Any,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    verifier_kind: str,
) -> tuple[str, str]:
    prompt_text = str(problem_statement or "").strip()
    if prompt_text:
        return prompt_text, "dataset"

    normalized_kind = normalize_verifier_kind(verifier_kind)
    task_label = {
        "pytest": "Resolve the failing Python tests for this task.",
        "go_test": "Resolve the failing Go tests for this task.",
        "node_test": "Resolve the failing JavaScript or TypeScript tests for this task.",
        "command": "Resolve the failing verifier commands for this task.",
    }[normalized_kind]
    preview = build_bounded_target_preview(
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
    )
    return f"{task_label}\n\nVerifier target preview:\n{preview}", "target_preview_fallback"


def build_bounded_target_preview(
    *,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    preview_limit: int = _TARGET_PREVIEW_LIMIT,
) -> str:
    lines = [
        _render_target_preview_line("FAIL_TO_PASS", fail_to_pass, preview_limit=preview_limit),
        _render_target_preview_line("PASS_TO_PASS", pass_to_pass, preview_limit=preview_limit),
    ]
    return "\n".join(line for line in lines if line)


def build_go_test_regex(targets: Sequence[str]) -> str:
    normalized_targets = [str(target).strip() for target in targets if str(target).strip()]
    if not normalized_targets:
        return "^$"
    escaped = [re.escape(target) for target in normalized_targets]
    return "^(?:" + "|".join(escaped) + ")$"


def stable_unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw_value in values:
        value = str(raw_value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _duplicate_targets(targets: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for raw_target in targets:
        target = str(raw_target).strip()
        if not target:
            continue
        if target in seen and target not in duplicates:
            duplicates.append(target)
            continue
        seen.add(target)
    return duplicates


def _normalize_problem_statement(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _render_target_preview_line(
    label: str,
    targets: Sequence[str],
    *,
    preview_limit: int,
) -> str:
    normalized_targets = [str(target).strip() for target in targets if str(target).strip()]
    if not normalized_targets:
        return f"- {label} (0)"

    preview_items = normalized_targets[: max(int(preview_limit), 1)]
    rendered = ", ".join(preview_items)
    remaining = len(normalized_targets) - len(preview_items)
    if remaining > 0:
        rendered += f", ... (+{remaining} more)"
    return f"- {label} ({len(normalized_targets)}): {rendered}"


def _validate_target_payload_size(label: str, targets: Sequence[str]) -> None:
    if len(targets) > _MAX_TARGETS_PER_GROUP:
        raise ValueError(
            f"{label} has {len(targets)} targets; max supported per row is {_MAX_TARGETS_PER_GROUP}."
        )
    for target in targets:
        if len(target) > _MAX_SELECTOR_TEXT_LENGTH:
            raise ValueError(
                f"{label} target exceeds {_MAX_SELECTOR_TEXT_LENGTH} characters: {target[:80]!r}"
            )
