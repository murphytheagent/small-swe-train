"""Deterministic teacher memory block builders for SDPO reprompts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_KNOWN_PATH_HEADER = "Known student-discovered paths (raw):"
_PATCH_HEADER = "Successful apply_patch calls through current turn:"
_MISSING_PATH = "<missing>"
_MAX_KNOWN_PATHS = 10
_TEXT_SEARCH_PATH_PATTERN = re.compile(r"^(.+?):\d+:")


@dataclass(frozen=True)
class TeacherMemoryBlocks:
    compressed_memory_block: str
    critical_facts_block: str


def _coerce_mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    rows: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        rows.append(item)
    return tuple(rows)


def _coerce_nested_sequence(value: Any) -> tuple[Sequence[Any], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    rows: list[Sequence[Any]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            return None
        rows.append(item)
    return tuple(rows)


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _coerce_exit_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _extract_known_paths(trajectory_steps: Sequence[Mapping[str, Any]]) -> list[str]:
    unique_paths: list[str] = []
    seen: set[str] = set()

    for step in reversed(trajectory_steps):
        if _coerce_exit_code(step.get("exit_code")) != 0:
            continue
        stdout = step.get("stdout")
        if not isinstance(stdout, str) or not stdout.strip():
            continue

        tool = str(step.get("tool", "")).strip()
        args = _coerce_mapping(step.get("args"))

        candidate_paths: list[str] = []
        if tool == "read":
            if args is None:
                continue
            raw_path = args.get("path")
            if isinstance(raw_path, str):
                normalized = raw_path.strip()
                if normalized:
                    candidate_paths.append(normalized)
        elif tool == "file_search":
            for line in stdout.splitlines():
                normalized = line.strip()
                if normalized:
                    candidate_paths.append(normalized)
        elif tool == "text_search":
            for line in stdout.splitlines():
                match = _TEXT_SEARCH_PATH_PATTERN.match(line)
                if match is None:
                    continue
                normalized = match.group(1).strip()
                if normalized:
                    candidate_paths.append(normalized)
        else:
            continue

        for path in candidate_paths:
            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)
            if len(unique_paths) >= _MAX_KNOWN_PATHS:
                return unique_paths

    return unique_paths


def _render_known_paths_block(trajectory_steps: Sequence[Mapping[str, Any]]) -> str:
    known_paths = _extract_known_paths(trajectory_steps)
    if not known_paths:
        return ""
    return "\n".join([_KNOWN_PATH_HEADER, *[f"- {path}" for path in known_paths]])


def _resolve_patch_prefix_step_count(
    sample: Mapping[str, Any],
    *,
    current_turn_index: int,
    include_student_attempt_for_teacher: bool,
) -> int | None:
    if current_turn_index < 0:
        return None
    turn_tool_blocks = _coerce_nested_sequence(sample.get("trajectory_turn_tool_response_blocks"))
    if turn_tool_blocks is None or current_turn_index >= len(turn_tool_blocks):
        return None
    end_index = current_turn_index + 1 if include_student_attempt_for_teacher else current_turn_index
    return sum(len(block) for block in turn_tool_blocks[:end_index])


def _render_patch_memory_block(
    sample: Mapping[str, Any],
    *,
    current_turn_index: int,
    include_student_attempt_for_teacher: bool,
    trajectory_steps: Sequence[Mapping[str, Any]],
) -> str:
    prefix_step_count = _resolve_patch_prefix_step_count(
        sample,
        current_turn_index=current_turn_index,
        include_student_attempt_for_teacher=include_student_attempt_for_teacher,
    )
    if prefix_step_count is None or prefix_step_count <= 0:
        return ""

    rendered_patches: list[str] = []
    for step in trajectory_steps[:prefix_step_count]:
        if str(step.get("tool", "")).strip() != "apply_patch":
            continue
        if _coerce_exit_code(step.get("exit_code")) != 0:
            continue
        args = _coerce_mapping(step.get("args"))
        if args is None:
            continue

        raw_patch = args.get("patch")
        if not isinstance(raw_patch, str):
            continue
        normalized_patch = raw_patch.strip()
        if not normalized_patch:
            continue

        raw_path_value = args.get("path")
        if isinstance(raw_path_value, str):
            normalized_path = raw_path_value.strip() or _MISSING_PATH
        else:
            normalized_path = _MISSING_PATH

        patch_index = len(rendered_patches) + 1
        rendered_patches.append(
            f"[PATCH {patch_index}]\n"
            f"raw_path: {normalized_path}\n"
            "raw_patch:\n"
            f"{normalized_patch}"
        )

    if not rendered_patches:
        return ""
    return f"{_PATCH_HEADER}\n\n" + "\n\n".join(rendered_patches)


def build_teacher_memory_blocks(
    sample: Mapping[str, Any],
    *,
    current_turn_index: int,
    include_student_attempt_for_teacher: bool = True,
    include_teacher_memory_blocks: bool = True,
) -> TeacherMemoryBlocks:
    if not include_teacher_memory_blocks:
        return TeacherMemoryBlocks(
            compressed_memory_block="",
            critical_facts_block="",
        )

    trajectory_steps = _coerce_mapping_sequence(sample.get("trajectory_steps"))
    if trajectory_steps is None:
        return TeacherMemoryBlocks(
            compressed_memory_block="",
            critical_facts_block="",
        )

    return TeacherMemoryBlocks(
        compressed_memory_block=_render_known_paths_block(trajectory_steps),
        critical_facts_block=_render_patch_memory_block(
            sample,
            current_turn_index=current_turn_index,
            include_student_attempt_for_teacher=include_student_attempt_for_teacher,
            trajectory_steps=trajectory_steps,
        ),
    )
