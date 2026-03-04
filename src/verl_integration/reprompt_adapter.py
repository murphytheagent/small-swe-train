"""Teacher-reprompt assembly helpers for self-distillation batches."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from data.feedback_canonicalizer import build_feedback_packet
from prompts.teacher_messages import build_teacher_output_contract_block
from teacher.memory_builder import build_teacher_memory_blocks
from teacher.prompt_builder import TeacherPromptInputs, build_teacher_prompt

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}
DEFAULT_NUM_RECENT_RAW_BLOCKS = 3
DEFAULT_MAX_REPROMPT_LEN = 12288
_TURN_SUPERVISION_NEXT = "next_turn"
_TURN_SUPERVISION_CURRENT = "current_turn"
_TURN_SUPERVISION_MODES = {_TURN_SUPERVISION_NEXT, _TURN_SUPERVISION_CURRENT}
LOGGER = logging.getLogger(__name__)
_VERIFIER_FEEDBACK_NONE = "none"
_VERIFIER_FEEDBACK_FINAL_TURN_ONLY = "final_turn_only"
_VERIFIER_FEEDBACK_ALL_TURNS = "all_turns"
_VERIFIER_FEEDBACK_MODES = {
    _VERIFIER_FEEDBACK_NONE,
    _VERIFIER_FEEDBACK_FINAL_TURN_ONLY,
    _VERIFIER_FEEDBACK_ALL_TURNS,
}
_VERIFIER_FEEDBACK_MARKER = "[VERIFIER_FEEDBACK]"
_LEGACY_GATING_RESOLVED_ONLY = "resolved_only"
_LEGACY_GATING_FEEDBACK_PRESENT = "feedback_present"
_LEGACY_GATING_ALWAYS = "always"
_LEGACY_GATING_POLICIES = {
    _LEGACY_GATING_RESOLVED_ONLY,
    _LEGACY_GATING_FEEDBACK_PRESENT,
    _LEGACY_GATING_ALWAYS,
}


def _token_count(text: str) -> int:
    return len(str(text).split())


def _truncate_text_to_token_budget(text: str, *, token_budget: int) -> str:
    if token_budget <= 0:
        return ""

    lines = str(text).split("\n")
    kept_lines: list[str] = []
    budget = token_budget

    for line in lines:
        words = line.split()
        if not words:
            if budget > 0:
                kept_lines.append(line)
            continue
        if len(words) <= budget:
            kept_lines.append(line)
            budget -= len(words)
        else:
            if budget > 0:
                kept_lines.append(" ".join(words[:budget]))
            break

    return "\n".join(kept_lines)


def _truncate_prompt_tokens(prompt: str, *, max_reprompt_len: int) -> tuple[str, bool]:
    if max_reprompt_len <= 0:
        raise ValueError("max_reprompt_len must be > 0")

    if _token_count(prompt) <= max_reprompt_len:
        return prompt, False

    return _truncate_text_to_token_budget(prompt, token_budget=max_reprompt_len), True


def _split_feedback_block(feedback_block: str) -> tuple[str, str]:
    rendered = str(feedback_block).strip()
    if not rendered:
        return "", ""
    marker_index = rendered.find(_VERIFIER_FEEDBACK_MARKER)
    if marker_index < 0:
        return rendered, ""
    feedback_main = rendered[:marker_index].rstrip()
    verifier_feedback = rendered[marker_index:].strip()
    return feedback_main, verifier_feedback


def _combine_feedback_sections(*, feedback_main: str, verifier_feedback_block: str) -> str:
    sections: list[str] = []
    main = str(feedback_main).strip()
    verifier = str(verifier_feedback_block).strip()
    if main:
        sections.append(main)
    if verifier:
        sections.append(verifier)
    return "\n\n".join(sections)


def _render_prompt_from_sections(sections: Mapping[str, str]) -> str:
    return build_teacher_prompt(
        TeacherPromptInputs(
            initial_prompt_block=sections["initial_prompt_block"],
            recent_raw_block=sections["recent_raw_block"],
            compressed_memory_block=sections["compressed_memory_block"],
            critical_facts_block=sections["critical_facts_block"],
            current_attempt_block=sections["current_attempt_block"],
            feedback_block=_combine_feedback_sections(
                feedback_main=sections["feedback_main"],
                verifier_feedback_block=sections["verifier_feedback_block"],
            ),
            output_contract_block=sections["output_contract_block"],
        )
    )


def _reduce_sections_to_budget(
    *,
    sections: dict[str, str],
    max_reprompt_len: int,
    reduction_order: Sequence[str],
    min_token_budget: Mapping[str, int],
) -> bool:
    changed = False
    prompt = _render_prompt_from_sections(sections)
    overage = _token_count(prompt) - max_reprompt_len
    if overage <= 0:
        return changed

    for key in reduction_order:
        if overage <= 0:
            break
        value = sections.get(key, "")
        current_tokens = _token_count(value)
        keep_floor = max(int(min_token_budget.get(key, 0)), 0)
        removable = max(current_tokens - keep_floor, 0)
        if removable <= 0:
            continue

        drop = min(removable, overage)
        keep = current_tokens - drop
        reduced = _truncate_text_to_token_budget(value, token_budget=keep)
        if reduced != value:
            sections[key] = reduced
            changed = True

        prompt = _render_prompt_from_sections(sections)
        overage = _token_count(prompt) - max_reprompt_len

    return changed


def _compact_teacher_prompt(
    *,
    initial_prompt_block: str,
    recent_raw_block: str,
    compressed_memory_block: str,
    critical_facts_block: str,
    current_attempt_block: str,
    feedback_block: str,
    output_contract_block: str,
    max_reprompt_len: int,
) -> tuple[str, bool]:
    if max_reprompt_len <= 0:
        raise ValueError("max_reprompt_len must be > 0")

    feedback_main, verifier_feedback_block = _split_feedback_block(feedback_block)
    sections: dict[str, str] = {
        "initial_prompt_block": str(initial_prompt_block),
        "recent_raw_block": str(recent_raw_block),
        "compressed_memory_block": str(compressed_memory_block),
        "critical_facts_block": str(critical_facts_block),
        "current_attempt_block": str(current_attempt_block),
        "feedback_main": str(feedback_main),
        "verifier_feedback_block": str(verifier_feedback_block),
        "output_contract_block": str(output_contract_block),
    }

    full_prompt = _render_prompt_from_sections(sections)
    if _token_count(full_prompt) <= max_reprompt_len:
        return full_prompt, False

    was_truncated = False
    optional_changed = _reduce_sections_to_budget(
        sections=sections,
        max_reprompt_len=max_reprompt_len,
        reduction_order=(
            "critical_facts_block",
            "compressed_memory_block",
            "recent_raw_block",
            "feedback_main",
        ),
        min_token_budget={},
    )
    was_truncated = was_truncated or optional_changed
    prompt = _render_prompt_from_sections(sections)
    if _token_count(prompt) <= max_reprompt_len:
        return prompt, True

    protected_token_floors = {
        "initial_prompt_block": min(_token_count(sections["initial_prompt_block"]), 24),
        "current_attempt_block": min(_token_count(sections["current_attempt_block"]), 16),
        "verifier_feedback_block": min(_token_count(sections["verifier_feedback_block"]), 16),
        "output_contract_block": min(_token_count(sections["output_contract_block"]), 32),
    }
    protected_changed = _reduce_sections_to_budget(
        sections=sections,
        max_reprompt_len=max_reprompt_len,
        reduction_order=(
            "initial_prompt_block",
            "current_attempt_block",
            "verifier_feedback_block",
            "output_contract_block",
        ),
        min_token_budget=protected_token_floors,
    )
    was_truncated = was_truncated or protected_changed
    prompt = _render_prompt_from_sections(sections)
    if _token_count(prompt) <= max_reprompt_len:
        return prompt, True

    hard_capped, hard_truncated = _truncate_prompt_tokens(prompt, max_reprompt_len=max_reprompt_len)
    return hard_capped, bool(was_truncated or hard_truncated)


def _coerce_step_index(value: Any, *, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise ValueError("step_index must be an integer >= 0")
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError("step_index must be an integer >= 0")
        coerced = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            coerced = int(stripped)
        except ValueError as exc:
            raise ValueError("step_index must be an integer >= 0") from exc
    else:
        raise ValueError("step_index must be an integer >= 0")

    if coerced < 0:
        raise ValueError("step_index must be an integer >= 0")
    return coerced


def _coerce_bool_flag(value: Any, *, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return fallback


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            rows.append(text)
    return rows


def _coerce_nested_text_list(value: Any) -> list[list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[list[str]] = []
    for item in value:
        rows.append(_coerce_text_list(item))
    return rows


def _coerce_int_list(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[int] = []
    for item in value:
        if isinstance(item, bool):
            rows.append(int(item))
        elif isinstance(item, int):
            rows.append(item)
        elif isinstance(item, float) and item.is_integer():
            rows.append(int(item))
        else:
            rows.append(0)
    return rows


def _coerce_binary_mask(value: Any) -> list[int]:
    rows = _coerce_int_list(value)
    return [1 if item else 0 for item in rows]


def _normalize_turn_supervision_mode(value: Any) -> str:
    if value is None:
        return _TURN_SUPERVISION_NEXT
    normalized = str(value).strip().lower()
    if not normalized:
        return _TURN_SUPERVISION_NEXT
    if normalized not in _TURN_SUPERVISION_MODES:
        supported = ", ".join(sorted(_TURN_SUPERVISION_MODES))
        raise ValueError(f"turn_supervision_mode must be one of: {supported}")
    return normalized

def _normalize_verifier_feedback_mode(value: Any) -> str:
    if value is None:
        return _VERIFIER_FEEDBACK_NONE
    normalized = str(value).strip().lower()
    if not normalized:
        return _VERIFIER_FEEDBACK_NONE
    if normalized not in _VERIFIER_FEEDBACK_MODES:
        supported = ", ".join(sorted(_VERIFIER_FEEDBACK_MODES))
        raise ValueError(f"verifier_feedback_mode must be one of: {supported}")
    return normalized


def _normalize_legacy_gating_policy(value: Any) -> str:
    if value is None:
        return _LEGACY_GATING_RESOLVED_ONLY
    normalized = str(value).strip().lower()
    if not normalized:
        return _LEGACY_GATING_RESOLVED_ONLY
    if normalized not in _LEGACY_GATING_POLICIES:
        supported = ", ".join(sorted(_LEGACY_GATING_POLICIES))
        raise ValueError(f"legacy_distillation_gating_policy must be one of: {supported}")
    return normalized


def _resolve_output_contract_block(
    sample: Mapping[str, Any],
    *,
    supervision_mode: str,
) -> str:
    explicit = sample.get("output_contract_block")
    if explicit is None:
        return build_teacher_output_contract_block(supervision_mode=supervision_mode)
    rendered = str(explicit).strip()
    if not rendered:
        return build_teacher_output_contract_block(supervision_mode=supervision_mode)
    return rendered


def _format_initial_prompt_block(sample: Mapping[str, Any]) -> str:
    explicit = str(sample.get("initial_prompt_block", "")).strip()
    if explicit:
        return explicit

    raw_prompt_messages = sample.get("_raw_prompt_messages")
    if isinstance(raw_prompt_messages, Sequence) and not isinstance(raw_prompt_messages, (str, bytes)):
        blocks: list[str] = []
        for item in raw_prompt_messages:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", "")).strip().upper()
            content = str(item.get("content", "")).strip()
            if not role or not content:
                continue
            blocks.append(f"[{role}]\n{content}")
        if blocks:
            return "\n\n".join(blocks)

    fallback = str(sample.get("prompt") or sample.get("task_block") or "").strip()
    if fallback:
        return f"[USER]\n{fallback}"
    return "[USER]\nSWE task prompt unavailable."


def _format_turn_block(*, turn_index: int, assistant_text: str, tool_response_blocks: Sequence[str]) -> str:
    parts = [
        f"[TURN_{turn_index}]",
        "[ASSISTANT]",
        assistant_text.strip(),
    ]
    for tool_index, block in enumerate(tool_response_blocks, start=1):
        block_text = str(block).strip()
        if not block_text:
            continue
        parts.extend(
            [
                f"[TOOL_RESPONSE_{tool_index}]",
                block_text,
            ]
        )
    return "\n".join(parts).strip()


def _normalize_turn_tool_blocks(
    turn_tool_blocks: Sequence[Sequence[str]],
    *,
    target_len: int,
) -> list[list[str]]:
    normalized = [_coerce_text_list(blocks) for blocks in turn_tool_blocks]
    if len(normalized) < target_len:
        normalized.extend([[] for _ in range(target_len - len(normalized))])
    elif len(normalized) > target_len:
        normalized = normalized[:target_len]
    return normalized


def _build_recent_raw_block(
    turn_blocks: Sequence[str],
    *,
    current_turn_index: int,
    num_recent_raw_blocks: int,
) -> str:
    window = max(int(num_recent_raw_blocks), 0)
    if current_turn_index <= 0 or window == 0:
        return ""
    start = max(0, current_turn_index - window)
    return "\n\n".join(block for block in turn_blocks[start:current_turn_index] if block.strip())


def _build_assistant_turn_spans(
    response_mask: Sequence[int],
    *,
    turn_count: int,
    turn_token_lengths: Sequence[int],
) -> list[tuple[int, int] | None]:
    if turn_count <= 0:
        return []

    generated_positions = [index for index, flag in enumerate(response_mask) if int(flag) != 0]
    if len(turn_token_lengths) >= turn_count and generated_positions:
        spans: list[tuple[int, int] | None] = []
        had_non_contiguous = False
        cursor = 0
        for turn_index in range(turn_count):
            token_length = max(int(turn_token_lengths[turn_index]), 0)
            if token_length <= 0 or cursor >= len(generated_positions):
                spans.append(None)
                continue
            selected = generated_positions[cursor : cursor + token_length]
            if not selected:
                spans.append(None)
                continue
            if any((left + 1) != right for left, right in zip(selected, selected[1:])):
                LOGGER.warning(
                    "Detected non-contiguous generated token positions for turn %s; disabling supervision for that turn.",
                    turn_index,
                )
                had_non_contiguous = True
                spans.append(None)
                cursor += len(selected)
                continue
            spans.append((selected[0], selected[-1] + 1))
            cursor += len(selected)
        if any(span is not None for span in spans) or had_non_contiguous:
            return spans

    spans = []
    current_start: int | None = None
    for index, flag in enumerate(response_mask):
        is_generated = int(flag) != 0
        if is_generated and current_start is None:
            current_start = index
        elif not is_generated and current_start is not None:
            spans.append((current_start, index))
            current_start = None
    if current_start is not None:
        spans.append((current_start, len(response_mask)))

    padded_spans: list[tuple[int, int] | None] = [None for _ in range(turn_count)]
    for index in range(min(turn_count, len(spans))):
        padded_spans[index] = spans[index]
    return padded_spans


def _build_mask_from_span(*, width: int, span: tuple[int, int] | None) -> list[int]:
    if width <= 0:
        return []
    if span is None:
        return [0 for _ in range(width)]
    start, end = span
    clipped_start = max(0, min(int(start), width))
    clipped_end = max(clipped_start, min(int(end), width))
    mask = [0 for _ in range(width)]
    for index in range(clipped_start, clipped_end):
        mask[index] = 1
    return mask


def _build_feedback_for_turn(
    *,
    tool: str,
    turn_index: int,
    tool_response_blocks: Sequence[str],
    include_student_attempt_for_teacher: bool,
) -> tuple[str, bool, dict[str, Any]]:
    feedback_text = "\n\n".join(block.strip() for block in tool_response_blocks if str(block).strip())
    tool_output: dict[str, Any]
    if feedback_text:
        tool_output = {"stdout": feedback_text, "stderr": "", "exit_code": 0}
    else:
        tool_output = {}

    feedback_packet = build_feedback_packet(
        step_index=turn_index,
        tool=tool,
        tool_input={},
        tool_output=tool_output,
        include_student_attempt_for_teacher=include_student_attempt_for_teacher,
    )
    feedback_block = (
        feedback_packet.canonical_feedback.actionable_error_text
        or feedback_packet.canonical_feedback.normalized_text
    )
    has_teacher_signal = feedback_packet.self_containment_checks.has_actionable_error_text
    return feedback_block, has_teacher_signal, feedback_packet.to_dict()


def _extract_verifier_feedback_block(sample: Mapping[str, Any]) -> str:
    verification_feedback = str(sample.get("verification_feedback", "")).strip()
    verification_error = str(sample.get("verification_error", "")).strip()
    if not verification_feedback and not verification_error:
        return ""

    status_lines: list[str] = []
    if sample.get("resolved") is not None:
        status_lines.append(f"resolved={_coerce_bool_flag(sample.get('resolved'), fallback=False)}")
    if sample.get("verification_missing") is not None:
        status_lines.append(
            f"verification_missing={_coerce_bool_flag(sample.get('verification_missing'), fallback=False)}"
        )

    sections: list[str] = []
    if verification_feedback:
        sections.append(f"feedback: {verification_feedback}")
    if verification_error:
        sections.append(f"error: {verification_error}")
    sections.extend(status_lines)
    if not sections:
        return ""
    return "\n".join(sections)


def _should_include_verifier_feedback(
    *,
    verifier_feedback_mode: str,
    current_turn_index: int,
    total_turn_count: int,
    turn_supervision_mode: str,
) -> bool:
    if verifier_feedback_mode == _VERIFIER_FEEDBACK_NONE:
        return False
    if verifier_feedback_mode == _VERIFIER_FEEDBACK_ALL_TURNS:
        return True
    if total_turn_count <= 0:
        return False

    if turn_supervision_mode == _TURN_SUPERVISION_CURRENT:
        final_prompt_turn_index = total_turn_count - 1
    else:
        final_prompt_turn_index = total_turn_count - 2

    return final_prompt_turn_index >= 0 and current_turn_index == final_prompt_turn_index


def _has_feedback_signal(
    sample: Mapping[str, Any],
    *,
    has_teacher_signal: bool,
    verifier_feedback_mode: str,
) -> bool:
    if has_teacher_signal:
        return True
    if verifier_feedback_mode != _VERIFIER_FEEDBACK_NONE:
        if str(sample.get("verification_feedback", "")).strip():
            return True
        if str(sample.get("verification_error", "")).strip():
            return True
    tool_output = sample.get("tool_output")
    if isinstance(tool_output, Mapping):
        if str(tool_output.get("stdout", "")).strip() or str(tool_output.get("stderr", "")).strip():
            return True
        for key, value in tool_output.items():
            if key in {"stdout", "stderr"}:
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, Mapping) and not value:
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and not value:
                continue
            return True
    tool_blocks = _coerce_text_list(sample.get("tool_response_blocks"))
    if tool_blocks:
        return True
    return False


def _resolve_legacy_distillation_active(
    *,
    policy: str,
    sample_resolved: bool,
    has_feedback_signal: bool,
) -> bool:
    if policy == _LEGACY_GATING_ALWAYS:
        return True
    if policy == _LEGACY_GATING_FEEDBACK_PRESENT:
        return sample_resolved or has_feedback_signal
    return sample_resolved


def _build_turn_prompt(
    sample: Mapping[str, Any],
    *,
    current_turn_index: int,
    supervision_mode: str,
    total_turn_count: int,
    turn_blocks: Sequence[str],
    turn_tool_blocks: Sequence[Sequence[str]],
    include_student_attempt_for_teacher: bool,
    max_reprompt_len: int,
    num_recent_raw_blocks: int,
    verifier_feedback_mode: str,
) -> tuple[str, bool, dict[str, Any]]:
    tool = str(sample.get("feedback_tool", "bash"))
    feedback_block, has_teacher_signal, feedback_packet = _build_feedback_for_turn(
        tool=tool,
        turn_index=current_turn_index,
        tool_response_blocks=turn_tool_blocks[current_turn_index],
        include_student_attempt_for_teacher=include_student_attempt_for_teacher,
    )

    memory_blocks = build_teacher_memory_blocks(sample, current_turn_index=current_turn_index)
    recent_raw_block = _build_recent_raw_block(
        turn_blocks,
        current_turn_index=current_turn_index,
        num_recent_raw_blocks=num_recent_raw_blocks,
    )

    current_attempt_block = turn_blocks[current_turn_index]
    if not include_student_attempt_for_teacher:
        current_attempt_block = ""

    verifier_feedback_block = ""
    if _should_include_verifier_feedback(
        verifier_feedback_mode=verifier_feedback_mode,
        current_turn_index=current_turn_index,
        total_turn_count=total_turn_count,
        turn_supervision_mode=supervision_mode,
    ):
        verifier_feedback_block = _extract_verifier_feedback_block(sample)
    combined_feedback_block = feedback_block
    if verifier_feedback_block:
        if combined_feedback_block:
            combined_feedback_block = f"{combined_feedback_block}\n\n[VERIFIER_FEEDBACK]\n{verifier_feedback_block}"
        else:
            combined_feedback_block = f"[VERIFIER_FEEDBACK]\n{verifier_feedback_block}"
        has_teacher_signal = True

    truncated_prompt, was_truncated = _compact_teacher_prompt(
        initial_prompt_block=_format_initial_prompt_block(sample),
        recent_raw_block=recent_raw_block,
        compressed_memory_block=memory_blocks.compressed_memory_block,
        critical_facts_block=memory_blocks.critical_facts_block,
        current_attempt_block=current_attempt_block,
        feedback_block=combined_feedback_block,
        output_contract_block=_resolve_output_contract_block(
            sample,
            supervision_mode=supervision_mode,
        ),
        max_reprompt_len=max_reprompt_len,
    )
    return truncated_prompt, has_teacher_signal, {
        "feedback_packet": feedback_packet,
        "prompt_truncated": was_truncated,
        "verifier_feedback_injected": bool(verifier_feedback_block),
    }


def _build_legacy_prompt_for_sample(
    sample: Mapping[str, Any],
    *,
    step_index: int,
    supervision_mode: str,
    include_student_attempt_for_teacher: bool,
    max_reprompt_len: int,
    verifier_feedback_mode: str,
) -> tuple[str, bool, dict[str, Any]]:
    tool = str(sample.get("feedback_tool", "bash"))
    tool_input = sample.get("feedback_tool_input", {})
    tool_output = sample.get("tool_output", {})
    if not isinstance(tool_input, Mapping):
        tool_input = {}
    if not isinstance(tool_output, Mapping):
        tool_output = {}

    feedback_packet = build_feedback_packet(
        step_index=step_index,
        tool=tool,
        tool_input=tool_input,
        tool_output=tool_output,
        include_student_attempt_for_teacher=include_student_attempt_for_teacher,
    )
    feedback_block = (
        feedback_packet.canonical_feedback.actionable_error_text
        or feedback_packet.canonical_feedback.normalized_text
    )
    current_attempt_block = str(sample.get("current_attempt_block") or sample.get("assistant_response") or "")
    if not include_student_attempt_for_teacher:
        current_attempt_block = ""

    verifier_feedback_block = ""
    if verifier_feedback_mode != _VERIFIER_FEEDBACK_NONE:
        verifier_feedback_block = _extract_verifier_feedback_block(sample)
    combined_feedback_block = feedback_block
    if verifier_feedback_block:
        if combined_feedback_block:
            combined_feedback_block = f"{combined_feedback_block}\n\n[VERIFIER_FEEDBACK]\n{verifier_feedback_block}"
        else:
            combined_feedback_block = f"[VERIFIER_FEEDBACK]\n{verifier_feedback_block}"

    memory_blocks = build_teacher_memory_blocks(sample, current_turn_index=step_index)
    truncated_prompt, was_truncated = _compact_teacher_prompt(
        initial_prompt_block=_format_initial_prompt_block(sample),
        recent_raw_block=str(sample.get("recent_raw_block", "")),
        compressed_memory_block=memory_blocks.compressed_memory_block,
        critical_facts_block=memory_blocks.critical_facts_block,
        current_attempt_block=current_attempt_block,
        feedback_block=combined_feedback_block,
        output_contract_block=_resolve_output_contract_block(
            sample,
            supervision_mode=supervision_mode,
        ),
        max_reprompt_len=max_reprompt_len,
    )
    has_teacher_signal = feedback_packet.self_containment_checks.has_actionable_error_text
    return truncated_prompt, has_teacher_signal, {
        "feedback_packet": feedback_packet.to_dict(),
        "prompt_truncated": was_truncated,
        "verifier_feedback_injected": bool(verifier_feedback_block),
    }


def build_self_distillation_batch(
    samples: Sequence[Mapping[str, Any]],
    *,
    include_student_attempt_for_teacher: bool = True,
    max_reprompt_len: int = DEFAULT_MAX_REPROMPT_LEN,
    num_recent_raw_blocks: int = DEFAULT_NUM_RECENT_RAW_BLOCKS,
    turn_supervision_mode: str = _TURN_SUPERVISION_NEXT,
    verifier_feedback_mode: str = _VERIFIER_FEEDBACK_NONE,
    legacy_distillation_gating_policy: str = _LEGACY_GATING_RESOLVED_ONLY,
) -> dict[str, Any]:
    """Build deterministic teacher prompts and mask fields for verl hooks."""
    normalized_turn_supervision_mode = _normalize_turn_supervision_mode(turn_supervision_mode)
    normalized_verifier_feedback_mode = _normalize_verifier_feedback_mode(verifier_feedback_mode)
    normalized_legacy_gating_policy = _normalize_legacy_gating_policy(
        legacy_distillation_gating_policy
    )

    teacher_prompts: list[str] = []
    self_distillation_mask: list[bool] = []
    feedback_packets: list[dict[str, Any]] = []
    prompt_truncated: list[bool] = []
    step_index_warnings: list[str] = []

    turn_teacher_prompts: list[list[str]] = []
    turn_response_masks: list[list[list[int]]] = []
    turn_distillation_mask: list[list[bool]] = []
    turn_feedback_packets: list[list[dict[str, Any]]] = []
    turn_prompt_truncated: list[list[bool]] = []

    for idx, sample in enumerate(samples):
        warning = ""
        try:
            step_index = _coerce_step_index(sample.get("step_index"), fallback=idx)
        except ValueError as exc:
            step_index = idx
            warning = str(exc)
        step_index_warnings.append(warning)

        assistant_turns = _coerce_text_list(sample.get("trajectory_assistant_turns"))
        per_turn_tool_blocks = _normalize_turn_tool_blocks(
            _coerce_nested_text_list(sample.get("trajectory_turn_tool_response_blocks")),
            target_len=len(assistant_turns),
        )
        response_mask = _coerce_binary_mask(sample.get("_response_mask"))
        turn_token_lengths = _coerce_int_list(sample.get("trajectory_assistant_turn_token_lengths"))

        sample_turn_teacher_prompts: list[str] = []
        sample_turn_response_masks: list[list[int]] = []
        sample_turn_mask: list[bool] = []
        sample_turn_feedback_packets: list[dict[str, Any]] = []
        sample_turn_prompt_truncated: list[bool] = []

        if assistant_turns and (
            normalized_turn_supervision_mode == _TURN_SUPERVISION_CURRENT or len(assistant_turns) >= 2
        ):
            turn_blocks = [
                _format_turn_block(
                    turn_index=turn_index,
                    assistant_text=assistant_turns[turn_index],
                    tool_response_blocks=per_turn_tool_blocks[turn_index],
                )
                for turn_index in range(len(assistant_turns))
            ]
            spans = _build_assistant_turn_spans(
                response_mask,
                turn_count=len(assistant_turns),
                turn_token_lengths=turn_token_lengths,
            )
            if normalized_turn_supervision_mode == _TURN_SUPERVISION_CURRENT:
                turn_indices = range(len(assistant_turns))
            else:
                turn_indices = range(len(assistant_turns) - 1)

            for current_turn_index in turn_indices:
                prompt, _has_teacher_signal, metadata = _build_turn_prompt(
                    sample,
                    current_turn_index=current_turn_index,
                    supervision_mode=normalized_turn_supervision_mode,
                    total_turn_count=len(assistant_turns),
                    turn_blocks=turn_blocks,
                    turn_tool_blocks=per_turn_tool_blocks,
                    include_student_attempt_for_teacher=include_student_attempt_for_teacher,
                    max_reprompt_len=max_reprompt_len,
                    num_recent_raw_blocks=num_recent_raw_blocks,
                    verifier_feedback_mode=normalized_verifier_feedback_mode,
                )
                if normalized_turn_supervision_mode == _TURN_SUPERVISION_CURRENT:
                    target_span = spans[current_turn_index] if current_turn_index < len(spans) else None
                else:
                    target_span = spans[current_turn_index + 1] if current_turn_index + 1 < len(spans) else None
                target_mask = _build_mask_from_span(width=len(response_mask), span=target_span)
                is_active = any(target_mask)

                sample_turn_teacher_prompts.append(prompt)
                sample_turn_response_masks.append(target_mask)
                sample_turn_mask.append(is_active)
                sample_turn_feedback_packets.append(metadata["feedback_packet"])
                sample_turn_prompt_truncated.append(bool(metadata["prompt_truncated"]))

        turn_teacher_prompts.append(sample_turn_teacher_prompts)
        turn_response_masks.append(sample_turn_response_masks)
        turn_distillation_mask.append(sample_turn_mask)
        turn_feedback_packets.append(sample_turn_feedback_packets)
        turn_prompt_truncated.append(sample_turn_prompt_truncated)

        if sample_turn_teacher_prompts:
            teacher_prompts.append(sample_turn_teacher_prompts[-1])
            feedback_packets.append(
                sample_turn_feedback_packets[-1] if sample_turn_feedback_packets else {}
            )
            prompt_truncated.append(any(sample_turn_prompt_truncated))
            self_distillation_mask.append(any(sample_turn_mask))
            continue

        legacy_prompt, has_teacher_signal, metadata = _build_legacy_prompt_for_sample(
            sample,
            step_index=step_index,
            supervision_mode=normalized_turn_supervision_mode,
            include_student_attempt_for_teacher=include_student_attempt_for_teacher,
            max_reprompt_len=max_reprompt_len,
            verifier_feedback_mode=normalized_verifier_feedback_mode,
        )
        teacher_prompts.append(legacy_prompt)

        # Keep alignment with SDPO batch semantics for non-turn trajectories.
        sample_resolved = _coerce_bool_flag(sample.get("resolved"), fallback=False)
        has_feedback_signal = _has_feedback_signal(
            sample,
            has_teacher_signal=has_teacher_signal,
            verifier_feedback_mode=normalized_verifier_feedback_mode,
        )
        self_distillation_mask.append(
            _resolve_legacy_distillation_active(
                policy=normalized_legacy_gating_policy,
                sample_resolved=sample_resolved,
                has_feedback_signal=has_feedback_signal,
            )
        )

        feedback_packets.append(metadata["feedback_packet"])
        prompt_truncated.append(bool(metadata["prompt_truncated"]))

    return {
        "teacher_prompts": teacher_prompts,
        "self_distillation_mask": self_distillation_mask,
        "feedback_packets": feedback_packets,
        "prompt_truncated": prompt_truncated,
        "step_index_warnings": step_index_warnings,
        "turn_teacher_prompts": turn_teacher_prompts,
        "turn_response_masks": turn_response_masks,
        "turn_distillation_mask": turn_distillation_mask,
        "turn_feedback_packets": turn_feedback_packets,
        "turn_prompt_truncated": turn_prompt_truncated,
    }
