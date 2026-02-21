"""Teacher-reprompt assembly helpers for self-distillation batches."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from data.feedback_canonicalizer import build_feedback_packet
from teacher.prompt_builder import TeacherPromptInputs, build_teacher_prompt

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


def _truncate_prompt_tokens(prompt: str, *, max_reprompt_len: int) -> tuple[str, bool]:
    if max_reprompt_len <= 0:
        raise ValueError("max_reprompt_len must be > 0")

    tokens = prompt.split()
    if len(tokens) <= max_reprompt_len:
        return prompt, False
    return " ".join(tokens[:max_reprompt_len]), True


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


def _build_prompt_for_sample(
    sample: Mapping[str, Any],
    *,
    step_index: int,
    include_student_attempt_for_teacher: bool,
    max_reprompt_len: int,
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

    prompt = build_teacher_prompt(
        TeacherPromptInputs(
            system_block=str(
                sample.get(
                    "system_block",
                    "You are a SWE coding agent. Follow the tool-use contract exactly.",
                )
            ),
            task_block=str(sample.get("task_block") or sample.get("prompt") or ""),
            recent_raw_block=str(sample.get("recent_raw_block", "")),
            compressed_memory_block=str(sample.get("compressed_memory_block", "")),
            critical_facts_block=str(sample.get("critical_facts_block", "")),
            current_attempt_block=current_attempt_block,
            feedback_block=feedback_block,
            output_contract_block=str(
                sample.get(
                    "output_contract_block",
                    "Return optional <think> and one-or-more <tool_call> blocks.",
                )
            ),
        )
    )

    truncated_prompt, was_truncated = _truncate_prompt_tokens(
        prompt,
        max_reprompt_len=max_reprompt_len,
    )

    has_teacher_signal = bool(feedback_block.strip())
    return truncated_prompt, has_teacher_signal, {
        "feedback_packet": feedback_packet.to_dict(),
        "prompt_truncated": was_truncated,
    }


def build_self_distillation_batch(
    samples: Sequence[Mapping[str, Any]],
    *,
    include_student_attempt_for_teacher: bool = True,
    max_reprompt_len: int = 10240,
) -> dict[str, Any]:
    """Build deterministic teacher prompts and mask fields for verl hooks."""
    teacher_prompts: list[str] = []
    self_distillation_mask: list[bool] = []
    feedback_packets: list[dict[str, Any]] = []
    prompt_truncated: list[bool] = []
    step_index_warnings: list[str] = []

    for idx, sample in enumerate(samples):
        warning = ""
        try:
            step_index = _coerce_step_index(sample.get("step_index"), fallback=idx)
        except ValueError as exc:
            step_index = idx
            warning = str(exc)

        teacher_prompt, has_teacher_signal, metadata = _build_prompt_for_sample(
            sample,
            step_index=step_index,
            include_student_attempt_for_teacher=include_student_attempt_for_teacher,
            max_reprompt_len=max_reprompt_len,
        )
        teacher_prompts.append(teacher_prompt)

        # Keep alignment with SDPO batch semantics: enable distillation only
        # when we have either explicit success labels or actionable feedback.
        sample_resolved = _coerce_bool_flag(sample.get("resolved"), fallback=False)
        self_distillation_mask.append(sample_resolved or has_teacher_signal)

        feedback_packets.append(metadata["feedback_packet"])
        prompt_truncated.append(bool(metadata["prompt_truncated"]))
        step_index_warnings.append(warning)

    return {
        "teacher_prompts": teacher_prompts,
        "self_distillation_mask": self_distillation_mask,
        "feedback_packets": feedback_packets,
        "prompt_truncated": prompt_truncated,
        "step_index_warnings": step_index_warnings,
    }
