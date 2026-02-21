"""Reward-function adapter for step-SDPO style rollout records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from data.feedback_canonicalizer import build_feedback_packet
from metrics.contracts import FormatMetrics, rate
from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn
from schemas import ActionEnvelope, validate_tool_call

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


def _parse_response_text(response_text: str, *, max_tool_calls: int) -> ActionEnvelope:
    stripped = response_text.strip()
    if stripped.startswith("<|im_start|>assistant"):
        return parse_chatml_assistant_turn(stripped, max_tool_calls=max_tool_calls)
    return parse_assistant_turn_payload(stripped, max_tool_calls=max_tool_calls)


def _thinking_delimiters_balanced(response_text: str) -> bool:
    return response_text.count("<think>") == response_text.count("</think>")


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


def reward_fn(
    data: Sequence[Mapping[str, Any]],
    *,
    max_tool_calls: int = 3,
) -> tuple[list[float], dict[str, list[Any]]]:
    """Compute per-sample binary rewards and rollout diagnostics.

    Expected sample keys:
    - ``response_text`` (or alias ``assistant_response``): assistant output text
    - ``resolved``: bool-like outcome flag from external evaluator
    - ``tool_output``: optional mapping for canonicalized feedback extraction
    """
    rewards: list[float] = []
    feedback: list[str] = []

    parse_valid_flags: list[bool] = []
    tool_presence_flags: list[bool] = []
    tool_count_flags: list[bool] = []
    submit_singleton_flags: list[bool] = []
    allowed_tool_flags: list[bool] = []
    required_arg_flags: list[bool] = []
    think_balance_flags: list[bool] = []
    validation_errors: list[list[str]] = []

    for index, sample in enumerate(data):
        response_text = str(sample.get("response_text") or sample.get("assistant_response") or "")
        resolved = _coerce_bool_flag(sample.get("resolved"), fallback=False)

        sample_errors: list[str] = []
        parse_valid = True
        tool_presence = False
        tool_count_valid = False
        submit_singleton_ok = False
        allowed_tools_ok = False
        required_args_ok = False
        envelope: ActionEnvelope | None = None

        try:
            envelope = _parse_response_text(response_text, max_tool_calls=max_tool_calls)
        except (TurnParseError, ValueError) as exc:
            parse_valid = False
            sample_errors.append(str(exc))

        if envelope is not None:
            tool_calls = envelope.tool_calls
            tool_presence = bool(tool_calls)
            tool_count_valid = 1 <= len(tool_calls) <= max_tool_calls
            submit_count = sum(1 for call in tool_calls if call.tool == "submit")
            submit_singleton_ok = submit_count in {0, 1} and not (submit_count == 1 and len(tool_calls) != 1)

            call_error_lists = [validate_tool_call(call) for call in tool_calls]
            sample_errors.extend(error for errors in call_error_lists for error in errors)
            allowed_tools_ok = all(call.tool in {"bash", "search", "edit", "submit"} for call in tool_calls)
            required_args_ok = all(not errors for errors in call_error_lists)

        try:
            step_index = _coerce_step_index(sample.get("step_index"), fallback=index)
        except ValueError as exc:
            step_index = index
            sample_errors.append(str(exc))

        reward_value = 1.0 if resolved and parse_valid and not sample_errors else 0.0
        rewards.append(reward_value)

        parse_valid_flags.append(parse_valid)
        tool_presence_flags.append(tool_presence)
        tool_count_flags.append(tool_count_valid)
        submit_singleton_flags.append(submit_singleton_ok)
        allowed_tool_flags.append(allowed_tools_ok)
        required_arg_flags.append(required_args_ok)
        think_balance_flags.append(_thinking_delimiters_balanced(response_text))
        validation_errors.append(sample_errors)

        tool_output = sample.get("tool_output")
        if sample_errors:
            feedback.append("; ".join(sample_errors))
        elif isinstance(tool_output, Mapping) and envelope is not None:
            first_call = envelope.tool_calls[0]
            packet = build_feedback_packet(
                step_index=step_index,
                tool=first_call.tool,
                tool_input=first_call.args,
                tool_output=tool_output,
                include_student_attempt_for_teacher=_coerce_bool_flag(
                    sample.get("include_student_attempt_for_teacher"),
                    fallback=True,
                ),
            )
            feedback_text = (
                packet.canonical_feedback.actionable_error_text
                or packet.canonical_feedback.normalized_text
            )
            feedback.append(feedback_text)
        else:
            feedback.append("")

    metrics = FormatMetrics(
        parse_valid_rate=rate(parse_valid_flags),
        tool_call_block_presence_rate=rate(tool_presence_flags),
        tool_call_count_valid_rate=rate(tool_count_flags),
        submit_singleton_rule_rate=rate(submit_singleton_flags),
        thinking_delimiter_balance_rate=rate(think_balance_flags),
        allowed_tool_rate=rate(allowed_tool_flags),
        required_arg_presence=rate(required_arg_flags),
    )

    info: dict[str, list[Any]] = {
        "feedback": feedback,
        "parse_valid": parse_valid_flags,
        "tool_call_block_presence": tool_presence_flags,
        "tool_call_count_valid": tool_count_flags,
        "submit_singleton_rule": submit_singleton_flags,
        "thinking_delimiter_balance": think_balance_flags,
        "allowed_tool": allowed_tool_flags,
        "required_arg_presence": required_arg_flags,
        "validation_errors": validation_errors,
        "format_metrics": [metrics.__dict__],
    }
    return rewards, info
