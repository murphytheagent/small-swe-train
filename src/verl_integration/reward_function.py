"""Reward-function adapter for step-SDPO style rollout records."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from data.feedback_canonicalizer import build_feedback_packet
from metrics.contracts import FormatMetrics, rate
from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn
from config import MAX_TOOL_CALLS_PER_TURN
from schemas import ALLOWED_TOOLS, TERMINAL_TOOL_NAME, ActionEnvelope, validate_tool_call

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}
_ALLOWED_TOOLS_SET = set(ALLOWED_TOOLS)


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


def _coerce_optional_bool_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS or normalized in {"pass", "passed", "success", "resolved"}:
            return True
        if normalized in _FALSE_STRINGS or normalized in {"fail", "failed", "error", "unresolved"}:
            return False
    if isinstance(value, Mapping):
        for key in ("passed", "resolved", "success", "status"):
            if key not in value:
                continue
            verdict = _coerce_optional_bool_flag(value.get(key))
            if verdict is not None:
                return verdict
    return None


def _coerce_test_name_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, Mapping):
        names = []
        for key in value:
            text = str(key).strip()
            if text:
                names.append(text)
        return set(names)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = []
        for item in value:
            text = str(item).strip()
            if text:
                names.append(text)
        return set(names)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _coerce_test_name_set(parsed)
        if "," in stripped:
            return {chunk.strip() for chunk in stripped.split(",") if chunk.strip()}
        return {stripped}
    return set()


def _iter_verification_sources(sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [sample]
    tool_output = sample.get("tool_output")
    if isinstance(tool_output, Mapping):
        metadata = tool_output.get("metadata")
        if isinstance(metadata, Mapping):
            sources.append(metadata)
    return tuple(sources)


def _lookup_verification_value(sample: Mapping[str, Any], *candidates: str) -> Any:
    candidate_keys = [str(key).strip() for key in candidates if str(key).strip()]
    if not candidate_keys:
        return None
    lowered_candidates = {key.lower() for key in candidate_keys}
    for source in _iter_verification_sources(sample):
        for key in candidate_keys:
            if key in source:
                return source[key]
        lowered_map = {
            str(source_key).strip().lower(): source_value
            for source_key, source_value in source.items()
            if isinstance(source_key, str)
        }
        for lowered_key in lowered_candidates:
            if lowered_key in lowered_map:
                return lowered_map[lowered_key]
    return None


def _resolve_test_group_verification(
    sample: Mapping[str, Any],
    *,
    key_root: str,
) -> tuple[set[str], bool | None, bool]:
    expected_tests = _coerce_test_name_set(
        _lookup_verification_value(sample, key_root, key_root.upper())
    )

    all_passed_raw = _lookup_verification_value(
        sample,
        f"{key_root}_all_passed",
        f"{key_root.upper()}_ALL_PASSED",
    )
    all_passed = _coerce_optional_bool_flag(all_passed_raw)
    if all_passed is not None:
        return expected_tests, all_passed, True

    passed_tests = _coerce_test_name_set(
        _lookup_verification_value(
            sample,
            f"{key_root}_passed",
            f"{key_root.upper()}_PASSED",
            f"{key_root}_passing",
            f"{key_root.upper()}_PASSING",
        )
    )
    failed_tests = _coerce_test_name_set(
        _lookup_verification_value(
            sample,
            f"{key_root}_failed",
            f"{key_root.upper()}_FAILED",
            f"{key_root}_failing",
            f"{key_root.upper()}_FAILING",
        )
    )

    results_raw = _lookup_verification_value(
        sample,
        f"{key_root}_results",
        f"{key_root.upper()}_RESULTS",
    )
    if isinstance(results_raw, Mapping):
        for test_name, status in results_raw.items():
            normalized_name = str(test_name).strip()
            if not normalized_name:
                continue
            verdict = _coerce_optional_bool_flag(status)
            if verdict is True:
                passed_tests.add(normalized_name)
            elif verdict is False:
                failed_tests.add(normalized_name)

    has_signal = (
        all_passed_raw is not None
        or results_raw is not None
        or bool(passed_tests)
        or bool(failed_tests)
    )
    if not has_signal:
        return expected_tests, None, False

    if expected_tests:
        if expected_tests.intersection(failed_tests):
            return expected_tests, False, True
        return expected_tests, expected_tests.issubset(passed_tests), True

    if failed_tests:
        return expected_tests, False, True
    return expected_tests, True, True


def _resolve_verifiable_resolution(sample: Mapping[str, Any]) -> dict[str, Any]:
    fail_expected, fail_verified, fail_signal = _resolve_test_group_verification(
        sample,
        key_root="fail_to_pass",
    )
    pass_expected, pass_verified, pass_signal = _resolve_test_group_verification(
        sample,
        key_root="pass_to_pass",
    )

    has_expected_tests = bool(fail_expected or pass_expected)
    has_any_signal = fail_signal or pass_signal

    if not has_any_signal:
        return {
            "resolved": None,
            "fail_to_pass_verified": fail_verified,
            "pass_to_pass_verified": pass_verified,
            "verification_missing": has_expected_tests,
        }

    fail_result = (
        fail_verified
        if fail_verified is not None
        else (False if fail_expected else True)
    )
    pass_result = (
        pass_verified
        if pass_verified is not None
        else (False if pass_expected else True)
    )
    return {
        "resolved": bool(fail_result and pass_result),
        "fail_to_pass_verified": fail_result,
        "pass_to_pass_verified": pass_result,
        "verification_missing": False,
    }


def reward_fn(
    data: Sequence[Mapping[str, Any]],
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
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
    terminal_submission_flags: list[bool] = []
    think_balance_flags: list[bool] = []
    validation_errors: list[list[str]] = []
    resolved_sources: list[str] = []
    fail_to_pass_verified: list[bool | None] = []
    pass_to_pass_verified: list[bool | None] = []
    reward_verification_missing: list[bool] = []

    step_index_warnings: list[str] = []

    for index, sample in enumerate(data):
        response_text = str(sample.get("response_text") or sample.get("assistant_response") or "")
        resolved_from_flag = _coerce_bool_flag(sample.get("resolved"), fallback=False)
        verification = _resolve_verifiable_resolution(sample)
        resolved_from_verification = _coerce_optional_bool_flag(verification.get("resolved"))
        if resolved_from_verification is None:
            resolved = resolved_from_flag
            resolved_sources.append("resolved_flag")
        else:
            resolved = resolved_from_verification
            resolved_sources.append("verifiable_tests")
        fail_to_pass_verified.append(bool(_coerce_optional_bool_flag(verification.get("fail_to_pass_verified"))))
        pass_to_pass_verified.append(bool(_coerce_optional_bool_flag(verification.get("pass_to_pass_verified"))))
        reward_verification_missing.append(bool(verification.get("verification_missing", False)))

        sample_errors: list[str] = []
        parse_valid = True
        tool_presence = False
        tool_count_valid = False
        submit_singleton_ok = False
        allowed_tools_ok = False
        required_args_ok = False
        terminal_submission_ok = False
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
            allowed_tools_ok = all(call.tool in _ALLOWED_TOOLS_SET for call in tool_calls)
            required_args_ok = all(not errors for errors in call_error_lists)
            terminal_submission_ok = (
                len(tool_calls) == 1 and tool_calls[0].tool == TERMINAL_TOOL_NAME
            )

        step_index_warning = ""
        try:
            step_index = _coerce_step_index(sample.get("step_index"), fallback=index)
        except ValueError as exc:
            step_index = index
            step_index_warning = str(exc)
        step_index_warnings.append(step_index_warning)

        reward_value = 1.0 if resolved and parse_valid and not sample_errors else 0.0
        rewards.append(reward_value)

        parse_valid_flags.append(parse_valid)
        tool_presence_flags.append(tool_presence)
        tool_count_flags.append(tool_count_valid)
        submit_singleton_flags.append(submit_singleton_ok)
        allowed_tool_flags.append(allowed_tools_ok)
        required_arg_flags.append(required_args_ok)
        terminal_submission_flags.append(terminal_submission_ok)
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
        terminal_submission_rate=rate(terminal_submission_flags),
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
        "terminal_submission": terminal_submission_flags,
        "validation_errors": validation_errors,
        "step_index_warnings": step_index_warnings,
        "resolved_source": resolved_sources,
        "fail_to_pass_verified": fail_to_pass_verified,
        "pass_to_pass_verified": pass_to_pass_verified,
        "reward_verification_missing": reward_verification_missing,
        "format_metrics": [metrics.__dict__],
    }
    return rewards, info
