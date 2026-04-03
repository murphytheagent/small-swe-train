"""Reward-function adapter for turn-SDPO SWE rollout records."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from config import MAX_TOOL_CALLS_PER_TURN, TERMINAL_VALIDITY_PENALTY
from metrics.contracts import FormatMetrics, rate
from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn
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


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
        names = [str(key).strip() for key in value]
        return {name for name in names if name}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = [str(item).strip() for item in value]
        return {name for name in names if name}
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
        if "\n" in stripped:
            return {chunk.strip() for chunk in stripped.splitlines() if chunk.strip()}
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
    expected_tests = _coerce_test_name_set(_lookup_verification_value(sample, key_root, key_root.upper()))

    all_passed_raw = _lookup_verification_value(
        sample,
        f"{key_root}_all_passed",
        f"{key_root.upper()}_ALL_PASSED",
    )
    all_passed = _coerce_optional_bool_flag(all_passed_raw)
    if all_passed is not None and expected_tests:
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
    has_non_empty_results = False
    if isinstance(results_raw, Mapping):
        has_non_empty_results = bool(results_raw)
        for test_name, status in results_raw.items():
            normalized_name = str(test_name).strip()
            if not normalized_name:
                continue
            verdict = _coerce_optional_bool_flag(status)
            if verdict is True:
                passed_tests.add(normalized_name)
            elif verdict is False:
                failed_tests.add(normalized_name)

    # Empty verifier-result mappings are non-informative when no expected target list
    # is present. Treat those as missing verification rather than implicit success.
    has_signal = (
        has_non_empty_results
        or bool(passed_tests)
        or bool(failed_tests)
        or (all_passed_raw is not None and bool(expected_tests))
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
    infra_invalid = _coerce_bool_flag(_lookup_verification_value(sample, "infra_invalid"), fallback=False)
    invalid_reason = str(_lookup_verification_value(sample, "invalid_reason") or "").strip()

    if infra_invalid:
        return {
            "resolved": None,
            "has_expected_tests": has_expected_tests,
            "fail_to_pass_verified": fail_verified,
            "pass_to_pass_verified": pass_verified,
            "verification_missing": True,
            "infra_invalid": True,
            "invalid_reason": invalid_reason or "infra_invalid",
        }

    if not has_any_signal:
        return {
            "resolved": None,
            "has_expected_tests": has_expected_tests,
            "fail_to_pass_verified": fail_verified,
            "pass_to_pass_verified": pass_verified,
            "verification_missing": True,
            "infra_invalid": False,
            "invalid_reason": "",
        }

    fail_result = fail_verified if fail_verified is not None else (False if fail_expected else True)
    pass_result = pass_verified if pass_verified is not None else (False if pass_expected else True)
    return {
        "resolved": bool(fail_result and pass_result),
        "has_expected_tests": has_expected_tests,
        "fail_to_pass_verified": bool(fail_result),
        "pass_to_pass_verified": bool(pass_result),
        "verification_missing": False,
        "infra_invalid": False,
        "invalid_reason": "",
    }


def _verification_feedback(sample: Mapping[str, Any], *, verification: Mapping[str, Any]) -> str:
    explicit_feedback = _lookup_verification_value(sample, "verification_feedback", "reward_feedback", "feedback")
    if isinstance(explicit_feedback, str) and explicit_feedback.strip():
        return explicit_feedback.strip()
    if bool(verification.get("infra_invalid", False)):
        invalid_reason = str(verification.get("invalid_reason", "")).strip() or "infra_invalid"
        return f"Verifier invalid: {invalid_reason}"
    if bool(verification.get("resolved", False)):
        return "Verifier: all FAIL_TO_PASS and PASS_TO_PASS tests passed."

    fail_verified = verification.get("fail_to_pass_verified")
    pass_verified = verification.get("pass_to_pass_verified")
    lines: list[str] = [
        f"Verifier resolution: fail_to_pass={bool(fail_verified)} pass_to_pass={bool(pass_verified)}"
    ]
    verification_error = _lookup_verification_value(sample, "verification_error")
    if isinstance(verification_error, str) and verification_error.strip():
        lines.append(f"Verifier error: {verification_error.strip()}")
    return "\n".join(lines)


def reward_fn(
    data: Sequence[Mapping[str, Any]],
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
) -> tuple[list[float], dict[str, list[Any]]]:
    """Compute baseline-minus-penalty rewards from verifier + terminal-submit validity."""
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
    validation_errors: list[bool] = []
    validation_error_messages: list[str] = []
    resolved_sources: list[str] = []
    verifier_statuses: list[str] = []
    fail_to_pass_verified: list[bool] = []
    pass_to_pass_verified: list[bool] = []
    reward_verification_missing: list[bool] = []
    terminal_submit_content: list[str] = []

    step_index_warnings: list[str] = []

    for sample in data:
        response_text = str(sample.get("response_text") or sample.get("assistant_response") or "")
        sample_errors: list[str] = []
        parse_valid = True
        tool_presence = False
        tool_count_valid = False
        submit_singleton_ok = False
        allowed_tools_ok = False
        required_args_ok = False
        parsed_has_submit = False
        parsed_format_valid = False
        terminal_submission_ok = False
        submit_contract_ok = True
        final_submit_text = ""
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
            submit_count = sum(1 for call in tool_calls if call.tool == TERMINAL_TOOL_NAME)
            submit_singleton_ok = submit_count in {0, 1} and not (submit_count == 1 and len(tool_calls) != 1)
            if submit_count > 0 and not submit_singleton_ok:
                submit_contract_ok = False

            call_error_lists = [validate_tool_call(call) for call in tool_calls]
            sample_errors.extend(error for errors in call_error_lists for error in errors)
            allowed_tools_ok = all(call.tool in _ALLOWED_TOOLS_SET for call in tool_calls)
            required_args_ok = all(not errors for errors in call_error_lists)
            if len(tool_calls) == 1 and tool_calls[0].tool == TERMINAL_TOOL_NAME:
                parsed_has_submit = True
                parsed_format_valid = required_args_ok
                raw_final_response = tool_calls[0].args.get("final_response")
                if isinstance(raw_final_response, str):
                    final_submit_text = raw_final_response.strip()
                elif raw_final_response is None:
                    final_submit_text = ""
                else:
                    final_submit_text = str(raw_final_response).strip()
                if not final_submit_text:
                    parsed_format_valid = False

        final_turn_has_submit = _coerce_optional_bool_flag(
            _lookup_verification_value(sample, "final_turn_has_submit", "FINAL_TURN_HAS_SUBMIT")
        )
        final_submit_format_valid = _coerce_optional_bool_flag(
            _lookup_verification_value(
                sample,
                "terminal_format_valid",
                "TERMINAL_FORMAT_VALID",
                "final_submit_format_valid",
                "FINAL_SUBMIT_FORMAT_VALID",
            )
        )
        explicit_final_response = _lookup_verification_value(
            sample,
            "submission_final_response",
            "SUBMISSION_FINAL_RESPONSE",
        )
        if explicit_final_response is not None:
            explicit_text = _as_text(explicit_final_response).strip()
            final_submit_text = explicit_text

        has_submit = parsed_has_submit
        if final_turn_has_submit is not None:
            has_submit = bool(final_turn_has_submit)
        format_valid = parsed_format_valid
        if final_submit_format_valid is not None:
            format_valid = bool(final_submit_format_valid)
        if not parse_valid and final_turn_has_submit:
            submit_contract_ok = False
        terminal_submission_ok = bool(has_submit and format_valid and submit_contract_ok)
        if not has_submit:
            final_submit_text = ""

        verification = _resolve_verifiable_resolution(sample)
        resolved_from_verification = _coerce_optional_bool_flag(verification.get("resolved"))
        has_expected_tests = bool(verification.get("has_expected_tests", False))
        verification_missing = bool(verification.get("verification_missing", False))
        infra_invalid = bool(verification.get("infra_invalid", False))
        invalid_reason = str(verification.get("invalid_reason", "")).strip()
        if infra_invalid:
            resolved_sources.append(invalid_reason or "infra_invalid")
        elif resolved_from_verification is None:
            if has_expected_tests:
                resolved_sources.append("missing_verifier")
            else:
                resolved_sources.append("missing_verifier_targets")
        else:
            resolved_sources.append("verifiable_tests")

        fail_verified_raw = _coerce_optional_bool_flag(verification.get("fail_to_pass_verified"))
        pass_verified_raw = _coerce_optional_bool_flag(verification.get("pass_to_pass_verified"))
        fail_verified = bool(fail_verified_raw) if fail_verified_raw is not None else False
        pass_verified = bool(pass_verified_raw) if pass_verified_raw is not None else False
        fail_to_pass_verified.append(fail_verified)
        pass_to_pass_verified.append(pass_verified)
        reward_verification_missing.append(verification_missing)
        if infra_invalid:
            verifier_statuses.append("invalid")
        elif resolved_sources[-1] != "verifiable_tests":
            verifier_statuses.append("missing")
        elif fail_verified and pass_verified:
            verifier_statuses.append("correct")
        else:
            verifier_statuses.append("incorrect")

        reward_value = 0.0
        if infra_invalid:
            reward_value = 0.0
        elif has_expected_tests:
            # Missing verifier signal on tasks with expected tests is treated as
            # an unresolved verification failure, not a neutral baseline.
            reward_value = -1.0 if verification_missing else 1.0
        if not infra_invalid and has_expected_tests and not verification_missing:
            if not fail_verified:
                reward_value -= 1.0
            if not pass_verified:
                reward_value -= 1.0
        if not infra_invalid and not terminal_submission_ok:
            reward_value -= TERMINAL_VALIDITY_PENALTY
        rewards.append(reward_value)
        feedback.append(_verification_feedback(sample, verification=verification))

        parse_valid_flags.append(parse_valid)
        tool_presence_flags.append(tool_presence)
        tool_count_flags.append(tool_count_valid)
        submit_singleton_flags.append(submit_singleton_ok)
        allowed_tool_flags.append(allowed_tools_ok)
        required_arg_flags.append(required_args_ok)
        terminal_submission_flags.append(terminal_submission_ok)
        think_balance_flags.append(_thinking_delimiters_balanced(response_text))
        validation_errors.append(bool(sample_errors))
        validation_error_messages.append("; ".join(sample_errors))
        terminal_submit_content.append(final_submit_text)
        step_index_warnings.append("")

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
        "validation_error_messages": validation_error_messages,
        "step_index_warnings": step_index_warnings,
        "resolved_source": resolved_sources,
        "verifier_resolution_source": resolved_sources,
        "verifier_status": verifier_statuses,
        "terminal_submit_content": terminal_submit_content,
        "fail_to_pass_verified": fail_to_pass_verified,
        "pass_to_pass_verified": pass_to_pass_verified,
        "reward_verification_missing": reward_verification_missing,
        "format_metrics": [metrics.__dict__],
    }
    return rewards, info
