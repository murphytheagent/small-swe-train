from __future__ import annotations

import config

from verl_integration.reward_function import reward_fn


def test_reward_fn_requires_terminal_submit_and_verifier_success() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": "passed"},
                    "verification_feedback": "all tests passed",
                }
            },
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0]
    assert info["terminal_submission"] == [True]
    assert info["resolved_source"] == ["verifiable_tests"]
    assert info["fail_to_pass_verified"] == [True]
    assert info["pass_to_pass_verified"] == [True]
    assert info["reward_verification_missing"] == [False]
    assert info["feedback"] == ["all tests passed"]
    assert info["terminal_submit_content"] == ["done"]


def test_reward_fn_returns_zero_when_verifier_signals_are_missing() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [0.0]
    assert info["resolved_source"] == ["missing_verifier"]
    assert info["reward_verification_missing"] == [True]
    assert info["fail_to_pass_verified"] == [False]
    assert info["pass_to_pass_verified"] == [False]


def test_reward_fn_applies_terminal_validity_penalty_when_submit_is_missing() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"search","args":{"query":"needle"}}</tool_call>',
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
                }
            },
        }
    ]

    rewards, info = reward_fn(data)

    expected_reward = 1.0 - config.TERMINAL_VALIDITY_PENALTY
    assert rewards == [expected_reward]
    assert info["terminal_submission"] == [False]
    assert info["resolved_source"] == ["verifiable_tests"]


def test_reward_fn_returns_zero_when_no_verifier_targets_exist() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
            "resolved": True,
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [0.0]
    assert info["resolved_source"] == ["missing_verifier_targets"]
    assert info["reward_verification_missing"] == [True]
    assert info["fail_to_pass_verified"] == [False]
    assert info["pass_to_pass_verified"] == [False]


def test_reward_fn_treats_empty_verifier_results_without_targets_as_missing() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
            "fail_to_pass_results": {},
            "pass_to_pass_results": {},
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [0.0]
    assert info["resolved_source"] == ["missing_verifier_targets"]
    assert info["reward_verification_missing"] == [True]
    assert info["fail_to_pass_verified"] == [False]
    assert info["pass_to_pass_verified"] == [False]


def test_reward_fn_ignores_vacuous_all_passed_without_targets() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
            "fail_to_pass": [],
            "pass_to_pass": [],
            "fail_to_pass_all_passed": True,
            "pass_to_pass_all_passed": True,
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [0.0]
    assert info["resolved_source"] == ["missing_verifier_targets"]
    assert info["reward_verification_missing"] == [True]
    assert info["fail_to_pass_verified"] == [False]
    assert info["pass_to_pass_verified"] == [False]


def test_reward_fn_applies_unresolved_and_terminal_penalties_for_invalid_payload() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"submit\",\"args\":{}}</tool_call>",
            "resolved": True,
        }
    ]

    rewards, info = reward_fn(data)

    expected_reward = 1.0 - 1.0 - config.TERMINAL_VALIDITY_PENALTY
    assert rewards == [expected_reward]
    assert info["parse_valid"] == [True]
    assert info["required_arg_presence"] == [False]
    assert info["validation_errors"] == [True]
    assert "final_response" in info["validation_error_messages"][0]


def test_reward_fn_rejects_terminal_metadata_when_submit_is_not_singleton() -> None:
    data = [
        {
            "response_text": (
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
                "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"needle\"}}</tool_call>"
            ),
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
                    "final_turn_has_submit": True,
                    "final_submit_format_valid": True,
                }
            },
        }
    ]

    rewards, info = reward_fn(data)

    expected_reward = 1.0 - config.TERMINAL_VALIDITY_PENALTY
    assert rewards == [expected_reward]
    assert info["terminal_submission"] == [False]


def test_reward_fn_coerces_explicit_submission_final_response_to_text() -> None:
    data = [
        {
            "response_text": '<tool_call>{"tool":"search","args":{"query":"needle"}}</tool_call>',
            "submission_final_response": {"summary": "done"},
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
                }
            },
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0]
    assert info["terminal_submission"] == [True]
    assert info["terminal_submit_content"] == ["{'summary': 'done'}"]
