from __future__ import annotations

from verl_integration.reward_function import reward_fn


def test_reward_fn_scores_resolved_and_valid_response() -> None:
    data = [
        {
            "response_text": (
                "<|im_start|>assistant\n"
                "<think>inspect failure</think>\n"
                "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"tests/test_math.py::test_add\"}}</tool_call>\n"
                "<|im_end|>"
            ),
            "resolved": True,
            "tool_output": {
                "stdout": "FAILED tests/test_math.py::test_add - AssertionError",
                "stderr": "",
                "exit_code": 1,
            },
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0]
    assert info["parse_valid"] == [True]
    assert info["required_arg_presence"] == [True]
    assert info["terminal_submission"] == [False]
    assert info["format_metrics"][0]["parse_valid_rate"] == 1.0
    assert info["format_metrics"][0]["terminal_submission_rate"] == 0.0


def test_reward_fn_returns_zero_for_invalid_payload() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{}}</tool_call>",
            "resolved": True,
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [0.0]
    assert info["parse_valid"] == [True]
    assert info["required_arg_presence"] == [False]
    assert info["validation_errors"][0]


def test_reward_fn_handles_invalid_step_index_without_aborting_batch() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"a\"}}</tool_call>",
            "resolved": True,
            "step_index": "nan",
            "tool_output": {"stdout": "ok", "stderr": "", "exit_code": 0},
        },
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"b\"}}</tool_call>",
            "resolved": True,
            "step_index": 4,
            "tool_output": {"stdout": "ok", "stderr": "", "exit_code": 0},
        },
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0, 1.0]
    assert info["step_index_warnings"][0] == "step_index must be an integer >= 0"
    assert info["step_index_warnings"][1] == ""
    assert info["validation_errors"][0] == []
    assert "STDOUT:" in info["feedback"][1]


def test_reward_fn_treats_string_false_resolved_as_unresolved() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"needle\"}}</tool_call>",
            "resolved": "false",
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [0.0]
    assert info["parse_valid"] == [True]
    assert info["validation_errors"] == [[]]


def test_reward_fn_tracks_terminal_submission_rate() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            "resolved": True,
        },
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"needle\"}}</tool_call>",
            "resolved": True,
        },
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0, 1.0]
    assert info["terminal_submission"] == [True, False]
    assert info["format_metrics"][0]["terminal_submission_rate"] == 0.5


def test_reward_fn_prefers_verifiable_fail_pass_signals_when_available() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"needle\"}}</tool_call>",
            "resolved": False,
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
            "tool_output": {
                "metadata": {
                    "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": "passed"},
                    "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
                }
            },
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0]
    assert info["resolved_source"] == ["verifiable_tests"]
    assert info["fail_to_pass_verified"] == [True]
    assert info["pass_to_pass_verified"] == [True]
    assert info["reward_verification_missing"] == [False]


def test_reward_fn_falls_back_to_resolved_flag_without_verification_signals() -> None:
    data = [
        {
            "response_text": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"needle\"}}</tool_call>",
            "resolved": True,
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
        }
    ]

    rewards, info = reward_fn(data)

    assert rewards == [1.0]
    assert info["resolved_source"] == ["resolved_flag"]
    assert info["reward_verification_missing"] == [True]
