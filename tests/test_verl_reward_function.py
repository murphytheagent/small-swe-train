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
    assert info["format_metrics"][0]["parse_valid_rate"] == 1.0


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

    assert rewards == [0.0, 1.0]
    assert "step_index must be an integer >= 0" in info["validation_errors"][0]
    assert "STDOUT:" in info["feedback"][1]
