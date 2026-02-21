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
