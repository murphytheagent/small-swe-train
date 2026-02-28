from __future__ import annotations

import config
from verl_integration.reward_loop_score import compute_score


def test_compute_score_uses_project_reward_logic() -> None:
    result = compute_score(
        data_source="dummy/dataset",
        solution_str='<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
        ground_truth={
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
        },
        extra_info={
            "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
            "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
            "verification_feedback": "all tests passed",
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
        },
    )

    assert result["score"] == 1.0
    assert result["terminal_submission"] is True
    assert result["resolved_source"] == "verifiable_tests"
    assert result["feedback"] == "all tests passed"


def test_compute_score_applies_terminal_penalty_without_submit() -> None:
    result = compute_score(
        data_source="dummy/dataset",
        solution_str='<tool_call>{"tool":"search","args":{"query":"needle"}}</tool_call>',
        ground_truth={
            "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_regression"],
        },
        extra_info={
            "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
            "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
        },
    )

    assert result["score"] == 1.0 - config.TERMINAL_VALIDITY_PENALTY
    assert result["terminal_submission"] is False
