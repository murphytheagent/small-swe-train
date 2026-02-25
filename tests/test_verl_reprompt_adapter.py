from __future__ import annotations

from verl_integration.reprompt_adapter import build_self_distillation_batch


def test_build_self_distillation_batch_contains_contract_blocks() -> None:
    samples = [
        {
            "prompt": "Fix failing test",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"pytest -q\"}}</tool_call>",
            "tool_output": {
                "stdout": "FAILED tests/test_math.py::test_add - AssertionError",
                "stderr": "",
                "exit_code": 1,
            },
            "resolved": False,
        }
    ]

    batch = build_self_distillation_batch(samples)

    prompt = batch["teacher_prompts"][0]
    assert "[SYSTEM_BLOCK]" in prompt
    assert "[TASK_BLOCK]" in prompt
    assert "[FEEDBACK_BLOCK]" in prompt
    assert batch["self_distillation_mask"] == [False]


def test_build_self_distillation_batch_honors_token_limit() -> None:
    long_attempt = " ".join(f"tok{i}" for i in range(80))
    samples = [
        {
            "prompt": "task",
            "assistant_response": long_attempt,
            "tool_output": {"stdout": "err", "stderr": "", "exit_code": 1},
        }
    ]

    batch = build_self_distillation_batch(samples, max_reprompt_len=20)

    assert len(batch["teacher_prompts"][0].split()) == 20
    assert batch["prompt_truncated"] == [True]


def test_build_self_distillation_batch_falls_back_on_invalid_step_index() -> None:
    samples = [
        {
            "prompt": "Fix failing test",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"pytest -q\"}}</tool_call>",
            "step_index": "bad",
            "tool_output": {
                "stdout": "FAILED tests/test_math.py::test_add - AssertionError",
                "stderr": "",
                "exit_code": 1,
            },
        }
    ]

    batch = build_self_distillation_batch(samples)

    assert batch["self_distillation_mask"] == [False]
    assert batch["step_index_warnings"] == ["step_index must be an integer >= 0"]
    assert batch["feedback_packets"][0]["step_index"] == 0


def test_build_self_distillation_batch_empty_tool_output_does_not_set_teacher_signal() -> None:
    """Regression: empty tool_output canonicalizes to '{}', which must not
    count as actionable teacher signal."""
    samples = [
        {
            "prompt": "Fix the thing",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"echo hi\"}}</tool_call>",
            "tool_output": {},
            "resolved": False,
        }
    ]

    batch = build_self_distillation_batch(samples)

    assert batch["self_distillation_mask"] == [False]


def test_build_self_distillation_batch_truncation_preserves_newlines() -> None:
    """Regression: truncation must not flatten multi-line prompt structure."""
    samples = [
        {
            "prompt": "task",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"x\"}}</tool_call>",
            "tool_output": {"stdout": "ok", "stderr": "", "exit_code": 0},
        }
    ]

    batch = build_self_distillation_batch(samples, max_reprompt_len=10)

    prompt = batch["teacher_prompts"][0]
    assert "\n" in prompt
    assert batch["prompt_truncated"] == [True]


def test_build_self_distillation_batch_treats_false_string_as_unresolved(monkeypatch) -> None:
    def _stub_prompt_builder(
        sample,
        *,
        step_index,
        include_student_attempt_for_teacher,
        max_reprompt_len,
    ):
        _ = (sample, step_index, include_student_attempt_for_teacher, max_reprompt_len)
        return "prompt", False, {"feedback_packet": {}, "prompt_truncated": False}

    monkeypatch.setattr(
        "verl_integration.reprompt_adapter._build_prompt_for_sample",
        _stub_prompt_builder,
    )

    samples = [{"resolved": "false"}]

    batch = build_self_distillation_batch(samples)

    assert batch["self_distillation_mask"] == [False]
