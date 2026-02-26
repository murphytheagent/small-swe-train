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
    assert "[INITIAL_PROMPT_BLOCK]" in prompt
    assert "[FEEDBACK_BLOCK]" in prompt
    assert "Teacher objective (turn-level SDPO):" in prompt
    assert "Assistant output contract:" in prompt
    assert batch["self_distillation_mask"] == [False]


def test_build_self_distillation_batch_allows_output_contract_override() -> None:
    samples = [
        {
            "prompt": "Fix failing test",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"pytest -q\"}}</tool_call>",
            "tool_output": {
                "stdout": "FAILED tests/test_math.py::test_add - AssertionError",
                "stderr": "",
                "exit_code": 1,
            },
            "output_contract_block": "CUSTOM CONTRACT BLOCK",
        }
    ]

    batch = build_self_distillation_batch(samples)
    prompt = batch["teacher_prompts"][0]
    assert "[OUTPUT_CONTRACT_BLOCK]\nCUSTOM CONTRACT BLOCK" in prompt


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
    def _stub_legacy_prompt_builder(
        sample,
        *,
        step_index,
        include_student_attempt_for_teacher,
        max_reprompt_len,
    ):
        _ = (sample, step_index, include_student_attempt_for_teacher, max_reprompt_len)
        return "prompt", False, {"feedback_packet": {}, "prompt_truncated": False}

    monkeypatch.setattr(
        "verl_integration.reprompt_adapter._build_legacy_prompt_for_sample",
        _stub_legacy_prompt_builder,
    )

    samples = [{"resolved": "false"}]
    batch = build_self_distillation_batch(samples)

    assert batch["self_distillation_mask"] == [False]


def test_build_self_distillation_batch_emits_turn_level_pairs() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 0, 1, 1, 0, 1, 1],
            "trajectory_assistant_turns": [
                "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"x\"}}</tool_call>",
                "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"pytest -q\"}}</tool_call>",
                "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
            ],
            "trajectory_assistant_turn_token_lengths": [2, 2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>search output</tool_response>"],
                ["<tool_response>pytest failed</tool_response>"],
                [],
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, num_recent_raw_blocks=3)

    assert batch["self_distillation_mask"] == [True]
    assert len(batch["turn_teacher_prompts"][0]) == 2
    assert batch["turn_distillation_mask"][0] == [True, True]
    assert batch["turn_response_masks"][0][0] == [0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert batch["turn_response_masks"][0][1] == [0, 0, 0, 0, 0, 0, 0, 1, 1]

    second_turn_prompt = batch["turn_teacher_prompts"][0][1]
    assert "[RECENT_RAW_BLOCK]" in second_turn_prompt
    assert "[TURN_0]" in second_turn_prompt
    assert "[CURRENT_ATTEMPT_BLOCK]" in second_turn_prompt
    assert "[TURN_1]" in second_turn_prompt


def test_build_self_distillation_batch_recent_raw_window_handles_short_histories() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 1, 0, 1, 0, 1, 0],
            "trajectory_assistant_turns": [
                "turn-0 assistant",
                "turn-1 assistant",
                "turn-2 assistant",
                "turn-3 assistant",
            ],
            "trajectory_assistant_turn_token_lengths": [1, 1, 1, 1],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>r0</tool_response>"],
                ["<tool_response>r1</tool_response>"],
                ["<tool_response>r2</tool_response>"],
                [],
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, num_recent_raw_blocks=99)

    # Prompt at current_turn=2 should include all available previous blocks
    # (turn-0, turn-1) because history is shorter than the requested window.
    prompt_current_turn_2 = batch["turn_teacher_prompts"][0][2]
    assert "[TURN_0]" in prompt_current_turn_2
    assert "[TURN_1]" in prompt_current_turn_2
    assert "[TURN_2]" in prompt_current_turn_2
