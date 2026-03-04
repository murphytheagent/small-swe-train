from __future__ import annotations

import pytest

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
    assert "current turn" in prompt.lower()
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


def test_feedback_present_gating_ignores_neutral_exit_code_only_tool_output() -> None:
    samples = [
        {
            "prompt": "Fix the thing",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"true\"}}</tool_call>",
            "tool_output": {"exit_code": 0},
            "resolved": False,
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        legacy_distillation_gating_policy="feedback_present",
    )

    assert batch["self_distillation_mask"] == [False]


def test_feedback_present_gating_uses_nonzero_exit_code_as_signal() -> None:
    samples = [
        {
            "prompt": "Fix the thing",
            "assistant_response": "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"false\"}}</tool_call>",
            "tool_output": {"exit_code": 2},
            "resolved": False,
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        legacy_distillation_gating_policy="feedback_present",
    )

    assert batch["self_distillation_mask"] == [True]


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


def test_turn_prompt_compaction_retains_protected_sections() -> None:
    long_tool_feedback = " ".join(f"tooltok{i}" for i in range(180))
    samples = [
        {
            "prompt": "ORIGINAL_PROBLEM_MUST_KEEP reduce flaky parser failures",
            "_response_mask": [1, 0, 1, 0, 1, 0],
            "trajectory_assistant_turns": [
                "TURN0 context",
                "CURRENT_STUDENT_ATTEMPT_MUST_KEEP patch parser branch and rerun",
                "TURN2 submit",
            ],
            "trajectory_assistant_turn_token_lengths": [1, 1, 1],
            "trajectory_turn_tool_response_blocks": [
                ["raw historical output " + " ".join(f"raw{i}" for i in range(160))],
                [long_tool_feedback],
                [],
            ],
            "verification_feedback": "VERIFIER_RESPONSE_MUST_KEEP parser regression still failing",
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="next_turn",
        verifier_feedback_mode="final_turn_only",
        max_reprompt_len=120,
    )

    prompt = batch["turn_teacher_prompts"][0][1]
    assert "ORIGINAL_PROBLEM_MUST_KEEP" in prompt
    assert "CURRENT_STUDENT_ATTEMPT_MUST_KEEP" in prompt
    assert "[VERIFIER_FEEDBACK]" in prompt
    assert "VERIFIER_RESPONSE_MUST_KEEP" in prompt
    assert "[OUTPUT_CONTRACT_BLOCK]" in prompt
    assert len(prompt.split()) <= 120
    assert batch["turn_prompt_truncated"][0][1] is True


def test_legacy_prompt_compaction_retains_protected_sections() -> None:
    samples = [
        {
            "prompt": "ORIGINAL_TASK_MUST_KEEP fix timeout handling in verifier pipeline",
            "assistant_response": "CURRENT_ATTEMPT_MUST_KEEP inspect failing command path",
            "tool_output": {
                "stdout": " ".join(f"stdout{i}" for i in range(220)),
                "stderr": "",
                "exit_code": 1,
            },
            "verification_feedback": "VERIFIER_FEEDBACK_MUST_KEEP command still exits non-zero",
            "resolved": False,
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        verifier_feedback_mode="all_turns",
        max_reprompt_len=120,
        legacy_distillation_gating_policy="feedback_present",
    )

    prompt = batch["teacher_prompts"][0]
    assert "ORIGINAL_TASK_MUST_KEEP" in prompt
    assert "CURRENT_ATTEMPT_MUST_KEEP" in prompt
    assert "[VERIFIER_FEEDBACK]" in prompt
    assert "VERIFIER_FEEDBACK_MUST_KEEP" in prompt
    assert "[OUTPUT_CONTRACT_BLOCK]" in prompt
    assert len(prompt.split()) <= 120
    assert batch["prompt_truncated"] == [True]


def test_build_self_distillation_batch_treats_false_string_as_unresolved(monkeypatch) -> None:
    def _stub_legacy_prompt_builder(
        sample,
        *,
        step_index,
        supervision_mode,
        include_student_attempt_for_teacher,
        max_reprompt_len,
        verifier_feedback_mode,
    ):
        _ = (
            sample,
            step_index,
            supervision_mode,
            include_student_attempt_for_teacher,
            max_reprompt_len,
            verifier_feedback_mode,
        )
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

    batch = build_self_distillation_batch(
        samples,
        num_recent_raw_blocks=3,
        turn_supervision_mode="next_turn",
    )

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


def test_build_self_distillation_batch_defaults_to_current_turn_supervision() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 0, 1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1", "turn-2"],
            "trajectory_assistant_turn_token_lengths": [2, 2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>a</tool_response>"],
                ["<tool_response>b</tool_response>"],
                ["<tool_response>c</tool_response>"],
            ],
        }
    ]

    batch = build_self_distillation_batch(samples)

    assert len(batch["turn_teacher_prompts"][0]) == 3
    assert batch["turn_distillation_mask"][0] == [True, True, True]
    assert batch["turn_response_masks"][0] == [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
    ]


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


def test_turn_supervision_next_turn_compatibility() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 0, 1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1", "turn-2"],
            "trajectory_assistant_turn_token_lengths": [2, 2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>a</tool_response>"],
                ["<tool_response>b</tool_response>"],
                ["<tool_response>c</tool_response>"],
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="next_turn")

    assert len(batch["turn_teacher_prompts"][0]) == 2
    assert batch["turn_distillation_mask"][0] == [True, True]
    assert batch["turn_response_masks"][0][0] == [0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert batch["turn_response_masks"][0][1] == [0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_turn_supervision_current_turn_exact_masks() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 0, 1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1", "turn-2"],
            "trajectory_assistant_turn_token_lengths": [2, 2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>a</tool_response>"],
                ["<tool_response>b</tool_response>"],
                ["<tool_response>c</tool_response>"],
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    assert len(batch["turn_teacher_prompts"][0]) == 3
    assert batch["turn_distillation_mask"][0] == [True, True, True]
    assert batch["turn_response_masks"][0] == [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
    ]


def test_current_turn_includes_first_and_last_turn_when_spans_exist() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 0, 1],
            "trajectory_assistant_turns": ["first", "middle", "last"],
            "trajectory_assistant_turn_token_lengths": [1, 0, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"], ["r2"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    assert len(batch["turn_teacher_prompts"][0]) == 3
    assert batch["turn_distillation_mask"][0] == [True, False, True]
    assert batch["turn_response_masks"][0][0] == [1, 0, 0, 0]
    assert batch["turn_response_masks"][0][2] == [0, 0, 0, 1]


def test_current_turn_handles_zero_length_and_span_mismatch() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 1, 0],
            "trajectory_assistant_turns": ["t0", "t1", "t2", "t3"],
            "trajectory_assistant_turn_token_lengths": [1, 0, 1, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"], ["r2"], ["r3"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    assert len(batch["turn_teacher_prompts"][0]) == 4
    assert batch["turn_distillation_mask"][0] == [True, False, True, False]
    assert all(len(mask) == 4 for mask in batch["turn_response_masks"][0])
    assert batch["turn_response_masks"][0][1] == [0, 0, 0, 0]
    assert batch["turn_response_masks"][0][3] == [0, 0, 0, 0]


def test_current_turn_falls_back_to_contiguous_spans_when_lengths_non_informative() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [0, 0],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    assert batch["turn_distillation_mask"][0] == [True, True]
    assert batch["turn_response_masks"][0] == [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ]


def test_current_turn_falls_back_when_turn_lengths_leave_generated_tokens_uncovered() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [1, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    assert batch["turn_distillation_mask"][0] == [True, True]
    assert batch["turn_response_masks"][0] == [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ]


def test_current_turn_non_contiguous_token_selection_disables_turn_supervision() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 1, 0],
            "trajectory_assistant_turns": ["turn-0"],
            "trajectory_assistant_turn_token_lengths": [2],
            "trajectory_turn_tool_response_blocks": [["<tool_response>r0</tool_response>"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    assert batch["turn_distillation_mask"][0] == [False]
    assert batch["turn_response_masks"][0][0] == [0, 0, 0, 0]


def test_invalid_turn_supervision_mode_raises() -> None:
    with pytest.raises(ValueError, match="turn_supervision_mode"):
        build_self_distillation_batch([], turn_supervision_mode="bad_mode")


def test_invalid_verifier_feedback_mode_raises() -> None:
    with pytest.raises(ValueError, match="verifier_feedback_mode"):
        build_self_distillation_batch([], verifier_feedback_mode="bad_mode")


def test_invalid_legacy_gating_policy_raises() -> None:
    with pytest.raises(ValueError, match="legacy_distillation_gating_policy"):
        build_self_distillation_batch([], legacy_distillation_gating_policy="bad_policy")


def test_current_turn_mode_omits_target_turn_attempt_text_when_enabled() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": [
                "TURN0_SECRET_STUDENT_ATTEMPT",
                "TURN1_SECRET_STUDENT_ATTEMPT",
            ],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>r0</tool_response>"],
                ["<tool_response>r1</tool_response>"],
            ],
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        include_student_attempt_for_teacher=True,
    )

    prompt_turn_0 = batch["turn_teacher_prompts"][0][0]
    prompt_turn_1 = batch["turn_teacher_prompts"][0][1]
    assert "TURN0_SECRET_STUDENT_ATTEMPT" not in prompt_turn_0
    assert "TURN1_SECRET_STUDENT_ATTEMPT" not in prompt_turn_1
    assert "<omitted_current_turn_target_text>" in prompt_turn_0
    assert "<omitted_current_turn_target_text>" in prompt_turn_1
    assert "current-turn reflection" in prompt_turn_0


def test_next_turn_mode_keeps_current_attempt_block_behavior() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": [
                "TURN0_SECRET_STUDENT_ATTEMPT",
                "TURN1_SECRET_STUDENT_ATTEMPT",
            ],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>r0</tool_response>"],
                ["<tool_response>r1</tool_response>"],
            ],
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="next_turn",
        include_student_attempt_for_teacher=True,
    )

    prompt_turn_0 = batch["turn_teacher_prompts"][0][0]
    assert "TURN0_SECRET_STUDENT_ATTEMPT" in prompt_turn_0
    assert "next turn" in prompt_turn_0.lower()


def test_current_turn_mode_omits_current_attempt_when_disabled() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": [
                "TURN0_SECRET_STUDENT_ATTEMPT",
                "TURN1_SECRET_STUDENT_ATTEMPT",
            ],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [
                ["<tool_response>r0</tool_response>"],
                ["<tool_response>r1</tool_response>"],
            ],
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        include_student_attempt_for_teacher=False,
    )

    prompt_turn_0 = batch["turn_teacher_prompts"][0][0]
    prompt_turn_1 = batch["turn_teacher_prompts"][0][1]
    assert "TURN0_SECRET_STUDENT_ATTEMPT" not in prompt_turn_0
    assert "TURN1_SECRET_STUDENT_ATTEMPT" not in prompt_turn_1


def test_verifier_feedback_all_turns_injection() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
            "verification_feedback": "Verifier: tests failed in parser path",
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )

    assert len(batch["turn_teacher_prompts"][0]) == 2
    assert "[VERIFIER_FEEDBACK]" in batch["turn_teacher_prompts"][0][0]
    assert "[VERIFIER_FEEDBACK]" in batch["turn_teacher_prompts"][0][1]


def test_verifier_feedback_final_turn_only_injection() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
            "verification_feedback": "Verifier: only final action should include this",
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="final_turn_only",
    )

    assert "[VERIFIER_FEEDBACK]" not in batch["turn_teacher_prompts"][0][0]
    assert "[VERIFIER_FEEDBACK]" in batch["turn_teacher_prompts"][0][1]


def test_verifier_feedback_final_turn_only_injection_next_turn_mode() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
            "verification_feedback": "Verifier: final distilled prompt should include this",
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="next_turn",
        verifier_feedback_mode="final_turn_only",
    )

    assert len(batch["turn_teacher_prompts"][0]) == 1
    assert "[VERIFIER_FEEDBACK]" in batch["turn_teacher_prompts"][0][0]


def test_submission_final_response_not_leaked_into_prompt() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
            "verification_feedback": "Verifier: still failing",
            "submission_final_response": "LEAK_ME_NEVER",
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )

    joined_prompts = "\n".join(batch["turn_teacher_prompts"][0])
    assert "LEAK_ME_NEVER" not in joined_prompts


def test_legacy_gating_policy_activation_matrix() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "assistant_response": "attempt",
            "tool_output": {"stdout": "pytest failed", "stderr": "", "exit_code": 1},
            "resolved": False,
        }
    ]

    resolved_only = build_self_distillation_batch(
        samples,
        legacy_distillation_gating_policy="resolved_only",
    )
    feedback_present = build_self_distillation_batch(
        samples,
        legacy_distillation_gating_policy="feedback_present",
    )
    always = build_self_distillation_batch(
        samples,
        legacy_distillation_gating_policy="always",
    )

    assert resolved_only["self_distillation_mask"] == [False]
    assert feedback_present["self_distillation_mask"] == [True]
    assert always["self_distillation_mask"] == [True]


def test_legacy_prompt_does_not_inject_status_only_verifier_block() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "assistant_response": "attempt",
            "resolved": True,
            "verification_missing": False,
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        verifier_feedback_mode="all_turns",
        legacy_distillation_gating_policy="feedback_present",
    )

    assert "[VERIFIER_FEEDBACK]" not in batch["teacher_prompts"][0]


def test_feedback_present_gating_respects_verifier_mode_and_payload_presence() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "assistant_response": "attempt",
            "resolved": False,
            "verification_feedback": "Verifier: failing tests",
        }
    ]

    verifier_disabled = build_self_distillation_batch(
        samples,
        verifier_feedback_mode="none",
        legacy_distillation_gating_policy="feedback_present",
    )
    verifier_enabled = build_self_distillation_batch(
        samples,
        verifier_feedback_mode="all_turns",
        legacy_distillation_gating_policy="feedback_present",
    )

    assert verifier_disabled["self_distillation_mask"] == [False]
    assert verifier_enabled["self_distillation_mask"] == [True]
    assert "[VERIFIER_FEEDBACK]" not in verifier_disabled["teacher_prompts"][0]
    assert "[VERIFIER_FEEDBACK]" in verifier_enabled["teacher_prompts"][0]


def test_current_turn_prompts_do_not_include_future_turn_blocks() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 1, 0],
            "trajectory_assistant_turns": [
                "turn-0 payload",
                "turn-1 payload",
                "turn-2 payload",
                "turn-3 payload",
            ],
            "trajectory_assistant_turn_token_lengths": [1, 1, 1, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"], ["r2"], ["r3"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")
    prompts = batch["turn_teacher_prompts"][0]
    assert len(prompts) == 4
    for prompt_index, prompt in enumerate(prompts):
        for future_turn in range(prompt_index + 1, 4):
            assert f"[TURN_{future_turn}]" not in prompt


def test_next_turn_prompts_do_not_include_future_turn_blocks() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 1, 0],
            "trajectory_assistant_turns": [
                "turn-0 payload",
                "turn-1 payload",
                "turn-2 payload",
                "turn-3 payload",
            ],
            "trajectory_assistant_turn_token_lengths": [1, 1, 1, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"], ["r2"], ["r3"]],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="next_turn")
    prompts = batch["turn_teacher_prompts"][0]
    assert len(prompts) == 3
    for prompt_index, prompt in enumerate(prompts):
        for future_turn in range(prompt_index + 1, 4):
            assert f"[TURN_{future_turn}]" not in prompt
