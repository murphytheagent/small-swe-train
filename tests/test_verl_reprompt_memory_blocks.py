from __future__ import annotations

from verl_integration.reprompt_adapter import build_self_distillation_batch


def test_teacher_memory_blocks_appear_in_composed_turn_prompts_when_enabled() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
            "trajectory_steps": [
                {
                    "tool": "read",
                    "args": {"path": "src/found.py"},
                    "stdout": "body",
                    "stderr": "",
                    "exit_code": 0,
                },
                {
                    "tool": "apply_patch",
                    "args": {"path": "src/found.py", "patch": "-old\n+new"},
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                },
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    prompt = batch["turn_teacher_prompts"][0][1]
    assert "Earlier attempt summary:" in prompt
    assert "Known student-discovered paths (raw):" in prompt
    assert "- src/found.py" in prompt
    assert "Key facts to keep in mind:" in prompt
    assert "Successful apply_patch calls through current turn:" in prompt


def test_current_turn_mode_includes_same_turn_successful_patch_memory() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1],
            "trajectory_assistant_turns": ["turn-0"],
            "trajectory_assistant_turn_token_lengths": [2],
            "trajectory_turn_tool_response_blocks": [["r0"]],
            "trajectory_steps": [
                {
                    "tool": "apply_patch",
                    "args": {"path": "src/current.py", "patch": "-broken\n+fixed"},
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                }
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    prompt = batch["turn_teacher_prompts"][0][0]
    assert "Successful apply_patch calls through current turn:" in prompt
    assert "raw_path: src/current.py" in prompt
    assert "-broken\n+fixed" in prompt


def test_current_turn_mode_omits_same_turn_patch_memory_when_student_attempt_context_is_disabled() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
            "trajectory_steps": [
                {
                    "tool": "apply_patch",
                    "args": {"path": "src/previous.py", "patch": "-old0\n+new0"},
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                },
                {
                    "tool": "apply_patch",
                    "args": {"path": "src/current.py", "patch": "-old1\n+new1"},
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                },
            ],
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        include_student_attempt_for_teacher=False,
    )

    prompt = batch["turn_teacher_prompts"][0][1]
    assert "raw_path: src/previous.py" in prompt
    assert "-old0\n+new0" in prompt
    assert "raw_path: src/current.py" not in prompt
    assert "-old1\n+new1" not in prompt


def test_future_known_paths_are_exposed_by_design_without_future_recent_raw_turn_leakage() -> None:
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0, 1, 0, 1, 0],
            "trajectory_assistant_turns": ["turn-0 visible", "turn-1 visible", "turn-2 secret"],
            "trajectory_assistant_turn_token_lengths": [1, 1, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"], ["r2"]],
            "trajectory_steps": [
                {
                    "tool": "read",
                    "args": {"path": "src/current.py"},
                    "stdout": "body",
                    "stderr": "",
                    "exit_code": 0,
                },
                {
                    "tool": "read",
                    "args": {"path": "src/current_turn.py"},
                    "stdout": "body",
                    "stderr": "",
                    "exit_code": 0,
                },
                {
                    "tool": "text_search",
                    "args": {"query": "future"},
                    "stdout": "src/future_only.py:17:match",
                    "stderr": "",
                    "exit_code": 0,
                },
            ],
        }
    ]

    batch = build_self_distillation_batch(samples, turn_supervision_mode="current_turn")

    prompt = batch["turn_teacher_prompts"][0][1]
    assert "src/future_only.py" in prompt
    assert "[TURN_2]" not in prompt
    assert "turn-2 secret" not in prompt


def test_tight_budget_reduces_recent_raw_before_memory_blocks() -> None:
    long_recent_raw = " ".join([*(f"rawtok{i}" for i in range(180)), "RECENT_TAIL_TOKEN"])
    samples = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1, 1],
            "trajectory_assistant_turns": ["turn-0 assistant", "turn-1 assistant"],
            "trajectory_assistant_turn_token_lengths": [2, 2],
            "trajectory_turn_tool_response_blocks": [[long_recent_raw], ["r1"]],
            "trajectory_steps": [
                {
                    "tool": "read",
                    "args": {"path": "src/kept.py"},
                    "stdout": "body",
                    "stderr": "",
                    "exit_code": 0,
                },
                {
                    "tool": "apply_patch",
                    "args": {"path": "src/kept.py", "patch": "-old\n+new"},
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                },
            ],
        }
    ]

    batch = build_self_distillation_batch(
        samples,
        turn_supervision_mode="current_turn",
        max_reprompt_len=380,
    )

    prompt = batch["turn_teacher_prompts"][0][1]
    assert batch["turn_prompt_truncated"][0][1] is True
    assert "Known student-discovered paths (raw):" in prompt
    assert "raw_path: src/kept.py" in prompt
    assert "RECENT_TAIL_TOKEN" not in prompt
