from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


def _load_pilot_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_teacher_reprompt_pilot.py"
    module_name = "teacher_reprompt_pilot"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pilot script module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_derive_turn_tool_response_blocks_groups_by_turn() -> None:
    pilot = _load_pilot_module()
    history = [
        "assistant-turn-0",
        '<tool_response>{"exit_code":1}</tool_response>',
        '<tool_response>{"stdout":"retry"}</tool_response>',
        "assistant-turn-1",
        '<tool_response>{"exit_code":0}</tool_response>',
        "assistant-turn-2",
    ]
    assistant_turns = ["assistant-turn-0", "assistant-turn-1", "assistant-turn-2"]

    tool_blocks = pilot.derive_turn_tool_response_blocks(
        history=history,
        assistant_turns=assistant_turns,
    )

    assert tool_blocks == [
        [
            '<tool_response>{"exit_code":1}</tool_response>',
            '<tool_response>{"stdout":"retry"}</tool_response>',
        ],
        ['<tool_response>{"exit_code":0}</tool_response>'],
        [],
    ]


def test_summarize_pair_rewards_computes_delta_statistics() -> None:
    pilot = _load_pilot_module()
    pairs = [
        {"student_reward": 0.0, "teacher_reward": 1.0, "reward_delta": 1.0},
        {"student_reward": 1.0, "teacher_reward": 1.0, "reward_delta": 0.0},
        {"student_reward": 1.0, "teacher_reward": 0.0, "reward_delta": -1.0},
    ]

    summary = pilot.summarize_pair_rewards(pairs)

    assert summary["pair_count"] == 3
    assert summary["student_mean_reward"] == 2.0 / 3.0
    assert summary["teacher_mean_reward"] == 2.0 / 3.0
    assert summary["mean_reward_delta"] == 0.0
    assert summary["improved_count"] == 1
    assert summary["worsened_count"] == 1
    assert summary["tied_count"] == 1


def test_teacher_generator_injects_reprompt_once_then_uses_fallback(monkeypatch) -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        assistant_turns=("turn-0", "turn-1", "turn-2"),
        turn_tool_response_blocks=((), (), ()),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )
    baseline_trace_map = {("task-1", 0): trace}
    posted_payloads: list[dict[str, object]] = []

    def _fallback_turn_generator(*, task, attempt_index, turn_index, step_index, history):
        _ = task, attempt_index, step_index, history
        return f"fallback-{turn_index}"

    def _fake_build_self_distillation_batch(*args, **kwargs):
        _ = args, kwargs
        return {"turn_teacher_prompts": [["prompt-turn-0", "prompt-turn-1", "prompt-turn-2"]]}

    def _fake_post_chat_completion(*, base_url, payload, timeout_sec):
        _ = base_url, timeout_sec
        posted_payloads.append(dict(payload))
        return {"choices": [{"message": {"content": "teacher-turn-1"}}]}

    def _fake_extract_assistant_content(payload):
        _ = payload
        return "teacher-turn-1"

    monkeypatch.setattr(pilot, "build_self_distillation_batch", _fake_build_self_distillation_batch)
    monkeypatch.setattr(pilot, "_post_chat_completion", _fake_post_chat_completion)
    monkeypatch.setattr(pilot, "_extract_assistant_content", _fake_extract_assistant_content)

    generator = pilot._build_teacher_turn_generator(
        baseline_trace_map=baseline_trace_map,
        fallback_turn_generator=_fallback_turn_generator,
        vllm_config=pilot.VLLMTurnGeneratorConfig(
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-model",
            request_timeout_sec=10,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            system_prompt="ignored",
        ),
        teacher_reprompt_turn_index=1,
        max_reprompt_len=1024,
        num_recent_raw_blocks=3,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )
    task = SimpleNamespace(task_id="task-1")

    turn_0 = generator(task=task, attempt_index=0, turn_index=0, step_index=0, history=[])
    turn_1 = generator(
        task=task,
        attempt_index=0,
        turn_index=1,
        step_index=0,
        history=["turn-0", '<tool_response>{"exit_code":1}</tool_response>'],
    )
    turn_2 = generator(
        task=task,
        attempt_index=0,
        turn_index=2,
        step_index=0,
        history=["turn-0", '<tool_response>{"exit_code":1}</tool_response>', "teacher-turn-1"],
    )

    assert turn_0 == "turn-0"
    assert turn_1 == "teacher-turn-1"
    assert turn_2 == "fallback-2"
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["messages"] == [{"role": "user", "content": "prompt-turn-1"}]
