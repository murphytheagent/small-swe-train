from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


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


def test_discover_latest_rft_manifest_prefers_newest_across_slurm_and_default_roots(tmp_path: Path) -> None:
    pilot = _load_pilot_module()
    local_manifest = tmp_path / "outputs" / "rft_runtime" / "run-local" / "rft_runtime_loop_manifest.json"
    slurm_manifest = tmp_path / "outputs" / "slurm" / "rft_runtime" / "run-slurm" / "rft_runtime_loop_manifest.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    slurm_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text("{}", encoding="utf-8")
    slurm_manifest.write_text("{}", encoding="utf-8")

    os.utime(local_manifest, (1_700_000_000, 1_700_000_000))
    os.utime(slurm_manifest, (1_800_000_000, 1_800_000_000))

    resolved = pilot._discover_latest_rft_manifest(project_root=tmp_path)
    assert resolved == slurm_manifest


def test_resolve_checkpoint_from_manifest_prefers_existing_candidates(tmp_path: Path) -> None:
    pilot = _load_pilot_module()
    checkpoint_root = tmp_path / "trainer_checkpoints" / "global_step_42"
    existing_hf = checkpoint_root / "huggingface"
    existing_hf.mkdir(parents=True)
    missing_merged = checkpoint_root / "huggingface_vllm_merged"
    manifest_path = tmp_path / "rft_runtime_loop_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "final_model_path": str(missing_merged),
                "latest_vllm_checkpoint": str(missing_merged),
                "latest_hf_checkpoint": str(existing_hf),
            }
        ),
        encoding="utf-8",
    )

    resolved = pilot._resolve_checkpoint_from_manifest(manifest_path=manifest_path)
    assert resolved == str(existing_hf)


def test_main_print_resolved_rft_checkpoint_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pilot = _load_pilot_module()
    checkpoint_path = tmp_path / "resolved-rft-checkpoint"
    checkpoint_path.mkdir(parents=True)
    manifest_path = tmp_path / "rft_runtime_loop_manifest.json"
    manifest_path.write_text(json.dumps({"final_model_path": str(checkpoint_path)}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_teacher_reprompt_pilot.py",
            "--print-resolved-rft-checkpoint",
            "--rft-manifest",
            str(manifest_path),
        ],
    )

    pilot.main()
    captured = capsys.readouterr()
    assert captured.out.strip() == str(checkpoint_path)


def test_resolve_teacher_reprompt_turn_index_supports_dynamic_middle() -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("turn-0", "turn-1", "turn-2", "turn-3"),
        turn_tool_response_blocks=((), (), (), ()),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )
    empty_trace = pilot.BaselineTrace(
        task_id="task-2",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=(),
        turn_tool_response_blocks=(),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )

    assert (
        pilot._resolve_teacher_reprompt_turn_index(
            trace=trace,
            teacher_reprompt_turn_index=1,
            teacher_reprompt_turn_index_mode="fixed",
        )
        == 1
    )
    assert (
        pilot._resolve_teacher_reprompt_turn_index(
            trace=trace,
            teacher_reprompt_turn_index=-1,
            teacher_reprompt_turn_index_mode="dynamic_middle",
        )
        == 1
    )
    assert (
        pilot._resolve_teacher_reprompt_turn_index(
            trace=empty_trace,
            teacher_reprompt_turn_index=-1,
            teacher_reprompt_turn_index_mode="dynamic_middle",
        )
        == 0
    )


def test_teacher_generator_injects_reprompt_once_then_uses_fallback(monkeypatch) -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(
            {"role": "system", "content": "system-contract"},
            {"role": "user", "content": "initial-user"},
        ),
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
        return {"choices": [{"message": {"content": "teacher-turn-2"}}]}

    def _fake_extract_assistant_content(payload):
        _ = payload
        return "teacher-turn-2"

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
        teacher_reprompt_turn_index_mode="fixed",
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
        history=[
            "turn-0",
            '<tool_response>{"exit_code":1}</tool_response>',
            "turn-1",
            '<tool_response>{"exit_code":0}</tool_response>',
        ],
    )
    turn_3 = generator(
        task=task,
        attempt_index=0,
        turn_index=3,
        step_index=0,
        history=[
            "turn-0",
            '<tool_response>{"exit_code":1}</tool_response>',
            "turn-1",
            '<tool_response>{"exit_code":0}</tool_response>',
            "teacher-turn-2",
        ],
    )

    assert turn_0 == "turn-0"
    assert turn_1 == "turn-1"
    assert turn_2 == "teacher-turn-2"
    assert turn_3 == "fallback-3"
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["messages"] == [
        {"role": "system", "content": "system-contract"},
        {"role": "user", "content": "prompt-turn-1"},
    ]


def test_teacher_generator_falls_back_when_teacher_completion_fails(monkeypatch) -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("turn-0", "turn-1"),
        turn_tool_response_blocks=((), ()),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )

    def _fallback_turn_generator(*, task, attempt_index, turn_index, step_index, history):
        _ = task, attempt_index, step_index, history
        return f"fallback-{turn_index}"

    def _fake_build_self_distillation_batch(*args, **kwargs):
        _ = args, kwargs
        return {"turn_teacher_prompts": [["prompt-turn-0", "prompt-turn-1", "prompt-turn-2"]]}

    def _raise_post_error(*, base_url, payload, timeout_sec):
        _ = base_url, payload, timeout_sec
        raise RuntimeError("temporary timeout")

    monkeypatch.setattr(pilot, "build_self_distillation_batch", _fake_build_self_distillation_batch)
    monkeypatch.setattr(pilot, "_post_chat_completion", _raise_post_error)

    generator = pilot._build_teacher_turn_generator(
        baseline_trace_map={("task-1", 0): trace},
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
        teacher_reprompt_turn_index_mode="fixed",
        max_reprompt_len=1024,
        num_recent_raw_blocks=3,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )
    task = SimpleNamespace(task_id="task-1")

    turn_2 = generator(
        task=task,
        attempt_index=0,
        turn_index=2,
        step_index=0,
        history=[
            "turn-0",
            '<tool_response>{"exit_code":1}</tool_response>',
            "turn-1",
            '<tool_response>{"exit_code":0}</tool_response>',
        ],
    )
    assert turn_2 == "fallback-2"


def test_teacher_generator_rejects_next_turn_mode() -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("turn-0", "turn-1", "turn-2"),
        turn_tool_response_blocks=((), (), ()),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )
    with pytest.raises(ValueError, match="current_turn"):
        pilot._build_teacher_turn_generator(
            baseline_trace_map={("task-1", 0): trace},
            fallback_turn_generator=lambda **_: "fallback",
            vllm_config=pilot.VLLMTurnGeneratorConfig(
                base_url="http://127.0.0.1:8000/v1",
                model_name="local-model",
                request_timeout_sec=10,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                system_prompt="pilot-system",
            ),
            teacher_reprompt_turn_index=1,
            teacher_reprompt_turn_index_mode="fixed",
            max_reprompt_len=1024,
            num_recent_raw_blocks=3,
            turn_supervision_mode="next_turn",
            verifier_feedback_mode="all_turns",
        )


def test_teacher_generator_dynamic_middle_mode_injects_at_middle_turn(monkeypatch) -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("turn-0", "turn-1", "turn-2", "turn-3"),
        turn_tool_response_blocks=((), (), (), ()),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )
    posted_payloads: list[dict[str, object]] = []

    def _fallback_turn_generator(*, task, attempt_index, turn_index, step_index, history):
        _ = task, attempt_index, step_index, history
        return f"fallback-{turn_index}"

    def _fake_build_self_distillation_batch(*args, **kwargs):
        _ = args, kwargs
        return {"turn_teacher_prompts": [["prompt-turn-0", "prompt-turn-1", "prompt-turn-2", "prompt-turn-3"]]}

    def _fake_post_chat_completion(*, base_url, payload, timeout_sec):
        _ = base_url, timeout_sec
        posted_payloads.append(dict(payload))
        return {"choices": [{"message": {"content": "teacher-turn-2"}}]}

    monkeypatch.setattr(pilot, "build_self_distillation_batch", _fake_build_self_distillation_batch)
    monkeypatch.setattr(pilot, "_post_chat_completion", _fake_post_chat_completion)
    monkeypatch.setattr(pilot, "_extract_assistant_content", lambda payload: payload["choices"][0]["message"]["content"])

    generator = pilot._build_teacher_turn_generator(
        baseline_trace_map={("task-1", 0): trace},
        fallback_turn_generator=_fallback_turn_generator,
        vllm_config=pilot.VLLMTurnGeneratorConfig(
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-model",
            request_timeout_sec=10,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            system_prompt="pilot-system",
        ),
        teacher_reprompt_turn_index=-1,
        teacher_reprompt_turn_index_mode="dynamic_middle",
        max_reprompt_len=1024,
        num_recent_raw_blocks=3,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )
    task = SimpleNamespace(task_id="task-1")
    turn_0 = generator(task=task, attempt_index=0, turn_index=0, step_index=0, history=[])
    turn_1 = generator(task=task, attempt_index=0, turn_index=1, step_index=0, history=["turn-0"])
    turn_2 = generator(task=task, attempt_index=0, turn_index=2, step_index=0, history=["turn-0", "turn-1"])
    turn_3 = generator(task=task, attempt_index=0, turn_index=3, step_index=0, history=["turn-0", "turn-1", "teacher-turn-2"])

    assert turn_0 == "turn-0"
    assert turn_1 == "turn-1"
    assert turn_2 == "teacher-turn-2"
    assert turn_3 == "fallback-3"
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["messages"][-1] == {"role": "user", "content": "prompt-turn-1"}


def test_teacher_generator_builds_reprompt_from_runtime_history_tool_blocks(monkeypatch) -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("turn-0", "turn-1", "turn-2"),
        turn_tool_response_blocks=(
            ('<tool_response>{"stdout":"BASELINE_ONLY"}</tool_response>',),
            (),
            (),
        ),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )
    captured: dict[str, Any] = {}

    def _fake_build_self_distillation_batch(samples, **kwargs):
        _ = kwargs
        captured["sample"] = samples[0]
        return {"turn_teacher_prompts": [["prompt-turn-0", "prompt-turn-1", "prompt-turn-2"]]}

    monkeypatch.setattr(pilot, "build_self_distillation_batch", _fake_build_self_distillation_batch)
    monkeypatch.setattr(
        pilot,
        "_post_chat_completion",
        lambda *, base_url, payload, timeout_sec: {"choices": [{"message": {"content": "teacher-turn-2"}}]},
    )
    monkeypatch.setattr(
        pilot,
        "_extract_assistant_content",
        lambda payload: payload["choices"][0]["message"]["content"],
    )

    generator = pilot._build_teacher_turn_generator(
        baseline_trace_map={("task-1", 0): trace},
        fallback_turn_generator=lambda **_: "fallback",
        vllm_config=pilot.VLLMTurnGeneratorConfig(
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-model",
            request_timeout_sec=10,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            system_prompt="pilot-system",
        ),
        teacher_reprompt_turn_index=1,
        teacher_reprompt_turn_index_mode="fixed",
        max_reprompt_len=1024,
        num_recent_raw_blocks=3,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )
    task = SimpleNamespace(task_id="task-1")
    _ = generator(
        task=task,
        attempt_index=0,
        turn_index=2,
        step_index=0,
        history=[
            "turn-0",
            '<tool_response>{"stdout":"RUNTIME_ONLY"}</tool_response>',
            "turn-1",
            '<tool_response>{"stdout":"RUNTIME_TURN_1"}</tool_response>',
        ],
    )

    sample = captured["sample"]
    assert sample["trajectory_assistant_turns"] == ["turn-0", "turn-1"]
    assert sample["trajectory_turn_tool_response_blocks"] == [
        ['<tool_response>{"stdout":"RUNTIME_ONLY"}</tool_response>'],
        ['<tool_response>{"stdout":"RUNTIME_TURN_1"}</tool_response>'],
    ]


def test_teacher_generator_short_trace_defaults_to_first_turn_replay() -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("only-turn",),
        turn_tool_response_blocks=((),),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )

    generator = pilot._build_teacher_turn_generator(
        baseline_trace_map={("task-1", 0): trace},
        fallback_turn_generator=lambda **kwargs: f"fallback-{kwargs['turn_index']}",
        vllm_config=pilot.VLLMTurnGeneratorConfig(
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-model",
            request_timeout_sec=10,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            system_prompt="pilot-system",
        ),
        teacher_reprompt_turn_index=-1,
        teacher_reprompt_turn_index_mode="dynamic_middle",
        max_reprompt_len=1024,
        num_recent_raw_blocks=3,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )
    task = SimpleNamespace(task_id="task-1")
    assert generator(task=task, attempt_index=0, turn_index=0, step_index=0, history=[]) == "only-turn"


def test_teacher_generator_out_of_range_turn_index_defaults_to_first_turn_replay() -> None:
    pilot = _load_pilot_module()
    trace = pilot.BaselineTrace(
        task_id="task-1",
        attempt_index=0,
        problem_statement="Fix issue",
        raw_prompt_messages=(),
        assistant_turns=("turn-0", "turn-1"),
        turn_tool_response_blocks=((), ()),
        verification_feedback="",
        verification_error="",
        resolved=False,
    )

    generator = pilot._build_teacher_turn_generator(
        baseline_trace_map={("task-1", 0): trace},
        fallback_turn_generator=lambda **kwargs: f"fallback-{kwargs['turn_index']}",
        vllm_config=pilot.VLLMTurnGeneratorConfig(
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-model",
            request_timeout_sec=10,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            system_prompt="pilot-system",
        ),
        teacher_reprompt_turn_index=99,
        teacher_reprompt_turn_index_mode="fixed",
        max_reprompt_len=1024,
        num_recent_raw_blocks=3,
        turn_supervision_mode="current_turn",
        verifier_feedback_mode="all_turns",
    )
    task = SimpleNamespace(task_id="task-1")
    assert generator(task=task, attempt_index=0, turn_index=0, step_index=0, history=[]) == "turn-0"
