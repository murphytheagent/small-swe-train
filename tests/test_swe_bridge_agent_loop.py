from __future__ import annotations

import asyncio
from pathlib import Path
import time

import pytest

from prompts import build_sdpo_rollout_followup_user_message
from verl_integration.swe_bridge_agent_loop import (
    BridgeLoopTaskContext,
    _apply_verification_metadata,
    _acquire_container_slot,
    _await_with_attempt_timeout,
    _cleanup_container_pool_after_acquire,
    _remaining_attempt_timeout_sec,
    _extract_final_submit_text,
    _initial_verification_extra_fields,
    _clip_prompt_for_rollout_context,
    _validate_rollout_context_alignment,
    _build_task_sample,
    _fallback_response_token_id,
    _get_container_slot_gate,
    append_response_tokens,
    build_agent_loop_messages,
    build_bridge_task_context,
    build_tool_response_messages,
    resolve_bridge_loop_runtime_config,
)

yaml = pytest.importorskip("yaml")


def _load_yaml_runtime_defaults() -> dict[str, int]:
    config_path = Path(__file__).resolve().parents[1] / "configs/verl/agent_loops/swe_bridge_agent.yaml"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, list) and parsed and isinstance(parsed[0], dict)
    config = parsed[0]
    return {
        "env_pool_size": int(config["env_pool_size"]),
        "tool_timeout_sec": int(config["tool_timeout_sec"]),
        "container_start_timeout_sec": int(config["container_start_timeout_sec"]),
        "cleanup_timeout_sec": int(config["cleanup_timeout_sec"]),
        "attempt_timeout_sec": int(config["attempt_timeout_sec"]),
        "max_tool_calls_per_turn": int(config["max_tool_calls_per_turn"]),
        "verifier_timeout_sec": int(config["verifier_timeout_sec"]),
    }


def test_build_bridge_task_context_requires_task_metadata() -> None:
    with pytest.raises(ValueError, match="task_id"):
        build_bridge_task_context(
            {
                "image_name": "sweb.eval.x86_64",
                "raw_prompt": [{"role": "user", "content": "Fix test."}],
            }
        )


def test_build_bridge_task_context_extracts_prompt_and_patch() -> None:
    context = build_bridge_task_context(
        {
            "task_id": "task-17",
            "image_name": "sweb.eval.x86_64",
            "patch": "diff --git ...",
            "raw_prompt": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "Fix failing tests."},
            ],
        }
    )
    assert context.task_id == "task-17"
    assert context.image_name == "sweb.eval.x86_64"
    assert context.prompt_text == "Fix failing tests."
    assert context.patch == "diff --git ..."


def test_build_task_sample_resolves_verifier_targets_from_reward_ground_truth() -> None:
    sample = _build_task_sample(
        task_context=BridgeLoopTaskContext(
            task_id="task-17",
            image_name="sweb.eval.x86_64",
            prompt_text="Fix tests.",
            patch=None,
        ),
        raw_kwargs={
            "reward_model": {
                "ground_truth": {
                    "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                    "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
                }
            }
        },
    )

    assert sample.fail_to_pass == ["tests/test_bug.py::test_bugfix"]
    assert sample.pass_to_pass == ["tests/test_ok.py::test_regression"]


def test_build_task_sample_prefers_explicit_targets_over_reward_ground_truth() -> None:
    sample = _build_task_sample(
        task_context=BridgeLoopTaskContext(
            task_id="task-17",
            image_name="sweb.eval.x86_64",
            prompt_text="Fix tests.",
            patch=None,
        ),
        raw_kwargs={
            "fail_to_pass": ["tests/test_bug.py::test_explicit_bugfix"],
            "pass_to_pass": ["tests/test_ok.py::test_explicit_regression"],
            "reward_model": {
                "ground_truth": {
                    "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
                    "pass_to_pass": ["tests/test_ok.py::test_regression"],
                }
            },
        },
    )

    assert sample.fail_to_pass == ["tests/test_bug.py::test_explicit_bugfix"]
    assert sample.pass_to_pass == ["tests/test_ok.py::test_explicit_regression"]


def test_verification_extra_fields_schema_stable_without_metadata() -> None:
    fields = _initial_verification_extra_fields(
        fail_to_pass=["tests/test_bug.py::test_bugfix"],
        pass_to_pass=["tests/test_ok.py::test_regression"],
    )

    assert fields["fail_to_pass"] == ["tests/test_bug.py::test_bugfix"]
    assert fields["pass_to_pass"] == ["tests/test_ok.py::test_regression"]
    assert fields["verification_feedback"] == ""
    assert fields["fail_to_pass_results"] == {}
    assert fields["pass_to_pass_results"] == {}
    assert fields["fail_to_pass_all_passed"] is None
    assert fields["pass_to_pass_all_passed"] is None
    assert fields["verification_missing"] is None
    assert fields["resolved"] is False


def test_apply_verification_metadata_preserves_schema_and_overrides_values() -> None:
    fields = _initial_verification_extra_fields(
        fail_to_pass=["tests/test_bug.py::test_bugfix"],
        pass_to_pass=["tests/test_ok.py::test_regression"],
    )
    baseline_keys = set(fields.keys())

    _apply_verification_metadata(
        fields,
        {
            "fail_to_pass_results": {"tests/test_bug.py::test_bugfix": True},
            "pass_to_pass_results": {"tests/test_ok.py::test_regression": True},
            "verification_feedback": "all tests passed",
            "fail_to_pass_all_passed": True,
            "pass_to_pass_all_passed": True,
            "fail_to_pass_verified": True,
            "pass_to_pass_verified": True,
            "verification_missing": False,
            "verification_error": "",
            "submission_final_response": "done",
            "resolved": True,
        },
    )

    assert set(fields.keys()) == baseline_keys
    assert fields["fail_to_pass_results"] == {"tests/test_bug.py::test_bugfix": True}
    assert fields["pass_to_pass_results"] == {"tests/test_ok.py::test_regression": True}
    assert fields["verification_feedback"] == "all tests passed"
    assert fields["verification_missing"] is False
    assert fields["resolved"] is True


def test_extract_final_submit_text_uses_terminal_submit_tool_call_when_steps_empty() -> None:
    submit_call = type(
        "_ToolCall",
        (),
        {"tool": "submit", "args": {"final_response": "patched successfully"}},
    )()

    text = _extract_final_submit_text([submit_call], [])

    assert text == "patched successfully"


def test_build_tool_response_messages_keeps_non_empty_blocks() -> None:
    messages = build_tool_response_messages(
        [
            "  ",
            "<tool_response>{\"stdout\":\"ok\"}</tool_response>",
        ]
    )
    assert messages == [{"role": "user", "content": "<tool_response>{\"stdout\":\"ok\"}</tool_response>"}]


def test_build_agent_loop_messages_adds_system_prompt_and_trailing_user_nudge() -> None:
    messages = build_agent_loop_messages(
        {
            "raw_prompt": [
                {"role": "user", "content": "Investigate the failing test."},
                {"role": "assistant", "content": "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"test\"}}</tool_call>"},
            ]
        }
    )

    assert messages[0]["role"] == "system"
    assert "Assistant output contract" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == build_sdpo_rollout_followup_user_message()


def test_build_agent_loop_messages_prepends_contract_to_existing_system_message() -> None:
    messages = build_agent_loop_messages(
        {
            "raw_prompt": [
                {"role": "system", "content": "Existing system guidance."},
                {"role": "user", "content": "Fix bug."},
            ]
        }
    )

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].endswith("Existing system guidance.")
    assert "Assistant output contract" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Fix bug."


def test_append_response_tokens_preserves_generated_vs_tool_masks() -> None:
    full_token_ids: list[int] = [100, 101]
    response_mask: list[int] = []
    response_logprobs: list[float] = []

    reached_limit = append_response_tokens(
        full_token_ids=full_token_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        token_ids=[11, 12],
        generated=True,
        max_response_length=4,
        token_logprobs=[-0.2, -0.3],
    )
    assert not reached_limit

    reached_limit = append_response_tokens(
        full_token_ids=full_token_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        token_ids=[21, 22],
        generated=False,
        max_response_length=4,
    )
    assert reached_limit
    assert full_token_ids == [100, 101, 11, 12, 21, 22]
    assert response_mask == [1, 1, 0, 0]
    assert response_logprobs == [-0.2, -0.3, 0.0, 0.0]


def test_append_response_tokens_clips_to_available_budget() -> None:
    full_token_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []

    reached_limit = append_response_tokens(
        full_token_ids=full_token_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        token_ids=[1, 2, 3],
        generated=True,
        max_response_length=2,
        token_logprobs=[-1.0, -2.0, -3.0],
    )

    assert reached_limit
    assert full_token_ids == [1, 2]
    assert response_mask == [1, 1]
    assert response_logprobs == [-1.0, -2.0]


def test_clip_prompt_for_rollout_context_applies_left_truncation() -> None:
    clipped = _clip_prompt_for_rollout_context([10, 11, 12, 13], prompt_length=2)
    assert clipped == [12, 13]


def test_fallback_response_token_id_prefers_eos() -> None:
    class _Tokenizer:
        eos_token_id = 42
        pad_token_id = 7

    assert _fallback_response_token_id(_Tokenizer()) == 42


def test_fallback_response_token_id_falls_back_to_zero() -> None:
    class _Tokenizer:
        eos_token_id = None
        pad_token_id = None
        bos_token_id = None
        unk_token_id = None

    assert _fallback_response_token_id(_Tokenizer()) == 0


def test_validate_rollout_context_alignment_accepts_consistent_state() -> None:
    _validate_rollout_context_alignment(
        canonical_prompt_ids=[101, 102],
        full_token_ids=[101, 102, 201, 202],
        response_mask=[1, 0],
        response_logprobs=[-0.1, 0.0],
    )


def test_validate_rollout_context_alignment_rejects_prefix_divergence() -> None:
    with pytest.raises(RuntimeError, match="prompt prefix diverged"):
        _validate_rollout_context_alignment(
            canonical_prompt_ids=[101, 102],
            full_token_ids=[101, 999, 201],
            response_mask=[1],
            response_logprobs=[-0.1],
        )


def test_validate_rollout_context_alignment_rejects_response_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="response length does not match response_mask"):
        _validate_rollout_context_alignment(
            canonical_prompt_ids=[101, 102],
            full_token_ids=[101, 102, 201],
            response_mask=[],
            response_logprobs=None,
        )


def test_resolve_bridge_loop_runtime_config_uses_yaml_aligned_defaults() -> None:
    yaml_defaults = _load_yaml_runtime_defaults()
    resolved = resolve_bridge_loop_runtime_config()

    assert resolved.env_pool_size == yaml_defaults["env_pool_size"]
    assert resolved.tool_timeout_sec == yaml_defaults["tool_timeout_sec"]
    assert resolved.container_start_timeout_sec == yaml_defaults["container_start_timeout_sec"]
    assert resolved.cleanup_timeout_sec == yaml_defaults["cleanup_timeout_sec"]
    assert resolved.attempt_timeout_sec == yaml_defaults["attempt_timeout_sec"]
    assert resolved.max_tool_calls_per_turn == yaml_defaults["max_tool_calls_per_turn"]
    assert resolved.verifier_timeout_sec == yaml_defaults["verifier_timeout_sec"]


def test_resolve_bridge_loop_runtime_config_honors_explicit_overrides() -> None:
    resolved = resolve_bridge_loop_runtime_config(
        env_pool_size=8,
        tool_timeout_sec=17,
        container_start_timeout_sec=44,
        cleanup_timeout_sec=13,
        attempt_timeout_sec=91,
        max_tool_calls_per_turn=2,
        verifier_timeout_sec=444,
    )

    assert resolved.env_pool_size == 8
    assert resolved.tool_timeout_sec == 17
    assert resolved.container_start_timeout_sec == 44
    assert resolved.cleanup_timeout_sec == 13
    assert resolved.attempt_timeout_sec == 91
    assert resolved.max_tool_calls_per_turn == 2
    assert resolved.verifier_timeout_sec == 444


def test_container_slot_gate_reuses_instances_per_pool_size() -> None:
    gate_a = _get_container_slot_gate(3)
    gate_b = _get_container_slot_gate(3)
    gate_c = _get_container_slot_gate(4)

    assert gate_a is gate_b
    assert gate_a is not gate_c


def test_container_slot_gate_enforces_capacity() -> None:
    gate = _get_container_slot_gate(1)
    acquired = gate.acquire(blocking=False)
    assert acquired is True
    try:
        assert gate.acquire(blocking=False) is False
    finally:
        gate.release()


def test_remaining_attempt_timeout_sec_accounts_for_elapsed_time() -> None:
    started_at = time.monotonic() - 1.5
    remaining = _remaining_attempt_timeout_sec(started_at=started_at, attempt_timeout_sec=5)
    assert 3.0 <= remaining <= 4.5


def test_await_with_attempt_timeout_includes_stage_in_error_message() -> None:
    async def _exercise() -> None:
        with pytest.raises(TimeoutError, match="stage 'acquire_container'"):
            await _await_with_attempt_timeout(
                asyncio.sleep(0),
                task_id="task-17",
                stage="acquire_container",
                started_at=time.monotonic() - 2.0,
                attempt_timeout_sec=1,
            )

    asyncio.run(_exercise())


def test_acquire_container_slot_times_out_with_stage_context() -> None:
    gate = _get_container_slot_gate(1)
    assert gate.acquire(blocking=False) is True

    async def _exercise() -> None:
        with pytest.raises(TimeoutError, match="stage 'wait_container_slot'"):
            await _acquire_container_slot(
                gate,
                task_id="task-17",
                stage="wait_container_slot",
                started_at=time.monotonic() - 2.0,
                attempt_timeout_sec=1,
            )

    try:
        asyncio.run(_exercise())
    finally:
        gate.release()


def test_cleanup_container_pool_waits_for_timed_out_acquire_task() -> None:
    call_order: list[str] = []

    class _FakePool:
        def acquire(self, _tasks: object) -> tuple[object, ...]:
            call_order.append("acquire_start")
            time.sleep(0.1)
            call_order.append("acquire_done")
            return ()

        def release_all(self) -> None:
            call_order.append("release_all")

    async def _exercise() -> None:
        pool = _FakePool()
        acquire_task = asyncio.create_task(asyncio.to_thread(pool.acquire, object()))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(acquire_task), timeout=0.01)
        await _cleanup_container_pool_after_acquire(
            pool=pool,
            acquire_task=acquire_task,
            task_id="task-17",
            stage="cleanup",
            cleanup_timeout_sec=1,
        )

    asyncio.run(_exercise())
    assert call_order.index("acquire_done") < call_order.index("release_all")
