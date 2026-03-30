from __future__ import annotations

import json
from pathlib import Path

import pytest

import trainer.rft_runtime as rft_runtime_module
from trainer.rft_runtime import (
    OnPolicyRFTRuntimeRequest,
    collect_onpolicy_rft_runtime_batch,
)


def test_collect_onpolicy_rft_runtime_batch_writes_runtime_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_collector = object()
    captured_turn_generator = {"fn": None}
    captured_collector_kwargs: dict[str, object] = {}

    def fake_build_onpolicy_collector(**kwargs):
        captured_turn_generator["fn"] = kwargs.get("turn_generator")
        captured_collector_kwargs.update(kwargs)
        return fake_collector

    def fake_collect_rft_sft_batch_for_steps(
        *,
        total_steps,
        start_step_index,
        collector,
        tokenizer,
        handoff_overrides,
        output_dir,
    ):
        assert total_steps == 2
        assert start_step_index == 3
        assert collector is fake_collector
        assert tokenizer == "tokenizer"
        assert handoff_overrides == {"require_resolved": True}
        assert output_dir == str(tmp_path)
        return {
            "rollout_rows": [{"task_id": "task-1"}, {"task_id": "task-2"}],
            "selected_rows": [{"task_id": "task-1"}],
            "rejected_rows": [{"task_id": "task-2", "rft_rejection_reason": "non_terminal,unresolved"}],
            "sft_batch": {"meta_info": {"selected_count": 1}},
            "dataproto_payload": {"meta_info": {"max_turn_level_generated_tokens": 16}},
        }

    monkeypatch.setattr(
        rft_runtime_module,
        "build_onpolicy_collector",
        fake_build_onpolicy_collector,
    )
    monkeypatch.setattr(
        rft_runtime_module,
        "collect_rft_sft_batch_for_steps",
        fake_collect_rft_sft_batch_for_steps,
    )

    request = OnPolicyRFTRuntimeRequest(
        data_config_name="on_policy_swe_smith",
        turn_generator_mode="proof_tool_chain",
        total_steps=2,
        start_step_index=3,
        runtime_overrides={"task_batch_size": 2},
        data_overrides={"dataset_split": "train"},
        handoff_overrides={"require_resolved": True},
        output_dir=str(tmp_path),
        task_partition="eval",
        task_eval_split_fraction=0.25,
        task_eval_min_rows=2,
        verify_submissions=True,
        stage_name="positive_rft",
    )
    result = collect_onpolicy_rft_runtime_batch(
        request=request,
        tokenizer="tokenizer",
    )

    assert len(result["selected_rows"]) == 1
    assert captured_collector_kwargs["task_partition"] == "eval"
    assert captured_collector_kwargs["task_eval_split_fraction"] == 0.25
    assert captured_collector_kwargs["task_eval_min_rows"] == 2
    assert captured_collector_kwargs["runtime_overrides"] == {"task_batch_size": 2, "verify_submissions": True}
    assert captured_collector_kwargs["stage_name"] == "positive_rft"
    turn_generator = captured_turn_generator["fn"]
    assert callable(turn_generator)
    submit_turn = turn_generator(
        task=None,
        attempt_index=0,
        turn_index=8,
        step_index=0,
        history=(),
    )
    assert '"tool":"submit"' in submit_turn

    manifest_path = tmp_path / "rft_runtime_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["total_steps"] == 2
    assert manifest_payload["start_step_index"] == 3
    assert manifest_payload["task_partition"] == "eval"
    assert manifest_payload["task_eval_split_fraction"] == 0.25
    assert manifest_payload["task_eval_min_rows"] == 2
    assert manifest_payload["verify_submissions"] is True
    assert manifest_payload["stage_name"] == "positive_rft"
    assert manifest_payload["rollout_count"] == 2
    assert manifest_payload["selected_count"] == 1
    assert manifest_payload["rejected_count"] == 1
    assert manifest_payload["rejection_reason_counts"] == {"non_terminal": 1, "unresolved": 1}
    assert manifest_payload["dataproto_meta_info"] == {"max_turn_level_generated_tokens": 16}


def test_collect_onpolicy_rft_runtime_batch_rejects_unknown_turn_generator_mode() -> None:
    request = OnPolicyRFTRuntimeRequest(turn_generator_mode="unsupported_mode")
    with pytest.raises(ValueError, match="turn_generator_mode"):
        collect_onpolicy_rft_runtime_batch(
            request=request,
            tokenizer="tokenizer",
        )


def test_collect_onpolicy_rft_runtime_batch_default_mode_uses_vllm_turn_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_collector = object()
    sentinel_turn_generator = object()
    captured_turn_generator = {"value": None}

    def fake_build_vllm_turn_generator():
        return sentinel_turn_generator

    def fake_build_onpolicy_collector(**kwargs):
        captured_turn_generator["value"] = kwargs.get("turn_generator")
        return fake_collector

    def fake_collect_rft_sft_batch_for_steps(
        *,
        total_steps,
        start_step_index,
        collector,
        tokenizer,
        handoff_overrides,
        output_dir,
    ):
        assert total_steps == 1
        assert start_step_index == 0
        assert collector is fake_collector
        assert tokenizer == "tokenizer"
        assert handoff_overrides is None
        assert output_dir is None
        return {
            "rollout_rows": [{"task_id": "task-1"}],
            "selected_rows": [{"task_id": "task-1"}],
            "rejected_rows": [],
            "sft_batch": {"meta_info": {"selected_count": 1}},
            "dataproto_payload": {"meta_info": {"max_turn_level_generated_tokens": 16}},
        }

    monkeypatch.setattr(
        rft_runtime_module,
        "build_vllm_turn_generator",
        fake_build_vllm_turn_generator,
    )
    monkeypatch.setattr(
        rft_runtime_module,
        "build_onpolicy_collector",
        fake_build_onpolicy_collector,
    )
    monkeypatch.setattr(
        rft_runtime_module,
        "collect_rft_sft_batch_for_steps",
        fake_collect_rft_sft_batch_for_steps,
    )

    request = OnPolicyRFTRuntimeRequest(turn_generator_mode="default")
    collect_onpolicy_rft_runtime_batch(request=request, tokenizer="tokenizer")

    assert captured_turn_generator["value"] is sentinel_turn_generator


def test_collect_onpolicy_rft_runtime_batch_rejects_negative_start_step_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rft_runtime_module,
        "build_onpolicy_collector",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        rft_runtime_module,
        "collect_rft_sft_batch_for_steps",
        lambda **_kwargs: {},
    )
    request = OnPolicyRFTRuntimeRequest(start_step_index=-1)
    with pytest.raises(ValueError, match="start_step_index"):
        collect_onpolicy_rft_runtime_batch(request=request, tokenizer="tokenizer")
