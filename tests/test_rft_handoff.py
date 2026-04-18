from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import resolve_rft_handoff_settings
from trainer.rft_handoff import (
    build_rft_handoff_result_from_rollout_rows,
    build_verl_sft_batch,
    collect_rft_sft_batch_for_steps,
    merge_rollout_and_preprocessed_rows,
)


class _EmptyCollector:
    def collect_step(self, step_index: int):
        assert step_index == 0
        return []


def test_collect_rft_sft_batch_for_steps_allows_empty_eval_collection(tmp_path: Path) -> None:
    result = collect_rft_sft_batch_for_steps(
        total_steps=1,
        collector=_EmptyCollector(),
        tokenizer=object(),
        output_dir=tmp_path,
    )

    assert result["rollout_rows"] == []
    assert result["selected_rows"] == []
    assert result["rejected_rows"] == []
    assert result["sft_batch"]["meta_info"]["selected_count"] == 0
    assert result["dataproto_payload"]["meta_info"]["selected_count"] == 0

    summary = json.loads((tmp_path / "rollout_artifact_summary.json").read_text(encoding="utf-8"))
    assert summary["rollout_row_count"] == 0
    assert summary["selected_count"] == 0
    assert summary["rejected_count"] == 0


def test_merge_rollout_and_preprocessed_rows_preserves_verifier_metadata() -> None:
    merged = merge_rollout_and_preprocessed_rows(
        rollout_rows=[
            {
                "task_id": "task-1",
                "attempt_index": 0,
                "turn_index": 1,
                "step_index": 2,
                "resolved": False,
                "verifier_kind": "go_test",
                "fail_to_pass": ["TestBug"],
                "pass_to_pass": ["TestRegression"],
                "task_family": "func_basic",
                "difficulty_band": "learnable",
                "difficulty_band_source": "instance_id_family:exact",
                "infra_invalid": True,
                "invalid_reason": "verifier_crash",
                "hit_generation_cap": True,
                "image_name": "img:1",
                "trajectory_format_valid": True,
                "final_turn_has_submit": True,
                "final_submit_format_valid": True,
            }
        ],
        preprocessed_rows=[
            {
                "input_ids": [1, 2, 3],
                "action_mask_rft": [1, 1, 1],
                "token_labels": ["a", "b", "c"],
                "format_valid": True,
            }
        ],
    )

    assert merged[0]["verifier_kind"] == "go_test"
    assert merged[0]["task_family"] == "func_basic"
    assert merged[0]["difficulty_band"] == "learnable"
    assert merged[0]["difficulty_band_source"] == "instance_id_family:exact"
    assert merged[0]["infra_invalid"] is True
    assert merged[0]["invalid_reason"] == "verifier_crash"
    assert merged[0]["hit_generation_cap"] is True


def test_build_verl_sft_batch_carries_difficulty_tags_in_grouping_metadata() -> None:
    batch = build_verl_sft_batch(
        [
            {
                "task_id": "task-1",
                "task_family": "func_basic",
                "difficulty_band": "learnable",
                "difficulty_band_source": "instance_id_family:exact",
                "attempt_index": 0,
                "step_index": 0,
                "turn_index": 0,
                "input_ids": [1, 2, 3],
                "action_mask_rft": [1, 1, 1],
                "token_labels": ["a", "b", "c"],
            }
        ],
        handoff_settings=resolve_rft_handoff_settings(),
    )

    assert batch["grouping_metadata"]["task_family"] == ["func_basic"]
    assert batch["grouping_metadata"]["difficulty_band"] == ["learnable"]
    assert batch["grouping_metadata"]["difficulty_band_source"] == [
        "instance_id_family:exact"
    ]


def test_build_verl_sft_batch_rejects_rows_above_handoff_length() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        build_verl_sft_batch(
            [
                {
                    "task_id": "task-1",
                    "attempt_index": 0,
                    "step_index": 0,
                    "turn_index": 0,
                    "input_ids": [1, 2, 3, 4],
                    "action_mask_rft": [1, 1, 1, 1],
                    "token_labels": ["a", "b", "c", "d"],
                }
            ],
            handoff_settings=resolve_rft_handoff_settings(overrides={"max_sequence_length": 3}),
        )


def test_collect_rft_sft_batch_for_steps_rejects_overlength_selected_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Collector:
        settings = SimpleNamespace(
            runtime=SimpleNamespace(max_tool_calls_per_turn=3),
        )

        def collect_step(self, step_index: int):
            assert step_index == 0
            return [
                {
                    "task_id": "task-1",
                    "image_name": "img:1",
                    "trajectory_format_valid": True,
                    "final_turn_has_submit": True,
                    "final_submit_format_valid": True,
                }
            ]

    monkeypatch.setattr(
        "trainer.rft_handoff.preprocess_trajectories",
        lambda rollout_rows, **kwargs: [
            {
                "input_ids": [1, 2, 3, 4],
                "action_mask_rft": [1, 1, 1, 1],
                "token_labels": ["a", "b", "c", "d"],
                "format_valid": True,
            }
        ],
    )
    monkeypatch.setattr(
        "trainer.rft_handoff.select_rft_attempt_rows",
        lambda rows, selection_policy: (list(rows), []),
    )

    result = collect_rft_sft_batch_for_steps(
        total_steps=1,
        collector=_Collector(),
        tokenizer=object(),
        handoff_overrides={"max_sequence_length": 3},
        output_dir=tmp_path,
    )

    assert result["selected_rows"] == []
    assert len(result["rejected_rows"]) == 1
    assert result["rejected_rows"][0]["rft_rejection_reason"] == "selected_over_handoff_length"
    assert result["rejected_rows"][0]["selected_over_budget"] is True
    assert result["sft_batch"]["meta_info"]["selected_count"] == 0


def test_build_rft_handoff_result_rejects_selected_rows_with_invalid_preprocessed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trainer.rft_handoff.preprocess_trajectories",
        lambda rollout_rows, **kwargs: [
            {
                "input_ids": [1, 2, 3],
                "action_mask_rft": [1, 1, 1],
                "token_labels": ["a", "b", "c"],
                "format_valid": True,
            },
            {
                "input_ids": None,
                "action_mask_rft": [1, 1],
                "token_labels": ["a", "b"],
                "format_valid": False,
                "parse_error": "Invalid tool_call JSON",
            },
        ],
    )
    monkeypatch.setattr(
        "trainer.rft_handoff.select_rft_attempt_rows",
        lambda rows, selection_policy: (list(rows), []),
    )

    result = build_rft_handoff_result_from_rollout_rows(
        rollout_rows=[
            {"task_id": "task-valid", "resolved": True},
            {"task_id": "task-invalid", "resolved": True},
        ],
        max_tool_calls=3,
        tokenizer=object(),
        handoff_overrides=None,
    )

    assert [row["task_id"] for row in result["selected_rows"]] == ["task-valid"]
    assert len(result["rejected_rows"]) == 1
    assert result["rejected_rows"][0]["task_id"] == "task-invalid"
    assert (
        result["rejected_rows"][0]["rft_rejection_reason"]
        == "selected_invalid_preprocessed_payload"
    )
    assert (
        result["rejected_rows"][0]["selected_payload_error"]
        == "rows[1].input_ids must be a sequence of ints."
    )
