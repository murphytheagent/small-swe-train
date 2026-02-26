from __future__ import annotations

import json
from pathlib import Path

from config import (
    OnPolicyDataConfig,
    OnPolicyDatasetColumns,
    OnPolicyRuntimeConfig,
    OnPolicySettings,
    resolve_rft_handoff_settings,
)
from rollout.onpolicy_collector import OnPolicyRolloutCollector
from verl_integration.onpolicy_rollout_adapter import (
    build_verl_sft_batch,
    collect_rft_sft_batch_for_steps,
    collect_rollouts_for_steps,
    evaluate_rft_rejection_reason,
    merge_rollout_and_preprocessed_rows,
    select_rft_attempt_rows,
)


class _CharTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens

        if isinstance(text, list):
            input_ids_batch = []
            offsets_batch = []
            for item in text:
                encoded = self(item, return_offsets_mapping=return_offsets_mapping)
                input_ids_batch.append(encoded["input_ids"])
                offsets_batch.append(encoded["offset_mapping"])
            return {"input_ids": input_ids_batch, "offset_mapping": offsets_batch}

        input_ids = [index + 1 for index, _char in enumerate(text)]
        offsets = [(index, index + 1) for index, _char in enumerate(text)]
        return {"input_ids": input_ids, "offset_mapping": offsets}


def test_collect_rollouts_for_steps_writes_jsonl_artifacts(tmp_path: Path) -> None:
    settings = OnPolicySettings(
        data=OnPolicyDataConfig(
            dataset_id="dummy/local",
            dataset_split="train",
            columns=OnPolicyDatasetColumns(
                image_name="image_name",
                problem_statement="problem_statement",
                fail_to_pass="FAIL_TO_PASS",
                pass_to_pass="PASS_TO_PASS",
            ),
        ),
        runtime=OnPolicyRuntimeConfig(
            enabled=True,
            rollout_only=True,
            task_batch_size=1,
            attempts_per_task=1,
            max_turns_per_attempt=1,
            env_pool_size=1,
            tool_timeout_sec=1,
            container_start_timeout_sec=1,
            attempt_timeout_sec=10,
            max_tool_calls_per_turn=3,
        ),
    )

    class _FakePool:
        def acquire(self, tasks):
            from env.container_pool import ContainerHandle

            return (
                ContainerHandle(
                    task_id=tasks[0].task_id,
                    image_name=tasks[0].image_name,
                    container_id="cid-1",
                    container_name="cname-1",
                ),
            )

        def release_all(self) -> None:
            return None

    class _FakeExecutor:
        def run(self, request):
            from env.runtime_protocol import ToolResponse

            return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=lambda **_kwargs: '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix bug",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
        ],
        pool_factory=lambda _runtime: _FakePool(),
        executor_factory=lambda _handle, _runtime: _FakeExecutor(),
    )

    rows = collect_rollouts_for_steps(total_steps=2, collector=collector, output_dir=tmp_path)

    assert len(rows) == 2
    assert len(rows[0]) == 1
    assert (tmp_path / "step_00000.jsonl").exists()
    assert (tmp_path / "step_00001.jsonl").exists()
    assert rows[0][0]["image_name"] == "img:1"
    assert "trajectory_steps" in rows[0][0]
    assert rows[0][0]["trajectory_history"]


def test_collect_rollouts_for_steps_applies_start_step_offset() -> None:
    seen_step_indexes: list[int] = []

    class _FakeCollector:
        def collect_step(self, step_index: int):
            seen_step_indexes.append(step_index)
            return [
                {
                    "task_id": f"task-{step_index}",
                    "step_index": step_index,
                    "attempt_index": 0,
                    "turn_index": 0,
                    "resolved": False,
                    "is_terminal": True,
                    "format_valid": True,
                    "prompt": "Fix bug",
                    "assistant_response": "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>",
                    "trajectory_history": [
                        "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"
                    ],
                    "trajectory_steps": [],
                    "trajectory_assistant_turns": [],
                    "trajectory_tool_validation_errors": [],
                    "final_turn_has_submit": True,
                    "final_submit_format_valid": True,
                }
            ]

    rows = collect_rollouts_for_steps(
        total_steps=2,
        start_step_index=5,
        collector=_FakeCollector(),  # type: ignore[arg-type]
    )

    assert seen_step_indexes == [5, 6]
    assert rows[0][0]["task_id"] == "task-5"
    assert rows[1][0]["task_id"] == "task-6"


def test_select_rft_attempt_rows_relabels_deterministically() -> None:
    selection = resolve_rft_handoff_settings().selection
    rows = [
        {
            "is_terminal": True,
            "format_valid": True,
            "resolved": True,
            "container_init_succeeded": True,
            "action_mask_rft": [1, 1],
            "input_ids": [1, 2],
        },
        {
            "is_terminal": False,
            "format_valid": False,
            "resolved": False,
            "timeout_error": "timed out",
            "container_init_succeeded": True,
            "action_mask_rft": [1, 1],
            "input_ids": [1, 2],
        },
    ]
    selected, rejected = select_rft_attempt_rows(
        rows,
        selection_policy=selection,
    )

    assert len(selected) == 1
    assert selected[0]["rft_label"] == "accept"
    assert len(rejected) == 1
    assert rejected[0]["rft_label"] == "reject"
    expected_reason = evaluate_rft_rejection_reason(rows[1], selection_policy=selection)
    assert expected_reason is not None
    assert rejected[0]["rft_rejection_reason"] == expected_reason


def test_build_verl_sft_batch_masks_last_token_and_pads() -> None:
    handoff = resolve_rft_handoff_settings(overrides={"max_sequence_length": 16, "pad_token_id": 99})
    batch = build_verl_sft_batch(
        [
            {
                "task_id": "task-1",
                "attempt_index": 0,
                "step_index": 1,
                "turn_index": 2,
                "resolved": True,
                "is_terminal": True,
                "format_valid": True,
                "input_ids": [11, 12, 13],
                "action_mask_rft": [1, 1, 1],
                "token_labels": ["tool_call", "tool_call", "tool_call"],
            },
            {
                "task_id": "task-2",
                "attempt_index": 0,
                "step_index": 2,
                "turn_index": 0,
                "resolved": True,
                "is_terminal": True,
                "format_valid": True,
                "input_ids": [21, 22],
                "action_mask_rft": [1, 1],
                "token_labels": ["tool_call", "tool_call"],
            },
        ],
        handoff_settings=handoff,
    )

    assert batch["tensors"]["input_ids"] == [[11, 12, 13], [21, 22, 99]]
    assert batch["tensors"]["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["tensors"]["loss_mask"] == [[1, 1, 0], [1, 0, 0]]
    assert batch["grouping_metadata"]["group_id"][0] == "task-1#attempt-0"
    assert batch["meta_info"]["selected_count"] == 2


def test_select_rft_attempt_rows_rejects_invalid_final_submit_even_when_format_flag_true() -> None:
    selection = resolve_rft_handoff_settings().selection
    selected, rejected = select_rft_attempt_rows(
        [
            {
                "is_terminal": True,
                "final_turn_has_submit": True,
                "final_submit_format_valid": False,
                "trajectory_format_valid": True,
                "format_valid": True,
                "resolved": True,
                "container_init_succeeded": True,
                "action_mask_rft": [1, 1],
                "input_ids": [1, 2],
            }
        ],
        selection_policy=selection,
    )

    assert selected == []
    assert len(rejected) == 1
    assert rejected[0]["rft_rejection_reason"] == "final_submit_invalid"


def test_select_rft_attempt_rows_rejects_container_init_failed_even_if_valid() -> None:
    selection = resolve_rft_handoff_settings().selection
    selected, rejected = select_rft_attempt_rows(
        [
            {
                "is_terminal": True,
                "final_turn_has_submit": True,
                "final_submit_format_valid": True,
                "trajectory_format_valid": True,
                "format_valid": True,
                "resolved": True,
                "container_init_succeeded": False,
                "action_mask_rft": [1, 1],
                "input_ids": [1, 2],
            }
        ],
        selection_policy=selection,
    )

    assert selected == []
    assert len(rejected) == 1
    assert rejected[0]["rft_rejection_reason"] == "container_init_failed"


def test_select_rft_attempt_rows_rejects_missing_container_init_status() -> None:
    selection = resolve_rft_handoff_settings().selection
    selected, rejected = select_rft_attempt_rows(
        [
            {
                "is_terminal": True,
                "final_turn_has_submit": True,
                "final_submit_format_valid": True,
                "trajectory_format_valid": True,
                "format_valid": True,
                "resolved": True,
                "action_mask_rft": [1, 1],
                "input_ids": [1, 2],
            }
        ],
        selection_policy=selection,
    )

    assert selected == []
    assert len(rejected) == 1
    assert rejected[0]["rft_rejection_reason"] == "container_init_failed"


def test_collect_rft_sft_batch_for_steps_filters_failed_attempts(tmp_path: Path) -> None:
    settings = OnPolicySettings(
        data=OnPolicyDataConfig(
            dataset_id="dummy/local",
            dataset_split="train",
            columns=OnPolicyDatasetColumns(
                image_name="image_name",
                problem_statement="problem_statement",
                fail_to_pass="FAIL_TO_PASS",
                pass_to_pass="PASS_TO_PASS",
            ),
        ),
        runtime=OnPolicyRuntimeConfig(
            enabled=True,
            rollout_only=True,
            task_batch_size=1,
            attempts_per_task=2,
            max_turns_per_attempt=2,
            env_pool_size=1,
            tool_timeout_sec=1,
            container_start_timeout_sec=1,
            attempt_timeout_sec=10,
            max_tool_calls_per_turn=3,
        ),
    )

    class _FakePool:
        def acquire(self, tasks):
            from env.container_pool import ContainerHandle

            return (
                ContainerHandle(
                    task_id=tasks[0].task_id,
                    image_name=tasks[0].image_name,
                    container_id="cid-1",
                    container_name="cname-1",
                ),
            )

        def release_all(self) -> None:
            return None

    class _FakeExecutor:
        def run(self, request):
            from env.runtime_protocol import ToolResponse

            return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)

    def turn_generator(**kwargs: object) -> str:
        attempt_index = int(kwargs["attempt_index"])
        turn_index = int(kwargs["turn_index"])
        if attempt_index == 0:
            if turn_index == 0:
                return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
            return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix bug",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
        ],
        pool_factory=lambda _runtime: _FakePool(),
        executor_factory=lambda _handle, _runtime: _FakeExecutor(),
    )

    result = collect_rft_sft_batch_for_steps(
        total_steps=1,
        collector=collector,
        tokenizer=_CharTokenizer(),
        output_dir=tmp_path,
    )

    assert len(result["rollout_rows"]) == 2
    assert len(result["selected_rows"]) == 1
    assert len(result["rejected_rows"]) == 1
    assert result["rejected_rows"][0]["rft_rejection_reason"]
    assert result["sft_batch"]["meta_info"]["selected_count"] == 1
    assert result["dataproto_payload"]["meta_info"]["selected_count"] == 1
    assert (tmp_path / "rollout_rows.jsonl").exists()
    assert (tmp_path / "rollout_artifact_summary.json").exists()
    assert (tmp_path / "selected_rows.jsonl").exists()
    assert (tmp_path / "rejected_rows.jsonl").exists()

    summary = json.loads((tmp_path / "rollout_artifact_summary.json").read_text(encoding="utf-8"))
    assert summary["rollout_row_count"] == 2
    assert summary["unique_task_ids"] == ["task-1"]
    assert summary["unique_image_names"] == ["img:1"]
    assert summary["rows_with_trajectory_steps"] >= 1


def test_collect_rft_sft_batch_for_steps_all_rejected_returns_empty_selected_batch(
    tmp_path: Path,
) -> None:
    settings = OnPolicySettings(
        data=OnPolicyDataConfig(
            dataset_id="dummy/local",
            dataset_split="train",
            columns=OnPolicyDatasetColumns(
                image_name="image_name",
                problem_statement="problem_statement",
                fail_to_pass="FAIL_TO_PASS",
                pass_to_pass="PASS_TO_PASS",
            ),
        ),
        runtime=OnPolicyRuntimeConfig(
            enabled=True,
            rollout_only=True,
            task_batch_size=1,
            attempts_per_task=2,
            max_turns_per_attempt=1,
            env_pool_size=1,
            tool_timeout_sec=1,
            container_start_timeout_sec=1,
            attempt_timeout_sec=10,
            max_tool_calls_per_turn=3,
        ),
    )

    class _FakePool:
        def acquire(self, tasks):
            from env.container_pool import ContainerHandle

            return (
                ContainerHandle(
                    task_id=tasks[0].task_id,
                    image_name=tasks[0].image_name,
                    container_id="cid-1",
                    container_name="cname-1",
                ),
            )

        def release_all(self) -> None:
            return None

    class _FakeExecutor:
        def run(self, request):
            from env.runtime_protocol import ToolResponse

            return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)

    def turn_generator(**_kwargs: object) -> str:
        return '<tool_call>{"tool":"submit","args":{}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix bug",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
        ],
        pool_factory=lambda _runtime: _FakePool(),
        executor_factory=lambda _handle, _runtime: _FakeExecutor(),
    )

    result = collect_rft_sft_batch_for_steps(
        total_steps=1,
        collector=collector,
        tokenizer=_CharTokenizer(),
        output_dir=tmp_path,
    )

    assert len(result["selected_rows"]) == 0
    assert len(result["rejected_rows"]) == 2
    assert result["sft_batch"]["meta_info"]["selected_count"] == 0
    assert result["sft_batch"]["meta_info"]["max_turn_level_generated_tokens"] == 0
    assert result["dataproto_payload"]["meta_info"]["selected_count"] == 0
    assert (tmp_path / "rejected_rows.jsonl").exists()
    meta = json.loads((tmp_path / "rft_sft_meta.json").read_text(encoding="utf-8"))
    assert meta["selected_count"] == 0
    assert meta["rejected_count"] == 2


def test_merge_rollout_and_preprocessed_rows_propagates_verifier_targets() -> None:
    merged = merge_rollout_and_preprocessed_rows(
        rollout_rows=[
            {
                "task_id": "task-1",
                "attempt_index": 0,
                "turn_index": 0,
                "step_index": 3,
                "resolved": False,
                "is_terminal": True,
                "format_valid": True,
                "final_turn_has_submit": True,
                "final_submit_format_valid": True,
                "fail_to_pass": ["tests/test_bug.py::test_bugfix"],
                "pass_to_pass": ["tests/test_ok.py::test_regression"],
            }
        ],
        preprocessed_rows=[{"input_ids": [1, 2], "action_mask_rft": [1, 1]}],
    )

    assert len(merged) == 1
    assert merged[0]["fail_to_pass"] == ["tests/test_bug.py::test_bugfix"]
    assert merged[0]["pass_to_pass"] == ["tests/test_ok.py::test_regression"]


def test_merge_rollout_and_preprocessed_rows_requires_non_empty_task_id() -> None:
    try:
        merge_rollout_and_preprocessed_rows(
            rollout_rows=[
                {
                    "task_id": "",
                }
            ],
            preprocessed_rows=[{}],
        )
    except ValueError as exc:
        assert "rows[0].task_id must be a non-empty string" in str(exc)
        return

    raise AssertionError("Expected merge_rollout_and_preprocessed_rows to fail on empty task_id")
