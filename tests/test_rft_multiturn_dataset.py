from __future__ import annotations

from pathlib import Path

import pytest

import trainer.rft_multiturn_dataset as dataset_module
from trainer.rft_multiturn_dataset import (
    build_multiturn_dataset_records,
    build_multiturn_messages,
    write_selected_rows_to_multiturn_parquet,
)


def test_build_multiturn_messages_uses_history_and_maps_tool_responses_to_user() -> None:
    row = {
        "prompt": "Fix the failing test.",
        "trajectory_history": [
            '<tool_call>{"tool":"bash","args":{"command":"pytest -q"}}</tool_call>',
            "<tool_response>{\"stdout\":\"1 failed\",\"stderr\":\"\"}</tool_response>",
            '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
        ],
    }

    messages = build_multiturn_messages(row, row_index=0)

    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "Fix the failing test."
    assert "1 failed" in messages[2]["content"]


def test_build_multiturn_messages_falls_back_to_assistant_response() -> None:
    row = {
        "prompt": "Write a patch.",
        "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"patched"}}</tool_call>',
    }

    messages = build_multiturn_messages(row, row_index=0)

    assert messages == [
        {"role": "user", "content": "Write a patch."},
        {
            "role": "assistant",
            "content": '<tool_call>{"tool":"submit","args":{"final_response":"patched"}}</tool_call>',
        },
    ]


def test_build_multiturn_messages_requires_assistant_turn() -> None:
    row = {
        "prompt": "Only observations exist.",
        "trajectory_history": [
            "<tool_response>{\"stdout\":\"no assistant turn\",\"stderr\":\"\"}</tool_response>",
        ],
    }

    with pytest.raises(ValueError, match="no assistant turns"):
        build_multiturn_messages(row, row_index=3)


def test_build_multiturn_dataset_records_keeps_metadata() -> None:
    rows = [
        {
            "prompt": "Task prompt",
            "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
            "task_id": "task-1",
            "image_name": "img:task-1",
            "attempt_index": 7,
            "step_index": 4,
            "turn_index": 2,
            "resolved": True,
            "format_valid": True,
            "final_turn_has_submit": True,
            "final_submit_format_valid": True,
        }
    ]

    records = build_multiturn_dataset_records(rows)

    assert len(records) == 1
    assert records[0]["task_id"] == "task-1"
    assert records[0]["image_name"] == "img:task-1"
    assert records[0]["attempt_index"] == 7
    assert records[0]["resolved"] is True
    assert records[0]["data_source"] == "small_swe_phase_d"
    assert records[0]["reward_model"]["ground_truth"]["task_id"] == "task-1"
    assert records[0]["reward_model"]["ground_truth"]["resolved"] is True
    assert records[0]["messages"][0]["role"] == "user"
    assert records[0]["messages"][-1]["role"] == "assistant"
    assert records[0]["prompt"] == [{"role": "user", "content": "Task prompt"}]


def test_build_multiturn_dataset_records_prompt_uses_preceding_context() -> None:
    rows = [
        {
            "prompt": "Task prompt",
            "trajectory_history": [
                '<tool_call>{"tool":"bash","args":{"command":"pytest -q"}}</tool_call>',
                "<tool_response>{\"stdout\":\"1 failed\",\"stderr\":\"\"}</tool_response>",
                '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
            ],
            "task_id": "task-1",
            "image_name": "img:task-1",
            "attempt_index": 0,
            "step_index": 0,
            "turn_index": 2,
        }
    ]

    records = build_multiturn_dataset_records(rows)

    assert [message["role"] for message in records[0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["role"] for message in records[0]["prompt"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert records[0]["prompt"][-1]["content"].startswith("<tool_response>")


def test_build_multiturn_dataset_records_requires_image_name() -> None:
    rows = [
        {
            "prompt": "Task prompt",
            "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
            "task_id": "task-1",
            "attempt_index": 0,
            "step_index": 0,
            "turn_index": 0,
        }
    ]

    with pytest.raises(ValueError, match=r"selected_rows\[0\]\.image_name"):
        build_multiturn_dataset_records(rows)


def test_write_selected_rows_to_multiturn_parquet_delegates_to_internal_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {"records": None, "path": None}

    def fake_writer(records, output_path):
        captured["records"] = list(records)
        captured["path"] = Path(output_path)

    monkeypatch.setattr(dataset_module, "_write_records_to_parquet", fake_writer)

    count = write_selected_rows_to_multiturn_parquet(
        [
            {
                "prompt": "Task prompt",
                "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
                "task_id": "task-1",
                "image_name": "img:task-1",
                "attempt_index": 0,
                "step_index": 0,
                "turn_index": 0,
            }
        ],
        tmp_path / "train.parquet",
    )

    assert count == 1
    assert captured["records"] is not None
    assert captured["records"][0]["messages"][0]["role"] == "user"
    assert captured["path"] == tmp_path / "train.parquet"
