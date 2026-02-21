from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.trajectory_ingestion import (
    build_episode_from_record,
    build_training_record,
    load_raw_records,
    run_ingestion,
    write_training_records,
)


class CharTokenizer:
    """Simple tokenizer for tests: one token per character with offsets."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del add_special_tokens
        token_count = len(text)
        payload: dict[str, Any] = {"input_ids": list(range(token_count))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(token_count)]
        return payload


def test_build_episode_from_swe_smith_record() -> None:
    record = {
        "instance_id": "swe-smith-001",
        "trajectory": [
            {
                "tool": "str_replace_editor",
                "args": {"command": "view", "path": "src/app.py"},
                "thinking": "Inspect the file first.",
                "output": {
                    "stdout": "src/app.py:12: AssertionError",
                    "stderr": "FAILED tests/test_app.py::test_x",
                    "exit_code": 1,
                },
            },
            {
                "tool": "answer",
                "args": {"answer": "Fixed and verified."},
                "output": {"stdout": "submitted", "exit_code": 0},
            },
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert episode.episode_id == "swe-smith-001"
    assert episode.source_format == "swe-smith"
    assert len(episode.environment_steps) == 2
    assert episode.environment_steps[0].request.tool == "search"
    assert episode.environment_steps[0].thinking == "Inspect the file first."
    assert episode.environment_steps[1].request.tool == "submit"
    assert episode.feedback_packets[0].tool == "search"
    assert episode.feedback_packets[1].tool == "submit"


def test_build_episode_from_swe_bench_history_record() -> None:
    record = {
        "instance_id": "swe-bench-001",
        "history": [
            {
                "role": "assistant",
                "thinking": "Run the failing test first.",
                "tool_call": {
                    "name": "bash",
                    "arguments": {"command": "pytest tests/test_app.py::test_x"},
                },
            },
            {
                "role": "tool",
                "content": "tests/test_app.py:12: AssertionError\nFAILED tests/test_app.py::test_x",
            },
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert episode.source_format == "swe-bench"
    assert len(episode.environment_steps) == 1
    assert episode.environment_steps[0].request.tool == "bash"
    assert episode.environment_steps[0].thinking == "Run the failing test first."
    assert episode.feedback_packets[0].canonical_feedback.actionable_error_text is not None
    assert "tests/test_app.py::test_x" in episode.feedback_packets[0].canonical_feedback.artifact_identities


def test_history_uses_tool_message_over_assistant_empty_content() -> None:
    record = {
        "instance_id": "swe-bench-empty-content",
        "history": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "bash",
                        "arguments": {"command": "pytest tests/test_app.py::test_x"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "real tool output",
            },
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert len(episode.environment_steps) == 1
    assert episode.environment_steps[0].response.stdout == "real tool output"


def test_history_uses_per_call_inline_outputs() -> None:
    record = {
        "instance_id": "swe-bench-inline-per-call",
        "history": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "bash",
                        "arguments": {"command": "echo first"},
                        "output": {"stdout": "first output", "exit_code": 0},
                    },
                    {
                        "name": "bash",
                        "arguments": {"command": "echo second"},
                        "output": {"stdout": "second output", "exit_code": 0},
                    },
                ],
            }
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert len(episode.environment_steps) == 2
    assert episode.environment_steps[0].response.stdout == "first output"
    assert episode.environment_steps[1].response.stdout == "second output"


def test_history_keeps_pending_calls_across_assistant_non_call_turns() -> None:
    record = {
        "instance_id": "swe-bench-pending-retained",
        "history": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "bash",
                        "arguments": {"command": "echo one"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Waiting for tool output.",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "one output",
            },
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert len(episode.environment_steps) == 1
    assert episode.environment_steps[0].request.args["command"] == "echo one"
    assert episode.environment_steps[0].response.stdout == "one output"


def test_history_matches_tool_outputs_by_tool_call_id() -> None:
    record = {
        "instance_id": "swe-bench-call-id-match",
        "history": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "bash",
                        "arguments": {"command": "echo first"},
                    },
                    {
                        "id": "call-2",
                        "name": "bash",
                        "arguments": {"command": "echo second"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": "second output",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "first output",
            },
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert len(episode.environment_steps) == 2
    assert episode.environment_steps[0].request.args["command"] == "echo second"
    assert episode.environment_steps[0].response.stdout == "second output"
    assert episode.environment_steps[1].request.args["command"] == "echo first"
    assert episode.environment_steps[1].response.stdout == "first output"


def test_step_prefers_non_empty_trajectory_variant() -> None:
    record = {
        "instance_id": "trajectory-fallback",
        "trajectory": [],
        "steps": [
            {
                "tool": "bash",
                "args": {"command": "pytest -q"},
                "output": {"stdout": "ok", "exit_code": 0},
            }
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert len(episode.environment_steps) == 1
    assert episode.environment_steps[0].request.tool == "bash"
    assert episode.environment_steps[0].response.stdout == "ok"


def test_step_reads_tool_call_local_outputs() -> None:
    record = {
        "instance_id": "tool-call-output",
        "trajectory": [
            {
                "tool_calls": [
                    {
                        "name": "bash",
                        "arguments": {"command": "echo first"},
                        "output": {"stdout": "first out", "exit_code": 0},
                    },
                    {
                        "name": "bash",
                        "arguments": {"command": "echo second"},
                        "output": {"stdout": "second out", "exit_code": 0},
                    },
                ]
            }
        ],
    }

    episode = build_episode_from_record(record, fallback_index=0)

    assert len(episode.environment_steps) == 2
    assert episode.environment_steps[0].response.stdout == "first out"
    assert episode.environment_steps[1].response.stdout == "second out"


def test_build_training_record_applies_stage_masks() -> None:
    record = {
        "instance_id": "mask-001",
        "trajectory": [
            {
                "tool": "str_replace_editor",
                "args": {"command": "insert", "path": "src/app.py", "new_str": "print('ok')"},
                "thinking": "Patch quickly and keep scope tight.",
                "output": {"stdout": "patched", "exit_code": 0},
            }
        ],
    }
    episode = build_episode_from_record(record, fallback_index=0)
    prepared = build_training_record(episode, tokenizer=CharTokenizer())

    input_ids = prepared["input_ids"]
    labels = prepared["token_labels"]
    rft_mask = prepared["action_mask_rft"]
    sdpo_mask = prepared["action_mask_step_sdpo"]

    assert len(input_ids) == len(labels) == len(rft_mask) == len(sdpo_mask)
    assert "tool_call" in labels
    assert "think" in labels

    for label, rft, sdpo in zip(labels, rft_mask, sdpo_mask):
        if label == "tool_call":
            assert rft is True
            assert sdpo is True
        if label == "think":
            assert rft is False
            assert sdpo is True
        if label == "other":
            assert rft is False
            assert sdpo is False


def test_load_raw_records_from_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        json.dumps({"instance_id": "a", "trajectory": []}) + "\n" + json.dumps({"instance_id": "b", "history": []}),
        encoding="utf-8",
    )

    loaded = load_raw_records(input_path)
    assert [record["instance_id"] for record in loaded] == ["a", "b"]


def test_write_training_records_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "prepared.jsonl"
    records = [{"episode_id": "ep-1", "input_ids": [1, 2], "token_labels": ["other", "tool_call"]}]
    write_training_records(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    decoded = json.loads(lines[0])
    assert decoded["episode_id"] == "ep-1"


def test_write_training_records_parquet(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")

    output_path = tmp_path / "prepared.parquet"
    records = [
        {
            "episode_id": "ep-1",
            "input_ids": [1, 2, 3],
            "token_labels": ["other", "tool_call", "other"],
            "action_mask_rft": [False, True, False],
            "action_mask_step_sdpo": [False, True, False],
            "environment_steps": [],
            "feedback_packets": [],
            "sequence_text": "",
            "source_format": "test",
            "num_steps": 0,
        }
    ]
    write_training_records(records, output_path)

    import pyarrow.parquet as pq

    table = pq.read_table(output_path)
    assert table.num_rows == 1
    assert table.column("episode_id")[0].as_py() == "ep-1"


def test_run_ingestion_respects_max_episodes_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "ok-1",
                        "trajectory": [
                            {
                                "tool": "bash",
                                "args": {"command": "echo ok"},
                                "output": {"stdout": "ok", "exit_code": 0},
                            }
                        ],
                    }
                ),
                json.dumps({"instance_id": "bad-2", "trajectory": [123]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "prepared.jsonl"

    monkeypatch.setattr("data.trajectory_ingestion.load_qwen_tokenizer", lambda _model: CharTokenizer())

    stats = run_ingestion(
        input_path=input_path,
        output_path=output_path,
        tokenizer_model="ignored-for-test",
        max_episodes=1,
    )

    assert stats == {"raw_records": 2, "episodes_ingested": 1, "records_written": 1}
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_run_ingestion_zero_max_episodes_skips_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "instance_id": "ok-1",
                "trajectory": [
                    {
                        "tool": "bash",
                        "args": {"command": "echo ok"},
                        "output": {"stdout": "ok", "exit_code": 0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "prepared.jsonl"

    def _should_not_load(_: str) -> CharTokenizer:
        raise AssertionError("tokenizer should not be loaded when no episodes are selected")

    monkeypatch.setattr("data.trajectory_ingestion.load_qwen_tokenizer", _should_not_load)

    stats = run_ingestion(
        input_path=input_path,
        output_path=output_path,
        tokenizer_model="ignored-for-test",
        max_episodes=0,
    )

    assert stats == {"raw_records": 1, "episodes_ingested": 0, "records_written": 0}
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""


def test_run_ingestion_rejects_negative_max_episodes(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "instance_id": "ok-1",
                "trajectory": [
                    {
                        "tool": "bash",
                        "args": {"command": "echo ok"},
                        "output": {"stdout": "ok", "exit_code": 0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match=r"max_episodes must be >= 0"):
        run_ingestion(
            input_path=input_path,
            output_path=output_path,
            tokenizer_model="ignored-for-test",
            max_episodes=-1,
        )
