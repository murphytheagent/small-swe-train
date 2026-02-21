from __future__ import annotations

from verl_integration.data_preprocessor import preprocess_trajectories


def test_preprocess_trajectories_from_assistant_response() -> None:
    trajectories = [
        {
            "prompt": "Fix test",
            "assistant_response": (
                "<|im_start|>assistant\n"
                "<think>debug quickly</think>\n"
                "<tool_call>{\"tool\":\"search\",\"args\":{\"query\":\"tests/test_math.py::test_add\"}}</tool_call>\n"
                "<|im_end|>"
            ),
            "tool_output": {"stdout": "Traceback", "stderr": "", "exit_code": 1},
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert len(rows) == 1
    row = rows[0]
    assert row["format_valid"] is True
    assert row["validation_errors"] == []
    assert row["action_mask_rft"]
    assert row["action_mask_step_sdpo"]
    assert row["feedback_packet"] is not None


def test_preprocess_trajectories_adapts_external_calls() -> None:
    trajectories = [
        {
            "prompt": "Submit answer",
            "external_tool_calls": [
                {"tool": "answer", "args": {"answer": "fixed"}},
            ],
            "tool_output": {"stdout": "", "stderr": "", "exit_code": 0},
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["tool_calls"][0]["tool"] == "submit"
    assert rows[0]["format_valid"] is True


def test_preprocess_trajectories_treats_null_assistant_response_as_absent() -> None:
    trajectories = [
        {
            "prompt": "Submit answer",
            "assistant_response": None,
            "external_tool_calls": [
                {"tool": "answer", "args": {"answer": "fixed"}},
            ],
            "tool_output": {"stdout": "", "stderr": "", "exit_code": 0},
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["assistant_response"] == ""
    assert rows[0]["tool_calls"][0]["tool"] == "submit"
    assert rows[0]["format_valid"] is True
    assert rows[0]["parse_error"] is None


def test_preprocess_trajectories_records_parse_error() -> None:
    trajectories = [
        {
            "assistant_response": "this is not a valid tool call payload",
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["format_valid"] is False
    assert rows[0]["parse_error"] is not None


def test_preprocess_trajectories_records_parse_error_for_non_mapping_external_call() -> None:
    trajectories = [
        {
            "prompt": "Submit answer",
            "assistant_response": "",
            "external_tool_calls": ["submit"],
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["format_valid"] is False
    assert rows[0]["parse_error"] is not None
    assert "external_tool_calls[0]" in rows[0]["parse_error"]


def test_preprocess_trajectories_rejects_string_external_tool_calls_field() -> None:
    trajectories = [
        {
            "prompt": "Submit answer",
            "assistant_response": "",
            "external_tool_calls": "submit",
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["format_valid"] is False
    assert rows[0]["parse_error"] == "external_tool_calls must be a sequence of call objects"


def test_preprocess_trajectories_records_parse_error_for_non_numeric_step_index() -> None:
    trajectories = [
        {
            "prompt": "bad index sample",
            "step_index": "not-a-number",
            "assistant_response": "",
            "external_tool_calls": [{"tool": "answer", "args": {"answer": "fixed"}}],
        },
        {
            "prompt": "valid fallback sample",
            "assistant_response": "",
            "external_tool_calls": [{"tool": "answer", "args": {"answer": "fixed"}}],
        },
    ]

    rows = preprocess_trajectories(trajectories)

    assert len(rows) == 2
    assert rows[0]["format_valid"] is False
    assert rows[0]["parse_error"] == "step_index must be an integer >= 0"
    assert rows[1]["format_valid"] is True
    assert rows[1]["parse_error"] is None
