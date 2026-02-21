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


def test_preprocess_trajectories_records_parse_error() -> None:
    trajectories = [
        {
            "assistant_response": "this is not a valid tool call payload",
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["format_valid"] is False
    assert rows[0]["parse_error"] is not None
