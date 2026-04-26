from __future__ import annotations

from typing import Any

from verl_integration.data_preprocessor import preprocess_trajectories


def test_preprocess_trajectories_from_assistant_response() -> None:
    trajectories = [
        {
            "prompt": "Fix test",
            "assistant_response": (
                "<|im_start|>assistant\n"
                "<think>debug quickly</think>\n"
                "<tool_call>{\"tool\":\"text_search\",\"args\":{\"query\":\"tests/test_math.py::test_add\"}}</tool_call>\n"
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
    assert row["action_mask_format_rft"]
    assert row["action_mask_positive_rft"]
    assert row["action_mask_turn_sdpo"]
    assert row["action_mask_step_sdpo"]
    assert row["assistant_action_token_count"] > 0
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


def test_preprocess_trajectories_coerces_include_student_flag_false_strings() -> None:
    trajectories = [
        {
            "prompt": "Submit answer",
            "assistant_response": "",
            "include_student_attempt_for_teacher": "false",
            "external_tool_calls": [
                {"tool": "answer", "args": {"answer": "fixed"}},
            ],
        },
        {
            "prompt": "Submit answer 2",
            "assistant_response": "",
            "include_student_attempt_for_teacher": "0",
            "external_tool_calls": [
                {"tool": "answer", "args": {"answer": "fixed"}},
            ],
        },
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["feedback_packet"]["include_student_attempt_for_teacher"] is False
    assert rows[1]["feedback_packet"]["include_student_attempt_for_teacher"] is False


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


def test_preprocess_trajectories_emits_label_blocks_metadata() -> None:
    trajectories = [
        {
            "prompt": "Fix test",
            "assistant_response": (
                "<|im_start|>assistant\n"
                "<think>debug quickly</think>\n"
                "<tool_call>{\"tool\":\"text_search\",\"args\":{\"query\":\"tests/test_math.py\"}}</tool_call>\n"
                "<|im_end|>"
            ),
        }
    ]

    rows = preprocess_trajectories(trajectories)

    blocks = rows[0]["label_blocks"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "think"
    assert blocks[0]["text"] == "debug quickly"
    assert blocks[1]["type"] == "tool_call"


def test_preprocess_trajectories_coerces_non_string_thinking() -> None:
    """Regression: non-string thinking field must not crash ActionEnvelope."""
    trajectories = [
        {
            "prompt": "Submit",
            "assistant_response": "",
            "external_tool_calls": [{"tool": "answer", "args": {"answer": "fixed"}}],
            "thinking": 42,
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert rows[0]["format_valid"] is True
    assert rows[0]["parse_error"] is None


class _CharTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del add_special_tokens
        payload: dict[str, Any] = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return payload


class _BatchTokenizer:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0

    def __call__(
        self,
        text: Any,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del add_special_tokens
        if isinstance(text, str):
            self.single_calls += 1
            payload: dict[str, Any] = {"input_ids": list(range(len(text)))}
            if return_offsets_mapping:
                payload["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return payload
        if isinstance(text, list):
            self.batch_calls += 1
            payload = {"input_ids": [list(range(len(item))) for item in text]}
            if return_offsets_mapping:
                payload["offset_mapping"] = [
                    [(i, i + 1) for i in range(len(item))]
                    for item in text
                ]
            return payload
        raise TypeError("Unsupported input type for tokenizer.")


def test_preprocess_trajectories_with_tokenizer_produces_aligned_masks() -> None:
    trajectories = [
        {
            "prompt": "Fix test",
            "assistant_response": (
                "<|im_start|>assistant\n"
                "<think>debug quickly</think>\n"
                "<tool_call>{\"tool\":\"text_search\",\"args\":{\"query\":\"tests/test_math.py\"}}</tool_call>\n"
                "<|im_end|>"
            ),
            "tool_output": {"stdout": "Traceback", "stderr": "", "exit_code": 1},
        }
    ]

    rows = preprocess_trajectories(trajectories, tokenizer=_CharTokenizer())

    row = rows[0]
    assert row["format_valid"] is True
    assert "input_ids" in row
    assert "canonical_text" in row
    assert len(row["input_ids"]) == len(row["token_labels"])
    assert len(row["token_labels"]) == len(row["action_mask_rft"])
    assert len(row["action_mask_rft"]) == len(row["action_mask_turn_sdpo"])
    assert row["action_mask_rft"] == row["action_mask_format_rft"]
    assert row["action_mask_step_sdpo"] == row["action_mask_turn_sdpo"]
    assert "tool_call" in row["token_labels"]
    assert "think" in row["token_labels"]


def test_preprocess_trajectories_uses_batched_tokenization_when_supported() -> None:
    trajectories = [
        {
            "prompt": "Fix test 1",
            "assistant_response": (
                "<|im_start|>assistant\n"
                "<think>debug quickly</think>\n"
                "<tool_call>{\"tool\":\"text_search\",\"args\":{\"query\":\"tests/test_math.py\"}}</tool_call>\n"
                "<|im_end|>"
            ),
            "tool_output": {"stdout": "Traceback", "stderr": "", "exit_code": 1},
        },
        {
            "prompt": "Fix test 2",
            "assistant_response": (
                "<|im_start|>assistant\n"
                "<think>inspect logs</think>\n"
                "<tool_call>{\"tool\":\"bash\",\"args\":{\"command\":\"pytest -q\"}}</tool_call>\n"
                "<|im_end|>"
            ),
            "tool_output": {"stdout": "", "stderr": "", "exit_code": 0},
        },
    ]
    tokenizer = _BatchTokenizer()

    rows = preprocess_trajectories(trajectories, tokenizer=tokenizer)

    assert tokenizer.batch_calls == 1
    assert tokenizer.single_calls == 0
    assert len(rows) == 2
    for row in rows:
        assert row["format_valid"] is True
        assert "input_ids" in row
        assert len(row["input_ids"]) == len(row["token_labels"])
        assert len(row["token_labels"]) == len(row["action_mask_rft"])
        assert len(row["action_mask_rft"]) == len(row["action_mask_turn_sdpo"])
        assert row["action_mask_rft"] == row["action_mask_format_rft"]
        assert row["action_mask_step_sdpo"] == row["action_mask_turn_sdpo"]


def test_preprocess_trajectories_without_tokenizer_omits_input_ids() -> None:
    trajectories = [
        {
            "prompt": "Fix test",
            "assistant_response": (
                "<tool_call>{\"tool\":\"text_search\",\"args\":{\"query\":\"a\"}}</tool_call>"
            ),
        }
    ]

    rows = preprocess_trajectories(trajectories)

    assert "input_ids" not in rows[0]
    assert "canonical_text" not in rows[0]
    assert rows[0]["token_labels"]
    assert rows[0]["action_mask_rft"]


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


def test_preprocess_trajectories_preserves_invalid_xml_calls_for_validation() -> None:
    trajectories = [
        {
            "prompt": "Fix test",
            "assistant_response": (
                '<tool_call name="bash">'
                "<command><![CDATA[pytest -q]]></command>"
                "<bogus><![CDATA[oops]]></bogus>"
                "</tool_call>"
            ),
        }
    ]

    rows = preprocess_trajectories(trajectories, tokenizer=_CharTokenizer())

    assert len(rows) == 1
    row = rows[0]
    assert row["format_valid"] is False
    assert row["parse_error"] is None
    assert row["validation_errors"] == ["tool_call[0]: Unknown arg 'bogus' for tool 'bash'"]
    assert row["label_blocks"] == [
        {
            "type": "tool_call",
            "text": '<tool_call>{"args": {"bogus": "oops", "command": "pytest -q"}, "tool": "bash"}</tool_call>',
        }
    ]
    assert row["canonical_text"] == (
        '<tool_call>{"args": {"bogus": "oops", "command": "pytest -q"}, "tool": "bash"}</tool_call>'
    )
    assert "input_ids" in row
    assert len(row["input_ids"]) == len(row["token_labels"])
