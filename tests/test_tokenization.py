from __future__ import annotations

from typing import Any

from data.tokenization import (
    build_labeled_spans,
    label_for_offset,
    tokenize_batch_with_labels,
    tokenize_with_labels,
)
from schemas import ActionEnvelope, ToolCall


class CharTokenizer:
    """One token per character with offset mapping."""

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


def test_build_labeled_spans_from_envelope() -> None:
    envelope = ActionEnvelope(
        tool_calls=(ToolCall(tool="bash", args={"command": "ls"}),),
        thinking="plan",
    )

    text, spans = build_labeled_spans(envelope)

    assert "<think>plan</think>" in text
    assert "<tool_call>" in text
    assert len(spans) == 2
    think_span = spans[0]
    assert think_span[2] == "think"
    assert text[think_span[0] : think_span[1]] == "<think>plan</think>"
    tool_span = spans[1]
    assert tool_span[2] == "tool_call"
    assert text[tool_span[0] : tool_span[1]].startswith("<tool_call>")
    assert text[tool_span[0] : tool_span[1]].endswith("</tool_call>")


def test_build_labeled_spans_no_thinking() -> None:
    envelope = ActionEnvelope(
        tool_calls=(ToolCall(tool="search", args={"query": "bug"}),),
    )

    text, spans = build_labeled_spans(envelope)

    assert "<think>" not in text
    assert len(spans) == 1
    assert spans[0][2] == "tool_call"
    assert spans[0][0] == 0


def test_tokenize_with_labels_aligns_to_offsets() -> None:
    envelope = ActionEnvelope(
        tool_calls=(ToolCall(tool="bash", args={"command": "ls"}),),
        thinking="plan",
    )
    text, spans = build_labeled_spans(envelope)
    tokenizer = CharTokenizer()

    input_ids, token_labels = tokenize_with_labels(text, spans, tokenizer)

    assert len(input_ids) == len(text)
    assert len(token_labels) == len(input_ids)
    assert "think" in token_labels
    assert "tool_call" in token_labels

    think_start, think_end, _ = spans[0]
    for i in range(think_start, think_end):
        assert token_labels[i] == "think"

    tool_start, tool_end, _ = spans[1]
    for i in range(tool_start, tool_end):
        assert token_labels[i] == "tool_call"


def test_label_for_offset_prioritizes_tool_call() -> None:
    spans = [
        (0, 10, "think"),
        (8, 20, "tool_call"),
    ]
    assert label_for_offset(9, 10, spans) == "tool_call"
    assert label_for_offset(0, 1, spans) == "think"
    assert label_for_offset(20, 25, spans) == "other"


def test_label_for_offset_empty_token() -> None:
    spans = [(0, 10, "think")]
    assert label_for_offset(5, 5, spans) == "other"


class _BatchAwareTokenizer:
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
        raise TypeError("Unsupported input type")


class _SingleOnlyTokenizer:
    def __init__(self) -> None:
        self.single_calls = 0

    def __call__(
        self,
        text: Any,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del add_special_tokens
        if not isinstance(text, str):
            raise TypeError("Batch input is unsupported.")
        self.single_calls += 1
        payload: dict[str, Any] = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return payload


def test_tokenize_batch_with_labels_prefers_batched_tokenization() -> None:
    first_text, first_spans = build_labeled_spans(
        ActionEnvelope(
            tool_calls=(ToolCall(tool="bash", args={"command": "ls"}),),
            thinking="plan",
        )
    )
    second_text, second_spans = build_labeled_spans(
        ActionEnvelope(
            tool_calls=(ToolCall(tool="search", args={"query": "bug"}),),
            thinking="inspect",
        )
    )
    tokenizer = _BatchAwareTokenizer()

    input_ids_batch, token_labels_batch = tokenize_batch_with_labels(
        [first_text, second_text],
        [first_spans, second_spans],
        tokenizer,
    )

    assert tokenizer.batch_calls == 1
    assert tokenizer.single_calls == 0
    assert len(input_ids_batch) == 2
    assert len(token_labels_batch) == 2
    assert all(len(ids) == len(labels) for ids, labels in zip(input_ids_batch, token_labels_batch))
    assert "tool_call" in token_labels_batch[0]
    assert "think" in token_labels_batch[1]


def test_tokenize_batch_with_labels_falls_back_to_sequential() -> None:
    first_text, first_spans = build_labeled_spans(
        ActionEnvelope(
            tool_calls=(ToolCall(tool="bash", args={"command": "ls"}),),
            thinking="plan",
        )
    )
    second_text, second_spans = build_labeled_spans(
        ActionEnvelope(
            tool_calls=(ToolCall(tool="search", args={"query": "bug"}),),
            thinking="inspect",
        )
    )
    tokenizer = _SingleOnlyTokenizer()

    input_ids_batch, token_labels_batch = tokenize_batch_with_labels(
        [first_text, second_text],
        [first_spans, second_spans],
        tokenizer,
    )

    assert tokenizer.single_calls == 2
    assert len(input_ids_batch) == 2
    assert len(token_labels_batch) == 2
    assert all(len(ids) == len(labels) for ids, labels in zip(input_ids_batch, token_labels_batch))
