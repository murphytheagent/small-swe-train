"""Shared tokenization utilities for offset-based label alignment."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, cast

from losses.action_masking import TokenLabel
from prompts.model_delimiters import ModelDelimiters, default_delimiters
from rollout.action_format import render_think_block, render_tool_call_block
from schemas import ActionEnvelope

LabeledSpan = tuple[int, int, TokenLabel]


class SupportsOffsetsTokenizer(Protocol):
    """Minimal tokenizer protocol requiring offset mapping support."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Mapping[str, Any]:
        ...


def load_qwen_tokenizer(model_name: str) -> SupportsOffsetsTokenizer:
    """Load a fast tokenizer for the configured Qwen chat format."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for tokenizer loading. "
            "Install with `pip install transformers`."
        ) from exc

    kwargs: dict[str, Any] = {
        "use_fast": True,
        "fix_mistral_regex": True,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    return cast(SupportsOffsetsTokenizer, tokenizer)


def label_for_offset(
    start: int,
    end: int,
    labeled_spans: Sequence[LabeledSpan],
) -> TokenLabel:
    """Map a single token's character span to a label.

    When a token overlaps multiple spans, ``tool_call`` wins over ``think``.
    """
    if end <= start:
        return "other"
    label: TokenLabel = "other"
    for span_start, span_end, span_label in labeled_spans:
        if end <= span_start or start >= span_end:
            continue
        if span_label == "tool_call":
            return "tool_call"
        if span_label == "think":
            label = "think"
    return label


def tokenize_with_labels(
    text: str,
    labeled_spans: Sequence[LabeledSpan],
    tokenizer: SupportsOffsetsTokenizer,
) -> tuple[list[int], list[TokenLabel]]:
    """Tokenize *text* and assign a label to each token via offset alignment."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if raw_ids is None or raw_offsets is None:
        raise ValueError("Tokenizer must return `input_ids` and `offset_mapping`.")

    input_ids = _normalize_ints(raw_ids)
    offsets = _normalize_offsets(raw_offsets)
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent input_ids and offset_mapping lengths.")

    token_labels = [label_for_offset(start, end, labeled_spans) for start, end in offsets]
    return input_ids, token_labels


def tokenize_batch_with_labels(
    texts: Sequence[str],
    labeled_spans_batch: Sequence[Sequence[LabeledSpan]],
    tokenizer: SupportsOffsetsTokenizer,
) -> tuple[list[list[int]], list[list[TokenLabel]]]:
    """Tokenize multiple texts with label alignment.

    Uses batched tokenization when the tokenizer supports it and falls back to
    deterministic per-sample tokenization otherwise.
    """
    if len(texts) != len(labeled_spans_batch):
        raise ValueError("texts and labeled_spans_batch must have the same length.")
    if not texts:
        return [], []

    batch_size = len(texts)
    if batch_size == 1:
        input_ids, token_labels = tokenize_with_labels(texts[0], labeled_spans_batch[0], tokenizer)
        return [input_ids], [token_labels]

    encoded = _try_batch_encode(texts, tokenizer)
    if encoded is None:
        return _tokenize_batch_sequential(texts, labeled_spans_batch, tokenizer)

    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if (
        raw_ids is None
        or raw_offsets is None
        or not _looks_like_batched_input_ids(raw_ids, batch_size=batch_size)
        or not _looks_like_batched_offsets(raw_offsets, batch_size=batch_size)
    ):
        return _tokenize_batch_sequential(texts, labeled_spans_batch, tokenizer)

    batch_input_ids: list[list[int]] = []
    batch_token_labels: list[list[TokenLabel]] = []
    for sample_ids, sample_offsets, labeled_spans in zip(raw_ids, raw_offsets, labeled_spans_batch):
        input_ids = _normalize_ints(sample_ids)
        offsets = _normalize_offsets(sample_offsets)
        if len(input_ids) != len(offsets):
            raise ValueError(
                "Tokenizer returned inconsistent input_ids and offset_mapping lengths."
            )
        token_labels = [label_for_offset(start, end, labeled_spans) for start, end in offsets]
        batch_input_ids.append(input_ids)
        batch_token_labels.append(token_labels)

    return batch_input_ids, batch_token_labels


def build_labeled_spans(
    envelope: ActionEnvelope,
    delimiters: ModelDelimiters | None = None,
    tool_call_fallback_payload_format: str | None = None,
) -> tuple[str, list[LabeledSpan]]:
    """Construct canonical text from *envelope* and track character-level label spans.

    Returns the concatenated text and a list of ``(start, end, label)`` spans
    covering the think and tool-call blocks (delimiters included).
    """
    d = delimiters or default_delimiters()
    chunks: list[str] = []
    spans: list[LabeledSpan] = []
    cursor = 0

    if envelope.thinking:
        block = render_think_block(envelope.thinking, delimiters=d)
        spans.append((cursor, cursor + len(block), "think"))
        chunks.append(block)
        cursor += len(block)

    for call in envelope.tool_calls:
        block = render_tool_call_block(
            call,
            delimiters=d,
            fallback_payload_format=tool_call_fallback_payload_format,
        )
        spans.append((cursor, cursor + len(block), "tool_call"))
        chunks.append(block)
        cursor += len(block)

    return "".join(chunks), spans


def _normalize_ints(raw_ids: Any) -> list[int]:
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("Tokenizer `input_ids` must be a sequence of ints.")
    return [int(item) for item in raw_ids]


def _normalize_offsets(raw_offsets: Any) -> list[tuple[int, int]]:
    if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes)):
        raise ValueError("Tokenizer `offset_mapping` must be a sequence of pairs.")
    normalized: list[tuple[int, int]] = []
    for offset in raw_offsets:
        if not isinstance(offset, Sequence) or isinstance(offset, (str, bytes)) or len(offset) != 2:
            raise ValueError("Each offset mapping entry must be a (start, end) pair.")
        normalized.append((int(offset[0]), int(offset[1])))
    return normalized


def _try_batch_encode(
    texts: Sequence[str],
    tokenizer: SupportsOffsetsTokenizer,
) -> Mapping[str, Any] | None:
    # Some tokenizers only accept a single string; treat batch mode as best-effort.
    try:
        encoded = cast(Any, tokenizer)(
            list(texts),
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    if not isinstance(encoded, Mapping):
        return None
    return encoded


def _tokenize_batch_sequential(
    texts: Sequence[str],
    labeled_spans_batch: Sequence[Sequence[LabeledSpan]],
    tokenizer: SupportsOffsetsTokenizer,
) -> tuple[list[list[int]], list[list[TokenLabel]]]:
    batch_input_ids: list[list[int]] = []
    batch_token_labels: list[list[TokenLabel]] = []
    for text, labeled_spans in zip(texts, labeled_spans_batch):
        input_ids, token_labels = tokenize_with_labels(text, labeled_spans, tokenizer)
        batch_input_ids.append(input_ids)
        batch_token_labels.append(token_labels)
    return batch_input_ids, batch_token_labels


def _looks_like_batched_input_ids(raw_ids: Any, *, batch_size: int) -> bool:
    if not _is_non_string_sequence(raw_ids) or len(raw_ids) != batch_size:
        return False
    return all(_is_non_string_sequence(sample_ids) for sample_ids in raw_ids)


def _looks_like_batched_offsets(raw_offsets: Any, *, batch_size: int) -> bool:
    if not _is_non_string_sequence(raw_offsets) or len(raw_offsets) != batch_size:
        return False

    for sample_offsets in raw_offsets:
        if not _is_non_string_sequence(sample_offsets):
            return False
        if not sample_offsets:
            continue
        first_offset = sample_offsets[0]
        if not _is_non_string_sequence(first_offset) or len(first_offset) != 2:
            return False
    return True


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
