"""Helpers for building and injecting verl-compatible response masks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from losses.action_masking import MaskStage, TokenLabel, build_action_token_mask

_ALLOWED_LABELS: set[str] = {"think", "tool_call", "other"}


def build_response_mask(labels: Sequence[str], *, stage: MaskStage) -> list[bool]:
    """Validate token labels and build a stage-aware boolean response mask."""
    normalized_labels: list[TokenLabel] = []
    for label in labels:
        if label not in _ALLOWED_LABELS:
            raise ValueError(f"Unsupported token label: {label!r}")
        normalized_labels.append(cast(TokenLabel, label))
    return build_action_token_mask(normalized_labels, stage=stage)


def inject_response_mask(
    batch: Sequence[Mapping[str, Any]],
    *,
    stage: MaskStage = "turn_sdpo",
    label_field: str = "token_labels",
    output_field: str = "response_mask",
) -> list[dict[str, Any]]:
    """Return a shallow-copied batch with ``response_mask`` populated per sample."""
    injected: list[dict[str, Any]] = []
    for sample in batch:
        labels_raw = sample.get(label_field, ())
        if isinstance(labels_raw, (str, bytes)) or not isinstance(labels_raw, Sequence):
            raise ValueError(f"Expected sequence at field '{label_field}', got {type(labels_raw).__name__}")

        labels = [str(label) for label in labels_raw]
        sample_copy = dict(sample)
        sample_copy[output_field] = build_response_mask(labels, stage=stage)
        injected.append(sample_copy)
    return injected
