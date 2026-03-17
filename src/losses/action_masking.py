"""Stage-aware token masking for format-RFT, positive-RFT, and turn-SDPO."""

from __future__ import annotations

from typing import Literal, Sequence

MaskStage = Literal["format_rft", "positive_rft", "turn_sdpo", "rft", "step_sdpo"]
TokenLabel = Literal["think", "tool_call", "other"]

_CANONICAL_STAGE_ALIASES = {
    "rft": "format_rft",
    "step_sdpo": "turn_sdpo",
}


def canonicalize_mask_stage(stage: MaskStage) -> Literal["format_rft", "positive_rft", "turn_sdpo"]:
    normalized = _CANONICAL_STAGE_ALIASES.get(stage, stage)
    if normalized not in {"format_rft", "positive_rft", "turn_sdpo"}:
        raise ValueError(f"Unsupported stage: {stage!r}")
    return normalized


def should_train_token(stage: MaskStage, label: TokenLabel) -> bool:
    """Return whether token with label should be included in loss for stage."""
    canonical_stage = canonicalize_mask_stage(stage)
    if canonical_stage in {"format_rft", "positive_rft", "turn_sdpo"}:
        return label in {"think", "tool_call"}
    raise ValueError(f"Unsupported stage: {stage!r}")


def build_action_token_mask(labels: Sequence[TokenLabel], stage: MaskStage) -> list[bool]:
    """Convert token labels into a boolean training mask."""
    return [should_train_token(stage, label) for label in labels]
