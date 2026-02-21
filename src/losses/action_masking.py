"""Stage-aware token masking for RFT and step-SDPO phases."""

from __future__ import annotations

from typing import Literal, Sequence

MaskStage = Literal["rft", "step_sdpo"]
TokenLabel = Literal["think", "tool_call", "other"]


def should_train_token(stage: MaskStage, label: TokenLabel) -> bool:
    """Return whether token with label should be included in loss for stage."""
    if stage == "rft":
        return label == "tool_call"
    if stage == "step_sdpo":
        return label in {"think", "tool_call"}
    raise ValueError(f"Unsupported stage: {stage!r}")


def build_action_token_mask(labels: Sequence[TokenLabel], stage: MaskStage) -> list[bool]:
    """Convert token labels into a boolean training mask."""
    return [should_train_token(stage, label) for label in labels]
