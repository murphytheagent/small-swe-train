"""Compatibility wrapper exposing trainer-owned RFT rejection-policy helpers."""

from trainer.rft_rejection import (
    RFTSelectionResult,
    apply_rft_selection,
    evaluate_rft_rejection_reason,
    select_rft_attempt_rows,
)

__all__ = [
    "RFTSelectionResult",
    "apply_rft_selection",
    "evaluate_rft_rejection_reason",
    "select_rft_attempt_rows",
]
