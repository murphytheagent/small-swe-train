"""Compatibility wrapper exposing trainer-owned RFT runtime orchestration."""

from trainer.rft_runtime import (
    OnPolicyRFTRuntimeRequest,
    collect_onpolicy_rft_runtime_batch,
)

__all__ = [
    "OnPolicyRFTRuntimeRequest",
    "collect_onpolicy_rft_runtime_batch",
]
