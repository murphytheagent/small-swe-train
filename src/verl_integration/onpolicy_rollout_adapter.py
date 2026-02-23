"""Compatibility wrapper for trainer-owned on-policy RFT handoff utilities."""

from trainer.rft_handoff import (
    build_dataproto_compatible_payload,
    build_onpolicy_collector,
    build_verl_sft_batch,
    collect_rft_sft_batch_for_steps,
    collect_rollouts_for_steps,
    merge_rollout_and_preprocessed_rows,
    write_jsonl_rows,
)
from trainer.rft_rejection import evaluate_rft_rejection_reason, select_rft_attempt_rows

__all__ = [
    "build_dataproto_compatible_payload",
    "build_onpolicy_collector",
    "build_verl_sft_batch",
    "collect_rft_sft_batch_for_steps",
    "collect_rollouts_for_steps",
    "evaluate_rft_rejection_reason",
    "merge_rollout_and_preprocessed_rows",
    "select_rft_attempt_rows",
    "write_jsonl_rows",
]
