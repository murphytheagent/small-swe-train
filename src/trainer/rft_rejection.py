"""Deterministic RFT rejection-policy evaluation with explicit typed outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from config import RFTSelectionPolicy

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}


@dataclass(frozen=True)
class RFTSelectionResult:
    selected_rows: list[dict[str, Any]]
    rejected_rows: list[dict[str, Any]]


def apply_rft_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_policy: RFTSelectionPolicy,
) -> RFTSelectionResult:
    """Apply centralized RFT rejection policy and annotate row-level labels."""
    selected_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for row in rows:
        mutable_row = dict(row)
        mutable_row["stage"] = str(mutable_row.get("stage", "format_rft"))
        rejection_reason = evaluate_rft_rejection_reason(
            mutable_row,
            selection_policy=selection_policy,
        )

        if rejection_reason is None:
            mutable_row["rft_selected"] = True
            mutable_row["stage_accepted"] = True
            mutable_row["rft_label"] = "accept"
            mutable_row["stage_decision_reason"] = "accepted"
            selected_rows.append(mutable_row)
            continue

        mutable_row["rft_selected"] = False
        mutable_row["stage_accepted"] = False
        mutable_row["rft_label"] = "reject" if selection_policy.relabel_rejected_attempts else "drop"
        mutable_row["rft_rejection_reason"] = rejection_reason
        mutable_row["stage_decision_reason"] = rejection_reason
        rejected_rows.append(mutable_row)

    return RFTSelectionResult(
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
    )


def select_rft_attempt_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_policy: RFTSelectionPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Backward-compatible tuple-return wrapper around ``apply_rft_selection``."""
    result = apply_rft_selection(
        rows,
        selection_policy=selection_policy,
    )
    return result.selected_rows, result.rejected_rows


def evaluate_rft_rejection_reason(
    row: Mapping[str, Any],
    *,
    selection_policy: RFTSelectionPolicy,
) -> str | None:
    """Return deterministic rejection reason list for one rollout attempt row."""
    reasons: list[str] = []

    has_terminal_submit = _coerce_bool(
        row.get("final_turn_has_submit"),
        fallback=_coerce_bool(row.get("is_terminal"), fallback=False),
    )
    trajectory_format_valid = _coerce_bool(
        row.get("trajectory_format_valid"),
        fallback=_coerce_bool(row.get("format_valid"), fallback=False),
    )
    final_submit_format_valid = _coerce_bool(
        row.get("terminal_format_valid", row.get("final_submit_format_valid")),
        fallback=trajectory_format_valid and has_terminal_submit,
    )
    container_init_succeeded = _coerce_bool(
        row.get("container_init_succeeded"),
        fallback=False,
    )
    infra_invalid = _coerce_bool(row.get("infra_invalid"), fallback=False)
    invalid_reason = str(row.get("invalid_reason", "")).strip()

    if not container_init_succeeded:
        reasons.append("container_init_failed")
    if infra_invalid:
        reasons.append(invalid_reason or "infra_invalid")

    if selection_policy.require_terminal and not has_terminal_submit:
        reasons.append("non_terminal")
    if selection_policy.require_format_valid and not trajectory_format_valid:
        reasons.append("format_invalid")
    if (
        selection_policy.reject_on_invalid_final_submit
        and has_terminal_submit
        and not final_submit_format_valid
    ):
        reasons.append("final_submit_invalid")
    if selection_policy.require_resolved and not _coerce_bool(row.get("resolved"), fallback=False):
        reasons.append("unresolved")

    if selection_policy.require_zero_exit_code:
        raw_exit_code = row.get("exit_code")
        if raw_exit_code is not None:
            try:
                exit_code = int(raw_exit_code)
            except (TypeError, ValueError):
                reasons.append("invalid_exit_code")
            else:
                if exit_code != 0:
                    reasons.append("nonzero_exit_code")

    if selection_policy.reject_on_collector_error and _has_error_text(row.get("collector_error")):
        reasons.append("collector_error")
    if selection_policy.reject_on_bridge_error and _has_error_text(row.get("bridge_error")):
        reasons.append("bridge_error")
    if selection_policy.reject_on_timeout_error and _has_error_text(row.get("timeout_error")):
        reasons.append("timeout_error")
    if selection_policy.reject_on_executor_error and _has_error_text(row.get("executor_error")):
        reasons.append("executor_error")
    if selection_policy.reject_on_parse_error and _has_error_text(row.get("parse_error")):
        reasons.append("parse_error")
    if selection_policy.reject_on_validation_errors and _has_nonempty_sequence(
        row.get("validation_errors")
    ):
        reasons.append("validation_errors")

    if not reasons:
        return None
    # Preserve deterministic first-seen order while removing duplicates.
    ordered_unique = list(dict.fromkeys(reasons))
    return ",".join(ordered_unique)


def _has_error_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_nonempty_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return bool(value)
    return isinstance(value, Sequence) and bool(value)


def _coerce_bool(value: Any, *, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return fallback
