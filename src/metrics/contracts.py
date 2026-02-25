"""Metric helpers for action-contract quality and gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FormatMetrics:
    parse_valid_rate: float
    tool_call_block_presence_rate: float
    tool_call_count_valid_rate: float
    submit_singleton_rule_rate: float
    thinking_delimiter_balance_rate: float
    allowed_tool_rate: float
    required_arg_presence: float
    terminal_submission_rate: float


def rate(flags: Iterable[bool]) -> float:
    """Return fraction of true values for iterable flags."""
    values = list(flags)
    if not values:
        return 0.0
    return sum(1 for item in values if item) / len(values)
