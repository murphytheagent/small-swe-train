"""Typed row contracts emitted by on-policy rollout collection."""

from __future__ import annotations

from typing import Any, TypedDict


class RolloutRowRequired(TypedDict):
    prompt: str
    assistant_response: str
    tool_output: dict[str, Any]
    resolved: bool
    step_index: int
    task_id: str
    attempt_index: int
    turn_index: int
    container_id: str


class RolloutRow(RolloutRowRequired, total=False):
    collector_error: str
    bridge_error: str
    timeout_error: str
    executor_error: str
    tool_name: str
    exit_code: int
    is_terminal: bool
    latency_ms: float
    task_patch_applied: bool
    batch_container_count: int
