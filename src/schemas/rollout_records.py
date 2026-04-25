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
    image_name: str
    attempt_index: int
    turn_index: int
    container_id: str


class RolloutRow(RolloutRowRequired, total=False):
    stage: str
    fail_to_pass: Any
    pass_to_pass: Any
    task_family: str
    difficulty_band: str
    difficulty_band_source: str
    fail_to_pass_results: Any
    pass_to_pass_results: Any
    fail_to_pass_failures: Any
    pass_to_pass_failures: Any
    fail_to_pass_stderr_tail: str
    pass_to_pass_stderr_tail: str
    fail_to_pass_all_passed: bool
    pass_to_pass_all_passed: bool
    fail_to_pass_verified: bool
    pass_to_pass_verified: bool
    verification_missing: bool
    verification_error: str
    verification_feedback: str
    submission_final_response: str
    collector_error: str
    bridge_error: str
    timeout_error: str
    executor_error: str
    container_init_succeeded: bool
    tool_name: str
    exit_code: int
    is_terminal: bool
    latency_ms: float
    task_patch_applied: bool
    batch_container_count: int
    trajectory_steps: list[dict[str, Any]]
    trajectory_history: list[str]
    trajectory_assistant_turns: list[str]
    trajectory_assistant_turn_count: int
    trajectory_tool_validation_errors: list[str]
    trajectory_format_valid: bool
    final_turn_has_submit: bool
    terminal_format_valid: bool
    final_submit_format_valid: bool
    verifier_status: str
    verifier_resolution_source: str
    initial_prompt_block: str
    _raw_prompt_messages: list[dict[str, Any]]
