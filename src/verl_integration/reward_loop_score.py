"""Custom verl reward-loop scorer backed by local SWE reward logic."""

from __future__ import annotations

from typing import Any, Mapping

from config import MAX_TOOL_CALLS_PER_TURN
from verl_integration.reward_function import reward_fn

_GROUND_TRUTH_KEYS = (
    "task_id",
    "image_name",
    "data_source",
    "fail_to_pass",
    "pass_to_pass",
)

_EXTRA_INFO_KEYS = (
    "fail_to_pass",
    "pass_to_pass",
    "fail_to_pass_results",
    "pass_to_pass_results",
    "fail_to_pass_all_passed",
    "pass_to_pass_all_passed",
    "fail_to_pass_verified",
    "pass_to_pass_verified",
    "verification_missing",
    "verification_error",
    "verification_feedback",
    "submission_final_response",
    "resolved",
    "final_turn_has_submit",
    "final_submit_format_valid",
    "bridge_error",
    "timeout_error",
    "executor_error",
)


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _copy_keys(
    target: dict[str, Any],
    source: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]


def compute_score(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **_: Any,
) -> dict[str, Any]:
    """Compute one sample reward for verl RewardLoopManager."""
    ground_truth_map = _coerce_mapping(ground_truth)
    extra_info_map = _coerce_mapping(extra_info)

    response_text = str(solution_str) if solution_str is not None else ""
    sample: dict[str, Any] = {
        "assistant_response": response_text,
        "response_text": response_text,
    }
    _copy_keys(sample, ground_truth_map, keys=_GROUND_TRUTH_KEYS)
    _copy_keys(sample, extra_info_map, keys=_EXTRA_INFO_KEYS)
    if "data_source" not in sample and data_source is not None:
        sample["data_source"] = str(data_source)

    metadata: dict[str, Any] = {}
    _copy_keys(metadata, sample, keys=_EXTRA_INFO_KEYS)
    if metadata:
        sample["tool_output"] = {"metadata": metadata}

    rewards, info = reward_fn([sample], max_tool_calls=MAX_TOOL_CALLS_PER_TURN)
    score = float(rewards[0]) if rewards else 0.0

    result: dict[str, Any] = {"score": score}
    for key, values in info.items():
        if isinstance(values, list) and values:
            result[str(key)] = values[0]
    return result
