"""Centralized runtime config loader with schema-consistent exports.

Runtime policy defaults live in ``configs/runtime/training_policy_defaults.v1.json``.
Dataset config for on-policy collection lives in ``configs/data/*.yaml``.
This module is the import surface for runtime knobs used by code paths across
rollout, prompting, trainer adapters, and environment orchestration.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from runtime_paths import (
    DEFAULT_ON_POLICY_BAD_TASK_CACHE_RELATIVE_DIR,
    DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR,
    resolve_on_policy_bad_task_cache_dir,
    resolve_sdpo_task_cache_dir,
)
from schemas import ALLOWED_TOOLS, TERMINAL_TOOL_NAME as SCHEMA_TERMINAL_TOOL_NAME
from verifier_utils import normalize_verifier_kind

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_DATA_CONFIGS_DIR = _CONFIGS_DIR / "data"
_TRAINING_POLICY_PATH = _CONFIGS_DIR / "runtime" / "training_policy_defaults.v1.json"
_PHASE_TRANSITION_GATES_PATH = _CONFIGS_DIR / "runtime" / "phase_transition_gates.v1.json"
_VERL_MODEL_DEFAULTS_PATH = _CONFIGS_DIR / "verl" / "model_defaults.yaml"
_MODEL_CONFIG_OVERRIDE_DIR = _CONFIGS_DIR / "model"
_BUNDLED_MODEL_CONFIGS_DIR = Path(__file__).resolve().parent / "prompts" / "model_configs"

DEFAULT_ON_POLICY_DATA_CONFIG_NAME = "on_policy_swe_smith"


@dataclass(frozen=True)
class OnPolicyDatasetColumns:
    image_name: str
    problem_statement: str
    fail_to_pass: str
    pass_to_pass: str


@dataclass(frozen=True)
class OnPolicyDifficultyBandConfig:
    strategy: str = "none"
    default_band: str = "unbanded"
    family_band_exact: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    family_band_prefix: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OnPolicyDataConfig:
    dataset_id: str
    dataset_split: str
    columns: OnPolicyDatasetColumns
    patch_is_bug_introducing: bool = True
    verifier_kind: str = "pytest"
    difficulty_banding: OnPolicyDifficultyBandConfig = field(
        default_factory=OnPolicyDifficultyBandConfig
    )


@dataclass(frozen=True)
class OnPolicyRuntimeConfig:
    enabled: bool
    rollout_only: bool
    task_batch_size: int
    attempts_per_task: int
    max_turns_per_attempt: int
    env_pool_size: int
    tool_timeout_sec: int
    container_start_timeout_sec: int
    attempt_timeout_sec: int
    max_tool_calls_per_turn: int
    verify_submissions: bool = False
    verifier_timeout_sec: int = 600
    max_in_flight_tasks: int = 4


@dataclass(frozen=True)
class OnPolicySettings:
    data: OnPolicyDataConfig
    runtime: OnPolicyRuntimeConfig


@dataclass(frozen=True)
class RFTSelectionPolicy:
    require_terminal: bool
    require_format_valid: bool
    require_resolved: bool
    require_zero_exit_code: bool
    reject_on_collector_error: bool
    reject_on_bridge_error: bool
    reject_on_timeout_error: bool
    reject_on_executor_error: bool
    reject_on_parse_error: bool
    reject_on_validation_errors: bool
    reject_on_invalid_final_submit: bool
    relabel_rejected_attempts: bool


@dataclass(frozen=True)
class RFTHandoffSettings:
    max_sequence_length: int
    pad_token_id: int
    selection: RFTSelectionPolicy


@dataclass(frozen=True)
class DeterministicTruncationSettings:
    head_tokens: int
    tail_tokens: int
    apply_after_payload_canonicalization: bool
    apply_before_teacher_prompt_wrapping: bool


@functools.lru_cache(maxsize=1)
def training_policy_defaults() -> dict[str, Any]:
    """Load and cache the full training policy defaults dict."""
    with _TRAINING_POLICY_PATH.open() as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=1)
def phase_transition_gates_defaults() -> dict[str, Any]:
    """Load and cache phase-transition gate thresholds."""
    with _PHASE_TRANSITION_GATES_PATH.open() as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=1)
def verl_model_defaults() -> dict[str, Any]:
    """Load and cache shared verl model defaults."""
    with _VERL_MODEL_DEFAULTS_PATH.open() as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Shared verl model defaults {_VERL_MODEL_DEFAULTS_PATH} must be a mapping."
        )
    return dict(payload)


def default_training_model_name() -> str:
    """Return the canonical training model name from shared verl config."""
    defaults = verl_model_defaults()
    model_defaults = defaults.get("model_defaults")
    if not isinstance(model_defaults, Mapping):
        raise ValueError(
            f"`model_defaults` block is missing from {_VERL_MODEL_DEFAULTS_PATH}."
        )
    model_name = model_defaults.get("primary_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            f"`model_defaults.primary_name` must be a non-empty string in {_VERL_MODEL_DEFAULTS_PATH}."
        )
    return model_name.strip()


@functools.lru_cache(maxsize=8)
def on_policy_data_defaults(
    config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
) -> dict[str, Any]:
    """Load a named on-policy dataset config from ``configs/data``."""
    normalized = config_name.strip()
    if not normalized:
        raise ValueError("on-policy data config name must be a non-empty string")

    config_path = _DATA_CONFIGS_DIR / f"{normalized}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"On-policy data config {normalized!r} not found at {config_path}."
        )

    with config_path.open() as fh:
        payload = yaml.safe_load(fh)

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"On-policy data config {config_path} must be a mapping at top level."
        )

    return dict(payload)


def on_policy_runtime_defaults() -> dict[str, Any]:
    """Return on-policy runtime defaults from centralized policy JSON."""
    defaults = training_policy_defaults()
    on_policy = defaults.get("on_policy")
    if not isinstance(on_policy, Mapping):
        raise ValueError("`on_policy` block is missing from training policy defaults.")
    return dict(on_policy)


def rft_handoff_defaults() -> dict[str, Any]:
    """Return centralized RFT handoff defaults from runtime policy JSON."""
    defaults = training_policy_defaults()
    handoff = defaults.get("rft_handoff")
    if not isinstance(handoff, Mapping):
        raise ValueError("`rft_handoff` block is missing from training policy defaults.")
    return dict(handoff)


def rft_runtime_defaults() -> dict[str, Any]:
    """Return centralized RFT runtime loop/vLLM defaults from runtime policy JSON."""
    defaults = training_policy_defaults()
    runtime_defaults = defaults.get("rft_runtime")
    if not isinstance(runtime_defaults, Mapping):
        raise ValueError("`rft_runtime` block is missing from training policy defaults.")
    return dict(runtime_defaults)


def feedback_processing_defaults() -> dict[str, Any]:
    """Return centralized feedback-processing defaults from runtime policy JSON."""
    defaults = training_policy_defaults()
    feedback_processing = defaults.get("feedback_processing")
    if not isinstance(feedback_processing, Mapping):
        raise ValueError("`feedback_processing` block is missing from training policy defaults.")
    return dict(feedback_processing)


@functools.lru_cache(maxsize=1)
def resolve_feedback_self_containment_signals_enabled() -> bool:
    """Resolve whether canonicalizer should extract self-containment signals."""
    payload = feedback_processing_defaults()
    raw_value = payload.get("extract_self_containment_signals", False)
    if not isinstance(raw_value, bool):
        raise ValueError("`feedback_processing.extract_self_containment_signals` must be a boolean.")
    return raw_value


def adaptation_defaults() -> dict[str, Any]:
    """Return centralized adaptation defaults from runtime policy JSON."""
    defaults = training_policy_defaults()
    adaptation = defaults.get("adaptation")
    if not isinstance(adaptation, Mapping):
        raise ValueError("`adaptation` block is missing from training policy defaults.")
    return dict(adaptation)


def resolve_rft_collector_max_in_flight_default(*, task_batch_size: int) -> int:
    """Resolve default collector in-flight task concurrency for an RFT loop run."""
    if isinstance(task_batch_size, bool) or task_batch_size < 1:
        raise ValueError("task_batch_size must be an integer >= 1.")

    runtime_defaults = rft_runtime_defaults()
    loop_defaults = runtime_defaults.get("loop")
    configured_value: int | None = None
    if isinstance(loop_defaults, Mapping):
        configured_value = _coerce_optional_positive_int(loop_defaults.get("collector_max_in_flight_tasks"))

    resolved = configured_value if configured_value is not None else int(task_batch_size)
    return max(1, min(resolved, int(task_batch_size)))


def resolve_rft_vllm_parallel_defaults(*, nproc_per_node: int) -> tuple[int, int]:
    """Resolve default vLLM TP/DP sizes for a given node world size."""
    if isinstance(nproc_per_node, bool) or nproc_per_node < 1:
        raise ValueError("nproc_per_node must be an integer >= 1.")

    runtime_defaults = rft_runtime_defaults()
    parallel_defaults = runtime_defaults.get("vllm_parallelism")
    resolved_tp: int | None = None
    resolved_dp: int | None = None

    if isinstance(parallel_defaults, Mapping):
        by_nproc = parallel_defaults.get("by_nproc_per_node")
        if isinstance(by_nproc, Mapping):
            keyed_defaults = by_nproc.get(str(int(nproc_per_node)))
            if isinstance(keyed_defaults, Mapping):
                resolved_tp = _coerce_optional_positive_int(keyed_defaults.get("tensor_parallel_size"))
                resolved_dp = _coerce_optional_positive_int(keyed_defaults.get("data_parallel_size"))
        if resolved_tp is None:
            resolved_tp = _coerce_optional_positive_int(parallel_defaults.get("default_tensor_parallel_size"))
        if resolved_dp is None:
            resolved_dp = _coerce_optional_positive_int(parallel_defaults.get("default_data_parallel_size"))

    if resolved_tp is None:
        resolved_tp = 2 if int(nproc_per_node) >= 4 else 1
    if int(nproc_per_node) % resolved_tp != 0:
        resolved_tp = 1
        resolved_dp = None

    max_supported_dp = max(1, int(nproc_per_node) // resolved_tp)
    if resolved_dp is None:
        resolved_dp = max_supported_dp
    else:
        resolved_dp = max(1, min(resolved_dp, max_supported_dp))

    return resolved_tp, resolved_dp


def output_contract_defaults() -> dict[str, Any]:
    """Return the output-contract defaults dictionary from runtime policy."""
    defaults = training_policy_defaults()
    output_contract = defaults.get("output_contract")
    if not isinstance(output_contract, Mapping):
        raise ValueError("`output_contract` block is missing from training policy defaults.")
    return dict(output_contract)


def _validate_terminal_tool_name(terminal_tool_name: str, *, allowed_tools: Sequence[str]) -> None:
    if terminal_tool_name not in allowed_tools:
        allowed = ", ".join(sorted(allowed_tools))
        raise ValueError(
            "Invalid terminal tool configured in training policy defaults: "
            f"{terminal_tool_name!r}. Expected one of: {allowed}."
        )


_output_contract = output_contract_defaults()

MIN_TOOL_CALLS_PER_TURN: int = int(_output_contract["min_tool_calls_per_turn"])
MAX_TOOL_CALLS_PER_TURN: int = int(_output_contract["max_tool_calls_per_turn"])
TERMINAL_TOOL_NAME: str = str(_output_contract["terminal_tool"]).strip().lower()
SUBMIT_MUST_BE_ONLY_TOOL_CALL: bool = bool(_output_contract["submit_must_be_only_tool_call"])
TERMINAL_VALIDITY_PENALTY: float = float(_output_contract.get("terminal_validity_penalty", 0.2))
DEFAULT_TRAINING_MODEL_NAME: str = default_training_model_name()

if MIN_TOOL_CALLS_PER_TURN < 1:
    raise ValueError("min_tool_calls_per_turn must be >= 1")
if MAX_TOOL_CALLS_PER_TURN < MIN_TOOL_CALLS_PER_TURN:
    raise ValueError("max_tool_calls_per_turn must be >= min_tool_calls_per_turn")
if TERMINAL_VALIDITY_PENALTY < 0.0:
    raise ValueError("terminal_validity_penalty must be >= 0")
_validate_terminal_tool_name(TERMINAL_TOOL_NAME, allowed_tools=ALLOWED_TOOLS)
if TERMINAL_TOOL_NAME != SCHEMA_TERMINAL_TOOL_NAME:
    raise ValueError(
        "training_policy_defaults terminal_tool is inconsistent with schema authority: "
        f"{TERMINAL_TOOL_NAME!r} != {SCHEMA_TERMINAL_TOOL_NAME!r}"
    )


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _coerce_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be a boolean.")


def _coerce_non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _coerce_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1.")
    return value


def _coerce_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return int(value)


def _coerce_non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an integer >= 0.")
    return value


def _coerce_non_empty_string_pairs(
    value: Any,
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping of non-empty strings.")

    items: list[tuple[str, str]] = []
    for raw_key, raw_mapped_value in value.items():
        key = _coerce_non_empty_str(raw_key, label=f"{label}.key")
        mapped_value = _coerce_non_empty_str(raw_mapped_value, label=f"{label}.{key}")
        items.append((key, mapped_value))
    return tuple(items)


def _parse_on_policy_data_config(payload: Mapping[str, Any]) -> OnPolicyDataConfig:
    dataset_id = _coerce_non_empty_str(payload.get("dataset_id"), label="on_policy.data.dataset_id")
    dataset_split = _coerce_non_empty_str(
        payload.get("dataset_split"),
        label="on_policy.data.dataset_split",
    )

    columns_payload = _require_mapping(payload.get("columns"), label="on_policy.data.columns")
    columns = OnPolicyDatasetColumns(
        image_name=_coerce_non_empty_str(
            columns_payload.get("image_name"),
            label="on_policy.data.columns.image_name",
        ),
        problem_statement=_coerce_non_empty_str(
            columns_payload.get("problem_statement"),
            label="on_policy.data.columns.problem_statement",
        ),
        fail_to_pass=_coerce_non_empty_str(
            columns_payload.get("fail_to_pass"),
            label="on_policy.data.columns.fail_to_pass",
        ),
        pass_to_pass=_coerce_non_empty_str(
            columns_payload.get("pass_to_pass"),
            label="on_policy.data.columns.pass_to_pass",
        ),
    )

    return OnPolicyDataConfig(
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        columns=columns,
        patch_is_bug_introducing=_coerce_bool(
            payload.get("patch_is_bug_introducing", True),
            label="on_policy.data.patch_is_bug_introducing",
        ),
        verifier_kind=normalize_verifier_kind(
            payload.get("verifier_kind", "pytest"),
            label="on_policy.data.verifier_kind",
        ),
        difficulty_banding=_parse_on_policy_difficulty_band_config(
            payload.get("difficulty_banding")
        ),
    )


def _parse_on_policy_runtime_config(payload: Mapping[str, Any]) -> OnPolicyRuntimeConfig:
    task_batch_size = _coerce_positive_int(
        payload.get("task_batch_size"),
        label="on_policy.task_batch_size",
    )
    attempts_per_task = _coerce_positive_int(
        payload.get("attempts_per_task"),
        label="on_policy.attempts_per_task",
    )
    max_turns_per_attempt = _coerce_positive_int(
        payload.get("max_turns_per_attempt"),
        label="on_policy.max_turns_per_attempt",
    )
    env_pool_size = _coerce_positive_int(
        payload.get("env_pool_size"),
        label="on_policy.env_pool_size",
    )
    tool_timeout_sec = _coerce_positive_int(
        payload.get("tool_timeout_sec"),
        label="on_policy.tool_timeout_sec",
    )
    container_start_timeout_sec = _coerce_positive_int(
        payload.get("container_start_timeout_sec"),
        label="on_policy.container_start_timeout_sec",
    )
    attempt_timeout_sec = _coerce_positive_int(
        payload.get("attempt_timeout_sec"),
        label="on_policy.attempt_timeout_sec",
    )
    max_tool_calls_per_turn = _coerce_positive_int(
        payload.get("max_tool_calls_per_turn"),
        label="on_policy.max_tool_calls_per_turn",
    )
    verify_submissions = _coerce_bool(
        payload.get("verify_submissions", False),
        label="on_policy.verify_submissions",
    )
    verifier_timeout_sec = _coerce_positive_int(
        payload.get("verifier_timeout_sec", 600),
        label="on_policy.verifier_timeout_sec",
    )
    max_in_flight_tasks = _coerce_positive_int(
        payload.get("max_in_flight_tasks", task_batch_size),
        label="on_policy.max_in_flight_tasks",
    )

    runtime = OnPolicyRuntimeConfig(
        enabled=_coerce_bool(payload.get("enabled"), label="on_policy.enabled"),
        rollout_only=_coerce_bool(payload.get("rollout_only"), label="on_policy.rollout_only"),
        task_batch_size=task_batch_size,
        attempts_per_task=attempts_per_task,
        max_turns_per_attempt=max_turns_per_attempt,
        env_pool_size=env_pool_size,
        tool_timeout_sec=tool_timeout_sec,
        container_start_timeout_sec=container_start_timeout_sec,
        attempt_timeout_sec=attempt_timeout_sec,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        verify_submissions=verify_submissions,
        verifier_timeout_sec=verifier_timeout_sec,
        max_in_flight_tasks=max_in_flight_tasks,
    )

    if runtime.env_pool_size < runtime.task_batch_size:
        raise ValueError(
            "on_policy.env_pool_size must be >= on_policy.task_batch_size for per-task containers."
        )
    if runtime.max_tool_calls_per_turn > MAX_TOOL_CALLS_PER_TURN:
        raise ValueError(
            "on_policy.max_tool_calls_per_turn exceeds centralized output contract max "
            f"({MAX_TOOL_CALLS_PER_TURN})."
        )

    return runtime


def _merge_on_policy_data_payload(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key == "columns" and isinstance(value, Mapping) and isinstance(merged.get("columns"), Mapping):
            columns = dict(merged["columns"])
            columns.update(value)
            merged["columns"] = columns
        elif (
            key == "difficulty_banding"
            and isinstance(value, Mapping)
            and isinstance(merged.get("difficulty_banding"), Mapping)
        ):
            banding = dict(merged["difficulty_banding"])
            banding.update(value)
            merged["difficulty_banding"] = banding
        else:
            merged[key] = value
    return merged


def _parse_on_policy_difficulty_band_config(
    payload: Any,
) -> OnPolicyDifficultyBandConfig:
    if payload is None:
        return OnPolicyDifficultyBandConfig()

    banding_payload = _require_mapping(
        payload,
        label="on_policy.data.difficulty_banding",
    )
    strategy = _coerce_non_empty_str(
        banding_payload.get("strategy", "none"),
        label="on_policy.data.difficulty_banding.strategy",
    )
    if strategy not in {"none", "instance_id_family"}:
        raise ValueError(
            "on_policy.data.difficulty_banding.strategy must be one of: "
            "'none', 'instance_id_family'."
        )

    return OnPolicyDifficultyBandConfig(
        strategy=strategy,
        default_band=_coerce_non_empty_str(
            banding_payload.get("default_band", "unbanded"),
            label="on_policy.data.difficulty_banding.default_band",
        ),
        family_band_exact=_coerce_non_empty_string_pairs(
            banding_payload.get("family_band_exact", {}),
            label="on_policy.data.difficulty_banding.family_band_exact",
        ),
        family_band_prefix=_coerce_non_empty_string_pairs(
            banding_payload.get("family_band_prefix", {}),
            label="on_policy.data.difficulty_banding.family_band_prefix",
        ),
    )


def _resolve_on_policy_runtime_scale_defaults() -> dict[str, int]:
    runtime_defaults = rft_runtime_defaults()
    loop_defaults = _require_mapping(
        runtime_defaults.get("loop"),
        label="rft_runtime.loop",
    )
    task_batch_size = _coerce_positive_int(
        loop_defaults.get("task_batch_size"),
        label="rft_runtime.loop.task_batch_size",
    )
    attempts_per_task = _coerce_positive_int(
        loop_defaults.get("samples_per_task"),
        label="rft_runtime.loop.samples_per_task",
    )
    max_in_flight_tasks = resolve_rft_collector_max_in_flight_default(
        task_batch_size=task_batch_size,
    )
    return {
        "task_batch_size": task_batch_size,
        "attempts_per_task": attempts_per_task,
        "env_pool_size": task_batch_size,
        "max_in_flight_tasks": max_in_flight_tasks,
    }


def resolve_on_policy_settings(
    *,
    data_config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    runtime_overrides: Mapping[str, Any] | None = None,
    data_overrides: Mapping[str, Any] | None = None,
) -> OnPolicySettings:
    """Resolve and validate merged on-policy settings from centralized configs."""
    runtime_payload = on_policy_runtime_defaults()
    runtime_payload.update(_resolve_on_policy_runtime_scale_defaults())
    if runtime_overrides is not None:
        runtime_payload.update(runtime_overrides)
        if (
            "task_batch_size" in runtime_overrides
            and "max_in_flight_tasks" not in runtime_overrides
        ):
            runtime_payload["max_in_flight_tasks"] = runtime_payload["task_batch_size"]
        if "task_batch_size" in runtime_overrides and "env_pool_size" not in runtime_overrides:
            runtime_payload["env_pool_size"] = runtime_payload["task_batch_size"]

    data_payload = on_policy_data_defaults(data_config_name)
    if data_overrides is not None:
        data_payload = _merge_on_policy_data_payload(data_payload, data_overrides)

    runtime = _parse_on_policy_runtime_config(runtime_payload)
    data = _parse_on_policy_data_config(data_payload)

    return OnPolicySettings(data=data, runtime=runtime)


def _parse_rft_selection_policy(payload: Mapping[str, Any]) -> RFTSelectionPolicy:
    return RFTSelectionPolicy(
        require_terminal=_coerce_bool(
            payload.get("require_terminal"),
            label="rft_handoff.selection.require_terminal",
        ),
        require_format_valid=_coerce_bool(
            payload.get("require_format_valid"),
            label="rft_handoff.selection.require_format_valid",
        ),
        require_resolved=_coerce_bool(
            payload.get("require_resolved"),
            label="rft_handoff.selection.require_resolved",
        ),
        require_zero_exit_code=_coerce_bool(
            payload.get("require_zero_exit_code"),
            label="rft_handoff.selection.require_zero_exit_code",
        ),
        reject_on_collector_error=_coerce_bool(
            payload.get("reject_on_collector_error"),
            label="rft_handoff.selection.reject_on_collector_error",
        ),
        reject_on_bridge_error=_coerce_bool(
            payload.get("reject_on_bridge_error"),
            label="rft_handoff.selection.reject_on_bridge_error",
        ),
        reject_on_timeout_error=_coerce_bool(
            payload.get("reject_on_timeout_error"),
            label="rft_handoff.selection.reject_on_timeout_error",
        ),
        reject_on_executor_error=_coerce_bool(
            payload.get("reject_on_executor_error"),
            label="rft_handoff.selection.reject_on_executor_error",
        ),
        reject_on_parse_error=_coerce_bool(
            payload.get("reject_on_parse_error"),
            label="rft_handoff.selection.reject_on_parse_error",
        ),
        reject_on_validation_errors=_coerce_bool(
            payload.get("reject_on_validation_errors"),
            label="rft_handoff.selection.reject_on_validation_errors",
        ),
        reject_on_invalid_final_submit=_coerce_bool(
            payload.get("reject_on_invalid_final_submit"),
            label="rft_handoff.selection.reject_on_invalid_final_submit",
        ),
        relabel_rejected_attempts=_coerce_bool(
            payload.get("relabel_rejected_attempts"),
            label="rft_handoff.selection.relabel_rejected_attempts",
        ),
    )


def resolve_rft_handoff_settings(
    *,
    overrides: Mapping[str, Any] | None = None,
) -> RFTHandoffSettings:
    """Resolve and validate RFT handoff settings from centralized runtime policy."""
    payload = rft_handoff_defaults()
    if overrides is not None:
        payload = dict(payload)
        for key, value in overrides.items():
            if (
                key == "selection"
                and isinstance(value, Mapping)
                and isinstance(payload.get("selection"), Mapping)
            ):
                selection_payload = dict(payload["selection"])
                selection_payload.update(value)
                payload["selection"] = selection_payload
            else:
                payload[key] = value

    selection_payload = _require_mapping(
        payload.get("selection"),
        label="rft_handoff.selection",
    )
    selection = _parse_rft_selection_policy(selection_payload)

    return RFTHandoffSettings(
        max_sequence_length=_coerce_positive_int(
            payload.get("max_sequence_length"),
            label="rft_handoff.max_sequence_length",
        ),
        pad_token_id=_coerce_non_negative_int(
            payload.get("pad_token_id"),
            label="rft_handoff.pad_token_id",
        ),
        selection=selection,
    )


@functools.lru_cache(maxsize=1)
def resolve_feedback_deterministic_truncation_settings() -> DeterministicTruncationSettings:
    """Resolve deterministic truncation settings for feedback/tool-response payloads."""
    feedback_payload = _require_mapping(
        feedback_processing_defaults().get("deterministic_truncation"),
        label="feedback_processing.deterministic_truncation",
    )
    return DeterministicTruncationSettings(
        head_tokens=_coerce_positive_int(
            feedback_payload.get("head_tokens"),
            label="feedback_processing.deterministic_truncation.head_tokens",
        ),
        tail_tokens=_coerce_positive_int(
            feedback_payload.get("tail_tokens"),
            label="feedback_processing.deterministic_truncation.tail_tokens",
        ),
        apply_after_payload_canonicalization=_coerce_bool(
            feedback_payload.get("apply_after_payload_canonicalization"),
            label=(
                "feedback_processing.deterministic_truncation."
                "apply_after_payload_canonicalization"
            ),
        ),
        apply_before_teacher_prompt_wrapping=_coerce_bool(
            feedback_payload.get("apply_before_teacher_prompt_wrapping"),
            label=(
                "feedback_processing.deterministic_truncation."
                "apply_before_teacher_prompt_wrapping"
            ),
        ),
    )


def resolve_model_config_path(model_family: str) -> Path:
    """Resolve a model-family delimiter config path with repo override fallback."""
    normalized = model_family.strip().lower()
    if not normalized:
        raise ValueError("model_family must be a non-empty string.")

    override_path = _MODEL_CONFIG_OVERRIDE_DIR / f"{normalized}.yaml"
    if override_path.exists():
        return override_path

    bundled_path = _BUNDLED_MODEL_CONFIGS_DIR / f"{normalized}.yaml"
    if bundled_path.exists():
        return bundled_path

    raise FileNotFoundError(
        f"No model config found for {normalized!r}. "
        f"Checked {override_path} and {bundled_path}."
    )
