"""Centralized runtime config loader with schema-consistent exports.

Runtime policy defaults live in ``configs/runtime/training_policy_defaults.v1.json``.
This module is the import surface for runtime knobs used by code paths across
rollout, prompting, and trainer adapters.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from schemas import ALLOWED_TOOLS, TERMINAL_TOOL_NAME as SCHEMA_TERMINAL_TOOL_NAME

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_TRAINING_POLICY_PATH = _CONFIGS_DIR / "runtime" / "training_policy_defaults.v1.json"
_PHASE_TRANSITION_GATES_PATH = _CONFIGS_DIR / "runtime" / "phase_transition_gates.v1.json"
_MODEL_CONFIG_OVERRIDE_DIR = _CONFIGS_DIR / "model"
_BUNDLED_MODEL_CONFIGS_DIR = Path(__file__).resolve().parent / "prompts" / "model_configs"


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

if MIN_TOOL_CALLS_PER_TURN < 1:
    raise ValueError("min_tool_calls_per_turn must be >= 1")
if MAX_TOOL_CALLS_PER_TURN < MIN_TOOL_CALLS_PER_TURN:
    raise ValueError("max_tool_calls_per_turn must be >= min_tool_calls_per_turn")
_validate_terminal_tool_name(TERMINAL_TOOL_NAME, allowed_tools=ALLOWED_TOOLS)
if TERMINAL_TOOL_NAME != SCHEMA_TERMINAL_TOOL_NAME:
    raise ValueError(
        "training_policy_defaults terminal_tool is inconsistent with schema authority: "
        f"{TERMINAL_TOOL_NAME!r} != {SCHEMA_TERMINAL_TOOL_NAME!r}"
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
