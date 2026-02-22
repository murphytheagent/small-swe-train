from __future__ import annotations

import pytest

import config
from prompts.chat_contract import build_assistant_contract_prompt
from schemas import ALLOWED_TOOLS, TERMINAL_TOOL_NAME as SCHEMA_TERMINAL_TOOL_NAME


def test_terminal_tool_is_supported_by_schema() -> None:
    assert config.TERMINAL_TOOL_NAME in ALLOWED_TOOLS
    assert config.TERMINAL_TOOL_NAME == SCHEMA_TERMINAL_TOOL_NAME


def test_output_contract_exports_match_runtime_defaults() -> None:
    output_contract = config.output_contract_defaults()

    assert config.MIN_TOOL_CALLS_PER_TURN == int(output_contract["min_tool_calls_per_turn"])
    assert config.MAX_TOOL_CALLS_PER_TURN == int(output_contract["max_tool_calls_per_turn"])
    assert config.TERMINAL_TOOL_NAME == str(output_contract["terminal_tool"]).strip().lower()
    assert config.SUBMIT_MUST_BE_ONLY_TOOL_CALL is bool(output_contract["submit_must_be_only_tool_call"])


def test_phase_transition_gates_defaults_load() -> None:
    gates = config.phase_transition_gates_defaults()
    assert "entry_gate_for_main_sdpo" in gates


def test_tool_call_bounds_are_valid() -> None:
    assert config.MIN_TOOL_CALLS_PER_TURN >= 1
    assert config.MAX_TOOL_CALLS_PER_TURN >= config.MIN_TOOL_CALLS_PER_TURN


def test_prompt_contract_uses_centralized_terminal_tool_default() -> None:
    prompt = build_assistant_contract_prompt()
    assert f"Terminal tool is '{config.TERMINAL_TOOL_NAME}'" in prompt


def test_terminal_tool_validator_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Invalid terminal tool"):
        config._validate_terminal_tool_name("not-a-tool", allowed_tools=ALLOWED_TOOLS)
