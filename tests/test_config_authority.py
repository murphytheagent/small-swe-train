from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

import pytest
import yaml

import config
from prompts.runtime_messages import (
    build_assistant_contract_prompt,
    build_onpolicy_system_prompt,
    build_sdpo_rollout_followup_user_message,
)
from prompts.teacher_messages import build_teacher_output_contract_block
from schemas import (
    ALLOWED_TOOLS,
    FILE_SEARCH_TOOL_NAME,
    TERMINAL_TOOL_NAME as SCHEMA_TERMINAL_TOOL_NAME,
    TEXT_SEARCH_TOOL_NAME,
    TOOL_SCHEMAS,
    ToolCall,
    validate_tool_call,
)


def test_terminal_tool_is_supported_by_schema() -> None:
    assert config.TERMINAL_TOOL_NAME in ALLOWED_TOOLS
    assert config.TERMINAL_TOOL_NAME == SCHEMA_TERMINAL_TOOL_NAME
    assert FILE_SEARCH_TOOL_NAME == "file_search"
    assert TEXT_SEARCH_TOOL_NAME == "text_search"


def test_output_contract_exports_match_runtime_defaults() -> None:
    output_contract = config.output_contract_defaults()

    assert config.MIN_TOOL_CALLS_PER_TURN == int(output_contract["min_tool_calls_per_turn"])
    assert config.MAX_TOOL_CALLS_PER_TURN == int(output_contract["max_tool_calls_per_turn"])
    assert config.TERMINAL_TOOL_NAME == str(output_contract["terminal_tool"]).strip().lower()
    assert config.SUBMIT_MUST_BE_ONLY_TOOL_CALL is bool(output_contract["submit_must_be_only_tool_call"])
    assert config.TERMINAL_VALIDITY_PENALTY == float(output_contract.get("terminal_validity_penalty", 0.2))
    assert config.ACTION_PAYLOAD_FORMAT == str(output_contract.get("action_payload_format", "json")).strip().lower()
    assert config.ACTION_PARSE_MODE == str(output_contract.get("action_parse_mode", "json_only")).strip().lower()


def test_default_training_model_name_is_loaded_from_shared_verl_config() -> None:
    defaults = config.verl_model_defaults()
    assert defaults["model_defaults"]["primary_name"] == config.DEFAULT_TRAINING_MODEL_NAME
    assert isinstance(config.DEFAULT_TRAINING_MODEL_NAME, str)
    assert config.DEFAULT_TRAINING_MODEL_NAME.strip()


def test_sdpo_task_cache_default_is_centralized() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert config.DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR == Path("data") / "sdpo_task_cache"
    assert config.resolve_sdpo_task_cache_dir(project_root=repo_root) == repo_root / "data" / "sdpo_task_cache"


def test_on_policy_bad_task_cache_default_is_centralized() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert config.DEFAULT_ON_POLICY_BAD_TASK_CACHE_RELATIVE_DIR == Path("data") / "on_policy_bad_task_cache"
    assert config.resolve_on_policy_bad_task_cache_dir(project_root=repo_root) == repo_root / "data" / "on_policy_bad_task_cache"


def test_phase_transition_gates_defaults_load() -> None:
    gates = config.phase_transition_gates_defaults()
    assert "entry_gate_for_main_sdpo" in gates
    assert "terminal_submission_rate_min" in gates["entry_gate_for_main_sdpo"]
    assert "thinking_delimiter_balance_rate_min" not in gates["entry_gate_for_main_sdpo"]


def test_tool_call_bounds_are_valid() -> None:
    assert config.MIN_TOOL_CALLS_PER_TURN >= 1
    assert config.MAX_TOOL_CALLS_PER_TURN >= config.MIN_TOOL_CALLS_PER_TURN


def test_prompt_contract_uses_centralized_terminal_tool_default() -> None:
    prompt = build_assistant_contract_prompt()
    assert f"Terminal tool is '{config.TERMINAL_TOOL_NAME}'" in prompt
    assert f"Allowed tools: {', '.join(ALLOWED_TOOLS)}." in prompt
    required_fields: list[str] = []
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not isinstance(schema, dict):
            continue
        required_raw = schema.get("required")
        if not isinstance(required_raw, (list, tuple, set)):
            continue
        for field_name in required_raw:
            if isinstance(field_name, str):
                required_fields.append(f"{tool_name}.{field_name}")
    required_args_text = ", ".join(required_fields) if required_fields else "-"
    assert f"Required args by tool: {required_args_text}." in prompt
    assert "read" in ALLOWED_TOOLS
    assert "read" in TOOL_SCHEMAS
    assert "file_search" in ALLOWED_TOOLS
    assert "text_search" in ALLOWED_TOOLS
    assert "file_search" in TOOL_SCHEMAS
    assert "text_search" in TOOL_SCHEMAS
    assert "search" not in ALLOWED_TOOLS


def test_prompt_contract_includes_read_and_direct_tool_call_rule() -> None:
    prompt = build_assistant_contract_prompt()

    assert "Begin with a tool-call block. Do not emit prose before the first tool call." in prompt
    assert "read args: required {path:str" in prompt
    assert "file_search args: required {query:str" in prompt
    assert "text_search args: required {query:str" in prompt
    assert "start_line:int" in prompt
    assert "end_line:int" in prompt
    assert "read.path" in prompt
    assert "Tool usage guardrails:" in prompt
    assert "Use 'file_search' to discover likely repo-relative file paths" in prompt
    assert "Use 'text_search' for exact fixed-string matches" in prompt
    assert "both 'path' and 'patch' in normal use." in prompt
    assert "Prefer using 'apply_patch' for file edits instead of bash heredocs, cat >, or sed -i." in prompt
    assert "use 'search'" not in prompt


def test_onpolicy_system_prompt_requires_repo_driven_validation_workflow() -> None:
    prompt = build_onpolicy_system_prompt()
    assert "First inspect surrounding code, related tests, and project configuration" in prompt
    assert "understand the root cause and broader context before editing" in prompt
    assert "do not wait for complete understanding before making progress." in prompt
    assert "Make focused, minimal changes that address the issue" in prompt
    assert "Use the bash tool to run repository-specific validation commands." in prompt
    assert "Do not assume a standard test command; infer the correct commands from this codebase." in prompt
    assert "If applicable and feasible, verify your changes with the repository's own tests, build, lint" in prompt
    assert "Start with the most specific relevant validation" in prompt
    assert "broaden to more general checks as confidence grows." in prompt
    assert "run a lightweight related validation instead of skipping validation entirely." in prompt
    assert "If the environment prevents validation, state that limitation briefly" in prompt
    assert "FAIL_TO_PASS" not in prompt
    assert "PASS_TO_PASS" not in prompt


def test_teacher_output_contract_block_wraps_shared_contract() -> None:
    prompt = build_teacher_output_contract_block()
    assert (
        "Now that you have seen the student's attempt, adhere to following contracts in your revised attempt:"
        in prompt
    )
    assert "Assistant output contract:" in prompt
    assert f"Terminal tool is '{config.TERMINAL_TOOL_NAME}'" in prompt
    assert "Teacher-specific tool guidance:" in prompt
    assert "Normal tool flow: use file_search to locate likely files" in prompt
    assert "use text_search to locate exact strings or symbols inside a known scope" in prompt
    assert "always include both args.path and args.patch" in prompt
    assert "You may reuse an exact repo-relative path the student already found" in prompt
    assert "Now correctly solve the original issue, focus only on what to do best in the next turn." in prompt
    assert "Do not repeat an identical previously-failed command without a new hypothesis." not in prompt
    assert "ls/search" not in prompt


def test_prompt_contract_renders_tool_examples_from_tool_schemas() -> None:
    prompt = build_assistant_contract_prompt(action_payload_format="json")
    assert "Realistic examples (one tool call each):" in prompt
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not isinstance(schema, Mapping):
            continue
        example_raw = schema.get("prompt_example")
        if isinstance(example_raw, Mapping):
            serialized = json.dumps(dict(example_raw), ensure_ascii=True, sort_keys=True)
            assert f"   - {tool_name}: {serialized}" in prompt


def test_default_system_prompt_uses_centralized_action_payload_format() -> None:
    prompt = build_onpolicy_system_prompt()

    if config.ACTION_PAYLOAD_FORMAT == "xml":
        assert '<tool_call name="tool_name"><arg_name><![CDATA[value]]></arg_name></tool_call>' in prompt
        assert "Use CDATA for string-valued args" in prompt
    else:
        assert '<tool_call>{"tool":"...","args":{...}}</tool_call>' in prompt


def test_prompt_contract_schema_text_is_rendered_from_tool_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    search_schema = dict(TOOL_SCHEMAS["file_search"])
    constraints = dict(search_schema["constraints"])
    constraints["query"] = {"min_length": 7}
    search_schema["constraints"] = constraints
    monkeypatch.setitem(TOOL_SCHEMAS, "file_search", search_schema)

    prompt = build_assistant_contract_prompt()
    assert "file_search args: required {query:str(min_len=7)}" in prompt


def test_prompt_contract_supports_xml_payload_mode() -> None:
    prompt = build_assistant_contract_prompt(action_payload_format="xml")

    assert '<tool_call name="tool_name"><arg_name><![CDATA[value]]></arg_name></tool_call>' in prompt
    assert "Every XML tool call MUST use a 'name' attribute for the tool and direct child elements for args." in prompt
    assert "Do not add an <args> wrapper." in prompt
    assert "Use CDATA for string-valued args" in prompt
    assert "<changed_paths><path><![CDATA[src/app.py]]></path></changed_paths>" in prompt
    assert "Do not emit namespaces, DTDs, processing instructions, extra attributes, or mixed JSON/XML payloads." in prompt
    assert '   - bash: <tool_call name="bash"><command><![CDATA[make test-target]]></command><cwd><![CDATA[.]]></cwd><timeout_sec>120</timeout_sec></tool_call>' in prompt


def test_teacher_output_contract_block_supports_xml_payload_mode() -> None:
    prompt = build_teacher_output_contract_block(action_payload_format="xml")

    assert "Assistant output contract:" in prompt
    assert '<tool_call name="tool_name"><arg_name><![CDATA[value]]></arg_name></tool_call>' in prompt
    assert "Teacher-specific tool guidance:" in prompt


def test_prompt_contract_examples_are_environment_neutral() -> None:
    prompt = build_assistant_contract_prompt()

    assert "/workspace/project" not in prompt
    assert "python -m pytest -q" not in prompt


def test_read_cross_field_validation_rejects_descending_range() -> None:
    errors = validate_tool_call(
        ToolCall(tool="read", args={"path": "src/app.py", "start_line": 10, "end_line": 9})
    )

    assert "Arg 'end_line': must be >= start_line" in errors


def test_sdpo_followup_message_uses_canonical_tool_names() -> None:
    message = build_sdpo_rollout_followup_user_message()

    assert "read" in message
    assert "file_search" in message
    assert "text_search" in message
    assert "apply_patch" in message
    assert "edit" not in message


def test_terminal_tool_validator_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Invalid terminal tool"):
        config._validate_terminal_tool_name("not-a-tool", allowed_tools=ALLOWED_TOOLS)


def test_action_format_contract_rejects_incompatible_payload_and_parse_modes() -> None:
    with pytest.raises(ValueError, match="JSON output cannot use xml_only parsing"):
        config._validate_action_format_contract(payload_format="json", parse_mode="xml_only")

    with pytest.raises(ValueError, match="XML output cannot use json_only parsing"):
        config._validate_action_format_contract(payload_format="xml", parse_mode="json_only")


def test_on_policy_runtime_defaults_load_from_central_json() -> None:
    on_policy = config.on_policy_runtime_defaults()
    assert on_policy["max_turns_per_attempt"] >= 1
    assert on_policy["tool_timeout_sec"] >= 1
    assert "task_batch_size" not in on_policy
    assert "attempts_per_task" not in on_policy
    assert "env_pool_size" not in on_policy
    assert "max_in_flight_tasks" not in on_policy


def test_adaptation_defaults_load_from_central_json() -> None:
    adaptation = config.adaptation_defaults()
    assert isinstance(adaptation.get("mode"), str)
    assert str(adaptation["mode"]).strip()
    assert isinstance(adaptation.get("compute_precision"), str)
    assert str(adaptation["compute_precision"]).strip()
    target_modules = adaptation.get("target_modules")
    assert isinstance(target_modules, list)
    assert target_modules
    assert all(isinstance(item, str) and item.strip() for item in target_modules)


@pytest.mark.parametrize(
    "config_relpath",
    [
        "configs/verl/rft_swe.yaml",
        "configs/verl/sdpo_swe.yaml",
    ],
)
def test_verl_model_config_mirrors_lora_rank_alpha_and_targets(
    config_relpath: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((repo_root / config_relpath).read_text(encoding="utf-8"))
    adaptation_defaults = config.adaptation_defaults()
    expected_targets = ".*(" + "|".join(str(item) for item in adaptation_defaults["target_modules"]) + ")$"
    model_cfg = payload["actor_rollout_ref"]["model"]
    lora_cfg = model_cfg["lora"]

    assert isinstance(model_cfg["lora_rank"], int)
    assert model_cfg["lora_rank"] >= 1
    assert isinstance(model_cfg["lora_alpha"], int)
    assert model_cfg["lora_alpha"] >= 1
    assert model_cfg["target_modules"] == expected_targets
    assert lora_cfg["rank"] == model_cfg["lora_rank"]
    assert lora_cfg["alpha"] == model_cfg["lora_alpha"]


def test_on_policy_data_defaults_load_from_configs_data() -> None:
    data_defaults = config.on_policy_data_defaults()
    assert isinstance(data_defaults.get("dataset_id"), str)
    assert str(data_defaults["dataset_id"]).strip()
    assert data_defaults["patch_is_bug_introducing"] is True
    assert data_defaults["verifier_kind"] == "pytest"
    columns = data_defaults.get("columns")
    assert isinstance(columns, Mapping)
    for key in ("image_name", "problem_statement", "fail_to_pass", "pass_to_pass"):
        assert isinstance(columns.get(key), str)
        assert str(columns[key]).strip()


def test_on_policy_go_data_defaults_load_from_configs_data() -> None:
    data_defaults = config.on_policy_data_defaults("on_policy_swe_smith_go")
    assert data_defaults["dataset_id"] == "SWE-bench/SWE-smith-go"
    assert data_defaults["patch_is_bug_introducing"] is True
    assert data_defaults["verifier_kind"] == "go_test"


def test_sdpo_config_disables_runtime_prompt_length_filter() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((repo_root / "configs/verl/sdpo_swe.yaml").read_text(encoding="utf-8"))
    data_cfg = payload["data"]
    assert data_cfg["filter_overlong_prompts"] is False


def test_verl_configs_fallback_project_root_when_env_unset() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sdpo_payload = yaml.safe_load((repo_root / "configs/verl/sdpo_swe.yaml").read_text(encoding="utf-8"))
    rft_payload = yaml.safe_load((repo_root / "configs/verl/rft_swe.yaml").read_text(encoding="utf-8"))

    assert "${oc.env:PROJECT_ROOT," in sdpo_payload["trainer"]["default_local_dir"]
    assert "${oc.env:PROJECT_ROOT," in sdpo_payload["custom_reward_function"]["path"]
    assert "${oc.env:PROJECT_ROOT," in rft_payload["trainer"]["default_local_dir"]


def test_resolve_on_policy_settings_merges_data_and_runtime_sources() -> None:
    settings = config.resolve_on_policy_settings()
    data_defaults = config.on_policy_data_defaults()
    runtime_defaults = config.rft_runtime_defaults()
    loop_defaults = runtime_defaults["loop"]
    assert settings.data.dataset_id == data_defaults["dataset_id"]
    assert settings.data.patch_is_bug_introducing is True
    assert settings.data.verifier_kind == "pytest"
    assert settings.runtime.task_batch_size == loop_defaults["task_batch_size"]
    assert settings.runtime.attempts_per_task == loop_defaults["samples_per_task"]
    assert settings.runtime.env_pool_size == settings.runtime.task_batch_size
    assert settings.runtime.max_tool_calls_per_turn <= config.MAX_TOOL_CALLS_PER_TURN
    assert settings.runtime.max_in_flight_tasks == config.resolve_rft_collector_max_in_flight_default(
        task_batch_size=settings.runtime.task_batch_size
    )


def test_resolve_on_policy_settings_aligns_in_flight_with_task_batch_override() -> None:
    settings = config.resolve_on_policy_settings(
        runtime_overrides={
            "task_batch_size": 8,
        },
    )
    assert settings.runtime.task_batch_size == 8
    assert settings.runtime.env_pool_size == 8
    assert settings.runtime.max_in_flight_tasks == 8


def test_resolve_on_policy_settings_respects_explicit_in_flight_override() -> None:
    settings = config.resolve_on_policy_settings(
        runtime_overrides={
            "task_batch_size": 8,
            "max_in_flight_tasks": 3,
        },
    )
    assert settings.runtime.task_batch_size == 8
    assert settings.runtime.env_pool_size == 8
    assert settings.runtime.max_in_flight_tasks == 3


def test_rft_runtime_defaults_load_loop_and_vllm_config() -> None:
    runtime_defaults = config.rft_runtime_defaults()
    assert runtime_defaults["loop"]["steps"] >= 1
    assert runtime_defaults["loop"]["samples_per_task"] >= 1
    assert runtime_defaults["loop"]["task_batch_size"] >= 1
    assert runtime_defaults["loop"]["collector_max_in_flight_tasks"] >= 1
    assert runtime_defaults["loop"]["train_batch_size"] >= 1
    assert 0.0 <= float(runtime_defaults["loop"]["eval_split_fraction"]) < 1.0
    assert runtime_defaults["loop"]["eval_min_rows"] >= 1
    assert runtime_defaults["loop"]["checkpoint_keep_last"] >= 1
    assert runtime_defaults["vllm"]["base_url"].startswith("http://")
    assert runtime_defaults["vllm_parallelism"]["default_tensor_parallel_size"] >= 1


def test_resolve_rft_collector_max_in_flight_default_clamps_to_task_batch_size() -> None:
    runtime_defaults = config.rft_runtime_defaults()
    configured = runtime_defaults["loop"].get("collector_max_in_flight_tasks")
    configured_int = int(configured) if isinstance(configured, int) else None

    for task_batch_size in (64, 16):
        resolved = config.resolve_rft_collector_max_in_flight_default(task_batch_size=task_batch_size)
        assert 1 <= resolved <= task_batch_size
        if configured_int is None:
            assert resolved == task_batch_size
        else:
            assert resolved == min(configured_int, task_batch_size)


@pytest.mark.parametrize("nproc_per_node", [8, 4])
def test_resolve_rft_vllm_parallel_defaults_returns_valid_tp_dp_pair(nproc_per_node: int) -> None:
    tp, dp = config.resolve_rft_vllm_parallel_defaults(nproc_per_node=nproc_per_node)
    assert tp >= 1
    assert dp >= 1
    assert nproc_per_node % tp == 0
    assert dp <= (nproc_per_node // tp)


def test_resolve_rft_handoff_settings_loads_selection_policy() -> None:
    settings = config.resolve_rft_handoff_settings()
    defaults = config.rft_handoff_defaults()
    selection_defaults = defaults["selection"]
    assert settings.max_sequence_length >= 2
    assert settings.pad_token_id >= 0
    for key in (
        "require_format_valid",
        "require_terminal",
        "require_resolved",
        "require_zero_exit_code",
        "reject_on_collector_error",
        "reject_on_bridge_error",
        "reject_on_timeout_error",
        "reject_on_executor_error",
        "reject_on_parse_error",
        "reject_on_validation_errors",
        "reject_on_invalid_final_submit",
    ):
        assert getattr(settings.selection, key) is bool(selection_defaults[key])


def test_verl_integration_has_no_config_dataclass_definitions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    integration_dir = repo_root / "src" / "verl_integration"
    config_named_classes: list[str] = []
    for path in sorted(integration_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith("Config"):
                config_named_classes.append(f"{path.name}:{node.name}")
    assert config_named_classes == []
