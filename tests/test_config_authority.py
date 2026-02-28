from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

import pytest
import yaml

import config
from prompts.runtime_messages import build_assistant_contract_prompt
from prompts.teacher_messages import build_teacher_output_contract_block
from schemas import ALLOWED_TOOLS, TERMINAL_TOOL_NAME as SCHEMA_TERMINAL_TOOL_NAME, TOOL_SCHEMAS


def test_terminal_tool_is_supported_by_schema() -> None:
    assert config.TERMINAL_TOOL_NAME in ALLOWED_TOOLS
    assert config.TERMINAL_TOOL_NAME == SCHEMA_TERMINAL_TOOL_NAME


def test_output_contract_exports_match_runtime_defaults() -> None:
    output_contract = config.output_contract_defaults()

    assert config.MIN_TOOL_CALLS_PER_TURN == int(output_contract["min_tool_calls_per_turn"])
    assert config.MAX_TOOL_CALLS_PER_TURN == int(output_contract["max_tool_calls_per_turn"])
    assert config.TERMINAL_TOOL_NAME == str(output_contract["terminal_tool"]).strip().lower()
    assert config.SUBMIT_MUST_BE_ONLY_TOOL_CALL is bool(output_contract["submit_must_be_only_tool_call"])
    assert config.TERMINAL_VALIDITY_PENALTY == float(output_contract.get("terminal_validity_penalty", 0.2))


def test_default_training_model_name_is_loaded_from_shared_verl_config() -> None:
    defaults = config.verl_model_defaults()
    assert defaults["model_defaults"]["primary_name"] == config.DEFAULT_TRAINING_MODEL_NAME
    assert isinstance(config.DEFAULT_TRAINING_MODEL_NAME, str)
    assert config.DEFAULT_TRAINING_MODEL_NAME.strip()


def test_sdpo_task_cache_default_is_centralized() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert config.DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR == Path("task") / "sdpo_task_cache"
    assert config.resolve_sdpo_task_cache_dir(project_root=repo_root) == repo_root / "task" / "sdpo_task_cache"


def test_phase_transition_gates_defaults_load() -> None:
    gates = config.phase_transition_gates_defaults()
    assert "entry_gate_for_main_sdpo" in gates
    assert "terminal_submission_rate_min" in gates["entry_gate_for_main_sdpo"]


def test_tool_call_bounds_are_valid() -> None:
    assert config.MIN_TOOL_CALLS_PER_TURN >= 1
    assert config.MAX_TOOL_CALLS_PER_TURN >= config.MIN_TOOL_CALLS_PER_TURN


def test_prompt_contract_uses_centralized_terminal_tool_default() -> None:
    prompt = build_assistant_contract_prompt()
    assert f"Terminal tool is '{config.TERMINAL_TOOL_NAME}'" in prompt
    assert f"4) Allowed tools: {', '.join(ALLOWED_TOOLS)}." in prompt
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
    assert f"5) Required args by tool: {required_args_text}." in prompt


def test_teacher_output_contract_block_wraps_shared_contract() -> None:
    prompt = build_teacher_output_contract_block()
    assert "Teacher objective (turn-level SDPO):" in prompt
    assert "Assistant output contract:" in prompt
    assert f"Terminal tool is '{config.TERMINAL_TOOL_NAME}'" in prompt
    assert "Do not repeat an identical previously-failed command without a new hypothesis." in prompt


def test_prompt_contract_renders_tool_examples_from_tool_schemas() -> None:
    prompt = build_assistant_contract_prompt()
    assert "9) Realistic examples (one tool call each):" in prompt
    for tool_name in ALLOWED_TOOLS:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not isinstance(schema, Mapping):
            continue
        example_raw = schema.get("prompt_example")
        if isinstance(example_raw, Mapping):
            serialized = json.dumps(dict(example_raw), ensure_ascii=True, sort_keys=True)
            assert f"   - {tool_name}: {serialized}" in prompt


def test_prompt_contract_schema_text_is_rendered_from_tool_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    search_schema = dict(TOOL_SCHEMAS["search"])
    constraints = dict(search_schema["constraints"])
    constraints["query"] = {"min_length": 7}
    search_schema["constraints"] = constraints
    monkeypatch.setitem(TOOL_SCHEMAS, "search", search_schema)

    prompt = build_assistant_contract_prompt()
    assert "search args: required {query:str(min_len=7)}" in prompt


def test_terminal_tool_validator_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Invalid terminal tool"):
        config._validate_terminal_tool_name("not-a-tool", allowed_tools=ALLOWED_TOOLS)


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
    columns = data_defaults.get("columns")
    assert isinstance(columns, Mapping)
    for key in ("image_name", "problem_statement", "fail_to_pass", "pass_to_pass"):
        assert isinstance(columns.get(key), str)
        assert str(columns[key]).strip()


def test_sdpo_config_disables_runtime_prompt_length_filter() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((repo_root / "configs/verl/sdpo_swe.yaml").read_text(encoding="utf-8"))
    data_cfg = payload["data"]
    assert data_cfg["filter_overlong_prompts"] is False


def test_resolve_on_policy_settings_merges_data_and_runtime_sources() -> None:
    settings = config.resolve_on_policy_settings()
    data_defaults = config.on_policy_data_defaults()
    runtime_defaults = config.rft_runtime_defaults()
    loop_defaults = runtime_defaults["loop"]
    assert settings.data.dataset_id == data_defaults["dataset_id"]
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
