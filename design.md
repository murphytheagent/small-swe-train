# small-swe-train: Research Mode v1.7 Design Revision Packet (on `main` branch)

Generated: 2026-02-21 06:24 UTC (original), updated 2026-02-21
Thread: 1771579678.414229
Supersedes: v1.6 at this same path

## 1) Chat contract for Qwen3-4B

### 1.1 Decision
- ChatML-style turn framing for Qwen3-4B.
- Turn boundaries: `<|im_start|>{role}` / `<|im_end|>`.

### 1.2 Intra-turn delimiters (assistant payload)
1. Optional thinking block: `<think> ... </think>`
2. One or more tool-call blocks: `<tool_call>{"tool":"...","args":{...}}</tool_call>`
3. Tool/environment feedback: `<tool_response> ... </tool_response>`

### 1.3 Implementation status
- **All delimiter strings are config-driven.** Loaded from YAML via `ModelDelimiters` dataclass.
- Bundled config: `src/prompts/model_configs/qwen3.yaml` (single source of truth).
- `load_delimiters(path)` for custom model families; `default_delimiters()` with `lru_cache` for default.
- `TurnParser` class accepts `ModelDelimiters`; module-level convenience functions use the default.
- Legacy constants (`CHATML_START`, etc.) are derived from the config, not hardcoded.

## 2) One tool vs multiple tools per turn

### 2.1 Policy
- Allow `1..M` tool calls per assistant turn (`M=3` default, configurable).
- Calls are executed sequentially in listed order.
- If `submit` appears, it must be the only call in that turn (terminal).

### 2.2 Implementation status
- Enforced by `ActionEnvelope.__post_init__` and `TurnParser`.

## 3) Tool set and terminal action

### 3.1 Canonical tools
`bash`, `search`, `edit`, `submit` — defined in `ALLOWED_TOOLS` ordered tuple.

### 3.2 Tool schema registry
- `TOOL_SCHEMAS` in `contracts.py` maps each tool to its TypedDict, required fields, and constraints.
- `validate_tool_call()` checks parsed `ToolCall` instances against the registry (required args, unknown args, types, min_length, min/max range).
- **JSON Schema files removed.** Validation is Python-native.

### 3.3 Legacy alias handling
- `answer` → `submit` during canonicalization.
- `str_replace_editor` subcommands mapped via `tool_schema_adapter.py`.

## 4) Default step-SDPO teacher prompt construction per turn

At turn `t`, teacher prompt is:
1. `SYSTEM_BLOCK` — role, tool schema, delimiter contract, execution policy.
2. `TASK_BLOCK` — issue statement, constraints, success condition.
3. `TRAJECTORY_BLOCK` — `RECENT_RAW_BLOCK` + `COMPRESSED_MEMORY_BLOCK` + `CRITICAL_FACTS_BLOCK`.
4. `CURRENT_ATTEMPT_BLOCK` — student attempt (included by default, flag retained for ablation).
5. `FEEDBACK_BLOCK` — canonicalized env feedback packet.
6. `OUTPUT_CONTRACT_BLOCK` — optional `<think>` plus one-or-more `<tool_call>` blocks.

### Implementation status
- `teacher/prompt_builder.py` — `TeacherPromptInputs` dataclass + `build_teacher_prompt()` implemented.
- Blocks are composed in design order.

## 5) Self-containment checks and canonicalization

### 5.1 Programmatic checks
Computed from canonical feedback fields: `has_failing_artifact_identity`, `has_actionable_error_text`, `has_localization_hint`.

### 5.2 Policy
- `include_student_attempt_for_teacher` defaults to `true` (not derived from self-containment in v1.6+).
- Flag retained in schema for future ablation.

### 5.3 Implementation status
- `data/feedback_canonicalizer.py` — fully implemented: `normalize_text`, `truncate_head_tail_tokens`, artifact/error/hint extraction, `build_feedback_packet()`.
- Tested via `test_feedback_canonicalizer.py`.

## 6) Stage-specific token masking policy

| Stage | Think tokens | Tool-call tokens |
|-------|-------------|-----------------|
| RFT   | excluded    | included        |
| step-SDPO | included | included       |

### Implementation status
- `losses/action_masking.py` — `should_train_token()` + `build_action_token_mask()` implemented and tested.

## 7) Tool schema alignment with SWE-bench / SWE-smith

### 7.1 Adapter mapping (deterministic)
- `bash` → `bash`
- `str_replace_editor.view` → `search`
- `str_replace_editor.create|str_replace|insert|undo_edit` → `edit`
- `submit`/`answer` → `submit`

### 7.2 Implementation status
- `data/tool_schema_adapter.py` — `adapt_external_tool_call()` implemented and tested.

## 8) Directory layout

```text
small-swe-train/
  IMPLEMENTATION_BLUEPRINT.md    # architecture, data flow, milestones
  pyproject.toml
  configs/
    runtime/
      training_policy_defaults.v1.json
      phase_transition_gates.v1.json
    verl/                        # verl/SDPO training configs
      sdpo_swe.yaml              # step-SDPO (main objective)
      rft_swe.yaml               # RFT supervised pre-training
      user.yaml                  # user-local path overrides
    data/
    experiments/
  src/
    schemas/
      contracts.py            # TOOL_SCHEMAS, validate_tool_call, data types
    prompts/
      model_delimiters.py      # ModelDelimiters, load_delimiters
      chat_contract.py         # build_assistant_contract_prompt
      model_configs/
        qwen3.yaml             # bundled delimiter config
    data/
      feedback_canonicalizer.py
      tool_schema_adapter.py
    env/
      runtime_protocol.py      # ToolRequest, ToolResponse, EnvironmentStep
    rollout/
      turn_parser.py           # TurnParser class
    trainer/
      sdpo_trainer.py          # STUB — SDPOTrainerScaffold
    losses/
      action_masking.py
    teacher/
      prompt_builder.py
    metrics/
      contracts.py             # FormatMetrics, rate()
    eval/
      swebench_lite.py         # STUB — EpisodeResult
    verl_integration/            # adapter layer: our modules ↔ verl
      reward_function.py         # PLANNED — verl reward_fn
      reprompt_adapter.py        # PLANNED — 6-block teacher prompt
      mask_injector.py           # PLANNED — stage-aware response_mask
      env_bridge.py              # PLANNED — Docker sandbox bridge
      data_preprocessor.py       # PLANNED — SWE trajectories → parquet
  scripts/
    prepare_rft_data.py
    run_rft.sh
    run_sdft.sh
    run_sdpo.sh
    eval_swebench_lite.sh
  tests/
    test_action_masking.py
    test_feedback_canonicalizer.py
    test_tool_schema_adapter.py
    test_turn_parser.py
```

## 9) Deeper adaptation notes for `lasgroup/SDPO`

### 9.1 Reuse points
- `verl/trainer/ppo/ray_trainer.py` for teacher-batch assembly and reprompt path.
- `verl/workers/actor/dp_actor.py` for EMA teacher update + distillation logit path.
- `verl/trainer/ppo/core_algos.py` for `compute_self_distillation_loss` and top-k distillation.

### 9.2 Required custom changes
1. Parser supports multi-tool-call turns with ordered block extraction.
2. Loss-mask builder supports stage-specific think-token behavior.
3. Canonicalization requires `actionable_error_text` key presence.
4. Teacher prompt builder always includes student attempt by default.
5. Terminal tool is `submit` (not `answer`).

## 10) Implementation progress

### Fully implemented (infrastructure layer)
| Module | Key exports | Tests |
|--------|------------|-------|
| `schemas/contracts.py` | `TOOL_SCHEMAS`, `validate_tool_call`, `ToolCall`, `ActionEnvelope`, `FeedbackPacket` | via adapter/parser tests |
| `prompts/model_delimiters.py` | `ModelDelimiters`, `load_delimiters`, `default_delimiters` | via turn_parser tests |
| `prompts/chat_contract.py` | `build_assistant_contract_prompt` | — |
| `data/feedback_canonicalizer.py` | `canonicalize_tool_feedback`, `build_feedback_packet` | `test_feedback_canonicalizer.py` |
| `data/tool_schema_adapter.py` | `adapt_external_tool_call` | `test_tool_schema_adapter.py` |
| `rollout/turn_parser.py` | `TurnParser`, `parse_chatml_assistant_turn` | `test_turn_parser.py` |
| `losses/action_masking.py` | `build_action_token_mask` | `test_action_masking.py` |
| `teacher/prompt_builder.py` | `build_teacher_prompt` | — |
| `env/runtime_protocol.py` | `ToolRequest`, `ToolResponse`, `EnvironmentStep` | — |
| `metrics/contracts.py` | `FormatMetrics`, `rate` | — |

### Stubs (interfaces defined, implementation pending)
| Module | What's missing |
|--------|---------------|
| `trainer/sdpo_trainer.py` | `run_rft_epoch`, `run_sdpo_step` raise `NotImplementedError` |
| `eval/swebench_lite.py` | `evaluate_swebench_lite` raises `NotImplementedError` |

### Not started
| Component | Description | Prerequisite for |
|-----------|-------------|-----------------|
| **Data ingestion pipeline** | Read SWE trajectories → tokenized training examples using adapter + parser + canonicalizer | RFT training |
| **Tokenization bridge** | Map `ActionEnvelope` / `EnvironmentStep` to token IDs + label masks | RFT training |
| **RFT training loop** | Model loading, supervised training on formatted tool calls | SDPO entry gate |
| **Environment executor** | Docker sandbox that runs tool calls, returns `ToolResponse` | On-policy rollouts |
| **SDFT stage** | Demo-conditioned self-distillation (optional pre-SDPO) | — |
| **Step-SDPO loop** | On-policy rollout → teacher prompt → KL distillation | Main objective |
| **Evaluation harness** | Run agent on SWE-bench Lite, score patches | Measuring progress |

## 11) Recommended next steps (priority order)

1. **Data ingestion pipeline** — stitch `tool_schema_adapter` + `turn_parser` + `feedback_canonicalizer` into a script that reads raw SWE trajectories (SWE-smith / SWE-bench format) and outputs tokenized training records. This is the prerequisite for all training.

2. **Tokenization bridge** — map `ActionEnvelope` + `EnvironmentStep` sequences to token ID tensors + per-token label masks (using `build_action_token_mask`). Requires choosing tokenizer (Qwen3-4B) and sequence format.

3. **RFT training loop** — implement `run_rft_epoch` using the tokenized data. Simple supervised cross-entropy on masked tokens. This teaches the model correct tool-call format.

4. **Environment executor** — Docker-based sandbox for `bash`/`search`/`edit` execution. Needed before on-policy SDPO can run.

5. **Step-SDPO loop** — the main training objective. Depends on (3) for format quality gates and (4) for on-policy rollouts.

## 12) Training infrastructure decision

### 12.1 Framework
- **Trainer & rollout**: `lasgroup/SDPO` (a fork of [verl](https://github.com/verl-project/verl)) used as-is.
- **Rollout engine**: vLLM (via verl's colocated rollout worker).
- **Training engine**: FSDP `FULL_SHARD` across 8 GPUs (via verl's `DataParallelPPOActor`).
- FSDP is needed not for model size (Qwen3-4B ≈ 8 GB bf16) but for **activation memory headroom** with long SWE-bench trajectories (8K–16K tokens).

### 12.2 Hardware target
- Single node, 8× A100 or H100 GPUs (80 GB each).
- GPUs alternate between vLLM inference and FSDP training each global step.

### 12.3 Config files
- `configs/verl/sdpo_swe.yaml` — step-SDPO training (main objective).
- `configs/verl/rft_swe.yaml` — RFT supervised pre-training.
- `configs/verl/user.yaml` — user-local path overrides.

### 12.4 Integration layer
- New package `src/verl_integration/` bridges our protocol modules with verl hooks.
- Full architecture, data flow, and milestone plan in `IMPLEMENTATION_BLUEPRINT.md`.

## 13) Sources
- Qwen3 tokenizer chat template: https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json
- SDPO baseline: https://github.com/lasgroup/SDPO
- verl framework: https://github.com/verl-project/verl
- Thread review: https://github.com/murphytheagent/small-swe-train/pull/2#discussion_r2835868321
- Implementation blueprint: `IMPLEMENTATION_BLUEPRINT.md` (this repo)
