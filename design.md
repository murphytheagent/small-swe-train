# small-swe-train: Research Mode v1.9 Design Revision Packet

Generated: 2026-02-21 06:24 UTC (original), updated 2026-02-22
Thread: 1771579678.414229
Supersedes: v1.8 at this same path

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
`bash`, `search`, `apply_patch`, `submit` — defined in `ALLOWED_TOOLS` ordered tuple.

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
- `str_replace_editor.create|str_replace|insert|undo_edit` → `apply_patch`
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
      tokenization.py          # shared offset-based tokenization + batch support
    env/
      runtime_protocol.py      # ToolRequest, ToolResponse, EnvironmentStep
    rollout/
      turn_parser.py           # TurnParser class
    trainer/
      sdpo_trainer.py          # SDPOTrainerScaffold (deterministic)
    losses/
      action_masking.py
    teacher/
      prompt_builder.py
    metrics/
      contracts.py             # FormatMetrics, rate()
    eval/
      swebench_lite.py         # EpisodeResult, summarize, compare
    verl_integration/            # adapter layer: our modules ↔ verl
      reward_function.py         # verl reward_fn (deterministic scaffold)
      reprompt_adapter.py        # 6-block teacher prompt assembly
      mask_injector.py           # stage-aware response_mask builder
      env_bridge.py              # multi-turn rollout ↔ executor bridge
      data_preprocessor.py       # rollout trajectories → verl-ready rows
  scripts/
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

SDPO launch ops note: on this machine, run `scripts/run_sdpo.sh` via Slurm with
`RAY_TMPDIR=/data/scratch/$USER/ray_tmp/$SLURM_JOB_ID`; if needed, clean stale
`/tmp/ray/session_*` only when no Ray daemons are running. The SDPO launcher also
defaults `TOKENIZERS_PARALLELISM=false` for Ray workers.

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
| `data/tokenization.py` | `tokenize_with_labels`, `tokenize_batch_with_labels`, `build_labeled_spans` | `test_tokenization.py` |
| `rollout/turn_parser.py` | `TurnParser`, `parse_chatml_assistant_turn` | `test_turn_parser.py` |
| `losses/action_masking.py` | `build_action_token_mask` | `test_action_masking.py` |
| `teacher/prompt_builder.py` | `build_teacher_prompt` | — |
| `env/runtime_protocol.py` | `ToolRequest`, `ToolResponse`, `EnvironmentStep` | — |
| `metrics/contracts.py` | `FormatMetrics`, `rate` | — |

### Implemented (verl integration scaffold — deterministic, no GPU)
| Module | Key exports | Tests |
|--------|------------|-------|
| `verl_integration/reward_function.py` | `reward_fn` — parse, validate, score, format metrics | `test_verl_reward_function.py` |
| `verl_integration/reprompt_adapter.py` | `build_self_distillation_batch` — 6-block teacher prompts + mask | `test_verl_reprompt_adapter.py` |
| `verl_integration/mask_injector.py` | `inject_response_mask` — stage-aware boolean masks | `test_verl_mask_injector.py` |
| `verl_integration/env_bridge.py` | `run_env_bridge_step` — parse + validate + dispatch | `test_verl_env_bridge.py` |
| `verl_integration/data_preprocessor.py` | `preprocess_trajectories` — rows + label blocks + approx masks | `test_verl_data_preprocessor.py` |
| `trainer/sdpo_trainer.py` | `SDPOTrainerScaffold` — deterministic end-to-end step | `test_sdpo_trainer.py` |
| `eval/swebench_lite.py` | `evaluate_swebench_lite`, `summarize_episode_results`, `compare_resolve_rates` | `test_swebench_lite.py` |

### Not started (require real model / GPU / Docker)
| Component | Description | Prerequisite for |
|-----------|-------------|-----------------|
| **Environment executor** | Docker sandbox implementing `ToolExecutor` protocol for live tool dispatch | RFT + SDPO rollouts |
| **RFT training loop** | On-policy: rollout N attempts per task in Docker, filter successful, train CE on masked tokens (LoRA) via verl | SDPO entry gate |
| **SDFT stage** | Demo-conditioned self-distillation (optional pre-SDPO) | — |
| **Step-SDPO loop** | On-policy rollout → teacher prompt → KL distillation via verl PPO trainer | Main objective |
| **Live evaluation harness** | Run agent on SWE-bench Lite with Docker sandboxes, score patches | Measuring progress |

## 11) Bug-fix log (v1.8, 2026-02-22)

Seven bugs were identified and fixed in the verl integration layer. All fixes
have regression tests (72 passed, 1 skipped).

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `env_bridge.py` | Submit early-return bypassed `validate_tool_call`, so malformed submit payloads (missing `final_response`) silently ended episodes | Validate submit call before terminal return; surface errors in `steps` and `tool_response_blocks` |
| 2 | `reprompt_adapter.py` | `has_teacher_signal` derived from `bool(feedback_block.strip())`; empty `tool_output` canonicalizes to `'{}'` which is truthy, flipping the distillation mask on for rows with no signal | Use `feedback_packet.self_containment_checks.has_actionable_error_text` instead |
| 3 | `reward_function.py` | `step_index` coercion failures were appended to `sample_errors` and gated the reward, zeroing out valid resolved samples with bad metadata | Separate `step_index_warnings` from format `validation_errors`; only format errors gate reward |
| 4 | `reprompt_adapter.py` | `_truncate_prompt_tokens` used `" ".join(tokens[:N])`, collapsing newlines into spaces and destroying the 6-block teacher prompt structure | Line-aware truncation: split by `\n`, count whitespace words per line, preserve newlines |
| 5 | `data_preprocessor.py` | `_labels_from_envelope` sized per-token masks via `len(text.split())` (whitespace words), producing masks that won't align with subword tokenizer output | Renamed to `_approx_labels_from_envelope` with warning; added `label_blocks` output with structured `{type, text}` block metadata for tokenizer-aligned mask generation |
| 6 | `data_preprocessor.py` | Non-string `thinking` field (e.g. int) passed to `ActionEnvelope` caused `AttributeError` on `.strip()` | Coerce `thinking` to `str(value)` when non-None and non-string |
| 7 | `rft_swe.yaml` / `sdpo_swe.yaml` | No LoRA configuration despite blueprint memory budget assuming LoRA (optimizer states ~0.05 GB vs ~8+ GB for full fine-tuning) | Added `lora:` block with `rank=64`, `alpha=128`, targets `q_proj/k_proj/v_proj/o_proj` |

## 12) Recommended next steps (priority order)

1. **Environment executor** — Implement a concrete `ToolExecutor` class backed by Docker containers. The `env_bridge.py` interface is stable; it needs a real executor behind `executor.run(request)` that dispatches `bash`/`search`/`apply_patch`/`submit` to per-instance containers. This is the prerequisite for both RFT and SDPO since both stages are on-policy.

2. **RFT training loop (on-policy)** — Roll out N attempts per SWE-bench task in Docker via `env_bridge.py`, filter to successful resolutions, then train CE on masked tokens (LoRA) via verl's SFT trainer. The tokenization bridge (`data/tokenization.py`) and preprocessor (`data_preprocessor.py`) are ready to convert rollout outputs into verl `DataProto` format with real token IDs and aligned masks. Config (`rft_swe.yaml`) and LoRA settings are in place.

3. **verl `reward_fn` signature adapter** — The current `reward_fn` takes `Sequence[Mapping]` and returns `list[float]`. verl expects `DataProto → (torch.Tensor, dict)`. Write a thin wrapper that unpacks/repacks between the two interfaces.

4. **Step-SDPO loop** — With (2) for RFT checkpoint + format gates, (1) for live rollouts, and (3) for verl-compatible reward, the on-policy SDPO loop can be wired. The reprompt adapter and mask injector are ready; the remaining work is plumbing them into verl's `RayPPOTrainer` hooks.

## 13) Training infrastructure decision

### 13.1 Framework
- **Trainer & rollout**: `lasgroup/SDPO` (a fork of [verl](https://github.com/verl-project/verl)) used as-is.
- **Rollout engine**: vLLM (via verl's colocated rollout worker).
- **Training engine**: FSDP `FULL_SHARD` across 8 GPUs (via verl's `DataParallelPPOActor`).
- FSDP is needed not for model size (Qwen3-4B ≈ 8 GB bf16) but for **activation memory headroom** with long SWE-bench trajectories (8K–16K tokens).

### 13.2 Hardware target
- Single node, 8× A100 or H100 GPUs (80 GB each).
- GPUs alternate between vLLM inference and FSDP training each global step.

### 13.3 Config files
- `configs/verl/sdpo_swe.yaml` — step-SDPO training (main objective).
- `configs/verl/rft_swe.yaml` — RFT supervised pre-training.
- `configs/verl/user.yaml` — user-local path overrides.

### 13.4 Integration layer
- New package `src/verl_integration/` bridges our protocol modules with verl hooks.
- Full architecture, data flow, and milestone plan in `IMPLEMENTATION_BLUEPRINT.md`.

## 14) Sources
- Qwen3 tokenizer chat template: https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json
- SDPO baseline: https://github.com/lasgroup/SDPO
- verl framework: https://github.com/verl-project/verl
- Thread review: https://github.com/murphytheagent/small-swe-train/pull/2#discussion_r2835868321
- Implementation blueprint: `IMPLEMENTATION_BLUEPRINT.md` (this repo)
