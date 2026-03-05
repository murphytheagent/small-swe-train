# Implementation Blueprint: step-SDPO on verl for SWE-Agent Training

> **Status**: Active implementation snapshot — 2026-03-05 00:00 UTC
> **Scope**: Runtime-integrated step-SDPO on `lasgroup/SDPO` (a verl fork) with
> on-policy RFT handoff and `swe_bridge_agent` multi-turn execution.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Mapping: Our Modules → verl/SDPO](#2-component-mapping-our-modules--verlsdpo)
3. [Training Pipeline Stages](#3-training-pipeline-stages)
4. [GPU Memory & Sequence Budget](#4-gpu-memory--sequence-budget)
5. [verl Integration Layer](#5-verl-integration-layer)
6. [Environment Executor Design](#6-environment-executor-design)
7. [Data Flow per SDPO Step](#7-data-flow-per-sdpo-step)
8. [File-Level Implementation Plan](#8-file-level-implementation-plan)
9. [Dependency Stack](#9-dependency-stack)
10. [Milestone Schedule](#10-milestone-schedule)
11. [Configuration & Type Authority](#11-configuration--type-authority)
12. [Current Build and Run Commands](#12-current-build-and-run-commands)
13. [Risks, Non-goals, Completion, Missing Items](#13-risks-non-goals-completion-missing-items)

---

## 1. Architecture Overview

### 1.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Single Node — 8× A100/H100 GPUs                     │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    verl Ray Trainer (Orchestrator)                  │ │
│  │  ray_trainer.py :: RayPPOTrainer                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  Training Loop (per global step):                            │  │ │
│  │  │   1. Rollout  →  2. Reward  →  3. Advantage                  │  │ │
│  │  │   4. Reprompt →  5. Train   →  6. EMA Update                 │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └──────────────┬──────────────────────┬─────────────────────────────┘ │
│                 │                      │                               │
│    ┌────────────▼──────────┐  ┌───────▼───────────────────────┐       │
│    │   Rollout Phase       │  │   Training Phase               │       │
│    │   (vLLM on 8 GPUs)   │  │   (FSDP on 8 GPUs)            │       │
│    │                       │  │                                │       │
│    │ • Student generation  │  │ • Student fwd → log_probs     │       │
│    │ • PagedAttention for  │  │   + top-k logits              │       │
│    │   long SWE contexts   │  │ • Teacher fwd (no_grad) with  │       │
│    │ • Multi-turn tool-use │  │   reprompted inputs → teacher  │       │
│    │   loop via env bridge │  │   log_probs + top-k logits    │       │
│    │ • n=8 rollouts/prompt │  │ • compute_self_distillation   │       │
│    │                       │  │   _loss (JSD, top-k, IS clip) │       │
│    └────────────┬──────────┘  │ • Optimizer step              │       │
│                 │             │ • EMA teacher update           │       │
│    ┌────────────▼──────────┐  └───────────────────────────────┘       │
│    │   Environment Bridge  │                                          │
│    │                       │  ┌────────────────────────────────┐       │
│    │ • Docker sandbox per  │  │   Our Protocol Layer           │       │
│    │   SWE-bench instance  │  │                                │       │
│    │ • Executes bash /     │  │ • TurnParser (format check)    │       │
│    │   search / edit       │  │ • feedback_canonicalizer       │       │
│    │ • Returns ToolResponse│  │ • contracts + turn parsing     │       │
│    │ • Terminal: submit    │  │ • reward_loop_score/reward_fn   │       │
│    │                       │  │ • reward_adapter               │       │
│    └───────────────────────┘  └────────────────────────────────┘       │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Shared Infrastructure                           │ │
│  │  • Flash Attention 2       • LoRA (q/k/v/o projections)           │ │
│  │  • Gradient Checkpointing  • bf16 compute throughout              │ │
│  │  • Dynamic micro-batching  • Remove-padding optimization          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Phase Alternation

The 8 GPUs serve **dual duty** — verl manages the lifecycle:

| Phase | Engine | GPU Usage | Duration |
|-------|--------|-----------|----------|
| **Rollout** | vLLM | Inference weights loaded, PagedAttention KV cache | ~60–70% of step time |
| **Training** | FSDP | Full model + optimizer states + activation memory | ~30–40% of step time |

verl's colocated worker pattern offloads vLLM weights before loading FSDP training
state, and vice versa, on every global step. The reference model (EMA teacher target)
shares GPU allocation with the actor.

### 1.3 Locked Decisions (Step-SDPO Slice)

These are already agreed and should not be reopened during this implementation slice.

1. No custom fine-grained token masking for step-SDPO runtime.
2. Step-SDPO supervision uses rollout-produced `response_mask` from the multi-turn agent loop.
3. Upstream SDPO divergence/loss internals remain unchanged unless a hard blocker is discovered.
4. Multi-turn rollout is required for SWE tool-use trajectories.
5. Local integration layers are narrow adapters only: custom agent loop (`swe_bridge_agent`),
   reward adapter (DataProto <-> row contract), self-distillation reprompt assembly hook.

---

## 2. Component Mapping: Our Modules → verl/SDPO

This section maps every module in `src/` to its integration point in the
`lasgroup/SDPO` codebase.

### 2.1 Direct reuse from SDPO (no modification needed)

| SDPO File | Function | Our Usage |
|-----------|----------|-----------|
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss()` | Core KL distillation loss with top-k, JSD, IS clipping |
| `verl/trainer/ppo/core_algos.py` | `agg_loss()` | Token-mean / seq-mean loss aggregation |
| `verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage()` | Outcome advantage for GRPO (no critic) |
| `verl/workers/actor/dp_actor.py` | `DataParallelPPOActor._update_teacher()` | EMA weight blending: `θ_t = (1-β)θ_t + β·θ_s` |
| `verl/workers/actor/dp_actor.py` | `DataParallelPPOActor._forward_micro_batch()` | Forward pass with top-k logit extraction |
| `verl/workers/actor/dp_actor.py` | `DataParallelPPOActor.update_policy()` | Full training step: student fwd → teacher fwd → loss → optim → EMA |

### 2.2 Custom integration (our code plugs into SDPO hooks)

| Our Module | SDPO Hook Point | Integration Method |
|------------|-----------------|-------------------|
| **`verl_integration/main_ppo_entry.py`** | SDPO launcher path (`verl.trainer.main_ppo`) | Imports registration side effects for `swe_bridge_agent` and applies `ppo_runtime_patch` before trainer starts |
| **`verl_integration/ppo_runtime_patch.py`** | `RayPPOTrainer._compute_or_extract_reward`, `RayPPOTrainer._maybe_build_self_distillation_batch` | Monkey-patches reward and self-distillation hooks in-process to keep upstream SDPO behavior intact |
| **`verl_integration/reward_adapter.py`** | Reward path / DataProto bridge | Converts rollout `DataProto` into row-level samples, then builds token-aligned reward tensors and extras |
| **`verl_integration/reward_loop_score.py`** | `custom_reward_function` entrypoint | Adapts local score outputs to expected `RewardLoopManager` field names when running via config-level scorer |
| **`verl_integration/reprompt_adapter.py`** | `RayPPOTrainer._maybe_build_self_distillation_batch` | Builds 6-block teacher prompts and turn-level self-distillation masks using local `teacher` prompt logic |
| **`verl_integration/swe_bridge_agent_loop.py`** | `verl.experimental.agent_loop` registry | Registers `swe_bridge_agent`, manages multi-turn tool calls, and returns per-turn response masks |
| **`verl_integration/env_bridge.py`** | Bridge execution | Runs parsed tool calls in Docker sandbox and returns normalized tool-response metadata |
| **`rollout/turn_parser.py`** | Bridge + reward path | Parses assistant response payloads and tool call blocks (including optional `<think>`) |
| **`schemas/contracts.py`** | Tool schema and terminal policy | `validate_tool_call` and terminal-tool checks gate execution/reward paths |
| **`data/feedback_canonicalizer.py`** | Reprompt + reward extras | Builds canonical feedback packets for logging, reprompt, and scoring context |

### 2.3 Mapping table: design doc §9.2 → implementation

| §9.2 Requirement | Implementation |
|-------------------|---------------|
| **1. Multi-tool-call turns** | `swe_bridge_agent_loop` parses and executes each ordered `ToolCall`; it appends each `<tool_response>` block before the next turn. |
| **2. Masking source** | SDPO uses rollout-produced `response_mask` (`assistant` tokens = 1, tool/observation tokens = 0). There is no custom SDPO-level fine-grained token-label injection in the runtime patch path. |
| **3. `actionable_error_text` in canonicalization** | `feedback_canonicalizer.build_feedback_packet()` generates normalized feedback text that is included in reprompt + reward-extra outputs. |
| **4. Teacher includes student attempt** | `reprompt_adapter` and config flags (`dont_reprompt_on_self_success`, `include_student_attempt_for_teacher`) control whether successful attempts stay in student context. |
| **5. Terminal tool is `submit`** | `validate_tool_call` + canonical tool normalization enforce terminal tool and argument checks; bridge/reward paths expect submit-formatted terminal turns. |

---

## 3. Training Pipeline Stages

```
  ┌─────────────┐     format gates      ┌──────────────┐     continuous
  │  Stage 1:   │    pass (§10 gates)?   │  Stage 2:    │     training
  │  RFT        │ ──────────────────────►│  step-SDPO   │ ──────────────►
  │ (on-policy) │     YES                │  (on-policy) │     convergence
  └─────────────┘                        └──────────────┘
        │                                      │
  Generate N rollouts per task           Self-distillation:
  in Docker, keep successes,             student ← teacher(EMA)
  supervised CE on tool-call tokens      Think tokens INCLUDED
  Think tokens EXCLUDED                  in loss mask
```

### Stage 1: RFT (Rejection Fine-Tuning)

- **Data**: On-policy — roll out N attempts per SWE-bench task in Docker sandboxes via `env_bridge.py`, keep only successful resolutions
- **Loss**: Supervised cross-entropy on masked tokens (tool-call tokens only, think tokens excluded per `action_masking.py` with `stage="rft"`)
- **Config**: `configs/verl/rft_swe.yaml`
- **Exit gate**: All 7 format quality metrics in `phase_transition_gates.v1.json` must pass (e.g., `parse_valid_rate ≥ 0.985`)
- **verl mode**: SFT trainer (`sft_trainer.yaml` base)

### Stage 2: step-SDPO (Self-Distilled Policy Optimization)

- **Loop**: On-policy rollout → environment execution → reward → reprompt → self-distillation training
- **Loss**: JSD between student logits and EMA-teacher logits (top-k=100, tail bucket, IS clip=2.0)
- **Masking**: Uses rollout-provided `response_mask` directly (`assistant=1`, tool/observation=0); no SDPO runtime token-label remapping.
- **Config**: `configs/verl/sdpo_swe.yaml`
- **verl mode**: PPO trainer with `loss_mode: sdpo`

### Stage 1.5 (optional): SDFT

- Demo-conditioned self-distillation using teacher-generated trajectories as conditioning
- Same infrastructure as step-SDPO but with curated rollouts instead of fully on-policy rollouts
- Controlled by `pipeline.sdft_enabled_default` in `training_policy_defaults.v1.json`

---

## 4. GPU Memory & Sequence Budget

### 4.1 Why FSDP for a 4B model

Qwen3-4B weights in bf16 ≈ 8 GB — fits on one GPU. But **activation memory from long
SWE-bench trajectories** is the bottleneck:

| Component | Memory per GPU (no FSDP) | Memory per GPU (FSDP 8-way) |
|-----------|--------------------------|------------------------------|
| Model weights (bf16) | 8 GB | 1 GB |
| LoRA trainable params | ~0.1 GB | ~0.01 GB |
| Optimizer states (AdamW, LoRA only) | ~0.4 GB | ~0.05 GB |
| Gradients | ~0.1 GB | ~0.01 GB |
| **Activations (16K seq, micro_bs=2)** | **~40–50 GB** | **~40–50 GB** |
| KV cache headroom | — | — |
| **Total** | **~50–60 GB** | **~42–52 GB** |

FSDP frees ~7 GB per GPU (parameters + optimizer + gradients), which is the margin
between fitting and OOM on 80 GB GPUs with 16K sequences. Combined with:

- **Gradient checkpointing**: Recomputes activations during backward pass, trading
  ~2× compute for ~4× activation memory reduction
- **Remove-padding** (`use_remove_padding: true`): Packs variable-length sequences to
  avoid wasted computation on padding tokens
- **Dynamic micro-batching** (`use_dynamic_bsz: true`): Adjusts micro-batch size per
  step to fit the longest sequences in the batch

### 4.2 Sequence length budget

```
┌──────────────────────────────────────────────────────────────┐
│              max_model_len = 18,944 tokens                    │
│                                                              │
│  ┌──────────┬──────────────────┬───────────────────────────┐ │
│  │  Prompt  │   Response       │   (Teacher reprompt only)  │ │
│  │  ≤ 2048  │   ≤ 8192         │   feedback ≤ 8192          │ │
│  │          │                  │   template overhead ~512    │ │
│  └──────────┴──────────────────┴───────────────────────────┘ │
│                                                              │
│  Student training input:  prompt + response ≤ 10240          │
│  Teacher reprompt input:  max_reprompt_len = 10240           │
└──────────────────────────────────────────────────────────────┘
```

A typical SWE-bench trajectory:
- System prompt: ~500 tokens
- Task description: ~500 tokens
- Per turn (5–15 turns): think (~200) + tool_call (~150) + tool_response (~500) ≈ 850 tokens
- Total for 10-turn episode: ~1000 + 10 × 850 = **~9500 tokens**

This fits within the 10240-token student budget with room to spare.

---

## 5. verl Integration Layer

### 5.1 Adapter Layer Implemented (between project and verl)

```
src/
  verl_integration/
    __init__.py
    main_ppo_entry.py        # entrypoint bootstrap + patch registration
    ppo_runtime_patch.py     # runtime monkeypatches for SDPO hooks
    reward_function.py        # verl reward_fn: parse + validate + score
    reward_loop_score.py      # verl reward-loop scorer wrapper
    reward_adapter.py         # DataProto bridge and reward tensor assembly
    reprompt_adapter.py       # override _maybe_build_self_distillation_batch
    swe_bridge_agent_loop.py  # custom multi-turn agent loop
    mask_injector.py          # build response_mask from action_masking.py
    env_bridge.py             # multi-turn rollout ↔ Docker sandbox
    data_preprocessor.py      # rollout trajectories → verl-ready rows
    submission_verifier.py    # verifies terminal submit payloads
```

### 5.2 `reward_function.py` — Reward Function

Local reward function accepts row-like samples (resolved from rollout output):
```python
def reward_fn(data: Sequence[Mapping[str, Any]], max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN) -> tuple[list[float], dict[str, list]]
```

Our implementation:
1. Parse assistant response text with `TurnParser`
2. Validate terminal and tool-call format via schema contracts
3. Resolve verification signals (`fail_to_pass` / `pass_to_pass`) and `verification_missing` states
4. Compute and return `(scores, extras)`; `extras` is propagated through reprompt and trainer metrics

### 5.3 `reprompt_adapter.py` — Teacher Prompt Override

Activated through `ppo_runtime_patch` in `main_ppo_entry.py`.
Patched `RayPPOTrainer._maybe_build_self_distillation_batch()`:
1. Call `feedback_canonicalizer.build_feedback_packet()` on raw env output
2. Construct `TeacherPromptInputs` with all 6 blocks
3. Call `build_teacher_prompt()` to produce teacher prompt text
4. Tokenize with Qwen3 tokenizer at `max_reprompt_len=10240`
5. Return `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`, and optional turn-level prompt tensors

### 5.4 `mask_injector.py` — Response Mask Override

Used for RFT/preprocessing adapter paths.
1. Label tokens via delimiter-aware token tags (`think`, `tool_call`, `other`)
2. Convert tags through `build_action_token_mask` with the requested stage (`rft`/`step_sdpo`)
3. Inject mask data into sample dictionaries for downstream RFT or local processing

### 5.5 `env_bridge.py` — Environment Bridge

`swe_bridge_agent_loop` drives multi-turn rollout; this bridge executes tool calls within one turn:
1. Receives generated text from vLLM
2. Parses with `TurnParser` → `ActionEnvelope`
3. For each `ToolCall` in the envelope:
   - Validates via `validate_tool_call()`
   - Constructs `ToolRequest`
   - Sends to Docker sandbox
   - Receives `ToolResponse`
4. Formats response as `<tool_response>...</tool_response>` block
5. Appends to conversation history for next vLLM generation
6. If `ActionEnvelope.is_terminal` (submit), ends the episode

### 5.6 Upstream SDPO Surfaces Assumed Stable

We treat these surfaces as stable integration points:
1. Rollout generation path in `RayPPOTrainer.fit`.
2. Reward computation hook path (DataProto-compatible interface).
3. `_maybe_build_self_distillation_batch(...)` path for teacher reprompt assembly.

### 5.7 Multi-Turn Assumptions

1. Multi-turn is supported but not default; explicit config enablement is required.
2. Agent loop selection is config-driven; default single-turn behavior is insufficient for SWE tool trajectories.
3. Role-aware mask correctness depends on agent loop behavior, not on generic trainer fallback mask computation.

---

## 6. Environment Executor Design

```
┌──────────────────────────────────────────────────────────────────┐
│  verl Rollout Worker (vLLM)                                      │
│                                                                  │
│  for turn in range(max_turns):                                   │
│    assistant_text = vllm.generate(conversation_history)           │
│    envelope = TurnParser.parse(assistant_text)                    │
│                                                                  │
│    if envelope.is_terminal:                                      │
│      feedback = run_submission_verifier(envelope.tool_calls[0])    │
│      break                                                       │
│                                                                  │
│    for tool_call in envelope.tool_calls:                         │
│      ┌──────────────────────────────────────────────────┐        │
│      │  Docker Sandbox (per SWE-bench instance)          │        │
│      │                                                    │        │
│      │  bash   → subprocess.run(cmd) → stdout/stderr     │        │
│      │  search → find/grep in repo   → file contents     │        │
│      │  edit   → apply patch to file → confirmation      │        │
│      └──────────────┬───────────────────────────────────┘        │
│                     │                                            │
│      tool_response = format_as_tool_response(result)             │
│      conversation_history.append(tool_response)                  │
│                                                                  │
│  return (conversation_history, feedback)                         │
└──────────────────────────────────────────────────────────────────┘
```

Each SWE-bench instance gets its own Docker container with:
- A checked-out repository for the selected task image
- Standard developer tools (Python, git, etc.)
- Network access where required by the task
- Read-write filesystem access
- Lifecycle reuse through `env.runtime_protocol` + `env.container_pool`
- Submission verification path for terminal `submit` tool outputs

The environment executor is the **main custom infrastructure** to build beyond the
verl framework.

---

## 7. Data Flow per SDPO Step

### 7.1 SWE Trajectory Record Contract (Minimum Required Fields)

Response masking contract (authoritative for step-SDPO runtime):
- `response_mask[t] = 1` iff token `t` is model-generated assistant output.
- `response_mask[t] = 0` iff token `t` is tool response, observation, or user-injected context.
- Non-goal: no token-label masking (`think`, `tool_call`, or similar) is injected for SDPO runtime.

Each sample must carry metadata for reward + reprompt paths (via DataProto non-tensors or equivalent):

- `prompt` (string): task prompt.
- `task_id` (string).
- `image_name` (string): container image used for tool execution.
- `assistant_response` (string): last assistant turn to score.
- `tool_output` (mapping): last tool execution payload (`stdout`, `stderr`, `exit_code`, metadata).
- `resolved` (bool): Phase-1 heuristic defined below.
- `step_index` (int).
- `attempt_index` (int).
- `turn_index` (int).
- `trajectory_steps` (list).
- `trajectory_tool_validation_errors` (list[str]).
- `final_turn_has_submit` (bool).
- `final_submit_format_valid` (bool).
- `executor_error` / `bridge_error` / `timeout_error` (optional strings when present).

Terminal submit edge-case contract:
- If terminal `submit` occurs without prior non-submit tool step, preserve terminal `assistant_response`.
- Set `tool_output = {}` for that row.
- Reward adapter must handle this deterministically (no shape/type special-case crash).

### 7.2 `resolved` Definition (Phase-1 Heuristic)

`resolved = true` iff all are true:
1. Terminal tool call is `submit`.
2. Submit payload/schema is valid.
3. No tool step has non-zero `exit_code`.
4. No bridge/executor/timeout error flags are present.

Phase-2 (future, out-of-scope here): replace heuristic with harness-based task resolution signal.

### 7.3 SDPO Step Trace

This traces one complete global step through the system.

```
Step 1: ROLLOUT
  Inputs:  batch of prompts (SWE-bench issues)
  Engine:  vLLM on 8 GPUs
  Process: multi-turn generation via env_bridge.py
  Outputs: DataProto with {input_ids, responses, attention_mask, ...}

     │
     ▼

Step 2: REWARD
  Inputs:  DataProto from rollout
  Process: reward_adapter -> reward_function
    a) Converts rollout rows through `dataproto_to_rows`
    b) Parses assistant responses and validates tool/terminal format
    c) Computes resolution/verification metrics and score metadata
  Outputs: reward_tensor (bs, response_len), feedback and extra metrics

     │
     ▼

Step 3: ADVANTAGE
  Inputs:  reward_tensor, response_mask
  Process: compute_grpo_outcome_advantage()
    • Groups by prompt UID, computes group-normalized advantage
    • No critic needed (outcome-level reward)
  Outputs: advantages tensor (bs, response_length)

     │
     ▼

Step 4: REPROMPT (self-distillation batch assembly)
  Inputs:  DataProto + reward outputs + extra fields
  Process: reprompt_adapter.py
    a) Resolve success from reward outputs
    b) feedback_canonicalizer.build_feedback_packet() on env output
    c) build_teacher_prompt() with 6-block structure:
       SYSTEM → TASK → TRAJECTORY → ATTEMPT → FEEDBACK → CONTRACT
    d) Tokenize teacher prompt at `max_reprompt_len=10240`
    e) Build `self_distillation_mask` and optional turn-expanded prompt tensors
  Outputs: `teacher_input_ids`, `teacher_attention_mask`,
            `teacher_position_ids`, `self_distillation_mask`

     │
     ▼

Step 5: TRAIN
  Inputs:  DataProto union'd with teacher batch
  Engine:  FSDP on 8 GPUs
  Process: dp_actor.update_policy()
    a) Patched trainer path uses rollout `response_mask` and optional `turn_response_mask`
    b) Student forward → log_probs + top-100 logits
    c) Teacher forward (no_grad, EMA model) with teacher_input_ids
       → teacher log_probs + top-100 logits (using student's top-k indices)
    d) compute_self_distillation_loss():
       • JSD (α=0.5) over top-100 logits + tail bucket
       • IS clipping at ratio 2.0
       • Masked by response_mask × self_distillation_mask
    e) Backward pass with gradient checkpointing
    f) Optimizer step (AdamW, lr=1e-5, grad_clip=1.0)
  Outputs: loss scalar, training metrics

     │
     ▼

Step 6: EMA UPDATE
  Inputs:  student model, teacher model, rate=0.005
  Process: dp_actor._update_teacher()
    θ_teacher = (1 - 0.005) × θ_teacher + 0.005 × θ_student
  Outputs: updated teacher weights (in-place)

     │
     ▼

  [Back to Step 1 with next batch of prompts]
```

---

## 8. File-Level Implementation Plan

### 8.1 Integration inventory (implemented)

| File | Purpose | Status |
|------|---------|--------|
| `src/verl_integration/__init__.py` | Package exports and import surface | Done |
| `src/verl_integration/main_ppo_entry.py` | Local SDPO launcher/bootstrap | Done |
| `src/verl_integration/ppo_runtime_patch.py` | Runtime monkeypatch for reward + distillation hooks | Done |
| `src/verl_integration/reward_adapter.py` | DataProto row adaptation + reward tensor assembly | Done |
| `src/verl_integration/reward_function.py` | SWE reward and scoring logic | Done |
| `src/verl_integration/reward_loop_score.py` | `custom_reward_function` compatibility wrapper | Done |
| `src/verl_integration/reprompt_adapter.py` | 6-block teacher prompt + turn expansion | Done |
| `src/verl_integration/mask_injector.py` | Stage-aware masking utility for RFT/preprocessing | Done |
| `src/verl_integration/env_bridge.py` | Tool-call execution bridge | Done |
| `src/verl_integration/swe_bridge_agent_loop.py` | Multi-turn agent loop registration + runtime | Done |
| `src/verl_integration/submission_verifier.py` | Submit payload verification | Done |
| `src/verl_integration/data_preprocessor.py` | RFT/preprocessing and masking support | Done |
| `src/verl_integration/onpolicy_rollout_adapter.py` | Runtime handoff adapter | Done |
| `configs/verl/sdpo_swe.yaml` | Step-SDPO runtime config | Done |
| `configs/verl/rft_swe.yaml` | RFT config | Done |
| `configs/verl/agent_loops/swe_bridge_agent.yaml` | `swe_bridge_agent` defaults | Done |
| `scripts/run_sdpo.sh` | SDPO entry + hygiene/resolution + watchdog | Done |
| `scripts/run_rft.sh` | RFT runtime entrypoint | Done |

### 8.2 Files to modify

| File | Change | Reason |
|------|--------|--------|
| `src/trainer/sdpo_trainer.py` | Keep compatibility shim for scaffold paths; runtime execution now flows through `run_sdpo.sh` + `main_ppo_entry.py` | Preserve older interfaces used by tests and docs |
| `src/verl_integration/main_ppo_entry.py` | Adjust entry/bootstrap defaults as SDPO dependencies or launch flags evolve | Maintain import-safe startup under Ray workers |
| `src/verl_integration/ppo_runtime_patch.py` | Keep turn-level expansion logic aligned with upstream SDPO surface changes | Avoid runtime regression from trainer API drift |
| `scripts/run_sdpo.sh` | Keep checkpoint/cache/validation logic current and add clean shutdown handling | Required for reliable one-step and monitored D6 runs |
| `scripts/SLURM_GPU_LAUNCH.md` | Keep run envelope examples aligned with hardened `run_sdpo.sh` and `RAY_TMPDIR` policy | Operational reproducibility |

Operational note for `scripts/run_sdpo.sh`:
- Prefer setting `RAY_TMPDIR` to a scratch location (for example `/data/scratch/$USER/ray_tmp/$SLURM_JOB_ID`) before long SDPO runs.
- Keep tokenizer deadlock guard enabled: `TOKENIZERS_PARALLELISM=false` (set by launcher by default).
- Run labels and optional cleanup are handled by the launcher/runtime; pair with `scripts/SLURM_GPU_LAUNCH.md` examples for submission context.
- Canonical launch examples live in `scripts/SLURM_GPU_LAUNCH.md`.

### 8.3 Files unchanged (protocol layer — already complete)

All files in `src/schemas/`, `src/prompts/`, `src/rollout/`,
`src/losses/`, `src/teacher/`, `src/metrics/`, `src/env/` remain as-is. They are
consumed by the integration layer. `src/data/` contains `feedback_canonicalizer.py`,
`tool_schema_adapter.py`, and `tokenization.py` (the offline ingestion module
`trajectory_ingestion.py` was removed in v1.9).

### 8.4 Step-SDPO Implementation Sequence (Exact Order)

Phase 0: Doc and policy alignment for masking pivot
- Objective: align design docs so step-SDPO mask source is rollout `response_mask`.
- Actions: remove SDPO runtime references to token-label masking; mark scaffold helpers as non-runtime.
- Exit criteria: no design doc claims SDPO runtime `response_mask` is produced by token-label injection.
- Status: complete.

Phase A: Runner hygiene and authoritative entrypoint
- Objective: make SDPO launch path as reliable as RFT launch path.
- Actions: `scripts/run_sdpo.sh` handles hygiene and `main_ppo_entry.py` is the entrypoint.
- Exit criteria: dry-run resolves a command importing local integration modules without path issues.
- Status: complete.

Phase B: Baseline upstream multi-turn + bridge loop
- Objective: confirm multi-turn plumbing before custom loop insertion.
- Actions: explicit multi-turn keys in `configs/verl/sdpo_swe.yaml`; route rollouts through `swe_bridge_agent`.
- Exit criteria: multi-turn active in resolved config; rollout `response_mask` semantics present.
- Status: complete.

Phase C: Task metadata propagation into SDPO rollouts
- Objective: guarantee rollout loop has `task_id`, `image_name`, and prompt data per sample.
- Actions: require metadata-rich parquet fields and validate at reward/reprompt boundaries.
- Exit criteria: agent loop receives `task_id`, `image_name`, and prompt text for each sample.
- Status: complete.

Phase D: Implement `swe_bridge_agent` loop
- Objective: replace generic tool-loop execution with SWE bridge semantics.
- Actions: deterministic Docker lifecycle, per-turn tool execution, correct `response_mask` semantics.
- Exit criteria: `swe_bridge_agent` active, no container leaks in smoke run, tool-response blocks present.
- Status: complete.

Phase E: Reward adapter for DataProto path
- Objective: reuse existing reward function in PPO DataProto flow without SDPO loss changes.
- Actions: map DataProto to rows, call `reward_fn(...)`, return reward tensor + extras.
- Exit criteria: reward computation runs without shape/type mismatch; `feedback` aligned.
- Status: complete.

Phase F: Self-distillation reprompt hook
- Objective: swap only teacher prompt construction to local SWE reprompt adapter.
- Actions: patch `_maybe_build_self_distillation_batch(...)`, enforce truncation, keep SDPO loss intact.
- Exit criteria: distillation tensors produced with stable shapes; hook integrates without fallback breakage.
- Status: complete.

Phase G: Verification and smoke E2E
- Objective: prove end-to-end wiring with minimal monitored run.
- Actions: one-step training run, verify bridge + reward + reprompt + masking behavior, clean up.
- Exit criteria: one complete SDPO training step finishes and D6 acceptance artifacts are captured.
- Status: one-step smoke complete; D6 acceptance artifacts pending.

### 8.5 Deliverables (D0..D6) and Acceptance Criteria

| ID | Deliverable | Primary Files | Acceptance Criteria | Status |
| --- | --- | --- | --- | --- |
| D0 | Root guiding plan | `step_sdpo_implementation_plan.md` (merged here) | Guiding plan tracked and current | Done |
| D1 | Multi-turn + loop defaults in SDPO config | `configs/verl/sdpo_swe.yaml` | Required keys explicit in YAML (not only CLI) | Done |
| D2 | SWE bridge agent loop integration | `configs/verl/agent_loops/swe_bridge_agent.yaml`, `src/verl_integration/swe_bridge_agent_loop.py` | Per-turn bridge call integrated; terminal submit edge case deterministic | Done |
| D3 | SDPO trainer reprompt hook integration | `src/verl_integration/main_ppo_entry.py` (patch module) | `_maybe_build_self_distillation_batch` uses local reprompt adapter; SDPO loss math unchanged | Done |
| D4 | DataProto reward adapter | `src/verl_integration/reward_adapter.py` + call site | Reward tensor and feedback extras produced with stable alignment | Done |
| D5 | SDPO launcher hygiene | `scripts/run_sdpo.sh` | Launcher exports PYTHONPATH, resolves checkpoints, executes local entry | Done |
| D6 | Monitored e2e step-SDPO run from RFT checkpoint | `outputs/integration/<run_label>/...` | Run satisfies Section 12.4 success gate with required artifacts | Pending |

---

## 9. Dependency Stack

### 9.1 Core training dependencies

```
# In pyproject.toml [project.optional-dependencies.train]
torch>=2.5.1
transformers>=4.46
flash-attn>=2.7               # compiled from source
peft>=0.13                     # LoRA
vllm>=0.8.4                   # rollout inference
ray>=2.40                      # distributed orchestration
verl @ git+https://github.com/lasgroup/SDPO.git  # SDPO fork package
omegaconf>=2.3                 # verl config
hydra-core>=1.3               # verl config
wandb>=0.19                    # experiment tracking
```

### 9.2 verl installation

```bash
# Clone the SDPO fork (includes verl + SDPO modifications)
git clone https://github.com/lasgroup/SDPO.git
cd SDPO
pip install -e .

# Our project is installed alongside
cd /path/to/small-swe-train
pip install -e ".[train]"
```

### 9.3 Docker (for environment executor)

```
docker>=24.0                   # container runtime
# SWE-bench instance images pulled at runtime
```

---

## 10. Milestone Schedule

### M1: Trajectory Preprocessing & Tokenization (prerequisite for all training)
- [x] `verl_integration/data_preprocessor.py` — deterministic trajectory rows for verl adapters (2026-02-21 09:55 UTC)
- [x] Stitch `tool_schema_adapter` + `turn_parser` + `feedback_canonicalizer` (2026-02-21 09:55 UTC)
- [x] Tokenization bridge with offset-aligned per-token label masks (`data/tokenization.py`) (2026-02-22)
- [x] Batch tokenization support (`tokenize_batch_with_labels`) with graceful fallback (2026-02-22)

### M2: RFT Training (on-policy)
- [x] `configs/verl/rft_swe.yaml` finalized (done — see `configs/verl/`)
- [x] `verl_integration/mask_injector.py` for RFT-stage masking (2026-02-21 09:55 UTC)
- [x] Launcher dry-run path validated via `tests/test_run_scripts.py` (2026-02-21 19:03 UTC)
- [x] Environment executor (Docker sandbox) wired into on-policy collector/runtime loop (2026-02-23 10:45 UTC)
- [x] RFT rollout loop: generate N attempts per task, filter by rejection policy, train CE on accepted trajectories, then restart vLLM from latest checkpoint (2026-02-23 10:45 UTC)

### M3: Environment Executor
- [x] `verl_integration/env_bridge.py` — deterministic rollout bridge with executor protocol (2026-02-21 09:55 UTC)
- [x] Tool execution: `bash`, `search`, `edit`, `submit` dispatch path implemented in bridge (2026-02-21 09:55 UTC)
- [x] `verl_integration/reward_function.py` — format checks + binary outcome reward scaffold (2026-02-21 09:55 UTC)
- [x] Integration test: single rollout episode end-to-end (unit-level with fake executor) (2026-02-21 09:55 UTC)

### M4: step-SDPO Integration
- [x] `verl_integration/reprompt_adapter.py` — 6-block teacher prompt scaffold (2026-02-21 09:55 UTC)
- [x] Wire SDPO runtime to use rollout-produced `response_mask` and local `reward_adapter` path instead of local token-mask remapping (2026-02-28 00:00 UTC)
- [x] `configs/verl/sdpo_swe.yaml` finalized (done — see `configs/verl/`)
- [x] End-to-end: rollout → reward → reprompt → train → EMA update (2026-02-27 20:00 UTC; `run_sdpo.sh --dry-run` and one-step smoke path in `outputs/turn_sdpo_runtime/.../global_step_1`)
- [x] Runtime wiring in progress: SDPO monitor + dataset checkpoint resolution + launch hygiene validated (2026-02-28 00:00 UTC; `scripts/run_sdpo.sh`)

### M5: Evaluation Harness
- [x] `eval/swebench_lite.py` — deterministic per-episode evaluator scaffold (2026-02-21 09:55 UTC)
- [x] Score patches, compute resolve rate (2026-02-21 19:03 UTC; `summarize_episode_results` + `scripts/eval_swebench_lite.py`)
- [x] Compare RFT baseline vs. step-SDPO (2026-02-21 19:03 UTC; `compare_resolve_rates` + `tests/test_eval_swebench_lite_script.py`)

### Ordering

```
M1 ──► M3 ──► M2 ──────────────► M5
                \                 ▲
                 ► M4 ────────────┘
```

M3 (env executor) is now prerequisite for M2 (RFT) since RFT is on-policy.
M4 (SDPO) requires M2 (RFT checkpoint) + M3 (env executor).
M5 (eval) can run against either M2 or M4 checkpoints.

### Progress Log

- [2026-02-21 09:55 UTC] Implemented `src/verl_integration/` with `data_preprocessor.py`, `mask_injector.py`, `reward_function.py`, `reprompt_adapter.py`, `env_bridge.py`, and package exports.
- [2026-02-21 09:55 UTC] Replaced `NotImplementedError` scaffolds in `src/trainer/sdpo_trainer.py` and `src/eval/swebench_lite.py` with deterministic adapter-based logic for local verification.
- [2026-02-21 09:55 UTC] Added unit tests for integration modules and updated trainer/eval behavior (`tests/test_verl_*.py`, `tests/test_sdpo_trainer.py`, `tests/test_swebench_lite.py`).
- [2026-02-21 09:55 UTC] Test status: `pytest` passing (`28 passed`).
- [2026-02-21 10:13 UTC] Addressed PR feedback: corrected SDFT launcher override to `actor_rollout_ref.actor.policy_loss.loss_mode=sdft`; updated preprocessor null handling so `assistant_response: null` falls back to `external_tool_calls`; added regression test coverage.
- [2026-02-21 10:13 UTC] Test status: `pytest` passing (`29 passed`).
- [2026-02-21 10:25 UTC] Addressed follow-up PR feedback: hardened `external_tool_calls` parsing to handle non-mapping entries/strings as per-row `parse_error` (no run-level crash), added two regression tests, and added `verl @ git+https://github.com/lasgroup/SDPO.git` to `[project.optional-dependencies.train]` so launcher install guidance is consistent.
- [2026-02-21 10:25 UTC] Test status: `pytest --override-ini addopts=''` passing (`31 passed`).
- [2026-02-21 19:03 UTC] Addressed follow-up PR review findings for malformed `step_index` handling by making preprocessor, reward adapter, and reprompt adapter fault-tolerant; added regressions for bad `step_index` and batch continuity.
- [2026-02-21 19:03 UTC] (removed) `scripts/prepare_rft_data.py` and `data/trajectory_ingestion.py` deleted in v1.9 — RFT is on-policy, offline ingestion pipeline was unnecessary.
- [2026-02-21 19:03 UTC] Extended evaluation harness with resolve-rate summaries/comparisons plus CLI (`scripts/eval_swebench_lite.py`) and non-invasive launcher dry-run checks for `run_rft.sh`, `run_sdft.sh`, `run_sdpo.sh`.
- [2026-02-21 19:03 UTC] Added deterministic end-to-end scaffold in `SDPOTrainerScaffold.run_end_to_end_global_step` covering rollout bridge, reward, reprompt assembly, SDPO step stats, and EMA-proxy updates.
- [2026-02-21 19:03 UTC] Test status: `pytest --override-ini addopts=''` passing (`44 passed`).
- [2026-02-21 23:32 UTC] Addressed new PR review findings on string-typed `resolved` values by adding explicit bool coercion in `src/verl_integration/reward_function.py`, `src/verl_integration/reprompt_adapter.py`, and `src/eval/swebench_lite.py`, with regressions in `tests/test_verl_reward_function.py`, `tests/test_verl_reprompt_adapter.py`, and `tests/test_swebench_lite.py`.
- [2026-02-21 23:32 UTC] Test status: `pytest --override-ini addopts=''` passing (`67 passed, 1 skipped`).
- [2026-02-22 06:10 UTC] Implemented centralized RFT handoff policy in `configs/runtime/training_policy_defaults.v1.json` + `src/config.py` (`resolve_rft_handoff_settings`) and added a single deterministic selection function (`evaluate_rft_rejection_reason` / `select_rft_attempt_rows`) reused by adapter tests and trainer entrypoint.
- [2026-02-22 06:10 UTC] Implemented direct on-policy RFT handoff path in `src/verl_integration/onpolicy_rollout_adapter.py`: rollout collection -> preprocessing -> centralized rejection -> SFT tensor assembly (`input_ids`, `attention_mask`, `position_ids`, `loss_mask`) + DataProto-compatible payload buckets with grouping metadata.
- [2026-02-22 06:10 UTC] Added custom verl SFT dataset `src/verl_integration/onpolicy_rft_dataset.py` and wired `configs/verl/rft_swe.yaml` (`data.custom_cls.path/name`) so RFT can source rollout rows in-memory instead of JSONL intermediates.
- [2026-02-22 06:10 UTC] Updated launchers (`scripts/run_rft.sh`, `scripts/run_rft_onpolicy_rollout_proof.sh`) to use `verl.trainer.fsdp_sft_trainer` via `torchrun` and proof-mode multi-turn tool-chain rollouts.
- [2026-02-22 06:10 UTC] Added/updated test coverage for centralized config authority, rollout adapter handoff behavior, trainer entrypoint reuse, and launcher command correctness.
- [2026-02-22 06:10 UTC] Test status: local `pytest --override-ini addopts=''` passing (`89 passed, 2 skipped`); remote targeted suite in `swe311` passing (`13 passed`) for updated adapter/trainer/script tests.
- [2026-02-22 06:10 UTC] GPU Slurm proof reached on-policy collection + dataset + trainer initialization, but final one-step run is blocked by remote environment dependencies (`flash_attn`) and transient SSH reachability during repeated retries.
- [2026-02-22 07:22 UTC] Added no-FlashAttention fallback entrypoint (`src/verl_integration/fsdp_sft_trainer_entry.py`) plus W&B logging wiring in proof launchers (`scripts/run_rft.sh`, `scripts/run_rft_onpolicy_rollout_proof.sh`); local + remote script regressions pass, but final Slurm one-step proof remains blocked because the only GPU partition node (`wth-gpu-01`) is currently `down` per `sinfo`.
- [2026-02-22 11:14 UTC] Addressed the remaining active PR #4 review threads: `src/rollout/onpolicy_collector.py` now keeps `assistant_response`/`tool_output` aligned to the same first executed tool call for the sampled turn and records executor failures from the first non-zero step; `src/verl_integration/data_preprocessor.py` now coerces bool-like `include_student_attempt_for_teacher` values instead of Python truthy casting.
- [2026-02-22 11:14 UTC] Added regressions in `tests/test_onpolicy_collector.py` and `tests/test_verl_data_preprocessor.py`; validation status: local `pytest --override-ini addopts=''` passing (`104 passed, 2 skipped`), plus Slurm GPU validation on `tianhaowang-gpu0` (`job 422`, `--gres=gpu:1 --mem=24G`) with CUDA visible and targeted suites passing (`19 passed`).
- [2026-02-22 11:17 UTC] Follow-up next-step implementation after PR #4 merge: aligned `onpolicy_collector` row `turn_index` with the sampled `assistant_response`/`tool_output` turn (instead of terminal submit turn), added regression assertions in `tests/test_onpolicy_collector.py`, and revalidated full local suite (`104 passed, 2 skipped`).
- [2026-02-22 21:27 UTC] Addressed PR #6 P1 reliability findings in `src/rollout/onpolicy_collector.py` and `src/env/docker_executor.py`: task patches are now streamed via stdin to `docker exec -i` (instead of embedding full base64 patch payload in argv), and task-env init executor exceptions are downgraded into row-level `executor_error` values so one failing task does not crash batch collection.
- [2026-02-22 21:27 UTC] Added regressions in `tests/test_onpolicy_collector.py` and `tests/test_docker_executor.py`; validation status: `python3 -m pytest tests/test_onpolicy_collector.py tests/test_onpolicy_rollout_adapter.py tests/test_sdpo_trainer.py tests/test_run_scripts.py tests/test_task_dataset.py tests/test_docker_executor.py -q` passing (`34 passed`).
- [2026-02-22 23:34 UTC] Added optional RFT checkpoint/saving scaffold in `SDPOTrainerScaffold.run_onpolicy_rft_step(...)`: when `checkpoint_dir` is provided, the trainer now writes `checkpoints/global_step_<n>/rft_step_manifest.json` plus `checkpoints/latest_checkpoint.txt`, and exposes `checkpoint_dir`/`checkpoint_exists` in `OnPolicyRFTStepArtifacts`.
- [2026-02-22 23:34 UTC] Added regression coverage in `tests/test_sdpo_trainer.py` for checkpoint manifest and latest-pointer writes.
- [2026-02-22 23:42 UTC] Hardened RFT checkpoint contract to require explicit `global_step` whenever `checkpoint_dir` is set; removed fallback to `total_steps` to prevent iterative runs from overwriting `global_step_1`.
- [2026-02-22 23:42 UTC] Added regression coverage in `tests/test_sdpo_trainer.py` asserting checkpoint writes fail fast without explicit `global_step`.
- [2026-02-22 23:50 UTC] Moved `global_step` checkpoint validation to run before `collect_rft_sft_batch_for_steps(...)` so invalid checkpoint requests fail fast before rollout/training side effects; added regression `test_run_onpolicy_rft_step_checkpoint_validation_fails_before_rollout`.
- [2026-02-23 02:30 UTC] Split RFT flow into dedicated `src/trainer/rft_trainer.py` (`RFTTrainerScaffold`) and kept `SDPOTrainerScaffold` as the SDPO-focused facade with compatibility delegation for `run_onpolicy_rft_step(...)`.
- [2026-02-23 02:30 UTC] Extended on-policy rollout artifacts for GPU runs: collector rows now include `image_name` plus serialized `trajectory_steps`/`trajectory_history`, and `collect_rft_sft_batch_for_steps(...)` now writes `rollout_rows.jsonl` + `rollout_artifact_summary.json` (unique task IDs, task-image pairs, trajectory counts) under `output_dir`.
- [2026-02-23 02:30 UTC] Hardened RFT handoff identity checks to fail fast on empty `task_id` before SFT batch assembly; added regression coverage in `tests/test_onpolicy_rollout_adapter.py`, `tests/test_rft_trainer.py`, and updated `tests/test_sdpo_trainer.py` to resolve settings from real dataset config name (`on_policy_swe_smith`) while preserving deterministic local fakes.
- [2026-02-23 03:08 UTC] Wired live on-policy RFT runtime orchestration into a dedicated module `src/verl_integration/rft_runtime.py` (typed request signature + `rft_runtime_manifest.json` artifact), moved rejection-policy selection logic into `src/verl_integration/rft_rejection.py`, and updated `OnPolicyRFTDataset` to route runtime collection through this explicit handoff layer.
- [2026-02-23 03:44 UTC] Refactored RFT runtime ownership so project-specific handoff logic now lives under `src/trainer/` (`rft_handoff.py`, `rft_runtime.py`, `rft_rejection.py`), while `src/verl_integration/` keeps thin compatibility wrappers only.
- [2026-02-23 10:15 UTC] Rewired `data.on_policy.turn_generator_mode=default` to a live vLLM OpenAI-compatible generator (`src/rollout/vllm_turn_generator.py`) and updated `scripts/run_rft.sh` to load centralized `rft_runtime.loop` defaults (`RFT_STEPS`, `SAMPLES_PER_TASK`, `RFT_TASK_BATCH_SIZE`, `RFT_SFT_NUM_EPOCH_PER_BATCH`) from `configs/runtime/training_policy_defaults.v1.json`.
- [2026-02-23 10:15 UTC] Updated collector/handoff/rejection flow to enforce trajectory-level RFT rejection criteria (any tool formatting failure, missing terminal submit, invalid terminal submit args) via rollout metadata fields (`trajectory_format_valid`, `final_turn_has_submit`, `final_submit_format_valid`), plus regression coverage in `tests/test_onpolicy_collector.py`, `tests/test_onpolicy_rollout_adapter.py`, `tests/test_rft_runtime.py`, and new `tests/test_vllm_turn_generator.py`.
- [2026-02-23 10:15 UTC] Test status: `python3 -m pytest -q` passing (`126 passed, 2 skipped`).
- [2026-02-23 10:45 UTC] Implemented an end-to-end RFT supervisor loop in `src/trainer/rft_runtime_loop.py`: per step it collects live trajectories, writes selected samples to `MultiTurnSFTDataset` parquet (`src/trainer/rft_multiturn_dataset.py`), trains via `torchrun -m verl.trainer.fsdp_sft_trainer` with per-step `data.train_files`, then resolves the newest `global_step_*` checkpoint and points vLLM to the new `huggingface/` snapshot for the next step.
- [2026-02-23 10:45 UTC] Updated `scripts/run_rft.sh` to use this loop by default (`RFT_RUNTIME_MODE=loop`) while preserving `RFT_RUNTIME_MODE=direct` for proof/legacy one-shot launches; updated `scripts/run_rft_onpolicy_rollout_proof.sh` to pin `direct` mode explicitly.
- [2026-02-23 10:45 UTC] Added regression coverage in `tests/test_rft_multiturn_dataset.py` and `tests/test_rft_runtime_loop.py`, plus launcher compatibility checks in `tests/test_run_scripts.py`.
- [2026-02-23 10:47 UTC] Grounded vLLM/verl launcher imports with explicit doc/source links in `scripts/run_rft.sh` and `src/trainer/rft_runtime_loop.py` to keep external module entrypoints tied to authoritative references.
- [2026-02-24 03:02 UTC] Locked realistic validated defaults in `configs/runtime/training_policy_defaults.v1.json` and config resolvers: `collector_max_in_flight_tasks=32` plus 8-GPU vLLM parallelism `TP/DP=2/4`; merged collector-concurrency PR #9 into PR #8 base.
- [2026-02-24 03:26 UTC] Hardened `scripts/run_rft.sh` TP/DP auto-resolution so non-divisible TP overrides now safely fall back to `DP=1` (instead of reusing default DP and producing invalid combinations); added regression `test_run_rft_script_dry_run_nondivisible_tp_override_falls_back_to_dp_one`.
- [2026-02-28 00:00 UTC] Activated runtime SDPO patch path: `main_ppo_entry.py` + `ppo_runtime_patch.py` + `reward_adapter.py` + `reward_loop_score.py`, plus `swe_bridge_agent_loop` registration and `run_sdpo.sh` checkpoint/cache validation. Added end-to-end one-step evidence under `outputs/turn_sdpo_runtime/.../global_step_1`.
- [2026-02-28 00:00 UTC] Updated this blueprint to mark D6 acceptance-run evidence as remaining gap: full monitored run artifacts (`acceptance_summary.md`) still not persisted under `outputs/integration/<run_label>` by automation.
- [2026-03-04 00:00 UTC] Merged `step_sdpo_implementation_plan.md` into this blueprint and formalized D0..D6 deliverables, acceptance gate, contracts, and strict missing-items list (D6 artifacts still absent under `outputs/integration/`).
- [2026-03-05 17:00 UTC] Corrected pilot + SDPO regressions: pilot reward now uses the real verifier-based `reward_fn` (no dummy scoring), SDPO reward is subtraction-based (fail-to-pass/pass-to-pass deltas plus terminal validity penalty), system prompt is injected for SDPO agent-loop messages and pilot teacher reprompts, and the Docker `search` tool now executes `grep -R` even after fallback without suppressing stderr.

---

## 11. Configuration & Type Authority

### 11.1 Type ownership boundaries

- `src/schemas/` owns runtime/domain contracts (tool-call schema, rollout/task sample dataclasses, reward payload contracts).
- `src/env/` owns environment protocol and env-runtime-specific types (request/response structs, container handle/pool structs).
- `src/verl_integration/` owns wiring/adapters only and must import domain types from `schemas`/`env` rather than redefining them.

### 11.2 Tool semantics authority

- Canonical tool names, argument schemas, and validation logic are defined in `src/schemas/contracts.py`.
- Runtime JSON config is allowed to provide policy knobs (counts, thresholds), but not redefine tool schemas.
- The terminal tool policy in config must be validated against schema authority at startup.

### 11.3 Runtime config authority

- `src/config.py` is the import surface for runtime defaults loaded from:
  - `configs/runtime/training_policy_defaults.v1.json`
  - `configs/runtime/phase_transition_gates.v1.json` (for gate thresholds used by trainer/eval logic)
- Shared output-contract exports (for example max/min tool calls, terminal tool) must come from `src/config.py`.
- Prompt/collector/reward adapters must import those shared exports; no hardcoded contract literals.

### 11.4 Model config resolution authority

- Delimiter config resolution order:
  1. `configs/model/<family>.yaml` (repo/user override)
  2. `src/prompts/model_configs/<family>.yaml` (packaged default)
- Bundled defaults remain in `src/prompts/model_configs/` for packaging stability.
- Local experimentation/customization should be done via `configs/model/` overrides.

### 11.5 Single-source policy checks

- Startup validation should fail fast if configured terminal tool is not in schema `ALLOWED_TOOLS` or tool-call bounds are invalid (`min < 1` or `max < min`).
- This keeps runtime config flexible while preserving schema correctness.

### 11.6 Authoritative Config Flow for Step-SDPO Runs

1. Baseline defaults live in `configs/verl/sdpo_swe.yaml`.
2. `scripts/run_sdpo.sh` is the authoritative launcher path and handles interpreter/import/runtime hygiene.
3. CLI/Hydra overrides are run-scoped only.

### 11.7 Source of Truth for Initial Checkpoint

Primary source:
- `outputs/rft_runtime/rft_runtime_loop_manifest.json` -> `.final_model_path`

Fallback source:
- Explicit human-provided completed RFT checkpoint path.

Checkpoint must resolve to a valid directory containing model export artifacts.

### 11.8 Required Runtime Keys for Acceptance Run

- `actor_rollout_ref.model.path=<RFT_CHECKPOINT_PATH>`
- `actor_rollout_ref.rollout.multi_turn.enable=true`
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=<N>`
- `actor_rollout_ref.rollout.multi_turn.max_user_turns=<N>`
- `actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent`
- `actor_rollout_ref.rollout.agent.agent_loop_config_path=<repo>/configs/verl/agent_loops/swe_bridge_agent.yaml`
- `trainer.total_training_steps=1`
- `trainer.default_local_dir=<outputs/integration/<run_label>>`
- rollout prompt dataset path (`data.train_files`)
- config-compatibility validation path (`data.val_files`) should mirror `data.train_files`
- for RL-style e2e acceptance, disable validation rollouts (`trainer.test_freq=0`)

---

## 12. Current Build and Run Commands

### 12.1 Build

Preferred (safe compile parallelism):
```bash
make build-train CORES=2
```

Equivalent `uv` command:
```bash
MAX_JOBS=2 uv sync --python 3.13 --extra train
```

Run tests:
```bash
python3 -m pytest -q
```

### 12.2 Run

Dry-run launcher resolution:
```bash
NPROC_PER_NODE=8 bash scripts/run_rft.sh --dry-run trainer.total_training_steps=1
```

SDPO dry-run:
```bash
SDPO_MONITOR_ENABLE=0 bash scripts/run_sdpo.sh --dry-run trainer.total_training_steps=1
```

Default runtime loop:
```bash
NPROC_PER_NODE=8 WANDB_MODE=offline bash scripts/run_rft.sh
```

SDPO one-step smoke (monitoring on):
```bash
SDPO_MONITOR_ENABLE=1 \
SDPO_TRAINER_LOG_PATH=outputs/turn_sdpo_runtime/$(date -u +%Y%m%dT%H%M%SZ)_sdpo_smoke.trainer.log \
bash scripts/run_sdpo.sh trainer.total_training_steps=1
```

SDPO acceptance scaffold (D6 target):
```bash
bash scripts/run_sdpo.sh \
  actor_rollout_ref.model.path=/path/to/rft/checkpoint \
  trainer.total_training_steps=1 \
  trainer.test_freq=0 \
  data.train_files=/path/to/turn_sdpo_train.parquet \
  data.val_files=/path/to/turn_sdpo_train.parquet
```

Canonical acceptance dry-run example (resolves overrides only):
```bash
RFT_MANIFEST="/path/to/rft_runtime_loop_manifest.json"
RFT_CKPT="$(jq -r '.final_model_path' "${RFT_MANIFEST}")"
test -d "${RFT_CKPT}"

bash scripts/run_sdpo.sh --dry-run \
  actor_rollout_ref.model.path="${RFT_CKPT}" \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$(pwd)/configs/verl/agent_loops/swe_bridge_agent.yaml" \
  trainer.total_training_steps=1
```

Realistic 2-step profile command:
```bash
RFT_STEPS=2 \
SAMPLES_PER_TASK=8 \
RFT_TASK_BATCH_SIZE=64 \
RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT=16 \
SMALL_SWE_VLLM_MAX_TOKENS=512 \
NPROC_PER_NODE=8 \
WANDB_MODE=offline \
bash scripts/run_rft.sh
```

Proof-mode direct path:
```bash
bash scripts/run_rft_onpolicy_rollout_proof.sh
```

Flash-attn constrained rebuild via Slurm:
```bash
bash scripts/run_flash_attn_rebuild.sh
```

### 12.3 Locked 8-GPU runtime defaults

- Source of truth: `configs/runtime/training_policy_defaults.v1.json`
- `rft_runtime.loop.collector_max_in_flight_tasks=32`
- `rft_runtime.vllm_parallelism.by_nproc_per_node.8.tensor_parallel_size=2`
- `rft_runtime.vllm_parallelism.by_nproc_per_node.8.data_parallel_size=4`
- SDPO runtime defaults are now resolved from `run_sdpo.sh` (`data.train_files`/`data.val_files`, checkpoint path, and `actor_rollout_ref.rollout` multi-turn/agent keys).

### 12.4 Acceptance Gate (D6, Authoritative)

Step-SDPO integration is accepted only if a single monitored run satisfies all conditions:

1. Checkpoint load correctness: runtime clearly shows `actor_rollout_ref.model.path` equals the RFT-derived checkpoint.
2. Multi-turn bridge path correctness: `swe_bridge_agent` runs and executes `generate -> bridge -> append tool response -> continue/submit`.
3. SDPO distillation hook correctness: self-distillation batch path executes via local reprompt hook without loss-path fallback errors.
4. Masking correctness: training uses rollout `response_mask` policy (assistant-only supervision), with no custom fine-mask injection in runtime.
5. Step completion: at least one SDPO global step completes and writes outputs.
6. Evidence completeness: run directory includes all required artifacts listed below.

### 12.5 Required Acceptance Artifacts (D6)

Under `outputs/integration/<run_label>/`:
- `launch_command.txt`
- `resolved_runtime_values.json`
- `train.log`
- `acceptance_summary.md`

`acceptance_summary.md` must include explicit pass/fail statements for all six success-gate conditions above.

### 12.6 Suggested Slurm Envelope for Acceptance Run

Heavy runtime must run in Slurm with explicit memory. For SDPO on this node, set
`RAY_TMPDIR` to scratch to avoid `/tmp` disk-pressure crashes. Keep
`TOKENIZERS_PARALLELISM=false` for Ray SDPO workers to avoid tokenizer deadlocks.

```bash
srun --mem=384G --gres=gpu:8 --cpus-per-task=32 --time=04:00:00 bash -lc '
  set -euo pipefail
  cd /path/to/small-swe-train
  export NPROC_PER_NODE=8
  export RAY_TMPDIR=/data/scratch/$USER/ray_tmp/${SLURM_JOB_ID:-manual}
  mkdir -p "$RAY_TMPDIR"

  RFT_MANIFEST=/path/to/rft_runtime_loop_manifest.json
  RFT_CKPT="$(jq -r ".final_model_path" "${RFT_MANIFEST}")"
  RUN_DIR="$(pwd)/outputs/integration/step_sdpo_e2e_from_rft_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "${RUN_DIR}"

  bash scripts/run_sdpo.sh \
    actor_rollout_ref.model.path="${RFT_CKPT}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$(pwd)/configs/verl/agent_loops/swe_bridge_agent.yaml" \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="${RUN_DIR}" \
    data.train_files=/path/to/train_data \
    data.val_files=/path/to/train_data \
    trainer.test_freq=0 \
    2>&1 | tee "${RUN_DIR}/train.log"
'
```

---

## 13. Risks, Non-goals, Completion, Missing Items

### 13.1 Main Risks and Mitigations

1. Agent-loop registration mismatch. Mitigation: local entry module imports/registers loop explicitly; add loop-instantiation test.
2. Metadata missing in rollout samples. Mitigation: strict validation at dataset/loop boundary with explicit field error messages.
3. Response-mask regression in loop path. Mitigation: unit tests asserting assistant tokens are `1`, tool/observation tokens are `0`.
4. Reward adapter schema mismatch. Mitigation: adapter tests for shape/type/alignment, including submit-only terminal row.
5. Distillation truncation mismatch. Mitigation: tokenizer-length truncation enforced in hook path.
6. Container lifecycle leaks during failures. Mitigation: explicit cleanup in success/error/finally paths and post-run leak checks.

### 13.2 Non-goals (This Slice)

1. No runtime introduction of fine-grained SDPO token-label masking.
2. No rewrite of upstream SDPO divergence/loss internals.
3. No long multi-step benchmark campaign before one-step acceptance gate passes.
4. No harness-grade `resolved` replacement in this implementation slice.

### 13.3 Completion Definition for This Slice

This slice is complete when:
1. This blueprint (merged guiding plan) is tracked and reviewed.
2. Runtime wiring in Sections 8-12 stays synchronized with repository state.
3. D6 acceptance artifacts are produced for a monitored SDPO run.

### 13.4 Strict Missing or Pending Items (as of 2026-03-04)

The items below are required for acceptance and are not yet present or captured:

- `outputs/integration/<run_label>/` is empty; no monitored D6 run directory exists yet.
- Required D6 artifacts are missing: `launch_command.txt`, `resolved_runtime_values.json`, `train.log`, `acceptance_summary.md`.
- No `acceptance_summary.md` exists that records explicit pass/fail for all six acceptance-gate conditions.
- Evidence for the six acceptance-gate conditions (checkpoint load correctness, multi-turn bridge path, distillation hook, response-mask policy, step completion, evidence completeness) has not been captured in a monitored run directory.
- No automation in-repo currently writes the D6 artifacts above (the only references are in this blueprint and the merged plan); artifact capture is still manual.
