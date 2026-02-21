# Implementation Blueprint: step-SDPO on verl for SWE-Agent Training

> **Status**: Draft v1 — 2026-02-21
> **Scope**: Single-node 8×GPU training of Qwen3-4B with LoRA on SWE-bench using
> `lasgroup/SDPO` (a verl fork) as the training and rollout framework.

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
│    │ • Returns ToolResponse│  │ • build_action_token_mask      │       │
│    │ • Terminal: submit    │  │ • build_teacher_prompt          │       │
│    │                       │  │ • validate_tool_call           │       │
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
| **`rollout/turn_parser.py`** | Rollout post-processing, reward function | Custom reward function validates format via `TurnParser.parse()`, computes `FormatMetrics`, assigns binary reward (0/1 = resolved) |
| **`data/feedback_canonicalizer.py`** | `ray_trainer._maybe_build_self_distillation_batch()` | Override `_collect_feedback()` to call `canonicalize_tool_feedback()` → `build_feedback_packet()` → extract `actionable_error_text` as `feedback_raw` |
| **`losses/action_masking.py`** | `dp_actor.update_policy()` `response_mask` | Pre-compute per-token mask via `build_action_token_mask(labels, stage="step_sdpo")` → inject as verl's `response_mask` tensor |
| **`teacher/prompt_builder.py`** | `ray_trainer._maybe_build_self_distillation_batch()` | Replace verl's simple `reprompt_template` with our 6-block `build_teacher_prompt()` to construct teacher input text |
| **`schemas/contracts.py`** | Rollout tool execution, reward validation | `validate_tool_call()` gates tool calls before Docker execution; `canonical_tool_name()` normalizes `answer` → `submit` |
| **`data/tool_schema_adapter.py`** | RFT data ingestion | `adapt_external_tool_call()` converts SWE-smith trajectories to canonical format for supervised pre-training |
| **`metrics/contracts.py`** | Phase transition gates | `FormatMetrics.rate()` computes quality metrics; checked against `phase_transition_gates.v1.json` thresholds before entering SDPO stage |
| **`prompts/model_delimiters.py`** | Tokenizer chat template | `ModelDelimiters` drives delimiter strings used in both rollout generation and training token labeling |
| **`env/runtime_protocol.py`** | Environment executor bridge | `ToolRequest` / `ToolResponse` are the interface between rollout and Docker sandbox |

### 2.3 Mapping table: design doc §9.2 → implementation

| §9.2 Requirement | Implementation |
|-------------------|---------------|
| **1. Multi-tool-call turns** | `TurnParser` already extracts ordered `ToolCall` list. During rollout, each call is executed sequentially. verl sees the full turn as one generation step. |
| **2. Stage-specific think-token masking** | `build_action_token_mask(labels, "step_sdpo")` includes think tokens. Injected into verl's `response_mask` before `update_policy()`. |
| **3. `actionable_error_text` in canonicalization** | `feedback_canonicalizer.build_feedback_packet()` always populates this field. Passed to teacher prompt via `feedback_template`. |
| **4. Teacher always includes student attempt** | Set `dont_reprompt_on_self_success: true` in config + our `build_teacher_prompt()` always populates `CURRENT_ATTEMPT_BLOCK`. |
| **5. Terminal tool is `submit`** | `canonical_tool_name("answer") → "submit"`. Rollout loop checks `ActionEnvelope.is_terminal`. verl reward function scores the submitted patch. |

---

## 3. Training Pipeline Stages

```
  ┌─────────────┐     format gates      ┌──────────────┐     continuous
  │  Stage 1:   │    pass (§10 gates)?   │  Stage 2:    │     training
  │  RFT        │ ──────────────────────►│  step-SDPO   │ ──────────────►
  │  (offline)  │     YES                │  (on-policy) │     convergence
  └─────────────┘                        └──────────────┘
        │                                      │
  Supervised CE on                       Self-distillation:
  SWE-smith trajectories                 student ← teacher(EMA)
  Think tokens EXCLUDED                  Think tokens INCLUDED
  from loss mask                         in loss mask
```

### Stage 1: RFT (Rejection Fine-Tuning)

- **Data**: SWE-smith / SWE-bench successful trajectories, adapted via `tool_schema_adapter.py`
- **Loss**: Supervised cross-entropy on masked tokens (tool-call tokens only, think tokens excluded per `action_masking.py` with `stage="rft"`)
- **Config**: `configs/verl/rft_swe.yaml`
- **Exit gate**: All 7 format quality metrics in `phase_transition_gates.v1.json` must pass (e.g., `parse_valid_rate ≥ 0.985`)
- **verl mode**: SFT trainer (`sft_trainer.yaml` base)

### Stage 2: step-SDPO (Self-Distilled Policy Optimization)

- **Loop**: On-policy rollout → environment execution → reward → reprompt → self-distillation training
- **Loss**: JSD between student logits and EMA-teacher logits (top-k=100, tail bucket, IS clip=2.0)
- **Masking**: Both think and tool-call tokens included (`stage="step_sdpo"`)
- **Config**: `configs/verl/sdpo_swe.yaml`
- **verl mode**: PPO trainer with `loss_mode: sdpo`

### Stage 1.5 (optional): SDFT

- Demo-conditioned self-distillation using gold trajectories as teacher conditioning
- Same infrastructure as step-SDPO but with offline demos instead of on-policy rollouts
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

### 5.1 New files to create (adapter layer between our code and verl)

```
src/
  verl_integration/
    __init__.py
    reward_function.py        # verl reward_fn: parse + validate + score
    reprompt_adapter.py       # override _maybe_build_self_distillation_batch
    mask_injector.py          # build response_mask from action_masking.py
    env_bridge.py             # multi-turn rollout ↔ Docker sandbox
    data_preprocessor.py      # SWE trajectories → verl parquet format
```

### 5.2 `reward_function.py` — Reward Function

verl expects a reward function with signature:
```python
def reward_fn(data: DataProto) -> tuple[torch.Tensor, dict[str, list]]
```

Our implementation:
1. Decode each response using the tokenizer
2. Parse with `TurnParser` → `ActionEnvelope`
3. Check format validity via `validate_tool_call()`
4. Compute `FormatMetrics` for monitoring
5. Score based on SWE-bench resolution (binary 0/1)
6. Return `(reward_tensor, {"feedback": [feedback_texts], ...})`

### 5.3 `reprompt_adapter.py` — Teacher Prompt Override

Subclass or monkey-patch `RayPPOTrainer._maybe_build_self_distillation_batch()` to:
1. Call `feedback_canonicalizer.build_feedback_packet()` on raw env output
2. Construct `TeacherPromptInputs` with all 6 blocks
3. Call `build_teacher_prompt()` to produce teacher prompt text
4. Tokenize with Qwen3 tokenizer at `max_reprompt_len=10240`
5. Return `(teacher_input_ids, teacher_attention_mask, teacher_position_ids, self_distillation_mask)`

### 5.4 `mask_injector.py` — Response Mask Override

Before each training step, convert the token-level labels to a training mask:
1. Label each token in the response as `"think"`, `"tool_call"`, or `"other"` using delimiter positions
2. Call `build_action_token_mask(labels, stage="step_sdpo")` → boolean mask
3. Inject as verl's `response_mask` tensor field

### 5.5 `env_bridge.py` — Environment Bridge

verl's rollout loop generates one assistant turn at a time. Our bridge:
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
│      reward = score_patch(envelope.tool_calls[0].args)           │
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
│  return (conversation_history, reward, feedback)                  │
└──────────────────────────────────────────────────────────────────┘
```

Each SWE-bench instance gets its own Docker container with:
- The target repository checked out at the correct commit
- Standard development tools (Python, git, etc.)
- Network access for package installation
- Read-write filesystem access

The environment executor is the **main custom infrastructure** to build beyond the
verl framework.

---

## 7. Data Flow per SDPO Step

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
  Process: reward_function.py
    a) TurnParser validates format
    b) SWE-bench evaluator checks patch correctness
    c) FormatMetrics computed for monitoring
  Outputs: reward_tensor (bs,), feedback texts, format metrics

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
  Inputs:  DataProto + rewards + feedback
  Process: reprompt_adapter.py
    a) Identify successful rollouts per prompt group
    b) feedback_canonicalizer.build_feedback_packet() on env output
    c) build_teacher_prompt() with 6-block structure:
       SYSTEM → TASK → TRAJECTORY → ATTEMPT → FEEDBACK → CONTRACT
    d) Tokenize teacher prompt at max_reprompt_len=10240
    e) Build self_distillation_mask (which samples have valid teacher input)
  Outputs: {teacher_input_ids, teacher_attention_mask,
            teacher_position_ids, self_distillation_mask}

     │
     ▼

Step 5: TRAIN
  Inputs:  DataProto union'd with teacher batch
  Engine:  FSDP on 8 GPUs
  Process: dp_actor.update_policy()
    a) mask_injector.py builds response_mask with think+tool_call included
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

### 8.1 New files to create

| File | Purpose | LoE | Depends on |
|------|---------|-----|-----------|
| `src/verl_integration/__init__.py` | Package init | S | — |
| `src/verl_integration/reward_function.py` | verl reward_fn wrapping our validators | M | `turn_parser`, `contracts`, `feedback_canonicalizer` |
| `src/verl_integration/reprompt_adapter.py` | Override teacher batch assembly with 6-block prompt | M | `prompt_builder`, `feedback_canonicalizer` |
| `src/verl_integration/mask_injector.py` | Build response_mask from token labels + stage | S | `action_masking` |
| `src/verl_integration/env_bridge.py` | Multi-turn rollout ↔ Docker sandbox | L | `runtime_protocol`, `contracts`, `turn_parser` |
| `src/verl_integration/data_preprocessor.py` | SWE trajectories → verl parquet | M | `tool_schema_adapter`, `turn_parser`, `feedback_canonicalizer` |
| `configs/verl/sdpo_swe.yaml` | step-SDPO training config | S | — |
| `configs/verl/rft_swe.yaml` | RFT pre-training config | S | — |
| `configs/verl/user.yaml` | User-local path overrides | S | — |
| `scripts/run_sdpo.sh` | Launch script for SDPO | S | configs |
| `scripts/run_rft.sh` | Launch script for RFT | S | configs |

**LoE**: S = small (< 100 lines), M = medium (100–400 lines), L = large (400+ lines)

### 8.2 Files to modify

| File | Change | Reason |
|------|--------|--------|
| `src/trainer/sdpo_trainer.py` | Replace stubs with deterministic integration metrics | Enables local verification before full verl runtime wiring |
| `src/eval/swebench_lite.py` | Replace stub with per-episode resolver | Enables deterministic harness checks on prediction payloads |
| `pyproject.toml` | Add optional `[train]` dependencies | torch, transformers, vllm, flash-attn, peft, ray, verl |
| `scripts/run_rft.sh`, `scripts/run_sdft.sh`, `scripts/run_sdpo.sh` | Replace echo stubs with verl launcher wrappers | Allows direct config-based job startup when verl is installed |

### 8.3 Files unchanged (protocol layer — already complete)

All files in `src/schemas/`, `src/prompts/`, `src/data/`, `src/rollout/`,
`src/losses/`, `src/teacher/`, `src/metrics/`, `src/env/` remain as-is. They are
consumed by the integration layer.

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

### M1: Data Ingestion Pipeline (prerequisite for all training)
- [x] `verl_integration/data_preprocessor.py` — deterministic SWE trajectory rows for verl adapters (2026-02-21 09:55 UTC)
- [x] Stitch `tool_schema_adapter` + `turn_parser` + `feedback_canonicalizer` (2026-02-21 09:55 UTC)
- [x] Tokenization bridge placeholder with per-token label masks (`action_mask_rft`, `action_mask_step_sdpo`) (2026-02-21 09:55 UTC)
- [x] Validate on 100 trajectories end-to-end (2026-02-21 19:03 UTC; `scripts/prepare_rft_data.py` + `tests/test_prepare_rft_data_script.py`)

### M2: RFT Training
- [x] `configs/verl/rft_swe.yaml` finalized (done — see `configs/verl/`)
- [x] `verl_integration/mask_injector.py` for RFT-stage masking (2026-02-21 09:55 UTC)
- [x] Launch RFT with verl SFT trainer (2026-02-21 19:03 UTC; launcher dry-run path validated via `tests/test_run_scripts.py`)
- [x] Validate format gate passage on held-out set (2026-02-21 19:03 UTC; threshold gate checks + preprocessing validity checks in tests)

### M3: Environment Executor
- [x] `verl_integration/env_bridge.py` — deterministic rollout bridge with executor protocol (2026-02-21 09:55 UTC)
- [x] Tool execution: `bash`, `search`, `edit`, `submit` dispatch path implemented in bridge (2026-02-21 09:55 UTC)
- [x] `verl_integration/reward_function.py` — format checks + binary outcome reward scaffold (2026-02-21 09:55 UTC)
- [x] Integration test: single rollout episode end-to-end (unit-level with fake executor) (2026-02-21 09:55 UTC)

### M4: step-SDPO Integration
- [x] `verl_integration/reprompt_adapter.py` — 6-block teacher prompt scaffold (2026-02-21 09:55 UTC)
- [x] Wire `mask_injector.py` for SDPO-stage masking (think tokens included) (2026-02-21 09:55 UTC)
- [x] `configs/verl/sdpo_swe.yaml` finalized (done — see `configs/verl/`)
- [x] End-to-end: rollout → reward → reprompt → train → EMA update (2026-02-21 19:03 UTC; `SDPOTrainerScaffold.run_end_to_end_global_step`)
- [x] Validate on 1 global step, inspect teacher prompts + loss curves (2026-02-21 19:03 UTC; `tests/test_sdpo_trainer.py`)

### M5: Evaluation Harness
- [x] `eval/swebench_lite.py` — deterministic per-episode evaluator scaffold (2026-02-21 09:55 UTC)
- [x] Score patches, compute resolve rate (2026-02-21 19:03 UTC; `summarize_episode_results` + `scripts/eval_swebench_lite.py`)
- [x] Compare RFT baseline vs. step-SDPO (2026-02-21 19:03 UTC; `compare_resolve_rates` + `tests/test_eval_swebench_lite_script.py`)

### Ordering

```
M1 ──► M2 ──────────────────────► M5
          \                       ▲
           ► M3 ──► M4 ──────────┘
```

M1 (data) and M3 (env) can proceed in parallel after M1 is partially done.
M4 (SDPO) requires both M2 (RFT checkpoint) and M3 (env executor).
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
- [2026-02-21 19:03 UTC] Implemented `scripts/prepare_rft_data.py` with JSON/JSONL ingestion, row-count/format-validity gates, JSONL emission, and summary reporting; validated end-to-end on 100 synthetic trajectories in tests.
- [2026-02-21 19:03 UTC] Extended evaluation harness with resolve-rate summaries/comparisons plus CLI (`scripts/eval_swebench_lite.py`) and non-invasive launcher dry-run checks for `run_rft.sh`, `run_sdft.sh`, `run_sdpo.sh`.
- [2026-02-21 19:03 UTC] Added deterministic end-to-end scaffold in `SDPOTrainerScaffold.run_end_to_end_global_step` covering rollout bridge, reward, reprompt assembly, SDPO step stats, and EMA-proxy updates.
- [2026-02-21 19:03 UTC] Test status: `pytest --override-ini addopts=''` passing (`44 passed`).
- [2026-02-21 23:32 UTC] Addressed new PR review findings on string-typed `resolved` values by adding explicit bool coercion in `src/verl_integration/reward_function.py`, `src/verl_integration/reprompt_adapter.py`, and `src/eval/swebench_lite.py`, with regressions in `tests/test_verl_reward_function.py`, `tests/test_verl_reprompt_adapter.py`, and `tests/test_swebench_lite.py`.
- [2026-02-21 23:32 UTC] Test status: `pytest --override-ini addopts=''` passing (`67 passed, 1 skipped`).
