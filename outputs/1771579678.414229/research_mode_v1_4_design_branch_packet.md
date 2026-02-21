# small-swe-train: Research Mode v1.4 Design Branch Packet

Generated: 2026-02-21 04:32 UTC
Thread: 1771579678.414229
Request reference: 2026-02-21 04:06 UTC (1771646811.801329)

This document answers the requested design points before coding restart.

## 1) Default Step-SDPO for SWE training (with per-turn teacher prompt construction)

### 1.1 Per-turn objects
- `h_t`: step history up to turn `t` (issue text, repo metadata, prior actions, prior observations).
- `a_t`: student action at turn `t` (strict JSON tool call).
- `o_{t+1}`: environment observation after executing `a_t`.
- `f_{t+1}`: canonicalized feedback payload derived from `o_{t+1}`.
- `M_t`: token mask over action tokens only (no observation/user tokens).

### 1.2 Default update
Main stage remains action-token masked Step-SDPO:

`L_step_sdpo = E[ sum_t sum_{n in M_t} KL( pi_theta(.|h_t,a_{t,<n}) || stopgrad(q_phi(.|h_t,f_{t+1},a_{t,<n})) ) ]`

Teacher is EMA-regularized:

`phi <- (1 - beta) * phi + beta * theta`, with v1 default `beta = 0.005`.

### 1.3 Teacher prompt template at each turn (default)
For each step `t`, build a teacher prompt packet in this exact order:

1. `SYSTEM_BLOCK`
- role constraints: "repair the issue; output exactly one JSON action object"
- allowed tools: `bash|search|edit|submit`
- schema reminder: strict envelope + per-tool args

2. `TASK_BLOCK`
- repo id / commit / issue text
- current objective and stop condition

3. `TRAJECTORY_BLOCK`
- compact prior turn summaries (`t-H_ctx ... t-1`)
- each prior turn includes action + short normalized observation digest

4. `CURRENT_ATTEMPT_BLOCK` (conditional)
- include student `a_t` iff feedback is not self-contained
- self-contained test uses locked predicate:
  - `A = has_failing_artifact_identity`
  - `B = has_actionable_error_text`
  - `C = has_localization_hint`
  - include attempt iff `not (A and B and C)`

5. `FEEDBACK_BLOCK` (always)
- canonicalized `f_{t+1}` from env response
- apply deterministic truncation after canonicalization, before wrapper text:
  - head `H=768`, tail `T=768`

6. `OUTPUT_CONTRACT_BLOCK`
- restate exact JSON schema for one action object
- no free-form text

### 1.4 Turn execution loop (default)
1. Student samples `a_t` from `pi_theta(.|h_t)`.
2. Execute `a_t` in dockerized repo env, capture `o_{t+1}`.
3. Normalize `o_{t+1}` -> `f_{t+1}`.
4. Construct teacher prompt using blocks above.
5. Run teacher distribution `q_phi` on same action prefix.
6. Compute masked KL on `M_t` only.
7. Optimizer step on `theta`; update EMA teacher `phi`.

This keeps SDPO single-step mathematically, but applies it at every turn in a multi-turn trajectory.

## 2) Dataset for RFT, and env/dataset map

### 2.1 Default RFT dataset (format stabilization)
RFT dataset default is a mixture focused on strict tool-call validity:

1. `SWE-bench/SWE-smith-trajectories` converted to single-step `(h_t -> a_t)` examples, restricted to our v1 tool set.
2. Synthetic schema-repair pairs generated from our JSON schemas:
- malformed JSON -> corrected JSON
- invalid tool/args -> corrected tool/args
3. Small held-out validation slice of the same schema domains for gate tracking.

Rationale: this directly trains the exact output contract we need before SDPO.

### 2.2 Environment and dataset usage by stage
- `Stage 0 (RFT)`: offline action-format data only (no env rollout).
- `Stage 1 (optional short SDFT)`: same tasks with demonstration-conditioned teacher, still tool-format focused.
- `Stage 2 (main SDPO)`: on-policy rollouts in dockerized SWE environments with textual feedback.

Default v1 environment/data plan:
- Training environments: `SWE-smith` dockerized repo tasks (train split only).
  - https://github.com/SWE-bench/SWE-smith
- Optional scale-up environment: `R2E-Gym` train split.
  - https://github.com/R2E-Gym/R2E-Gym
- Optional compatibility baseline env: `SWE-Gym`.
  - https://github.com/SWE-Gym/SWE-Gym
- Held-out benchmark (no training data leakage): SWE-bench Lite style episodes only.
  - https://github.com/SWE-bench/SWE-bench

## 3) Tech stack (vLLM and trainer source)

Default stack:
- Base model family: 4B coder/instruct checkpoint (LoRA-only adaptation, bf16 compute).
- Core trainer: SDPO codepath built on `verl` (from official SDPO repo).
  - https://github.com/lasgroup/SDPO
  - https://github.com/volcengine/verl
- Rollout/inference backend: `vLLM` workers for high-throughput sampling.
- Environment runner: Docker-backed executor with strict allowlist + denylist policy.
- Data layer: Hugging Face datasets + parquet/arrow preprocessing.
- Tracking: structured JSONL metrics + W&B run logs.

Why this stack:
- We inherit the closest existing SDPO implementation instead of re-deriving optimizer internals.
- vLLM is already part of the SDPO implementation path and is practical for on-policy rollouts.
- `verl` gives scalable distributed trainer plumbing without forcing us into unsupported SDPO abstractions.

## 4) Is there ready-to-use SDPO implementation on GitHub?

Short answer: yes for SDPO core, no for our exact SWE Step-SDPO pipeline.

Current landscape checked on 2026-02-21:
- Official SDPO implementation: https://github.com/lasgroup/SDPO
  - Includes SDPO training code, data prep scripts, and verl-based training/inference integration.
- Official SDFT implementation (for optional pre-stage): https://github.com/sail-sg/sdft

Readiness assessment:
- `lasgroup/SDPO` is usable as the algorithmic base trainer.
- It is not a drop-in end-to-end multi-turn SWE agent trainer with our exact `bash/search/edit/submit` action schema, so we still need project-specific adapters:
  - trajectory-to-step segmentation,
  - teacher prompt builder for per-turn tool feedback,
  - action-token masking and schema-validity gating,
  - SWE-specific env wrappers/eval harness integration.

## 5) Tentative project directories and design rationale

Proposed v1 repository layout (inside `projects/small-swe-train/`):

```text
small-swe-train/
  README.md
  pyproject.toml
  configs/
    model/
    runtime/
    data/
    experiments/
  data/
    manifests/
    prepared/
    cache/
  src/small_swe_train/
    schemas/
    prompts/
    data/
    env/
    rollout/
    trainer/
    losses/
    teacher/
    metrics/
    eval/
  scripts/
    prepare_rft_data.py
    run_rft.sh
    run_sdft.sh
    run_sdpo.sh
    eval_swebench_lite.sh
  benchmarks/
    swebench_lite/
      episodes/
      metadata/
      results/
  outputs/
    <run_label>/
```

Why this structure:
- `schemas/` is isolated so action/feedback contracts are versionable and testable.
- `teacher/` and `prompts/` are separate from `trainer/` so prompt construction can evolve without optimizer churn.
- `env/` is isolated because docker/runtime policy is high-risk and should be independently testable.
- `rollout/` separates vLLM interaction from training logic.
- `losses/` keeps SDPO/SDFT/RFT math implementations explicit and composable.
- `benchmarks/` is separated from `data/` to avoid accidental train/eval contamination.
- `outputs/` remains run-labeled for reproducibility and auditability.

## Design status
- Research mode remains active.
- No training loop regeneration in this packet.
- Next action after human approval: convert this layout and stack into implementation scaffolding on `design` branch.
