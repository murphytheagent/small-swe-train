# step-SDPO Guiding Implementation Plan (Mainline)

- Updated: 2026-02-24 09:40 UTC
- Source thread: `1771841520.464849`
- Base branch target: `main`
- Base reference for this plan: collaborator refined PDF (`F0AGVFQFHTN`) plus PR deliverables (`D0..D6`)
- Scope of this PR phase: planning artifact only (no runtime implementation in this commit)

## 0. Change Log vs Previous PR Plan

This document replaces the earlier concise plan and is now the authoritative guiding doc for step-SDPO implementation.

Major changes in this revision:

1. Expanded to match the refined PDF structure (contracts, phases, risks, and concrete integration boundaries).
2. Kept the locked masking pivot explicit: SDPO training uses rollout-produced `response_mask` only.
3. Added precise deliverables (`D0..D6`) with owner files and acceptance criteria.
4. Made config authority explicit: baseline YAML -> launcher defaults/hygiene -> run-scoped CLI overrides.
5. Defined final success as a monitored end-to-end step-SDPO run initialized from an RFT checkpoint (`final_model_path`).

## 1. Locked Decisions

These are already agreed and should not be reopened during this implementation slice.

1. No custom fine-grained token masking for step-SDPO in runtime.
2. Step-SDPO supervision source is rollout `response_mask` from multi-turn agent loop.
3. Upstream SDPO divergence/loss internals remain unchanged unless a hard blocker is discovered.
4. Multi-turn rollout is required for SWE tool-use trajectories.
5. Local integration layers are allowed only as narrow adapters:
- custom agent loop (`swe_bridge_agent`),
- reward adapter (DataProto <-> row contract),
- self-distillation reprompt assembly hook.

## 2. Definitions and Contracts

### 2.1 Response Masking Contract (Authoritative)

For step-SDPO runtime training:

- `response_mask[t] = 1` iff token `t` is model-generated assistant output.
- `response_mask[t] = 0` iff token `t` is tool response, observation, or user-injected context.

Non-goal in this slice:

- No token-label masking (`think`, `tool_call`, or similar) is injected for SDPO runtime.
- Existing masking helpers may remain as scaffold/test utilities, but they are not the source of truth for SDPO training behavior.

### 2.2 SWE Trajectory Record Contract

Each sample needs metadata available to reward + reprompt path (via DataProto non-tensors or equivalent).

Minimum required fields per sample:

- `prompt` (string): task prompt.
- `task_id` (string).
- `image_name` (string): container image used for tool execution.
- `assistant_response` (string): last assistant turn to score.
- `tool_output` (mapping): last tool execution payload (`stdout`, `stderr`, `exit_code`, metadata).
- `resolved` (bool): current resolved label (Phase-1 heuristic defined below).
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

### 2.3 `resolved` Definition

Phase-1 (integration unblocker heuristic):

`resolved = true` iff all are true:

1. Terminal tool call is `submit`.
2. Submit payload/schema is valid.
3. No tool step has non-zero `exit_code`.
4. No bridge/executor/timeout error flags are present.

Phase-2 (future, out-of-scope here):

- Replace heuristic with harness-based task resolution signal.

## 3. Upstream SDPO Surfaces We Build On

### 3.1 Trainer Architecture Assumptions

We treat these surfaces as stable integration points:

1. Rollout generation path in `RayPPOTrainer.fit`.
2. Reward computation hook path (DataProto-compatible interface).
3. `_maybe_build_self_distillation_batch(...)` path for teacher reprompt assembly.

### 3.2 Multi-turn Assumptions

1. Multi-turn is supported but not default; explicit config enablement is required.
2. Agent loop selection is config-driven; default single-turn behavior is insufficient for SWE tool trajectories.
3. Role-aware mask correctness depends on agent loop behavior, not on generic trainer fallback mask computation.

## 4. Current Local State in `small-swe-train`

### 4.1 Reusable Components Already Present

- Bridge execution primitive:
  - `src/verl_integration/env_bridge.py`
- Docker tool execution and command protocol:
  - `src/env/docker_executor.py`
- Container lifecycle manager:
  - `src/env/container_pool.py`
- On-policy rollout row schema patterns:
  - `src/rollout/onpolicy_collector.py`
- Reward logic used by collector path:
  - `src/verl_integration/reward_function.py`
- Teacher reprompt builder used by local scaffold:
  - `src/verl_integration/reprompt_adapter.py`

### 4.2 Gaps for Real SDPO Runtime Wiring

1. `configs/verl/sdpo_swe.yaml` does not yet explicitly lock required multi-turn + loop keys for this path.
2. No SWE-aware custom multi-turn agent loop is wired into SDPO runtime path.
3. No dedicated DataProto reward adapter in SDPO path that wraps existing row-based reward function.
4. `run_sdpo.sh` does not yet fully enforce launcher/import hygiene equivalent to proven `run_rft.sh` patterns.
5. SDPO entrypoint patch path for reprompt hook is not yet finalized.
6. Metadata propagation guarantees for `task_id/image_name/...` into rollout loop need explicit enforcement checks.

## 5. Implementation Sequence (Exact Order)

### Phase 0: Doc and Policy Alignment for Masking Pivot

Objective:

- Align all design docs so step-SDPO mask source is rollout `response_mask`.

Actions:

1. Update docs to remove language that step-SDPO runtime mask comes from token-label masking utilities.
2. Mark scaffold mask helpers as non-runtime for SDPO path.

Exit criteria:

- No design doc claims that SDPO runtime `response_mask` is produced by token-label injection.

### Phase A: Runner Hygiene and Authoritative Entrypoint

Objective:

- Make SDPO launch path as reliable as RFT launch path.

Actions:

1. Update `scripts/run_sdpo.sh`:
- export `PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"`.
- pick `python3` fallback consistently.
- launch project-local SDPO entry module for registration/patch side effects.
2. Add local SDPO entry module (for narrow runtime glue only).

Exit criteria:

- `bash scripts/run_sdpo.sh --dry-run ...` resolves a command that imports local integration modules without path issues.

### Phase B: Baseline Upstream Multi-turn Sanity (No Bridge Loop Yet)

Objective:

- Confirm multi-turn plumbing works before custom loop insertion.

Actions:

1. Explicitly set these keys in `configs/verl/sdpo_swe.yaml`:
- `actor_rollout_ref.rollout.multi_turn.enable`
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns`
- `actor_rollout_ref.rollout.multi_turn.max_user_turns`
- `actor_rollout_ref.rollout.agent.default_agent_loop`
- `actor_rollout_ref.rollout.agent.agent_loop_config_path` (for `swe_bridge_agent` stage)
2. Sanity run with baseline loop (`tool_agent`) before switching to `swe_bridge_agent`.

Exit criteria:

- Multi-turn is active in resolved config.
- Rollout batch includes non-trivial `response_mask` semantics.

### Phase C: Task Metadata Propagation into SDPO Rollouts

Objective:

- Guarantee rollout loop has `task_id/image_name/...` per sample.

Actions:

Preferred path:

1. Add SDPO prompt dataset adapter that emits prompt plus required metadata fields.
2. Wire dataset class in `sdpo_swe.yaml`.
3. Add strict validation fail-fast when required metadata keys are missing.

Fallback path:

- If upstream data path preserves columns directly, keep it but still add explicit validation and clear errors.

Exit criteria:

- Agent loop receives `task_id`, `image_name`, prompt text for each sample.

### Phase D: Implement `swe_bridge_agent` Loop

Objective:

- Replace generic tool-loop execution with local SWE bridge semantics while preserving SDPO multi-turn behavior.

Turn-state behavior (per sample):

1. Generate assistant turn.
2. Run `run_env_bridge_step(...)` using local executor.
3. Append assistant turn plus tool-response blocks to context.
4. Continue until terminal submit.

Environment lifecycle requirements:

1. Start per-sample container from `image_name`.
2. Optionally apply patch using collector-consistent logic.
3. Reuse container across turns within sample.
4. Always cleanup on normal completion and all error paths.

Response-mask requirements:

1. Assistant generated tokens -> `1`.
2. Tool/observation inserted context -> `0`.
3. No extra fine-mask override.

Exit criteria:

- `swe_bridge_agent` selected and instantiated.
- Docker lifecycle is deterministic and leak-free for smoke run.
- Multi-turn trajectory includes tool-response blocks and correct mask semantics.

### Phase E: Reward Adapter for DataProto Path

Objective:

- Reuse existing reward function in PPO DataProto flow without changing SDPO loss internals.

Actions:

1. Build reward adapter module:
- map DataProto sample -> reward row mapping,
- call existing `reward_fn(...)`,
- return reward tensor + extras (`feedback` at minimum).
2. Ensure submit-only terminal rows (`tool_output={}`) are handled without special crashes.

Exit criteria:

- PPO reward computation runs without shape/type mismatch.
- `feedback` extras remain aligned with sample indices.

### Phase F: Self-distillation Reprompt Hook

Objective:

- Swap only teacher prompt construction to local SWE reprompt adapter.

Actions:

1. Override or patch `_maybe_build_self_distillation_batch(...)` in local entry path.
2. Use `reprompt_adapter` to build teacher prompts.
3. Produce required tensors (`teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`, `self_distillation_mask`).
4. Enforce tokenizer-length-safe truncation at hook boundary.

Important:

- Keep upstream SDPO divergence and core loss math unchanged.

Exit criteria:

- Distillation tensors are produced with stable shapes.
- Hook integrates without fallback path breakage.

### Phase G: Verification and Smoke E2E

Objective:

- Prove end-to-end wiring with minimal but real run.

Validation order:

1. Dry-run command render and config snapshot.
2. One-step training run (`trainer.total_training_steps=1`).
3. Confirm runtime evidence:
- RFT checkpoint path loaded,
- `swe_bridge_agent` active,
- reward extras present,
- distillation batch path active,
- rollout `response_mask` policy in effect.
4. Verify cleanup:
- no leaked containers,
- no lingering orphan processes.

Exit criteria:

- One complete SDPO training step finishes with required artifacts.

## 6. Deliverables (`D0..D6`) and Acceptance Criteria

| ID | Deliverable | Primary Files | Acceptance Criteria |
| --- | --- | --- | --- |
| D0 | Root guiding plan in PR | `step_sdpo_implementation_plan.md` | This full guiding plan is tracked in PR, superseding concise version. |
| D1 | Explicit multi-turn and loop defaults in SDPO config | `configs/verl/sdpo_swe.yaml` | Required keys are explicit in YAML (not only CLI overrides). |
| D2 | SWE bridge agent loop integration | `configs/verl/agent_loops/swe_bridge_agent.yaml`, `src/verl_integration/swe_bridge_agent_loop.py` | Per-turn bridge call integrated between generation and next prompt assembly; terminal submit edge case deterministic. |
| D3 | SDPO trainer reprompt hook integration | `src/verl_integration/main_ppo_entry.py` (or equivalent patch module) | `_maybe_build_self_distillation_batch` uses local reprompt adapter; upstream SDPO loss math unchanged. |
| D4 | DataProto reward adapter | `src/verl_integration/reward_adapter.py` + integration call site | Reward tensor and feedback extras are produced in PPO path with stable sample alignment. |
| D5 | SDPO launcher hygiene | `scripts/run_sdpo.sh` | Launcher exports PYTHONPATH, uses consistent python fallback, and executes local SDPO entry path. |
| D6 | Monitored e2e step-SDPO run from RFT checkpoint | `outputs/integration/<run_label>/...` | One run from RFT checkpoint completes and satisfies Section 8 success gate with required evidence artifacts. |

## 7. Authoritative Config Flow

This flow is mandatory for all step-SDPO runs.

1. Baseline defaults are defined in `configs/verl/sdpo_swe.yaml`.
2. `scripts/run_sdpo.sh` is the authoritative launcher path and handles interpreter/import/runtime hygiene.
3. CLI/Hydra overrides are run-scoped only.
4. Initial model checkpoint must be sourced from a completed RFT run.

### 7.1 Source of Truth for Initial Checkpoint

Primary source:

- `outputs/rft_runtime/rft_runtime_loop_manifest.json` -> `.final_model_path`

Fallback source:

- Explicit human-provided completed RFT checkpoint path.

Checkpoint must resolve to a valid directory containing model export artifacts.

### 7.2 Required Runtime Keys for Acceptance Run

- `actor_rollout_ref.model.path=<RFT_CHECKPOINT_PATH>`
- `actor_rollout_ref.rollout.multi_turn.enable=true`
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=<N>`
- `actor_rollout_ref.rollout.multi_turn.max_user_turns=<N>`
- `actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent`
- `actor_rollout_ref.rollout.agent.agent_loop_config_path=<repo>/configs/verl/agent_loops/swe_bridge_agent.yaml`
- `trainer.total_training_steps=1`
- `trainer.default_local_dir=<outputs/integration/<run_label>>`
- rollout prompt dataset path (`data.train_files`)
- for RL-style e2e acceptance, clear offline validation file binding (`~data.val_files`)

### 7.3 Canonical Dry-run Example

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

## 8. Final Success Gate (Authoritative)

Step-SDPO integration is accepted only if a single monitored run satisfies all conditions below.

1. Checkpoint load correctness:
- runtime clearly shows `actor_rollout_ref.model.path` equals RFT-derived checkpoint.
2. Multi-turn bridge path correctness:
- `swe_bridge_agent` runs and executes `generate -> bridge -> append tool response -> continue/submit`.
3. SDPO distillation hook correctness:
- self-distillation batch path executes via local reprompt hook without loss-path fallback errors.
4. Masking correctness:
- training uses rollout `response_mask` policy (assistant-only supervision), with no custom fine-mask injection in runtime.
5. Step completion:
- at least one SDPO global step completes and writes outputs.
6. Evidence completeness:
- run directory includes all required artifacts listed below.

### Required Acceptance Artifacts (`D6`)

Under `outputs/integration/<run_label>/`:

- `launch_command.txt`
- `resolved_runtime_values.json`
- `train.log`
- `acceptance_summary.md`

`acceptance_summary.md` must include explicit pass/fail statements for all six success-gate conditions above.

### Suggested Slurm Envelope for Acceptance Run

Heavy runtime must run in Slurm with explicit memory.

```bash
srun --mem=384G --gres=gpu:8 --cpus-per-task=32 --time=04:00:00 bash -lc '
  set -euo pipefail
  cd /home/murphy/projects/small-swe-train-runtime
  export NPROC_PER_NODE=8

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
    ~data.val_files \
    2>&1 | tee "${RUN_DIR}/train.log"
'
```

## 9. Main Risks and Mitigations

1. Agent-loop registration mismatch.
- Mitigation: local entry module imports/registers loop explicitly; add loop-instantiation test.
2. Metadata missing in rollout samples.
- Mitigation: strict validation at dataset/loop boundary with explicit field error messages.
3. Response-mask regression in loop path.
- Mitigation: unit tests asserting assistant tokens are `1`, tool/observation tokens are `0`.
4. Reward adapter schema mismatch.
- Mitigation: adapter tests for shape/type/alignment, including submit-only terminal row.
5. Distillation truncation mismatch.
- Mitigation: tokenizer-length truncation enforced in hook path.
6. Container lifecycle leaks during failures.
- Mitigation: explicit cleanup in success/error/finally paths and post-run leak checks.

## 10. Non-goals (This Slice)

1. No runtime introduction of fine-grained SDPO token-label masking.
2. No rewrite of upstream SDPO divergence/loss internals.
3. No long multi-step benchmark campaign before one-step acceptance gate passes.
4. No harness-grade `resolved` replacement in this implementation slice.

## 11. Completion Definition for This Planning PR

This planning PR phase is complete when:

1. This root guiding plan is tracked and reviewed.
2. Implementation work follows Sections 5-8 in order.
3. Runtime coding starts only after collaborator confirms this guiding doc is acceptable.
