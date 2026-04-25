# Stability Modules Implementation Plan

Generated: 2026-03-11 03:34 UTC
Updated: 2026-03-16 23:08 UTC
Thread: 1773182046.432339
Status: planning, no implementation yet

## 1. Goal

Stabilize the training pipeline around a clear staged contract:

- `format_rft`
- `positive_rft` as an optional stage
- `turn_sdpo`

This plan replaces the old `rft -> sdft_optional -> sdpo_main` story with a code-grounded implementation sequence that removes SDFT, adds optional positive-RFT to the canonical pipeline contract, then makes RFT outer-loop resume safe, and finally implements positive-RFT as a verifier-backed stage.

## 2. Implementation Order

### Phase 1: Cleanup and pipeline contract update

First:

- delete SDFT from the active runtime, config, and docs path
- add `positive_rft` as an optional stage in the canonical pipeline contract
- clean up masking and telemetry contracts so later stages build on one stable definition

### Phase 2: RFT outer-loop resume

Second:

- implement resume for RFT only
- scope resume to the latest committed outer-loop checkpoint only
- do not support resume from arbitrary older steps

### Phase 3: Positive-RFT implementation

Third:

- implement `positive_rft` as a real stage
- require verifier-backed correctness for selection
- do not use heuristic resolution as the positive selector

## 3. Code-Grounded Current State

### 3.1 Pipeline drift

Current `main` still reflects the old pipeline in multiple places:

- `configs/runtime/training_policy_defaults.v1.json` still advertises `rft -> sdft_optional -> sdpo_main`
- `scripts/run_sdft.sh` still exists as a first-class launcher
- `../README.md`, `design.md`, and launcher docs still contain `sdft`, `sdpo_main`, and `step_sdpo` naming

### 3.2 Masking drift

Mask semantics are currently split across multiple authorities:

- `src/losses/action_masking.py`
  - `rft` trains only `tool_call`
  - `step_sdpo` trains `think` plus `tool_call`
- `configs/runtime/training_policy_defaults.v1.json`
  - says both `rft` and `step_sdpo` exclude thinking and include only `tool_call`
- `src/verl_integration/swe_bridge_agent_loop.py`
  - uses `response_mask` so every assistant-generated token is trainable and environment/tool feedback is masked out

The cleanup phase should reduce this to one authority: assistant-turn supervision on the instruct path.

### 3.3 RFT selection and verifier state

Current RFT stays format-first:

- `src/trainer/rft_rejection.py` selects on terminal submit plus format-validity checks and optional error filters
- `require_resolved` is `false` by default
- `src/trainer/rft_runtime.py` forces `verify_submissions=false` for the shared RFT runtime helper

Current reward logic already exposes the verifier signals needed for a later positive-RFT stage:

- `src/verl_integration/reward_function.py` emits `fail_to_pass_verified`
- `src/verl_integration/reward_function.py` emits `pass_to_pass_verified`
- `src/verl_integration/reward_function.py` emits `reward_verification_missing`
- `src/verl_integration/reward_function.py` distinguishes `missing_verifier` from `missing_verifier_targets` through `resolved_source`

### 3.4 Resume is not safe yet

Current RFT artifacts are useful but not sufficient for resume:

- `src/trainer/rft_runtime_loop.py` writes per-step summaries, train/eval parquet shards, and checkpoints
- `scripts/run_sdpo.sh` can already warm-start from RFT manifests

But current behavior is not replay-safe:

- `src/trainer/rft_runtime_loop.py` unconditionally calls `reset_step_artifacts(step_dir)` before each outer step
- old checkpoints and payloads are pruned by `checkpoint_keep_last` recency, not by committed state
- `scripts/run_sdpo.sh` resolves from manifest fields, not from an authoritative latest committed checkpoint record

## 4. Target Pipeline Contract

After Phase 1, the canonical pipeline story should be:

- `format_rft -> positive_rft -> turn_sdpo`

Where:

- `format_rft` is the required initial stage
- `positive_rft` exists in the contract but remains optional
- `turn_sdpo` stays the final stage

Important distinction:

- Phase 1 adds `positive_rft` to the official contract and deletes SDFT
- Phase 3 is where `positive_rft` becomes a real verifier-backed runtime stage

## 5. Phase 1 Plan: Cleanup and Contract Unification

### 5.1 Objectives

- remove SDFT from the active runtime/config/docs path
- add `positive_rft` as the optional intermediate stage in the canonical pipeline contract
- unify masking around assistant-turn supervision
- normalize telemetry names and stage vocabulary before resume or positive-RFT implementation

### 5.2 Concrete changes

#### A. Remove SDFT from active surfaces

- update `configs/runtime/training_policy_defaults.v1.json`
  - replace `sdft_optional` / `sdpo_main` naming
  - add `positive_rft` as the optional stage
- remove `scripts/run_sdft.sh` from the active launcher set
- update `../README.md`, `design.md`, and Slurm docs
- remove or archive SDFT-specific references that still imply it is part of the intended path

#### B. Make the stage vocabulary consistent

Normalize stage naming across configs, docs, and runtime-facing outputs:

- `format_rft`
- `positive_rft`
- `turn_sdpo`

Do not keep a second active vocabulary such as `sdpo_main` or `step_sdpo` for the pipeline story. Internal module names can remain temporarily if needed, but user-facing and config-facing stage names should be unified.

#### C. Collapse masking to assistant-turn supervision

For this instruct path:

- everything assistant-generated is trainable action
- tool responses and environment feedback stay masked out
- do not rely on `think` vs `tool_call` loss-mask distinctions

Expected cleanup:

- update `src/losses/action_masking.py`
- update `src/verl_integration/data_preprocessor.py`
- update `configs/runtime/training_policy_defaults.v1.json`
- keep any remaining token labels only if they are still useful for debugging or analytics, not as training authority

#### D. Add minimal mask and stage telemetry

Introduce shared telemetry fields that later phases can rely on without inferring from overloaded booleans.

Required shared fields:

- `stage`
- `stage_accepted`
- `stage_decision_reason`
- `trajectory_format_valid`
- `final_turn_has_submit`
- `terminal_format_valid`
- `assistant_action_token_count`
- `trajectory_assistant_turns`

Verifier fields to preserve code-grounded distinctions:

- `verifier_status` in `{correct, incorrect, missing}`
- `verifier_resolution_source` in `{verifiable_tests, missing_verifier, missing_verifier_targets}`

Alias existing names deliberately:

- `rft_selected` -> `stage_accepted`
- `rft_rejection_reason` -> `stage_decision_reason`
- `final_submit_format_valid` -> `terminal_format_valid`

Do not create a second parallel telemetry family that duplicates old and new names indefinitely.

### 5.3 Exit criteria

Phase 1 is complete when:

- SDFT is removed from the active runtime story
- `positive_rft` exists as the optional stage in the canonical contract
- masking no longer depends on `think` vs `tool_call` distinctions on the instruct path
- telemetry names are stable enough for resume and positive-RFT to consume

## 6. Phase 2 Plan: RFT Outer-Loop Resume

### 6.1 Scope

Resume support in this phase is intentionally narrow:

- RFT only
- outer loop only
- latest committed outer-loop checkpoint only

Explicit non-goals:

- no inner trainer-state resume
- no SDPO continuation semantics
- no resume from arbitrary older steps
- no attempt to salvage half-finished steps
- no user-facing step-targeted resume API

### 6.2 Required contract

Add one authoritative resume record for the outer loop. This can be a committed-step journal or a single latest-checkpoint pointer, but the contract must be:

- there is one authoritative latest committed outer-loop checkpoint
- resume always starts from that checkpoint
- anything after that committed point is uncommitted and may be replayed or discarded

Each committed record should capture at least:

- stage name
- outer step index
- checkpoint path(s) that define the next model start point
- the selection contract used for that step
- whether correctness signals were verifier-backed or heuristic

### 6.3 Required runtime changes

Resume is not safe unless the runtime changes alongside the journal:

- make `reset_step_artifacts(...)` commit-aware
- make checkpoint pruning commit-aware
- make payload pruning commit-aware
- ensure latest committed checkpoint discovery happens before any destructive cleanup
- make `scripts/run_sdpo.sh` prefer the authoritative latest committed checkpoint instead of raw manifest heuristics

### 6.4 Recovery rule

- Resume only from the latest committed outer-loop checkpoint.
- If that checkpoint is missing, incomplete, or inconsistent, fail closed or require a fresh run.
- Do not silently fall back to older steps.

### 6.5 Exit criteria

Phase 2 is complete when:

- rerunning an RFT output dir does not destroy the authoritative latest committed checkpoint
- resume deterministically starts from the latest committed outer-loop checkpoint
- pruning cannot delete the checkpoint resume depends on

## 7. Phase 3 Plan: Verifier-Backed Positive-RFT

### 7.1 Stage purpose

`positive_rft` is a short follow-up stage after `format_rft` that selects only verifier-positive trajectories for additional RFT training.

This stage must use the verifier for positive selection. It is not a heuristic-resolved stage.

### 7.2 Launch semantics

- start from the final `format_rft` checkpoint
- open a new run directory and its own manifest/journal namespace
- enable verifier-backed correctness collection for this stage

### 7.3 Selection contract

Selector rules:

- select only explicit verifier-positive rows
- do not use reward sign as the selector
- do not use heuristic `resolved` fallback as the positive selector
- do not gate selection on terminal submit format or final submit presence

Telemetry must still surface:

- terminal-format status
- malformed-but-correct rows
- verifier-missing rows
- verifier-missing-targets rows
- selector yield per batch

### 7.4 Verifier semantics

For this stage:

- `verify_submissions` must be enabled
- positive rows are rows with explicit verifier-backed correctness
- rows with `missing_verifier` on verifier-eligible tasks are not positive
- rows with `missing_verifier_targets` must be surfaced separately and must not be silently treated as verifier negatives

### 7.5 Stop condition

Evaluate stop only on a finalized collection batch.

Definitions:

- `collected_count` = total trajectories collected in the batch
- `positive_count` = explicit verifier-positive trajectories
- `eligible_count` = trajectories whose verifier source is not `missing_verifier_targets`
- `positive_rate = positive_count / eligible_count`

Default stop rule:

- `collected_count >= 2048`
- `positive_count >= 40`
- `positive_rate >= 0.02`

If the stage runs only on verifier-targeted tasks, then `eligible_count == collected_count`. Otherwise, `missing_verifier_targets` must remain visible and excluded from the rate denominator.

### 7.6 Exit criteria

Phase 3 is complete when:

- `positive_rft` runs as a real stage rather than a contract placeholder
- selection uses explicit verifier-backed correctness
- the stage can report malformed-but-correct and missing-verifier behavior cleanly
- the stop condition is enforced on finalized collection batches

## 8. Shared Telemetry Contract

### 8.1 Per-trajectory fields

Minimum shared record:

- `run_id`
- `stage`
- `outer_step`
- `trajectory_id`
- `instance_id`
- `stage_accepted`
- `stage_decision_reason`
- `trajectory_format_valid`
- `final_turn_has_submit`
- `terminal_format_valid`
- `verifier_status`
- `verifier_resolution_source`
- `assistant_action_token_count`
- `trajectory_assistant_turns`
- `reward_total` when applicable

### 8.2 Per-step summary fields

- `collected_count`
- `stage_accepted_count`
- `stage_accepted_rate`
- `verifier_correct_count`
- `verifier_correct_rate`
- `verifier_missing_count`
- `verifier_missing_rate`
- `verifier_missing_targets_count`
- `verifier_eligible_count`
- `terminal_format_valid_count`
- `terminal_format_valid_rate`
- `malformed_but_correct_count`
- `empty_action_mask_count`
- `collection_duration_s`
- `train_duration_s`
- existing train and val losses

## 9. Risks and Guardrails

### 9.1 Cleanup risk

If Phase 1 changes naming but leaves old stage or mask semantics live in configs, the codebase will keep drifting under a new label instead of getting simpler.

### 9.2 Resume risk

If Phase 2 adds a journal without changing reset and pruning behavior, resume will still be unsafe.

### 9.3 Positive-RFT risk

If Phase 3 uses reward sign or heuristic `resolved` instead of explicit verifier results, the stage will not match its intended contract.

### 9.4 Format risk after masking cleanup

Once masking stops enforcing old tool-call-only biases, format behavior must be enforced by stage selection plus reward and telemetry, not by implicit loss-mask behavior.

## 10. Summary

The plan is:

1. Clean up the pipeline, delete SDFT, add `positive_rft` as an optional stage in the canonical contract, and unify masking plus telemetry.
2. Implement RFT resume for the outer loop only, from the latest committed outer-loop checkpoint only.
3. Implement `positive_rft` as a real verifier-backed stage that selects only explicit verifier-positive rows.
