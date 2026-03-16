# Stability Modules Follow-Up Plan

Generated: 2026-03-11 03:34 UTC
Updated: 2026-03-11 04:02 UTC
Thread: 1773182046.432339
Status: review-only, no implementation yet

This PR note is intended to carry the full detailed plan from the latest review packet, not a shortened summary. It reflects the current thread decisions and the current code-grounded state of `main`.

## 1. Updated decisions from the thread

- Keep the canonical active runtime path as `format_rft -> turn_sdpo` for now.
- Remove SDFT from the active pipeline instead of keeping it as an optional stage.
- Do not rely on think-token parsing or token-type-aware loss masks on the instruct path.
- Treat everything the assistant generates as trainable action and keep masking only at the assistant-turn vs environment-feedback boundary.
- Keep the first resume milestone entirely inside RFT.
- Keep positive-RFT as a fresh short run after format-RFT, with correctness-only selection and no terminal-format gate.
- Use a batch-level stop condition for positive-RFT: with the default `256 * 8 = 2048` collected trajectories, stop only once the batch contains about `40` verifier-correct rows, i.e. both `positive_count >= 40` and `positive_rate >= 0.02`.

## 2. Code-grounded state of main

### 2.1 Pipeline drift is still real

- `configs/runtime/training_policy_defaults.v1.json` still advertises `rft -> sdft_optional -> sdpo_main`.
- `README.md`, `IMPLEMENTATION_BLUEPRINT.md`, and launch docs still carry `step_sdpo` / `sdpo_main` / `sdft` naming and assumptions in different places.
- `scripts/run_sdft.sh` still exists as a first-class launcher even though the latest human direction says SDFT is not part of the intended pipeline.

### 2.2 Masking contracts are inconsistent across surfaces

- Offline preprocessing still labels `think`, `tool_call`, and `other` spans in:
  - `src/data/tokenization.py`
  - `src/verl_integration/data_preprocessor.py`
  - `src/losses/action_masking.py`
- RFT currently trains only `tool_call` tokens, while the SDPO-side helper says `step_sdpo` trains `think` plus `tool_call`.
- The live SDPO rollout path already uses a simpler contract:
  - `src/verl_integration/swe_bridge_agent_loop.py` builds `response_mask` so every assistant-generated token is trainable and every tool-response/user-feedback token is masked out.
- The codebase therefore already has two competing mask authorities. Cleanup should reduce them to one.

### 2.3 RFT selection is still format-first

- `src/trainer/rft_rejection.py` currently accepts rows based on:
  - terminal submit present
  - trajectory / final submit format validity
  - optional collector / bridge / parse / validation filters
- `require_resolved` is `false` by default, so correctness is not part of the live mainline selector.
- The shared RFT runtime helper also forces `verify_submissions=false`, so even stricter handoff overrides would still default to heuristic resolution unless that is changed deliberately for positive-RFT.

### 2.4 Reward already exposes the signals needed for correctness-only selection

- `src/verl_integration/reward_function.py` already separates:
  - verifier correctness
  - terminal submit / format validity
  - the terminal-format penalty applied to total reward
- The reward info already exposes:
  - `fail_to_pass_verified`
  - `pass_to_pass_verified`
  - `reward_verification_missing`
  - format metrics
- This is enough to define a correctness-only positive selector without overloading reward sign.

### 2.5 RFT already has most of the artifact surface needed for a replay-safe resume

- `src/trainer/rft_runtime_loop.py` already writes:
  - collector JSONL/meta files
  - accepted train/eval parquet shards
  - `rft_step_summary.json`
  - `rft_runtime_loop_manifest.json`
  - per-step HF checkpoints and optional vLLM-ready merged checkpoints
- `scripts/run_sdpo.sh` already knows how to consume those manifests to warm-start SDPO from an RFT checkpoint.
- What is missing is not more artifacts; it is an authoritative committed-step cursor with atomic finalization semantics.

## 3. Athena cross-check

Athena converged on the same overall order and agreed with the main code-grounded conclusion: keep `terminal_format_valid`, `verifier_status`, and `stage_accepted` separate everywhere. Athena added one useful correction: do not put `positive_rft` into the canonical pipeline contract until the stage actually exists, or the cleanup pass will create a second round of drift immediately after deleting SDFT.

## 4. Revised implementation order

### 4.1 Contract cleanup first

Goals for the cleanup patch:

- remove SDFT from the active runtime/docs/config path
- make the active runtime story explicitly `format_rft -> turn_sdpo`
- collapse mask authority onto assistant-turn supervision
- add minimal mask-sanity telemetry in the same patch

The mask-sanity telemetry matters because a bad simplification can silently produce all-zero or near-zero trainable masks while launches still appear healthy.

### 4.2 Shared telemetry second

The telemetry patch should come before resume and positive-RFT so both later modules depend on clean, non-overloaded signals instead of inferring state from reward sign or generic success booleans.

### 4.3 Replay-safe RFT resume third

The first resume milestone should be narrow:

- only outer-loop RFT
- no inner trainer-state continuation
- no PPO / SDPO continuation

The recovery boundary should be "last fully committed outer step" only.

### 4.4 Positive-RFT fourth

Positive-RFT should be implemented only after the format-RFT shell is replay-safe and telemetry can already show:

- correctness-positive rate
- malformed-but-correct rate
- selector yield per batch
- empty / too-small selected sets

## 5. Minimal telemetry contract

### 5.1 Shared per-trajectory record

Keep this local-only in JSONL or Parquet and join it to W&B via stable IDs.

Required keys:

- `run_id`
- `stage` in `{format_rft, positive_rft, turn_sdpo}`
- `outer_step`
- `trajectory_id`
- `instance_id`
- `stage_accepted`
- `stage_decision_reason`
- `final_turn_has_submit`
- `terminal_format_valid`
- `terminal_format_failure_code` (nullable)
- `verifier_status` in `{correct, incorrect, missing}`
- `assistant_action_token_count`
- `trajectory_assistant_turns`
- `reward_total` when the stage actually uses reward

Everything else already exposed today, such as `trajectory_steps`, `trajectory_turn_tool_response_blocks`, verification payloads, or SDPO self-distillation internals, can stay as optional stage-specific extras rather than becoming mandatory shared columns.

### 5.2 Shared per-step summary

Emit the following to W&B and mirror them into a small local step JSON:

- `collected_count`
- `stage_accepted_count`
- `stage_accepted_rate`
- `verifier_correct_count`
- `verifier_correct_rate`
- `verifier_missing_count`
- `verifier_missing_rate`
- `terminal_format_valid_count`
- `terminal_format_valid_rate`
- `malformed_but_correct_count`
- `empty_action_mask_count`
- `collection_duration_s`
- `train_duration_s`
- existing train and val losses

`malformed_but_correct_count` is the extra field worth insisting on, because positive-RFT deliberately keeps correctness separate from terminal format.

## 6. RFT-only resume contract

### 6.1 What should become authoritative

Add a committed-step journal separate from analytics manifests. Each committed record should minimally capture:

- stage name
- committed outer step index
- checkpoint path(s) that define the next start model
- the selection contract used for that step
- whether correctness signals were verifier-backed or heuristic

### 6.2 Recovery rule

- Resume only from the last committed outer step.
- Any later step directory without a committed journal entry is uncommitted and should be replayed or discarded.
- Pruning must happen only below the committed cursor.

### 6.3 Explicit non-goals for this milestone

- no inner trainer resume
- no attempt to salvage a half-finished step
- no SDPO continuation semantics

## 7. Positive-RFT contract

### 7.1 Launch semantics

- Start from the final format-RFT checkpoint.
- Open a new run directory and new manifest/journal namespace.
- Recollect trajectories with verifier-backed correctness enabled.

### 7.2 Selector semantics

- Select only explicit verifier-positive rows.
- Do not use total reward sign.
- Do not gate on terminal submit format or final submit presence.
- Keep terminal-format fields in telemetry so malformed-but-correct behavior is visible.

### 7.3 Stop condition

Evaluate only on a fully finalized collection batch.

Definitions:

- `collected_count` = total collected trajectories in the batch
- `positive_count` = number of trajectories with `verifier_status=correct`
- `positive_rate = positive_count / collected_count`

Default stop rule:

- `collected_count >= 2048`
- `positive_count >= 40`
- `positive_rate >= 0.02`

This matches the human's example and avoids a smaller-than-expected batch passing on fraction alone.

### 7.4 Failure modes to call out explicitly

- malformed-but-correct rows are admissible by design
- verifier-missing rows stay in the denominator and never count as positive
- positive selection can be too small for a meaningful train step, so the stage should surface this as a first-class stop / continue reason rather than silently skipping into trainer behavior

## 8. Recommended implementation decisions

1. Keep the canonical runtime path as `format_rft -> turn_sdpo` until `positive_rft` is actually implemented.
2. Remove SDFT from the active runtime/docs/config path in the first cleanup patch.
3. Unify masking around assistant-turn supervision across preprocessing and rollout, without `think` vs `tool_call` distinctions on this instruct path.
4. Ship that masking change together with `assistant_action_token_count` and `empty_action_mask_count` telemetry.
5. Keep `terminal_format_valid`, `verifier_status`, and `stage_accepted` separate everywhere.
6. Add an RFT-only committed-step journal and resume only from the last committed outer step.
7. Implement positive-RFT as a fresh verifier-backed run from the format-RFT checkpoint, with correctness-only selection and the explicit `(count >= 40) AND (rate >= 0.02)` stop rule on a finalized batch.
8. Do not re-open the previously ambiguous question; the current thread already answers the scope needed for the review packet.

## 9. Main warning to carry into implementation

The masking simplification removes one of the old implicit format biases. After cleanup, format behavior must be enforced by stage selection and reward/penalty signals, not by tool-call-only loss masking.
