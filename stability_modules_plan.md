# Stability Modules Follow-Up Plan

Generated: 2026-03-11 03:34 UTC
Thread: 1773182046.432339
Status: review-only, no implementation yet

## Scope decisions locked by the latest thread

- Canonical active runtime path stays `format_rft -> turn_sdpo` for now. Do not advertise `positive_rft` in the mainline contract until that stage actually exists.
- SDFT is no longer part of the intended active pipeline. Delete SDFT-facing launcher/config/doc drift as part of the first cleanup pass instead of keeping it optional.
- Do not use think-token-aware masking on the instruct path. Treat all assistant-generated tokens as trainable action and keep masking at assistant-turn vs environment-feedback granularity.
- The first resume milestone is RFT-only. SDPO warm-start can keep consuming an RFT checkpoint, but true SDPO continuation stays out of scope.

## Code-grounded current state

- `configs/runtime/training_policy_defaults.v1.json` still advertises `rft -> sdft_optional -> sdpo_main`.
- `src/losses/action_masking.py`, `src/data/tokenization.py`, and `src/verl_integration/data_preprocessor.py` still encode token-level `think` vs `tool_call` distinctions.
- `src/verl_integration/swe_bridge_agent_loop.py` already emits a coarser `response_mask` where all assistant-generated tokens are trainable and tool-response blocks are masked out.
- `src/trainer/rft_rejection.py` is still format-first: default selection requires terminal submit plus format-valid final submit, while correctness is not required.
- `src/verl_integration/reward_function.py` already keeps verifier correctness separate from terminal-format penalty, so correctness-only positive selection is feasible without inventing a new scorer.
- `src/trainer/rft_runtime_loop.py` already writes step artifacts, manifests, and checkpoints, but it does not yet have an authoritative committed-step resume cursor and it launches the inner trainer with `trainer.resume_mode=disable`.

## Revised implementation order

1. Contract cleanup:
   - remove SDFT from the active runtime/docs/config path
   - make the runtime contract explicitly `format_rft -> turn_sdpo`
   - simplify masking to assistant-turn supervision and add mask-sanity telemetry
2. Shared telemetry:
   - per-trajectory records plus per-step summaries for both RFT and turn-SDPO
   - keep `terminal_format_valid`, `verifier_status`, and `stage_accepted` separate everywhere
3. Replay-safe RFT resume:
   - resume only from the last fully committed outer RFT step
   - no inner trainer resume and no SDPO continuation in this milestone
4. Optional positive-RFT:
   - fresh run initialized from the format-RFT checkpoint
   - correctness-only selector, independent of terminal submit format
   - batch-level stop condition based on explicit verifier-positive count and fraction

## Module contracts

### Telemetry

- Shared per-trajectory keys:
  - `run_id`, `stage`, `outer_step`, `trajectory_id`, `instance_id`
  - `stage_accepted`, `stage_decision_reason`
  - `final_turn_has_submit`, `terminal_format_valid`, `terminal_format_failure_code`
  - `verifier_status` in `{correct, incorrect, missing}`
  - `assistant_action_token_count`, `trajectory_assistant_turns`
  - `reward_total` when the stage actually uses reward
- Shared per-step aggregates:
  - collected, accepted, verifier-correct, verifier-missing, and terminal-format-valid counts/rates
  - `malformed_but_correct_count`
  - `empty_action_mask_count`
  - collection/train durations and existing train/val losses
- W&B should receive compact scalars only. High-cardinality detail stays in local JSONL or Parquet keyed by the shared IDs above.

### RFT resume

- Add a committed-step journal separate from analytics manifests.
- A step is resumable only after its checkpoint path, selection contract, and committed step index are atomically recorded.
- Pruning must happen only below the last committed step.
- Partial later steps should be discarded or replayed, never inferred as committed.
- Single-writer protection is required for one output directory.

### Positive-RFT

- Launch as a new short RFT run from the format-RFT checkpoint, not as an appended phase inside the old run journal.
- Enable verifier-backed correctness during recollection.
- Select rows only from explicit verifier-positive outcomes; do not use reward sign and do not gate on terminal-format validity.
- Log malformed-but-correct rows explicitly so this behavior is visible rather than accidental.
- Stop only on a fully finalized collection batch once both:
  - `positive_count >= 40`
  - `positive_rate >= 0.02`
  over the total collected trajectories in that batch.

## Main warning to carry into implementation

The masking simplification removes one of the old implicit format biases. After cleanup, format behavior must be enforced by stage selection and reward/penalty signals, not by tool-call-only loss masking.
