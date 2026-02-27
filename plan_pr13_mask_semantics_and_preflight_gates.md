# PR Plan: Mask Semantics Hardening and SDPO Preflight Gates

- Created: 2026-02-27 00:19 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab

## Why this PR exists
The runtime currently reuses `response_mask` in contexts where attention-validity and distillation-loss targeting may diverge. This can silently degrade teacher conditioning. We also need hard preflight checks to prevent expensive training on incorrect turn masks or prompt leakage.

## Desired behavior
1. Separate semantics:
   - attention-validity mask (tokens model can attend to),
   - distillation-loss mask (tokens we optimize on).
2. Add offline integrity gates and tiny-run gates before long SDPO runs.
3. Make mask failures fail-fast with actionable diagnostics.

## Proposed implementation scope
- `src/verl_integration/ppo_runtime_patch.py`
  - Introduce explicit attention mask construction for teacher path independent of distillation target mask.
  - Preserve `turn_response_mask` exclusively for optimization targeting.
  - Add debug metrics for mask density / invalid overlap.
- `src/verl_integration/reward_adapter.py`
  - Strengthen invariants for `_response_mask` presence/shape on SWE batches (avoid silent all-ones fallback when unintended).
- `scripts/` (new)
  - Add `check_sdpo_turn_integrity.py` preflight script for offline gates.
- `tests/test_ppo_runtime_patch.py`, `tests/test_verl_reward_adapter.py`
  - Add regression tests for mask-role separation and invariant failures.

## Acceptance checks
1. Unit tests prove attention mask includes valid sequence context while loss mask remains task-targeted.
2. Preflight script validates:
   - turn-target mask subset relation,
   - no tool-token supervision leakage,
   - no target-turn prompt leakage.
3. Tiny SDPO gate run has no NaN/Inf and no degenerate mask ratios.

## Non-goals
- Teacher prompt content redesign.
- Verifier-feedback prompt fusion.
- Curriculum/data-mix changes.
