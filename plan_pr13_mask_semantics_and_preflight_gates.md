# PR Plan: Mask Semantics Hardening and SDPO Preflight Gates

- Created: 2026-02-27 00:19 UTC
- Updated: 2026-02-27 01:24 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab
- Consult risk rating: High

## Why this PR exists
The runtime currently reuses `response_mask` in places where attention-validity and distillation-loss targeting can diverge. This can degrade teacher conditioning silently. We also need low-cost integrity gates before expensive SDPO runs.

## Scope and boundaries
This PR separates mask roles, hardens SWE response-mask invariants, and adds preflight/tiny-run go-no-go checks.

Non-goals:
- redesigning teacher prompt content (PR #19/#20),
- changing turn target mapping logic (PR #18).

## Dependency note
Recommended after PR #18-#20 so preflight checks validate final intended behavior (current-turn mapping + leakage-safe prompts + verifier fusion).

## Expanded implementation checklist
1. Teacher attention mask decoupling
- File: `src/verl_integration/ppo_runtime_patch.py`
- Introduce explicit teacher response-validity mask for attention (not reusing loss mask semantics).
- Keep loss-targeting masks unchanged:
  - row-level: `response_mask` for loss targeting,
  - turn-level: `turn_response_mask` for loss targeting.
- Apply decoupling in both row-level and turn-level teacher tensor builders.
- Add debug metrics:
  - teacher-attention valid-token ratio,
  - supervised-token ratio,
  - invalid-overlap count (`turn_response_mask` outside teacher-valid mask).

2. SWE response-mask invariant hardening
- File: `src/verl_integration/reward_adapter.py`
- Detect SWE rows via trajectory turn metadata keys.
- For SWE rows, fail fast when `_response_mask` is missing/empty instead of silently falling back to all-ones.
- Preserve non-SWE fallback compatibility where strict mask semantics are unavailable.

3. Turn-span contiguity safety
- File: `src/verl_integration/reprompt_adapter.py`
- In `_build_assistant_turn_spans`, add contiguity checks for selected generated positions.
- On non-contiguous selections, emit diagnostics and use explicit safe fallback behavior (sparse-safe handling or inactive-turn fallback with warning).

4. Offline integrity preflight script
- New file: `scripts/check_sdpo_turn_integrity.py`
- Input: row-level export format used for reprompt construction.
- Required checks:
  - `turn_response_mask` subset of `_response_mask`,
  - no tool-token supervision leakage,
  - no target-turn prompt leakage in current-turn mode,
  - shape/cardinality alignment of prompt/mask lists,
  - truncation-rate summary.
- CLI flags should include supervision mode and truncation fail thresholds.

5. Tiny-run gate harness
- Add a lightweight SDPO gate runner (or extend existing scaffold runner) for 10-20 update steps on tiny workload.
- Enforce finite loss/gradients and non-degenerate mask density before allowing full runs.

## Required invariants
1. Mask-role separation
- Teacher attention validity mask must include all non-padding response tokens needed for conditioning.
- Loss masks (`response_mask` / `turn_response_mask`) must not be reused as attention-validity masks.

2. Overlap safety
- No supervised token may lie outside teacher attention-valid region.

3. SWE strictness
- SWE rows cannot pass with synthetic all-ones `_response_mask` fallback.

4. Preflight determinism
- Integrity script produces deterministic pass/fail with actionable reasons.

## Test matrix
1. Unit tests (`tests/test_ppo_runtime_patch.py`)
- Add tests showing teacher attention mask behavior differs from loss mask behavior where expected.
- Add tests ensuring loss-targeting masks remain unchanged after refactor.

2. Unit tests (`tests/test_verl_reward_adapter.py`)
- Add regression test: SWE row with missing `_response_mask` fails fast.
- Add compatibility test: non-SWE row fallback behavior remains intact.

3. Unit tests (`tests/test_verl_reprompt_adapter.py`)
- Add contiguity-edge tests for `_build_assistant_turn_spans` safety behavior.

4. Script tests
- Add tests for `scripts/check_sdpo_turn_integrity.py`:
  - valid fixture exits 0,
  - tool-token supervision violation exits non-zero,
  - prompt leakage violation exits non-zero.

## Go/No-Go thresholds before expensive SDPO
1. Offline integrity gate (required)
- 0 subset violations.
- 0 tool-token supervision leakage events.
- 0 target-turn prompt leakage events in current-turn mode.
- Prompt/mask cardinality alignment = 100%.
- Prompt truncation rate <= 5% (or explicitly overridden with rationale).

2. Tiny-run gate (required)
- 20 update steps complete with no NaN/Inf loss or gradients.
- Active-turn ratio >= 0.70 on intended turn-level setting.
- Supervised-token density per row non-degenerate for >=95% rows.

## Rollback and compatibility strategy
- Provide config switch for teacher attention-mask mode if incremental rollout is needed.
- If regressions appear, revert to compatibility attention mode while keeping integrity script available.
- Keep strict SWE invariants enabled; only non-SWE fallback remains permissive.

## Merge gate (must all pass)
1. Unit tests for mask-role separation + SWE strict invariants pass.
2. Preflight script exists, tested, and returns actionable failures.
3. Tiny-run gate criteria are codified and runnable.
4. No regressions in existing non-turn or non-SWE compatibility tests.

## Operator checklist after merge
1. Run preflight script on recent rollout rows before every long SDPO run.
2. Run tiny-run gate whenever prompt/mask logic changes.
3. Block expensive training if either gate fails.
