# PR Plan: Align Turn-Level SDPO to Current-Turn Supervision

- Created: 2026-02-27 00:19 UTC
- Updated: 2026-02-27 01:12 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab
- Consult risk rating: High

## Why this PR exists
Current turn-level distillation builds a teacher prompt at turn `t` but supervises tokens from turn `t+1` (`target_span = spans[current_turn_index + 1]`). This contradicts the current project intent to distill each turn directly with same-turn supervision.

## Scope and boundaries
This PR introduces supervision-mode plumbing and same-turn targeting mechanics.

Non-goals:
- teacher-contract wording updates (PR #19),
- verifier-feedback fusion and gating policy redesign (PR #20),
- attention-mask vs loss-mask decoupling and preflight scripts (PR #21).

## Dependency note (blocking)
`current_turn` must not become production default until PR #19 leakage-safe prompting is merged. This PR should ship mode support and validation, with runtime default kept at `next_turn` for compatibility.

## Expanded implementation checklist
1. Config compatibility wiring
- File: `src/small_swe_runtime_patches.py`
- Extend `SelfDistillationConfig` compatibility patch to accept a new keyword `turn_supervision_mode`.
- Normalize accepted values to `next_turn | current_turn` and fail fast on invalid values.

2. Training config surfacing
- File: `configs/verl/sdpo_swe.yaml`
- Add `actor_rollout_ref.actor.self_distillation.turn_supervision_mode`.
- Keep default `next_turn` in this PR (safety until PR #19).
- Document the default-flip dependency on PR #19 in config comments.

3. RePrompt batch generation changes
- File: `src/verl_integration/reprompt_adapter.py`
- Extend `build_self_distillation_batch(..., turn_supervision_mode: str = "next_turn")`.
- For `next_turn`, preserve existing behavior byte-for-byte.
- For `current_turn`:
  - iterate all assistant turns, not `range(len(assistant_turns) - 1)`,
  - target `spans[current_turn_index]` (same turn),
  - include turn 0 and final turn whenever span exists,
  - build per-turn masks as width-aligned vectors,
  - mark turns inactive only when same-turn span is absent or empty.
- Keep stable list shapes across prompt/mask/metadata outputs.

4. Runtime pass-through
- File: `src/verl_integration/ppo_runtime_patch.py`
- Read `turn_supervision_mode` from self-distillation config.
- Pass the mode into `build_self_distillation_batch`.
- Emit a debug metric/log field for selected supervision mode at batch build.

## Required invariants
1. Same-turn exactness (current-turn mode)
- For each active turn `t`, `turn_response_mask[t]` exactly matches `spans[t]`.
- No index may be active where `_response_mask[index] == 0`.

2. Coverage
- Turn 0 and final turn are represented whenever they have non-empty spans.

3. Output alignment
- `len(turn_teacher_prompts) == len(turn_response_masks) == len(turn_distillation_mask)` per sample.
- Mask width equals `len(_response_mask)` for every turn row.

4. Compatibility
- `next_turn` mode remains behavior-compatible with pre-PR outputs.

## Test matrix
1. Unit tests (`tests/test_verl_reprompt_adapter.py`)
- Add `test_turn_supervision_next_turn_compatibility()`.
- Add `test_turn_supervision_current_turn_exact_masks()`.
- Add `test_current_turn_includes_first_and_last_turn_when_spans_exist()`.
- Add edge-case tests for zero-length turn tokens and span/turn-count mismatch.

2. Integration-style test
- Through reward-row -> reprompt-batch path, assert:
  - mode-dependent mask cardinality,
  - no out-of-width indices,
  - no tool-token supervision leakage.

## Rollback and compatibility strategy
- Immediate rollback path: set `turn_supervision_mode=next_turn` in training config.
- Keep both modes available for controlled A/B comparisons during migration.
- Do not remove next-turn path in this PR.

## Merge gate (must all pass)
1. New mode-plumbing tests green.
2. Same-turn invariants proven in unit tests.
3. Explicit compatibility test proves no next-turn regression.
4. Config default remains `next_turn` in this PR.

## Operator checklist after merge (before enabling current_turn)
1. Merge PR #19 (teacher contract + leakage-safe prompting).
2. Flip config to `turn_supervision_mode=current_turn` in a dedicated change.
3. Run PR #21 preflight gates before expensive SDPO runs.
