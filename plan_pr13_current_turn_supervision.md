# PR Plan: Align Turn-Level SDPO to Current-Turn Supervision

- Created: 2026-02-27 00:19 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab

## Why this PR exists
Current turn-level distillation builds a teacher prompt at turn `t` but supervises tokens from turn `t+1` (`target_span = spans[current_turn_index + 1]`). This contradicts the current project intent: distill every turn using feedback from that same turn.

## Desired behavior
For each assistant turn `t` with valid generated-token span:
1. Build teacher prompt for turn `t`.
2. Distill tokens from turn `t` (not turn `t+1`).
3. Include turn `0` and final turn when spans exist.

## Proposed implementation scope
- `src/verl_integration/reprompt_adapter.py`
  - Add explicit supervision mode (`current_turn` default for SWE turn-level path).
  - In `current_turn` mode:
    - iterate all turns with valid spans,
    - set target span to `spans[current_turn_index]`.
  - Keep `next_turn` mode available only as compatibility fallback.
- `tests/test_verl_reprompt_adapter.py`
  - Add direct assertions that target masks for each pair correspond to same-turn token ranges.
  - Add regression tests ensuring turn 0 coverage is present when spans are available.

## Acceptance checks
1. Unit tests prove same-turn targeting:
   - no selected target index where `_response_mask[index] == 0`,
   - same-turn masks match expected token ranges for a multi-turn fixture.
2. `turn_pair_count_per_sample` no longer drops turn 0 by construction.
3. No regression in legacy (non-turn) behavior.

## Non-goals
- Teacher prompt contract wording updates.
- Verifier-feedback injection.
- Attention-mask/loss-mask decoupling.
