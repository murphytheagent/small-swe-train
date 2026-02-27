# PR Plan: Teacher Contract and Leakage-Safe Prompting for Current-Turn SDPO

- Created: 2026-02-27 00:19 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab

## Why this PR exists
The current teacher contract text explicitly optimizes for the *next* turn. If supervision is aligned to current turn, contract wording must align too. Also, current teacher prompt construction includes the current attempt block by default, which becomes target leakage under current-turn distillation.

## Desired behavior
1. Teacher prompt contract can be selected by supervision mode (`current_turn` vs compatibility `next_turn`).
2. In `current_turn` mode, target-turn assistant content is excluded from teacher prompt context.
3. Prompt tests enforce no next-turn language and no target leakage for same-turn distillation.

## Proposed implementation scope
- `src/prompts/teacher_messages.py`
  - Add explicit contract builders for `current_turn` and `next_turn`.
  - Keep existing wording only in `next_turn` compatibility path.
- `src/verl_integration/reprompt_adapter.py`
  - When supervising current turn, avoid injecting target-turn assistant text into `CURRENT_ATTEMPT_BLOCK`.
  - Keep behavior configurable for controlled ablations.
- `tests/test_verl_reprompt_adapter.py`
  - Add assertions for contract wording and leakage-safe prompt structure.
- `tests/test_teacher_messages.py` (new or extended)
  - Validate contract text selection by mode.

## Acceptance checks
1. `current_turn` mode prompts contain explicit same-turn objective text and no `next turn` instruction.
2. Leakage guard test: prompt for turn `t` does not contain raw assistant text for turn `t`.
3. Compatibility mode keeps previous prompt semantics for controlled comparisons.

## Non-goals
- Verifier-feedback fusion.
- Distillation gating changes.
- Attention/loss mask decoupling.
