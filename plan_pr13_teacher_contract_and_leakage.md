# PR Plan: Teacher Contract and Leakage-Safe Prompting for Current-Turn SDPO

- Created: 2026-02-27 00:19 UTC
- Updated: 2026-02-27 01:16 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab
- Consult risk rating: Medium

## Why this PR exists
Current teacher contract language is explicitly next-turn oriented, and current teacher prompt construction includes target-turn assistant content by default. Under same-turn distillation this introduces objective mismatch and target leakage risk.

## Scope and boundaries
This PR aligns teacher prompt semantics with supervision mode and guarantees leakage-safe prompt construction for current-turn supervision.

Non-goals:
- span/mask alignment implementation (PR #18),
- verifier-feedback injection and legacy gating policy (PR #20),
- teacher attention/loss mask decoupling and preflight scripts (PR #21).

## Dependency note
This PR should be merged before enabling `turn_supervision_mode=current_turn` in production configs.

## Expanded implementation checklist
1. Contract-mode support in prompt message builders
- File: `src/prompts/teacher_messages.py`
- Extend `build_teacher_output_contract_block` with `supervision_mode: next_turn | current_turn`.
- Keep existing wording for `next_turn` compatibility.
- Add current-turn wording that explicitly avoids "next turn" objective language.

2. Prompt builder interface upgrade
- File: `src/teacher/prompt_builder.py`
- Extend `TeacherPromptInputs` with `supervision_mode` (or an explicit contract block selector field).
- Ensure `build_teacher_prompt` uses the mode-aware contract selection path.
- Preserve existing block ordering and delimiters.

3. RePrompt leakage guard
- File: `src/verl_integration/reprompt_adapter.py`
- Thread supervision mode into `_build_turn_prompt`.
- In `current_turn` mode, force `current_attempt_block` to empty for target-turn leakage safety.
- In `next_turn` mode, preserve existing include/exclude behavior via `include_student_attempt_for_teacher`.
- Optional ablation knob may exist, but default must remain leakage-safe (`false`).

4. Config surfacing
- Add mode/guard knobs into runtime-facing config path used by training (`configs/verl/sdpo_swe.yaml` and compatibility patch layer as needed).
- Ensure defaults preserve current behavior until explicit mode flip.

## Required invariants
1. Contract-language alignment
- `current_turn` prompts contain explicit same-turn objective text.
- `current_turn` prompts contain no "next turn" or equivalent phrasing.

2. Leakage safety
- For each active turn `t` in current-turn mode, prompt `t` must not contain raw assistant text from turn `t`.

3. Backward compatibility
- `next_turn` mode keeps previous contract wording and attempt-block behavior.

## Test matrix
1. Unit tests (`tests/test_teacher_messages.py`)
- Add `test_contract_current_turn_has_no_next_turn_language()`.
- Add `test_contract_next_turn_keeps_legacy_language()`.

2. Unit tests (`tests/test_verl_reprompt_adapter.py`)
- Add marker-based leakage test with unique assistant turn strings.
- Add mode-switch test proving:
  - current-turn omits target-turn attempt text,
  - next-turn preserves compatibility behavior.

3. Prompt-shape sanity test
- Verify prompt remains well-formed for tokenizer/chat-template path after current-attempt omission.

## Rollback and compatibility strategy
- Rollback behavior by setting supervision/contract mode back to `next_turn`.
- Keep current-turn logic behind explicit mode; avoid removing compatibility code in this PR.

## Merge gate (must all pass)
1. Contract text tests for both modes pass.
2. Leakage guard tests pass with zero target-turn prompt leakage.
3. Compatibility test confirms no unintended next-turn behavior drift.
4. PR #18 mode plumbing is available (or mirrored here) so mode selection is deterministic.

## Operator checklist after merge
1. Enable `current_turn` mode only after PR #18 + PR #19 are both merged.
2. Run preflight integrity checks from PR #21 before long SDPO runs.
