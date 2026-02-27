# PR Plan: Verifier-Feedback Fusion and Distillation Gating Semantics

- Created: 2026-02-27 00:19 UTC
- Updated: 2026-02-27 01:20 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab
- Consult risk rating: Medium

## Why this PR exists
Two gaps reduce teacher-signal quality and can silently disable useful distillation:
1. Final verifier feedback is produced but not injected into per-turn teacher prompts.
2. Legacy fallback gating can collapse unresolved failures to inactive rows when self-containment extraction is disabled.

## Scope and boundaries
This PR adds verifier-feedback prompt fusion and explicit, testable gating semantics.

Non-goals:
- changing turn span target mapping (PR #18),
- contract/leakage safety mechanics (PR #19),
- teacher attention-mask refactor and preflight script framework (PR #21).

## Dependency note
Recommended after PR #19 so verifier sections are added on top of leakage-safe prompt construction.

## Expanded implementation checklist
1. Verifier-feedback extraction helper
- File: `src/verl_integration/reprompt_adapter.py`
- Add helper to extract verifier section from sample metadata with stable headers.
- Allowed metadata fields to inject:
  - `verification_feedback`,
  - `verification_error`,
  - coarse pass/resolved flags where present.
- Hard exclusion: never inject `submission_final_response` into teacher prompts.

2. Verifier injection policy knob
- Add `verifier_feedback_mode: none | final_turn_only | all_turns`.
- Policy behavior:
  - `none`: no verifier section.
  - `final_turn_only`: inject on final distilled turn only.
  - `all_turns`: inject final verifier feedback into each turn prompt.

3. Legacy distillation gating policy knob
- Add `legacy_distillation_gating_policy: resolved_only | feedback_present | always`.
- Define behavior in legacy/single-turn fallback path:
  - `resolved_only`: active if `sample_resolved`.
  - `feedback_present`: active if resolved or tool/verifier feedback exists.
  - `always`: always active.
- Ensure fallback behavior is deterministic and never silently all-false when failure feedback exists.

4. Turn-level semantics lock
- For turn-level SWE rows, preserve intent that all valid turn pairs are active by span validity (not legacy fallback heuristics).

5. Config wiring
- Surface both knobs in active training config path and compatibility patch layer where required.
- If `configs/runtime/training_policy_defaults.v1.json` is updated, ensure runtime uses the knobs (avoid documentation-only no-op keys).

## Required invariants
1. Verifier section correctness
- When verifier metadata exists and mode != `none`, configured prompts contain the verifier header block.
- Prompt text must never include `submission_final_response`.

2. Gating determinism
- Under `feedback_present`, unresolved rows with non-empty tool/verifier feedback must be active.
- No policy may silently produce all-false masks when feedback exists.

3. Turn-level intent protection
- Turn-level valid spans remain active independent of legacy self-containment extraction defaults.

## Test matrix
1. Unit tests (`tests/test_verl_reprompt_adapter.py`)
- Add `test_verifier_feedback_all_turns_injection()`.
- Add `test_verifier_feedback_final_turn_only_injection()`.
- Add `test_submission_final_response_not_leaked_into_prompt()`.
- Add gating-policy activation-count tests for all policies.
- Add `feedback_present` regression test for unresolved failure rows with tool/verifier feedback.

2. Integration-style test
- Build synthetic SWE turn-level sample with verifier metadata and assert:
  - expected prompt-level verifier block placement,
  - expected distillation mask activation counts,
  - no forbidden field leakage.

## Rollback and compatibility strategy
- Rollback verifier fusion by setting `verifier_feedback_mode=none`.
- Rollback gating behavior by choosing conservative `resolved_only` mode.
- Keep prior fallback semantics available as compatibility option during migration if needed.

## Merge gate (must all pass)
1. Verifier-injection placement tests pass for all modes.
2. Leakage exclusion test for `submission_final_response` passes.
3. Gating-policy tests prove activation behavior is deterministic and non-silent.
4. Turn-level path remains active for valid spans under configured turn-level training.

## Operator checklist after merge
1. Choose `all_turns` vs `final_turn_only` intentionally based on ablation goals.
2. Run PR #21 preflight + tiny-run gates before expensive SDPO runs.
