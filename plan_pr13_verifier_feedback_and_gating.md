# PR Plan: Verifier-Feedback Fusion and Distillation Gating Semantics

- Created: 2026-02-27 00:19 UTC
- Thread: 1772102085.104579
- Base branch: task/1771927570-step-sdpo-stage-ab

## Why this PR exists
Two gaps currently reduce teacher-signal quality:
1. Final verifier feedback is produced by the rollout pipeline but not injected into turn teacher prompts.
2. Legacy gating depends on `has_actionable_error_text`, while self-containment extraction is disabled in defaults, collapsing many unresolved rows to `self_distillation_mask=false`.

## Desired behavior
1. Turn teacher prompts can include final verifier feedback with clear section labeling.
2. Distillation activation semantics are explicit and aligned to intent (`distill_every_turn` for turn-level trajectories).
3. Legacy single-turn fallback has deterministic, configurable gating that does not silently disable failure learning.

## Proposed implementation scope
- `src/verl_integration/reprompt_adapter.py`
  - Extend `_build_feedback_for_turn` to optionally include `verification_feedback` / `verification_error` from sample metadata.
  - Add explicit gating policy enum/config for legacy fallback (`resolved_only`, `feedback_present`, `always`).
  - Default gate for SWE turn-level path: all valid same-turn pairs active.
- `configs/runtime/training_policy_defaults.v1.json`
  - Add explicit knobs for verifier-feedback inclusion and fallback gating policy.
  - Re-evaluate `extract_self_containment_signals` in policy to avoid hidden disablement.
- `tests/test_verl_reprompt_adapter.py`
  - Add tests for verifier-feedback inclusion and gating-policy behavior.

## Acceptance checks
1. When verifier feedback exists, all configured turn prompts include it with a stable section header.
2. Gating-policy tests prove expected activation counts under each mode.
3. No silent all-false legacy mask when failure feedback is present.

## Non-goals
- Current-turn token-span alignment implementation.
- Attention-mask/loss-mask separation.
- End-to-end training benchmarks.
