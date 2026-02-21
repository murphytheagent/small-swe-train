# SDPO Adaptation Plan (v2)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229
Source baseline: `https://github.com/lasgroup/SDPO` (main branch)

## 1) Reusable SDPO components

- `verl/trainer/ppo/ray_trainer.py`
  - feedback collection and teacher-batch assembly path.
- `verl/workers/actor/dp_actor.py`
  - teacher regularization (`ema`), student/teacher forward, distillation call site.
- `verl/trainer/ppo/core_algos.py`
  - `compute_self_distillation_loss` with top-k/full-logit support.
- `verl/trainer/config/actor/actor.yaml`
  - self-distillation configuration surface.

## 2) Required custom layers for our codebase

1. Turn formatter/parser
- parse ChatML assistant boundaries.
- parse optional `<think>` and ordered `1..M` `<tool_call>` JSON blocks.
- enforce `submit` singleton rule per turn.

2. Teacher prompt builder
- block-structured prompt with hybrid trajectory context and canonicalized feedback packet.
- include student attempt by default in v1.6.

3. Feedback canonicalization
- deterministic normalization/extraction plus self-containment diagnostics.
- require `actionable_error_text` key presence (`string | null`).

4. SWE environment bridge
- Docker execution wrapper, tool-level traces, benchmark split isolation.

5. Dataset adapters
- SWE-smith trajectory conversion to canonical tool schema (`answer|submit` -> `submit`).

6. Stage-aware masking
- RFT mask excludes think tokens.
- step-SDPO mask includes think and tool-call tokens.

## 3) File-level adaptation targets in our planned tree

- `src/teacher/prompt_builder.py`
  - replaces SDPO generic reprompt templates.
- `src/data/feedback_canonicalizer.py`
  - produces schema-valid feedback packets.
- `src/rollout/turn_parser.py`
  - ChatML boundary parsing + delimiter parsing + multi-call extraction.
- `src/losses/action_masking.py`
  - stage-specific token mask generation.
- `src/trainer/sdpo_trainer.py`
  - integration wrapper around reused SDPO/verl components.

## 4) Integration sequence

1. Lock parser/schema unit tests.
2. Implement canonicalization + self-containment derivation tests.
3. Wire teacher prompt builder with deterministic truncation.
4. Integrate stage-aware masking and distillation loss path.
5. Run minimal offline smoke with fixed trajectories.
6. Run small on-policy SWE-smith train split smoke.

## 5) Primary risks and controls

- Risk: parser/schema drift between rollout and trainer.
  - Control: strict schema validation at ingestion boundary.
- Risk: masking mismatch across stages.
  - Control: explicit per-stage mask diagnostics and golden-token tests.
- Risk: feedback canonicalization instability.
  - Control: versioned canonicalizer + deterministic hashes.
