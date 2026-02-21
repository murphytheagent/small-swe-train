# SDPO Adaptation Plan (v1)

Generated: 2026-02-21 21:37 UTC
Thread: 1771579678.414229
Source baseline: `https://github.com/lasgroup/SDPO` (inspected at commit `c52586ba45633a817879f59e2612cc62c55c8479`)

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
- parse optional `<think>` and one `<tool_call>` JSON block.
- emit action-token mask for tool JSON only.

2. Teacher prompt builder
- block-structured prompt with hybrid trajectory context and canonicalized feedback packet.

3. Feedback canonicalization
- deterministic normalization/extraction plus derived self-containment flags.

4. SWE environment bridge
- Docker execution wrapper, tool-level traces, benchmark split isolation.

5. Dataset adapters
- SWE-smith trajectory conversion to canonical tool schema (`submit` -> `answer`).

## 3) File-level adaptation targets in our planned tree

- `src/teacher/prompt_builder.py`
  - replaces SDPO generic reprompt templates.
- `src/data/feedback_canonicalizer.py`
  - produces schema-valid feedback packets.
- `src/rollout/turn_parser.py`
  - delimiter parsing and tool-call extraction.
- `src/losses/action_masking.py`
  - tool-token mask generation.
- `src/trainer/sdpo_trainer.py`
  - integration wrapper around reused SDPO/verl components.

## 4) Integration sequence

1. Lock parser/schema unit tests.
2. Implement canonicalization + self-containment derivation tests.
3. Wire teacher prompt builder with deterministic truncation.
4. Integrate masking and distillation loss path.
5. Run minimal offline smoke with fixed trajectories.
6. Run small on-policy SWE-smith train split smoke.

## 5) Primary risks and controls

- Risk: parser/schema drift between rollout and trainer.
  - Control: strict schema validation at ingestion boundary.
- Risk: over-masking (lost learning signal) or under-masking (reasoning leakage).
  - Control: explicit mask diagnostics per batch.
- Risk: feedback canonicalization instability.
  - Control: versioned canonicalizer + deterministic hashes.
