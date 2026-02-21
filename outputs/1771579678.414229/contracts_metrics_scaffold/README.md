# Contracts and Metrics Scaffold (v2, review-only)

Generated: 2026-02-21 21:37 UTC
Thread: 1771579678.414229

Purpose:
- Encode current v1 research-mode contracts as machine-readable artifacts before training-loop coding.
- Lock parser/schema/feedback derivation rules to prevent rollout-vs-trainer drift.

Out of scope:
- No model training loop implementation.
- No optimizer/runtime wiring.
- No benchmark execution logic.

Contents:
- `schemas/action_envelope.schema.json`: assistant tool-call envelope with optional `thinking` and tools `bash|search|edit|answer`.
- `schemas/tool_args.schema.json`: per-tool argument schemas.
- `schemas/feedback_packet.schema.json`: canonicalized feedback packet with enforced derived self-containment flags.
- `config/phase_transition_gates.v1.json`: numeric gates for entering main SDPO.
- `config/training_policy_defaults.v1.json`: locked defaults and parsing/prompting policy.
- `metrics/metric_definitions.v1.md`: metric formulas and definitions.
- `metrics/minimal_experiment_matrix.v1.md`: initial run plan and ablation plan.
- `docs/self_containment_and_canonicalization.v1.md`: programmatic checks and canonicalization algorithm.
- `docs/tool_schema_alignment.v1.md`: SWE-smith/SWE-bench to canonical schema mapping.
- `docs/sdpo_adaptation_plan.v1.md`: detailed `lasgroup/SDPO` adaptation blueprint.

Validation sequence intended for runtime:
1. Parse optional `<think>...</think>` span.
2. Parse exactly one `<tool_call>...</tool_call>` JSON object.
3. Validate `action_envelope` and tool-specific args schema.
4. Canonicalize env response and compute self-containment checks.
5. Validate `feedback_packet` including derived-flag consistency.
