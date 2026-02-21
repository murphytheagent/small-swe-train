# Contracts and Metrics Scaffold (v3, frozen design snapshot)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229

Purpose:
- Encode the approved v1.6 research-mode contracts as machine-readable artifacts.
- Lock parser/schema/feedback derivation rules to prevent rollout-vs-trainer drift.

Implementation status:
- Runtime scaffolding now lives under `src/` and is tested.
- This folder remains the frozen design reference for the thread run.

Contents:
- `schemas/action_envelope.schema.json`: assistant turn envelope with optional thinking and ordered multi-tool call support (`bash|search|edit|submit`).
- `schemas/tool_args.schema.json`: per-tool argument schemas.
- `schemas/feedback_packet.schema.json`: canonicalized feedback packet with self-containment diagnostics and configurable student-attempt flag.
- `config/phase_transition_gates.v1.json`: numeric gates for entering main SDPO.
- `config/training_policy_defaults.v1.json`: locked defaults and parsing/prompting policy.
- `metrics/metric_definitions.v1.md`: metric formulas and definitions.
- `metrics/minimal_experiment_matrix.v1.md`: initial run plan and ablation plan.
- `docs/self_containment_and_canonicalization.v1.md`: programmatic checks and canonicalization algorithm.
- `docs/tool_schema_alignment.v1.md`: SWE-smith/SWE-bench to canonical schema mapping.
- `docs/sdpo_adaptation_plan.v1.md`: detailed `lasgroup/SDPO` adaptation blueprint.

Validation sequence intended for runtime:
1. Parse assistant turn boundaries under ChatML (`<|im_start|>assistant ... <|im_end|>`).
2. Parse optional `<think>...</think>` span.
3. Parse `1..M` ordered `<tool_call>...</tool_call>` JSON objects.
4. Validate `action_envelope` and tool-specific args schema.
5. Canonicalize env response and compute self-containment diagnostics.
6. Validate `feedback_packet` including required canonical fields.
