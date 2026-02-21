# Contracts and Metrics Scaffold (v1, review-only)

Generated: 2026-02-21 03:43 UTC
Thread: 1771579678.414229

Purpose:
- Encode approved v1 contracts and gate thresholds as machine-readable scaffold files.
- Keep implementation risk low by reviewing interfaces and metrics before coding the training loop.

Out of scope:
- No model training loop.
- No optimizer/runtime implementation.
- No benchmark execution logic.

Contents:
- `schemas/action_envelope.schema.json`: strict single-action JSON envelope.
- `schemas/tool_args.schema.json`: per-tool argument schemas.
- `schemas/feedback_packet.schema.json`: tool feedback packet and self-containment checks.
- `config/phase_transition_gates.v1.json`: numeric gates for entering main SDPO.
- `config/training_policy_defaults.v1.json`: locked policy constants and toggles.
- `metrics/metric_definitions.v1.md`: metric formulas and operational definitions.
- `metrics/minimal_experiment_matrix.v1.md`: first-pass run plan and ablations.

Validation sequence intended for runtime:
1. Parse one JSON object from model output.
2. Validate envelope schema.
3. Validate `args` by tool-specific schema.
4. Emit invalid-step event if parser/schema checks fail after one repair pass.
