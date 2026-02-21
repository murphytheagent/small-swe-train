# small-swe-train: Research Mode v1.3 Spec Freeze

Generated: 2026-02-21 03:43 UTC
Thread: 1771579678.414229
Approval event: 2026-02-21 03:41 UTC (1771645263.619809) -> `Approve defaults 1-11`

## Freeze scope
- This freeze locks the v1 research design, numeric gates, and runtime policy defaults.
- This freeze does not restart full coding of the training loop.
- This checkpoint only adds review-first contracts and metrics scaffolding.

## Locked math objectives

1. Format supervision
- `L_format = E_{(h_t, a_t^*)}[-log pi_theta(a_t^* | h_t)]`

2. Optional short SDFT warm-start (flagged)
- `L_sdft = E_{a_t ~ pi_theta(.|h_t)} [log pi_theta(a_t|h_t) - log q_phi(a_t|h_t,d_t)]`

3. Main step-level SDPO (action-token masking only)
- `L_step_sdpo = E[ sum_t sum_{n in M_t} KL( pi_theta(.|h_t,a_{t,<n}) || stopgrad(q_phi(.|h_t,f_{t+1},a_{t,<n})) ) ]`
- `M_t` masks only action tokens.

4. EMA teacher regularization
- `phi <- (1 - beta) * phi + beta * theta`
- v1 default: `beta = 0.005` (sweep range `[0.001, 0.01]`).

## Locked policy decisions (1-11)

1. JSON envelope: `{"tool": "bash|search|edit|submit", "args": {...}}` with strict per-tool args schema.
2. Invalid-output handling: one deterministic repair pass; then hard invalid step with env error feedback.
3. Main-stage gate window `N=200` with thresholds:
   - `parse_valid_rate >= 0.985`
   - `allowed_tool_rate >= 0.995`
   - `required_arg_presence >= 0.985`
   - `single_object_rate >= 0.98`
4. Teacher-context attempt inclusion rule: include student attempt only when feedback is not self-contained.
5. Deterministic truncation constants: `H=768`, `T=768` after payload canonicalization and before teacher wrapper text.
6. Pre-main sequence: `RFT -> short SDFT (flagged) -> SDPO`.
7. SDPO teacher regularization: EMA default.
8. Top-K distillation: fixed `K=100` for v1 (schedule support present but off).
9. Adaptation mode: LoRA-only, attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), bf16 compute by default.
10. Runtime policy: Docker sandbox + explicit command allowlist (denylist also enforced).
11. Exit checklist: schema/gates/context/truncation/regularization/top-K/sandbox/eval protocol/experiment matrix all locked.

## Review-only scaffolding produced
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/README.md`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/schemas/action_envelope.schema.json`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/schemas/tool_args.schema.json`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/schemas/feedback_packet.schema.json`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/config/phase_transition_gates.v1.json`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/config/training_policy_defaults.v1.json`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/metrics/metric_definitions.v1.md`
- `projects/small-swe-train/outputs/1771579678.414229/contracts_metrics_scaffold/metrics/minimal_experiment_matrix.v1.md`

## Final design-review questions before coding restart

1. For `edit.args.patch`, should v1 lock to unified diff only, or allow a structured edit-ops format as an alternate schema branch?
2. For `search.args`, do you want domain restriction fields in v1 (`domains`, `recency_days`) or keep query-only for lower complexity?
3. For the command allowlist, do you want explicit test-runner binaries pinned per repository, or a generic category allowlist with repo-level overrides?
4. For self-containment checks, should localization hints require file path strings, or can test/function names alone satisfy localization?
5. For metric dashboards, do you want gating metrics computed per-repo and globally, or only globally for v1?

Research mode remains active until these review points are confirmed.
