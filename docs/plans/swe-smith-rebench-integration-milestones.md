# SWE-smith + SWE-rebench integration milestones

Last updated: 2026-03-30 UTC.
Owner thread: Slack `1772422639.171219`.

## Goal
Integrate additional SWE datasets into `small-swe-train` without breaking current Python-first training stability.

Datasets in scope:
- `SWE-bench/SWE-smith-go`
- `SWE-bench/SWE-smith-js`
- `SWE-bench/SWE-smith-ts`
- `nebius/SWE-rebench`

## Decision summary
- Integrate first: `SWE-smith-go`.
- Integrate later: `SWE-smith-js`, `SWE-smith-ts` after language-aware verifier path is production-ready.
- Hold for now in on-policy RL path: `SWE-rebench` until patch semantics and command-based verification are explicit.

## Why this sequencing
Current codebase constraints:
- `src/env/task_dataset.py` requires non-empty `problem_statement`, `fail_to_pass`, and `pass_to_pass`.
- `src/verl_integration/submission_verifier.py` is Python/pytest-specific.
- `src/prompts/runtime_messages.py` now renders only the task objective in the initial user prompt, so large verifier target lists no longer bloat the first-turn prompt body.
- High-cardinality verifier targets still flow through structured task metadata (`reward_model.ground_truth` and related task rows), so future multi-language adapters still need bounded preview/report rules instead of dumping raw target arrays back into prompts or logs.
- `src/rollout/onpolicy_collector.py` still applies task-init patches directly when present, so rebench-style patch semantics must stay explicit.

These assumptions are safe for current SWE-smith-py settings, but block robust multi-language + rebench ingestion.

## Milestones

### Milestone 0: dataset contract + verifier abstraction baseline
Delivery boundary:
- Introduce dataset-normalization and verification-plan contracts, but do not enable new datasets in default runtime configs.

Implementation scope:
- Add a typed dataset adapter layer to normalize raw row fields before `TaskSample` construction.
- Introduce explicit patch semantics in adapter output:
  - `bug_patch` (apply before rollout),
  - `gold_fix_patch` (never auto-apply for training attempts).
- Introduce verifier backend interface with language-aware dispatch (`python`, `go`, `js`, `ts`, `command`).
- Add bounded verifier-target preview/report guardrails for very large test lists:
  - keep the initial prompt focused on the task objective,
  - keep the full verifier target plan in structured metadata.

Expected touchpoints:
- `src/env/task_dataset.py`
- `src/rollout/onpolicy_collector.py`
- `src/prompts/runtime_messages.py`
- `src/verl_integration/submission_verifier.py`
- `configs/runtime/training_policy_defaults.v1.json`
- `tests/` for adapter + verifier contract coverage

Acceptance gates:
- Existing SWE-smith-py path remains bit-for-bit equivalent by default config.
- New adapter/verifier contracts have regression tests for schema validation and fallback behavior.
- No runtime config switches to new datasets yet.

Out of scope:
- No change to default training mix.
- No rebench activation.

### Milestone 1: gated SWE-smith-go integration
Delivery boundary:
- Enable `SWE-smith-go` in a dedicated data config and verifier backend path behind explicit config selection.

Implementation scope:
- Add Go dataset config under `configs/data/`.
- Map SWE-smith-go fields to normalized task contract.
- Implement Go verifier backend strategy (suite-level command or target-aware command mode).
- Add dataset filters for empty/invalid rows with deterministic logging.

Expected touchpoints:
- `configs/data/`
- `src/env/task_dataset.py`
- `src/verl_integration/submission_verifier.py`
- `scripts/run_rft.sh` and/or `scripts/run_sdpo.sh` argument docs if new knobs are needed
- `tests/` verifier and ingestion tests

Acceptance gates:
- End-to-end dry-run ingestion over SWE-smith-go split succeeds with stable row counts.
- On-policy collector verifies with Go backend without Python verifier regressions.
- Prompt size remains bounded for high-cardinality target cases.

Out of scope:
- JS/TS datasets not enabled.
- Rebench not enabled.

### Milestone 2: gated SWE-smith-js + SWE-smith-ts integration
Delivery boundary:
- Add JS/TS adapters and verifier backends with bounded verifier-target preview/report safeguards; still gated by non-default config.

Implementation scope:
- Add dataset configs for JS/TS.
- Implement JS/TS verifier command execution strategy.
- Add strict safeguards for large PASS/FAIL target arrays (truncate display, preserve full execution plan outside prompt body).
- Add cost/time guardrails for high target-count verification.

Expected touchpoints:
- `configs/data/`
- `src/prompts/runtime_messages.py`
- `src/verl_integration/submission_verifier.py`
- `src/env/task_dataset.py`
- `tests/` prompt-shape and verifier coverage

Acceptance gates:
- JS/TS ingestion passes validation with deterministic filtered-row accounting.
- Verification can run in bounded mode and reports actionable failure payloads.
- Python + Go behavior remains unchanged under legacy configs.

Out of scope:
- No SWE-rebench on-policy activation yet.

### Milestone 3: SWE-rebench pilot (explicit patch semantics only)
Delivery boundary:
- Add rebench adapter + verifier-plan support for offline/pilot path only; default on-policy training still unchanged unless explicit opt-in is set.

Implementation scope:
- Map rebench patch fields into explicit semantics (`gold_fix_patch` must not be auto-applied in student attempts).
- Support install/test command metadata routing into verifier backend.
- Add explicit policy checks preventing accidental gold-patch leakage into rollout initialization.

Expected touchpoints:
- `src/env/task_dataset.py`
- `src/rollout/onpolicy_collector.py`
- `src/verl_integration/submission_verifier.py`
- `configs/data/` and runtime policy knobs
- `tests/` for rebench patch-policy invariants

Acceptance gates:
- Rebench adapter tests prove `gold_fix_patch` is never auto-applied in training attempts.
- Command-based verification runs through backend abstraction with explicit allowlist.
- Feature remains opt-in and off by default.

Out of scope:
- No default production mix flip to include rebench.

## PR slicing proposal
Each milestone should be delivered as its own PR with a strict boundary:
1. PR-A (`M0`) contract/refactor only.
2. PR-B (`M1`) SWE-smith-go enablement only.
3. PR-C (`M2`) SWE-smith-js/ts enablement only.
4. PR-D (`M3`) SWE-rebench pilot-only adapter + guards.

Review policy:
- Do not mix milestones in one PR.
- Each PR must include: design delta, migration risk, tests added, and explicit out-of-scope section.

## Delivery boundary summary (human-facing)
- Immediate deliverable: M0 + M1.
- Deferred until verifier maturity: M2.
- Explicit hold/pilot-only: M3.

This keeps training stability while unlocking incremental dataset diversity.
