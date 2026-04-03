# SWE-smith language expansion plan

Last updated: 2026-04-03 UTC.
Owner thread: Slack `1775202373.974779`.

## Goal
Integrate the non-Python SWE-smith language variants into `small-swe-train` without destabilizing the current `SWE-smith-py` training path.

Language variants in scope:
- `SWE-bench/SWE-smith-go`
- `SWE-bench/SWE-smith-js`
- `SWE-bench/SWE-smith-ts`

Out of scope:
- any non-SWE-smith dataset family
- changing the default training mix before the new verifier paths are proven

## Decision summary
- Integrate first: `SWE-smith-go`.
- Integrate later: `SWE-smith-js`, `SWE-smith-ts` after the language-aware verifier path is stable and bounded for large target sets.
- Keep `SWE-smith-py` as the default path until each added language clears its own dry-run and verifier gates.

## Why this sequencing
Current codebase constraints:
- `src/env/task_dataset.py` requires non-empty `problem_statement`, `fail_to_pass`, and `pass_to_pass`.
- `src/verl_integration/submission_verifier.py` is still Python/pytest-specific.
- `src/prompts/runtime_messages.py` now keeps the initial user prompt focused on the task objective, but large verifier target lists still need bounded preview/report rules outside the prompt body.
- Current runtime configs and regression tests are still centered on the Python path.

Because the language-specific SWE-smith variants stay within the same dataset family, the main missing surface is verifier and config generalization rather than a new dataset-policy program.

## Milestones

### Milestone 0: language-aware dataset + verifier baseline
Delivery boundary:
- Introduce the shared contracts needed for non-Python SWE-smith variants, but do not enable any new language in default runtime configs.

Implementation scope:
- Add a typed verifier backend interface with language-aware dispatch (`python`, `go`, `js`, `ts`).
- Add dataset-config plumbing for selecting SWE-smith language variants explicitly.
- Add bounded verifier-target preview/report safeguards so large target lists stay out of the first-turn prompt body.
- Keep the current `SWE-smith-py` path bit-for-bit equivalent under existing configs.

Expected touchpoints:
- `configs/data/`
- `src/env/task_dataset.py`
- `src/prompts/runtime_messages.py`
- `src/verl_integration/submission_verifier.py`
- `tests/` for verifier contract and config coverage

Acceptance gates:
- Existing `SWE-smith-py` behavior remains unchanged by default config.
- New shared contracts have regression tests for schema validation and fallback behavior.
- No runtime config flips to a new language yet.

Out of scope:
- No default mix change.
- No language-specific enablement yet.

### Milestone 1: gated `SWE-smith-go` integration
Delivery boundary:
- Enable `SWE-smith-go` behind an explicit non-default data config and a working Go verifier backend.

Implementation scope:
- Add a Go dataset config under `configs/data/`.
- Map `SWE-smith-go` rows to the normalized task contract.
- Implement a Go verifier execution strategy.
- Add deterministic filtering and logging for empty/invalid rows.

Expected touchpoints:
- `configs/data/`
- `src/env/task_dataset.py`
- `src/verl_integration/submission_verifier.py`
- `scripts/run_rft.sh` and/or `scripts/run_sdpo.sh` docs if new knobs are needed
- `tests/` for ingestion and verifier coverage

Acceptance gates:
- End-to-end dry-run ingestion over the `SWE-smith-go` split succeeds with stable row counts.
- The on-policy collector verifies with the Go backend without Python-path regressions.
- Prompt size remains bounded for high-cardinality target cases.

Out of scope:
- `SWE-smith-js` and `SWE-smith-ts` are not enabled yet.

### Milestone 2: gated `SWE-smith-js` + `SWE-smith-ts` integration
Delivery boundary:
- Add JS/TS adapters and verifier backends with bounded verifier-target preview/report safeguards, still behind non-default configs.

Implementation scope:
- Add dataset configs for JS and TS.
- Implement JS/TS verifier command execution strategies.
- Add strict safeguards for very large PASS/FAIL target arrays.
- Add cost/time guardrails for high target-count verification.

Expected touchpoints:
- `configs/data/`
- `src/prompts/runtime_messages.py`
- `src/verl_integration/submission_verifier.py`
- `src/env/task_dataset.py`
- `tests/` for prompt-shape and verifier coverage

Acceptance gates:
- JS/TS ingestion passes validation with deterministic filtered-row accounting.
- Verification runs in bounded mode and reports actionable failure payloads.
- Python and Go behavior remain unchanged under legacy configs.

Out of scope:
- No additional dataset families.

## PR slicing proposal
Each milestone should be delivered as its own PR with a strict boundary:
1. PR-A (`M0`) shared verifier/config baseline only.
2. PR-B (`M1`) `SWE-smith-go` enablement only.
3. PR-C (`M2`) `SWE-smith-js` + `SWE-smith-ts` enablement only.

Review policy:
- Do not mix milestones in one PR.
- Each PR must include: design delta, migration risk, tests added, and explicit out-of-scope section.

## Delivery boundary summary (human-facing)
- Immediate deliverable: `M0` + `M1`.
- Deferred until verifier maturity: `M2`.

This keeps the current Python training path stable while making the next language additions explicit and reviewable.
