# SWE-smith language expansion plan

Last updated: 2026-04-04 05:10 UTC.
Owner thread: Slack `1775202373.974779`.
Companion plan: `microcoder_transfer_plan.md` covers the verifier/data-quality/length-policy work that should land before or alongside language expansion.

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
- Reuse the shared verifier/data-quality contract from `microcoder_transfer_plan.md` rather than inventing a second policy path for non-Python SWE-smith rows.
- `SWE-smith-go` is now merged and node-smoke-proved.
- Integrate later: `SWE-smith-js`, `SWE-smith-ts` after the language-aware verifier path is stable and bounded for large target sets.
- Keep `SWE-smith-py` as the default path until each added language clears its own dry-run and verifier gates.
- Treat large verifier target lists as a first-class constraint: the full test plan must stay in structured metadata, while prompts and human-readable reports use bounded previews only.
- Judge language readiness on selector-valid rows, and keep verifier/backend/setup failures separate from ordinary unresolved-task outcomes.
- Do not treat `node_test` as one command. JS/TS enablement needs a repo-aware Node verifier adapter layer with per-family selector resolution and runner dispatch.

## Current status after PR `#31`

- Merged `PR #31` closes the shared contract work plus the Go enablement object:
  - config-side `patch_is_bug_introducing` + `verifier_kind`,
  - duplicate-target / FAIL-PASS overlap rejection and logical-task dedupe,
  - bounded target-preview fallback when issue text is missing,
  - task-level verifier smoke in preflight,
  - runtime Go verification, reward wiring, and node-backed minimal SDPO proof.
- JS/TS are still not runtime-complete after that merge:
  - config and some preflight plumbing now recognize `node_test`,
  - the runtime verifier still only executes `pytest` and `go_test`,
  - there is no repo-aware selector-to-command adapter for JS/TS yet.

## Live JS/TS findings that change the plan

- Target cardinality is materially larger than Go in the rows I checked:
  - JS can carry up to `18394` `PASS_TO_PASS` selectors.
  - TS can carry up to `5890` `PASS_TO_PASS` selectors.
- TS selectors are not plain test names everywhere. At least one live family uses workspace-qualified selectors like `|effect| test/Micro.test.ts`.
- The command contract is repo-family-specific, not just language-specific:
  - TS `effect` expects package-local `pnpm exec vitest run <file>` under the right workspace.
  - JS `mongoose` is Mocha-driven and wants file/grep-style dispatch through the repo's Node toolchain.
- Because of that, the next JS/TS milestone cannot honestly be "add two more configs." It needs:
  - selector normalization into structured metadata,
  - repo-aware runner resolution,
  - bounded verification policy for very large PASS target sets,
  - live container smokes on representative JS and TS families before default enablement.

## Why this sequencing
Current codebase constraints:
- `src/env/task_dataset.py` requires non-empty `problem_statement`, `fail_to_pass`, and `pass_to_pass`.
- `src/verl_integration/submission_verifier.py` now supports `pytest` and `go_test`, but it still has no JS/TS runtime adapter.
- `src/prompts/runtime_messages.py` now keeps the initial user prompt focused on the task objective, but large verifier target lists still need bounded preview/report rules outside the prompt body.
- Current runtime configs and regression tests are still centered on the Python and Go paths.

Because the language-specific SWE-smith variants stay within the same dataset family, the main missing surface is verifier and config generalization rather than a new dataset-policy program.

## Non-negotiable constraint
Some non-Python SWE-smith tasks will carry very large PASS/FAIL target lists. That does not fit the current prompt/report surface if we simply dump raw targets into the user-visible text.

Required behavior for every milestone:
- keep the initial prompt focused on the task objective rather than the full test list,
- preserve the full verifier target plan in structured metadata,
- expose only bounded previews and bounded failure summaries in user-visible text and logs,
- validate selector sets before enablement (dedupe exact duplicate targets, reject `fail_to_pass` / `pass_to_pass` overlap, and fail early on absurd selector payloads).

## Milestones

### Milestone 0: language-aware dataset + verifier baseline
Delivery boundary:
- Introduce the shared contracts needed for non-Python SWE-smith variants, but do not enable any new language in default runtime configs.

Implementation scope:
- Add a typed verifier backend interface with language-aware dispatch (`python`, `go`, `js`, `ts`).
- Add dataset-config plumbing for selecting SWE-smith language variants explicitly.
- Refactor preflight around `verifier_kind` rather than dataset language labels, with image-level runner checks kept separate from task-level target-smoke probes.
- Add row-static selector sanity checks before enablement so obviously bad target sets get filtered before runtime verification.
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
- Large verifier target lists stay out of the first-turn prompt and produce bounded previews/reports.
- Selector-invalid rows are accounted for separately from verifier/backend/setup failures and from genuine unresolved-task outcomes.

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
- Add deterministic selector validation and a Go target-smoke preflight before enablement.

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
- Failure payloads stay readable even when the underlying target list is large.
- Reports distinguish selector-invalid rows, verifier/backend/setup failures, and genuine unresolved-task outcomes.

Out of scope:
- `SWE-smith-js` and `SWE-smith-ts` are not enabled yet.

### Milestone 2: gated `SWE-smith-js` + `SWE-smith-ts` integration
Delivery boundary:
- Add JS/TS adapters and verifier backends with bounded verifier-target preview/report safeguards, still behind non-default configs.

Implementation scope:
- Add dataset configs for JS and TS.
- Implement a repo-aware Node verifier adapter instead of a single hardcoded `node_test` command.
- Normalize JS/TS selectors into structured verifier metadata before command execution.
  - Minimum useful fields: repo-family/runner hint, optional workspace/package hint, file/path target, and optional test-name/grep fragment.
- Add runner-specific execution strategies for the first real families observed in-container:
  - workspace-aware `pnpm` + Vitest dispatch for TS families like `effect`,
  - `npm` + Mocha file/grep dispatch for JS families like `mongoose`.
- Detect the effective package-manager / workspace root inside the task container instead of assuming repo root is always the command root.
- Add strict safeguards for very large PASS/FAIL target arrays.
- Define a bounded verification contract for large PASS sets before enablement.
  - Do not naively expand thousands of PASS selectors into one-turn prompts or one-test-per-selector shell loops.
  - The first pass should prefer repo-aware batch/file dispatch where the runner supports it, and only then layer on finer-grained name filtering.
- Carry the same selector-validation and verifier-kind preflight rules forward for JS/TS backends.
- Add cost/time guardrails for high target-count verification.
- Run live container smokes on at least one representative TS family and one representative JS family before calling the backend wired.

Expected touchpoints:
- `configs/data/`
- `src/prompts/runtime_messages.py`
- `src/verl_integration/submission_verifier.py`
- `src/env/task_dataset.py`
- `src/env/preflight_onpolicy_dataset.py`
- `tests/` for prompt-shape and verifier coverage

Acceptance gates:
- JS/TS ingestion passes validation with deterministic filtered-row accounting.
- Verification resolves selectors through the repo-aware adapter rather than a language-only shell template.
- At least one live TS container smoke and one live JS container smoke prove runner resolution plus bounded command dispatch on real task images.
- Verification runs in bounded mode and reports actionable failure payloads even when the underlying PASS target arrays are very large.
- Python and Go behavior remain unchanged under legacy configs.
- Very large PASS/FAIL target arrays never get dumped verbatim into prompts or user-visible logs.
- JS/TS readiness accounting keeps selector-invalid and verifier-unusable rows separate from ordinary unresolved tasks.

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
