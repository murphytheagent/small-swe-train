# STATUS

Last updated: 2026-04-26.

## Current State
- Canonical staged pipeline: `format_rft -> positive_rft -> turn_sdpo`.
- Repository notes identify the held-out `2040` chain as the latest validated E2E proof.
- `2040` proved `format_rft -> positive_rft` end to end, but positive-stage training still selected `0` rows while eval reached `6/64`; there is not yet a usable positive-RFT checkpoint.
- The next executable artifact is the first full `SWE-smith-py` difficulty-probe cache against the `2040` format-stage checkpoint.
- There is no active benchmark stage yet. `turn_sdpo` has in-loop verifier-backed validation only; the stale SWE-bench Lite evaluator path has been removed.
- JS/TS work should stay focused on repo-aware Node verifier adapters, not a generic `node_test` command toggle.
- Root documentation policy is now: keep only `AGENTS.md`, `README.md`, and `STATUS.md` at repo root; keep research, design, migration, and evaluation plans under `docs/`.

## Active TODO
- Materialize the `SWE-smith-py` difficulty-probe cache against the `2040` format-stage checkpoint.
- Diagnose why positive-stage training selected `0` rows despite nonzero held-out eval resolution.
- Continue JS/TS verifier planning around repo-aware selector normalization, runner detection, and bounded target reporting.
- Run the first `turn_sdpo` teacher-entropy pilot before adding an entropy-gated objective.
- Define a fresh benchmark target, artifact contract, and scoring runner before adding any post-training benchmarking stage.
- Keep packaging metadata aligned with `src/` imports when adding top-level modules or package data.

## Reference Docs
- `docs/design.md` - current architecture/design packet.
- `docs/stability_modules_plan.md` - staged pipeline stabilization plan.
- `docs/microcoder_transfer_plan.md` - verifier/data-quality/length-policy transfer plan.
- `docs/swe_smith_language_expansion_plan.md` - SWE-smith Go/JS/TS expansion plan.
- `docs/rft-eval-plan.md` - fixed held-out RFT checkpoint evaluation plan.
- `docs/turn_sdpo_entropy_gate_plan.md` - teacher-entropy-gated `turn_sdpo` ablation plan.
- `docs/research_*.md` - adapter and migration research notes.
