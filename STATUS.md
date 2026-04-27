# STATUS

Last updated: 2026-04-27.

## Current State
- Canonical staged pipeline: `format_rft -> positive_rft -> turn_sdpo`.
- Repository notes identify the held-out `2040` chain as the latest validated E2E proof.
- `2040` proved `format_rft -> positive_rft` end to end, but positive-stage training still selected `0` rows while eval reached `6/64`; there is not yet a usable positive-RFT checkpoint.
- The next executable artifact is the first full `SWE-smith-py` difficulty-probe cache against the `2040` format-stage checkpoint.
- There is no active benchmark stage yet. `turn_sdpo` has in-loop verifier-backed validation only; the stale SWE-bench Lite evaluator path has been removed.
- RFT convergence telemetry now uses a fixed valid-task holdout count (`rft_runtime.loop.eval_task_count`, default `50`) collected once per outer step. Step 0 evaluates the initial model, both format and positive RFT use the path, and the local SFT trainer entrypoint disables verl's inner validation path for this signal.
- The disabled-validation SFT entrypoint now tears down the verl global process group with `finally` after initialization, including dataset-construction and `trainer.fit()` failures.
- Qwen3-8B system optimization is implemented behind the live defaults: `Qwen/Qwen3-8B`, thinking-off tokenizer/vLLM defaults, RFT pre-tokenized full-transcript cache consumption, length bucketing without packing, conservative RFT/SDPO memory defaults, and `profiler/*` telemetry.
- Cached RFT inner SFT length bucketing now uses the DP-normalized per-rank train batch size for the sampler and `StatefulDataLoader`, so `drop_last=True` no longer treats the global target as a per-rank batch.
- PR #35 review feedback is addressed locally: RFT token caches now use the active checkpoint tokenizer/fingerprint for each outer step, and explicit `enable_thinking` caller kwargs no longer receive a contradictory no-thinking chat-template map.
- JS/TS work should stay focused on repo-aware Node verifier adapters, not a generic `node_test` command toggle.
- Root documentation policy is now: keep only `AGENTS.md`, `README.md`, and `STATUS.md` at repo root; keep research, design, migration, and evaluation plans under `docs/`.
- JSON remains the default assistant payload format; XML action format is implemented as an opt-in path with reusable schema derivation, dual parsing, prompt/render support, and JSON fallback for non-XML-representable structured tool calls.
- PR #34 Codex P1 is addressed: XML rendering now rejects list values for scalar schema args so structured vLLM fallback preserves malformed scalar-list payloads as JSON for downstream validation.

## Active TODO
- Materialize the `SWE-smith-py` difficulty-probe cache against the `2040` format-stage checkpoint.
- Run a real E2E JSON-vs-XML rollout comparison before deciding whether XML is effective enough to become the default assistant payload surface.
- Use the fixed outer-step RFT eval telemetry to diagnose positive-stage convergence and row-selection behavior.
- Continue JS/TS verifier planning around repo-aware selector normalization, runner detection, and bounded target reporting.
- Run the first `turn_sdpo` teacher-entropy pilot before adding an entropy-gated objective.
- Define a fresh benchmark target, artifact contract, and scoring runner before adding any post-training benchmarking stage.
- Keep packaging metadata aligned with `src/` imports when adding top-level modules or package data.
- Use `docs/system_optimization_8b.md` to track 8B memory/profiler follow-ups, especially eager-vs-cuda-graph measurement and RFT vLLM restart cost.

## Reference Docs
- `docs/design.md` - current architecture/design packet.
- `docs/stability_modules_plan.md` - staged pipeline stabilization plan.
- `docs/microcoder_transfer_plan.md` - verifier/data-quality/length-policy transfer plan.
- `docs/swe_smith_language_expansion_plan.md` - SWE-smith Go/JS/TS expansion plan.
- `docs/rft-eval-plan.md` - fixed held-out RFT checkpoint evaluation plan.
- `docs/turn_sdpo_entropy_gate_plan.md` - teacher-entropy-gated `turn_sdpo` ablation plan.
- `docs/system_optimization_8b.md` - Qwen3-8B system optimization tracker.
- `docs/research_*.md` - adapter and migration research notes.
