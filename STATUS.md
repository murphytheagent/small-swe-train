# STATUS

Last updated: 2026-04-28.

## Current State
- Canonical staged pipeline: `format_rft -> positive_rft -> turn_sdpo`.
- Repository notes identify the held-out `2040` chain as the latest validated E2E proof.
- `2040` proved `format_rft -> positive_rft` end to end, but positive-stage training still selected `0` rows while eval reached `6/64`; there is not yet a usable positive-RFT checkpoint.
- The full `SWE-smith-py` difficulty-probe cache against the `2040` format-stage checkpoint is intentionally deferred until cheaper preflights show that the current default model can produce useful positive-RFT signal.
- Immediate preflight work should use the real `on_policy_swe_smith` train path (`configs/data/on_policy_swe_smith.yaml`, the full `SWE-bench/SWE-smith-py` train split). The bounded work comes from RFT/pilot task batching over that dataset, not from a separate reduced dataset slice.
- RFT/pilot sampling is code-driven: `src/rollout/onpolicy_collector.py` calls `load_task_batch(...)`, and `src/env/task_dataset.py` chooses tasks deterministically from the full train or held-out partition using `step_index`, `batch_size`, and wraparound. The experiment controls are therefore outer steps, task batch size, attempts per task, train batch/min-row settings, and held-out eval count, and the RFT values must be defined through `configs/` rather than ad hoc launch-env tuning.
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
- Review cleanup is applied for the preflight branch: the stability plan is marked historical, root docs describe Python-native schema validators and the cached RFT dataset path, the tracked SWE-bench Lite placeholder dirs are removed, preflight runtime policies are tracked under `configs/runtime/`, RFT token-cache writing is deferred until after upsampling and reuses written records for profiler masks, and the teacher-reprompt pilot has configurable partial-output flush/summary intervals.
- The 4-GPU preflight matrix under `outputs/preflight_runs/20260428T010552Z_4gpu` was intentionally stopped on 2026-04-28 before producing usable experiment results. Jobs `2948`-`2955` are no longer in `squeue`; `2948` had already exited during vLLM startup, while the remaining launched/pending jobs were canceled on request. Wait for explicit relaunch instruction before submitting replacement jobs.
- Replacement 4-GPU preflights were queued on 2026-04-28 under `outputs/preflight_runs/20260428T092008Z_4gpu`: format RFT jobs `3024` JSON and `3025` XML, dependent positive RFT jobs `3026` JSON after `3024` and `3027` XML after `3025`, and teacher-reprompt pilot replicates `3028`/`3029` JSON plus `3030`/`3031` XML. At submission check, the jobs were pending for resources/priority/dependencies on the mixed GPU node.

## Active TODO
- Preflight setup is encoded in checked-in policy configs under `configs/runtime/training_policy_preflight_*.v1.json`. Launch-only selectors remain stage name, output directory, vLLM port, and checkpoint handoff.
- Monitor queued preflight matrix `3024`-`3031` for startup health, especially vLLM readiness and clean Slurm GPU allocation. JSON remains the default contract unless matched JSON-vs-XML evidence says otherwise; XML is an explicit opt-in config for this comparison.
- Preflight 1: monitor real `format_rft` JSON/XML jobs with `Qwen/Qwen3-8B`, thinking off, `5` outer steps, `task_batch_size=512`, `samples_per_task=4`, `train_batch_size=train_min_rows=32`, and `eval_task_count=50`. Track selected-row count, format-valid rate, held-out resolved/selected counts, and whether held-out eval improves from step 0.
- Preflight 2: monitor dependent positive RFT JSON/XML jobs initialized from the matching format job manifest, with `5` outer steps and the same collection/eval settings, but `train_batch_size=train_min_rows=4` for a 4-GPU run. Treat failure to select enough verifier-positive rows for the threshold or repeated skipped inner optimization steps as a hard blocker for full difficulty banding.
- Preflight 3: monitor two JSON and two XML teacher-vs-student reprompt replicates on fixed `step_index=0`, `task_batch_size=512`, and `attempts_per_task=4`; average `reward_delta`, resolved/pass deltas, and task-level win/loss counts before deciding whether the teacher is actually stronger than the student for `turn_sdpo`.
- After the JSON and XML preflights, decide whether to materialize the full `SWE-smith-py` difficulty-probe cache or debug positive-RFT row selection, batch/min-row settings, payload format, and teacher prompting first.
- Continue JS/TS verifier planning around repo-aware selector normalization, runner detection, and bounded target reporting.
- Run the first `turn_sdpo` teacher-entropy pilot before adding an entropy-gated objective, but only after the teacher-vs-student reprompt pilot shows positive teacher signal on the matched deterministic task batch.
- Define a fresh benchmark target, artifact contract, and scoring runner before adding any post-training benchmarking stage.
- Keep packaging metadata aligned with `src/` imports when adding top-level modules or package data.
- Use `docs/system_optimization_8b.md` to track 8B memory/profiler follow-ups, especially eager-vs-cuda-graph measurement and RFT vLLM restart cost.

## Reference Docs
- `docs/design.md` - current architecture/design packet.
- `docs/stability_modules_plan.md` - historical staged pipeline stabilization plan; current status is tracked here.
- `docs/microcoder_transfer_plan.md` - verifier/data-quality/length-policy transfer plan.
- `docs/swe_smith_language_expansion_plan.md` - SWE-smith Go/JS/TS expansion plan.
- `docs/rft-eval-plan.md` - fixed held-out RFT checkpoint evaluation plan.
- `docs/turn_sdpo_entropy_gate_plan.md` - teacher-entropy-gated `turn_sdpo` ablation plan.
- `docs/system_optimization_8b.md` - Qwen3-8B system optimization tracker.
- `docs/research_*.md` - adapter and migration research notes.
