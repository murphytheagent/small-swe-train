# Fixed Held-Out Smith Checkpoint Eval for RFT

## Summary

Add a default-on fixed-checkpoint benchmark for RFT that evaluates the run’s initial checkpoint and every committed outer-step checkpoint on the same frozen 100-task SWE-smith held-out set. Use this as the cross-step accuracy curve for
format_rft; keep existing inner train/val losses as local inner_sft_* sanity telemetry only.

The held-out benchmark is telemetry only during format_rft. It must not change format-stage selection, stopping, or rejection behavior. The runner should be reusable later by positive_rft, where verifier-backed accuracy becomes
contract-critical per stability_modules_plan.md.

## Public Interfaces / Config Changes

- Add a new default-on rft_runtime.fixed_eval config block in configs/runtime/training_policy_defaults.v1.json with:
    - enabled: true
    - data_config_name: on_policy_swe_smith
    - task_manifest_path: benchmarks/on_policy_swe_smith/heldout_100_task_ids.json
    - every_n_steps: 1
    - attempts_per_task: 1
    - max_in_flight_tasks: 32
    - temperature: 0.0
    - top_p: 1.0
    - baseline mode fixed to “delta vs initial checkpoint”
- Extend OnPolicyDataConfig in src/config.py with an optional task-ID allowlist/manifest field so the task loader can materialize only the frozen held-out tasks.
- Expose minimal run-script / CLI escape hatches to disable or override fixed eval, at minimum:
    - enable/disable
    - manifest path
    - cadence

## Implementation Changes

- Add a committed benchmark manifest at benchmarks/on_policy_swe_smith/heldout_100_task_ids.json.
    - Source of truth is the committed file, not a runtime-derived split.
    - Initial contents are frozen from the current filtered SWE-bench/SWE-smith-py:train pool and never regenerated automatically at runtime.
- Extend task-pool loading in src/env/task_dataset.py so an allowlist manifest filters the task pool before batching.
    - Keep batching deterministic.
    - Make the task-pool cache allowlist-content aware, not just path-string aware.
    - Continue honoring the bad-task cache before final held-out pool construction.
- Add a dedicated fixed-eval runner instead of reusing collect_onpolicy_rft_runtime_batch.
    - Build the collector directly so verify_submissions=true is allowed.
    - Reuse the existing verifier path as-is in format_rft.
    - Require manage_vllm=true; fail closed with a clear runtime error if fixed eval is enabled while checkpoint identity is externally managed.
    - Add step-summary and W&B fields:
        - fixed_eval_total
        - fixed_eval_resolved
        - fixed_eval_resolve_rate
        - fixed_eval_resolve_rate_delta_vs_initial
        - fixed_eval_verifier_missing_count
        - fixed_eval_duration_sec
        - fixed_eval_manifest_path
- Keep format-stage contracts unchanged.
    - Do not enable verifier-backed selection for training data in format_rft.
    - Do not use fixed-eval score for step acceptance, row selection, or stop conditions.
    - Keep inner SFT loss fields, but relabel/document them as local sanity metrics rather than comparable accuracy metrics.

## Test Plan

- Config parsing accepts the new fixed-eval block and optional allowlist manifest field.
- Task-pool filtering returns only held-out manifest task IDs and remains deterministic with bad-task filtering enabled.
- Baseline eval is written once, reused on resume, and step deltas stay anchored to the initial checkpoint.
- Fixed eval uses verify_submissions=true and deterministic decoding, while the existing format-stage training collector path still keeps verifier disabled.
- Per-step summaries and W&B logs include fixed-eval metrics when enabled.
- The loop fails clearly when fixed eval is enabled with unmanaged/external vLLM.
- Benchmark artifact paths are stable and survive normal checkpoint/payload pruning.

## Assumptions and Defaults

- Fixed eval is the default cross-step accuracy curve for format_rft, but telemetry only.
- The current verifier implementation is acceptable for format-stage held-out telemetry.
- The frozen held-out set is exactly 100 committed SWE-smith task IDs.
- One attempt per task is the benchmark contract; this is a deterministic checkpoint score, not pass@k.
- Future positive_rft should reuse the same fixed-eval runner and artifact format without changing metric semantics.