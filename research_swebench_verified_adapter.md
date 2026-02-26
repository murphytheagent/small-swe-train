# Research: SWE-bench-Verified Adapter for `small-swe-train`

Generated: 2026-02-26 05:17 UTC
Status: research draft (no implementation in this PR)

## Goal
Define a concrete adapter plan so `small-swe-train` can be evaluated on SWE-bench-Verified with minimal train/deploy mismatch.

## Source-Backed Facts
- SWE-bench introduced a `SWE-bench Verified` subset of 500 human-validated solvable problems.
  - Source: https://github.com/SWE-bench/SWE-bench
- Official SWE-bench harness uses `python -m swebench.harness.run_evaluation` and writes outputs under `evaluation_results/`.
  - Source: https://github.com/SWE-bench/SWE-bench
- Terminal-Bench registry includes a `swebench-verified` dataset adapter entry (`terminal_bench_version >=0.2.4`).
  - Source: https://raw.githubusercontent.com/laude-institute/terminal-bench/main/registry.json
- Harbor already ships a SWE-bench adapter using generated per-task directories (`task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`, `solution/solve.sh`).
  - Source: https://raw.githubusercontent.com/laude-institute/harbor/main/adapters/swebench/README.md

## Recommended Adapter Shape
### 1. Keep one internal runtime contract
- Keep current internal action schema as canonical runtime contract:
  - `bash`, `search`, `apply_patch`, `submit`
- Do not mirror SWE-bench-specific schema internally.
- Adapter boundary is translation/scoring, not runtime policy.

### 2. Split adapter into two layers
- `execution adapter`
  - Builds task workspace and launches agent loop.
  - Produces trajectory + final patch + metadata.
- `scoring adapter`
  - Runs SWE-bench verifier via official harness-compatible path.
  - Produces pass/fail + verifier details.

### 3. Unified prediction artifact
Use one normalized prediction record for every run:
```json
{
  "instance_id": "...",
  "model_name_or_path": "...",
  "patch": "...",
  "completed": true,
  "exit_reason": "submitted|timeout|runtime_error",
  "trajectory_path": "...",
  "timings": {"wall_seconds": 0.0}
}
```
This allows sending the same run output to either SWE-bench harness tooling or our own analytics.

### 4. Minimal module contract
- `src/eval/adapters/swebench_verified/spec.py`
  - task dataclasses and prediction schema.
- `src/eval/adapters/swebench_verified/build_task.py`
  - creates run directory and task materialization.
- `src/eval/adapters/swebench_verified/run_agent.py`
  - invokes current bridge loop against one task env.
- `src/eval/adapters/swebench_verified/score.py`
  - invokes verifier and emits normalized metrics.
- `src/eval/adapters/swebench_verified/cli.py`
  - batch entrypoint.

## Proposed Evaluation Flow
1. Load SWE-bench-Verified split and choose task IDs.
2. Materialize per-task workspace and metadata.
3. Run existing multi-turn agent loop until `submit` or stop condition.
4. Emit normalized prediction JSONL.
5. Score with verifier and store structured results.
6. Aggregate report: resolve rate, timeout rate, infra-failure rate, cost/time.

## Required Guardrails
- Runtime failures vs. model failures must be separated.
- Tool-format errors and non-zero tool exits are model feedback, not infra aborts.
- Infra/container failures are the only run-level hard failures.
- Save raw verifier outputs for auditing disagreements.

## Acceptance Criteria (for implementation PR)
- Reproducible CLI entrypoint for SWE-bench-Verified batch eval.
- One prediction JSONL format reused across local and harness scoring.
- Verifier-backed summary report with per-instance traceability.
- Regression tests for task materialization, prediction schema, and scorer parsing.

## Open Decisions
- Whether to consume SWE-bench tasks directly or through Harbor/Terminal-Bench dataset wrappers for initial v1.
- Whether initial v1 supports only single-attempt per instance or configurable multi-attempt.

## References
- SWE-bench repo and harness usage:
  - https://github.com/SWE-bench/SWE-bench
- Terminal-Bench registry (`swebench-verified`):
  - https://raw.githubusercontent.com/laude-institute/terminal-bench/main/registry.json
- Harbor SWE-bench adapter structure:
  - https://raw.githubusercontent.com/laude-institute/harbor/main/adapters/swebench/README.md
