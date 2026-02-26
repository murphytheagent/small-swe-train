# Research: TerminalBench Adapter Plan (v1 + v2)

Generated: 2026-02-26 05:17 UTC
Status: research draft (no implementation in this PR)

## Goal
Define how `small-swe-train` should evaluate on both TerminalBench 1.x and TerminalBench 2.x without changing the core agent runtime contract.

## Source-Backed Facts
- Terminal-Bench repository points new users to Harbor for running Terminal-Bench 2.0.
  - Source: https://raw.githubusercontent.com/laude-institute/terminal-bench/main/README.md
- Terminal-Bench Core v0.1.1 is a registry-defined dataset with 80 task IDs.
  - Source: https://raw.githubusercontent.com/laude-institute/terminal-bench/main/registry.json
- Harbor registry entry `terminal-bench@2.0` states “more tasks, harder, and higher quality than 1.0”, with 89 tasks on the registry page.
  - Source: https://harborframework.com/registry/terminal-bench/2.0
- Harbor usage for v2 dataset can be launched via `uvx harbor run -d terminal-bench@2.0`.
  - Source: https://harborframework.com/registry/terminal-bench/2.0

## Design Principle
Use one shared execution abstraction, and mount benchmark-specific task adapters on top.

- Shared execution layer:
  - same model runtime, same tool policy, same stop-reason logging.
- Benchmark-specific adapters:
  - task materialization
  - result normalization
  - scorer binding

## Recommended Adapter Split
### A. `terminalbench_v1_adapter`
Scope: Terminal-Bench-Core v0.1.1 compatibility.

Responsibilities:
- consume v1 task metadata conventions
- preserve v1 scoring semantics
- emit normalized record format compatible with shared analytics

### B. `terminalbench_v2_adapter`
Scope: Harbor `terminal-bench@2.0` dataset execution.

Responsibilities:
- integrate Harbor dataset/task contract
- keep identical normalized output schema as v1 adapter
- run v2 scoring and aggregate with shared report template

## Unified Result Schema
```json
{
  "benchmark": "terminalbench_v1|terminalbench_v2",
  "task_id": "...",
  "attempt_id": 0,
  "completed": true,
  "exit_reason": "submitted|timeout|runtime_error",
  "score": {"resolved": false, "raw": {}},
  "artifacts": {
    "trajectory": "...",
    "stdout": "...",
    "stderr": "..."
  }
}
```

## Implementation Sequence
1. Build shared adapter interfaces:
   - `prepare_task()`
   - `run_attempt()`
   - `score_attempt()`
   - `to_normalized_result()`
2. Implement v1 adapter first (smaller/known baseline).
3. Implement v2 adapter with same normalized output.
4. Add cross-benchmark aggregator that merges both schemas into one report.

## Metrics to Track
- solve/pass rate
- timeout rate
- infra-failure rate
- median wall time per task
- stop-reason distribution
- tool-interaction density per solved vs. unsolved tasks

## Risk Notes
- Dataset/version drift across v1 and v2 can silently break comparability; pin explicit dataset versions.
- Scoring protocol drift (especially partial-credit behavior) must be reflected in `score.raw` and not collapsed prematurely.
- Resource-heavy tasks in v2 can bias runtime-failure metrics if not split from model-failure metrics.

## Acceptance Criteria (for implementation PR)
- Separate runnable entrypoints for v1 and v2.
- One shared normalized JSONL schema for both versions.
- Version-pinned metadata in run manifest.
- Regression tests for adapter input parsing + output normalization for both versions.

## References
- Terminal-Bench README:
  - https://raw.githubusercontent.com/laude-institute/terminal-bench/main/README.md
- Terminal-Bench registry definitions:
  - https://raw.githubusercontent.com/laude-institute/terminal-bench/main/registry.json
- Harbor registry page (terminal-bench@2.0):
  - https://harborframework.com/registry/terminal-bench/2.0
