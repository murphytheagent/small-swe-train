# Research: TerminalBench Adapter Plan (v1 + v2)

Generated: 2026-02-26 05:17 UTC
Updated: 2026-02-26 10:36 UTC (deep consult + codebase-context pass)
Status: research draft (implementation-ready plan; no code changes in this PR)

## Executive Decision Summary
- Keep `small-swe-train` internal runtime contract unchanged (`bash/search/apply_patch/submit`, JSON tool-call payloads).
- Add a shared TerminalBench adapter core plus version-specific adapters for v1 and v2.
- Run benchmark-faithful scoring per version, then normalize into one cross-version result schema.
- Keep raw version-specific scorer outputs in artifacts; never collapse away benchmark semantics.
- Add explicit version locks (dataset/registry commit + manifest hashes) and drift checks.
- Introduce failure taxonomy with strict `agent` vs `infra` vs `verifier` vs `schema` categories.
- Start with small canary tasks on each version before broad rollout.
- Treat v1-v2 score comparison as controlled normalization, not direct semantic equivalence.

## v1 vs v2 Benchmark Delta Map

| Dimension | v1 | v2 | Implication |
| --- | --- | --- | --- |
| Primary distribution | TerminalBench core registry entry | TerminalBench 2.0 / Harbor-style tasks | Separate loaders and lock files required |
| Task metadata format | YAML-oriented task descriptors | TOML + instruction-style metadata | Need dual parser layer |
| Environment style | Commonly compose/multi-service patterns | Commonly single-container task specs | Need v1 compose manager and v2 single-container manager |
| Scoring output | Parser/script-driven pass signal | Verifier reward artifact style | Preserve raw outputs and normalize post-score |
| Timeout controls | Task-level script/runtime timeouts | Task-level agent/verifier timeout fields | Use per-task resolved timeouts, not one global default |
| Difficulty/variant expression | Often encoded in IDs or metadata | Explicit metadata in newer task specs | Normalize into shared `difficulty` but keep raw labels |

## Gap Analysis Against Current Codebase

| Capability | Present now | Missing | Risk |
| --- | --- | --- | --- |
| Canonical tool schema + validation | Yes (`contracts.py`) | None | Low |
| Multi-turn parser and bridge execution | Yes (`turn_parser.py`, `env_bridge.py`) | None | Low |
| Docker tool execution | Yes (`docker_executor.py`) | None | Medium |
| Multi-turn collection and trace structure | Yes (`onpolicy_collector.py`) | None | Medium |
| Dedicated TerminalBench adapter package | No | `src/eval/adapters/terminalbench/*` | High |
| Version-aware dataset loading and pinning | No | v1/v2 loaders + lock files | High |
| Benchmark-faithful scoring integration | No | version-specific scoring modules | High |
| Cross-version normalization and comparison tooling | No | schema + compare script | High |
| Drift detection (upstream schema/image/version) | No | lock + drift checker scripts | Critical |

## Adapter Architecture

### Proposed file structure
- `src/eval/adapters/terminalbench/types.py`
- `src/eval/adapters/terminalbench/result_schema.py`
- `src/eval/adapters/terminalbench/normalization.py`
- `src/eval/adapters/terminalbench/runner.py`
- `src/eval/adapters/terminalbench/report.py`
- `src/eval/adapters/terminalbench/v1/dataset.py`
- `src/eval/adapters/terminalbench/v1/env_compose.py`
- `src/eval/adapters/terminalbench/v1/scoring.py`
- `src/eval/adapters/terminalbench/v1/version_lock.json`
- `src/eval/adapters/terminalbench/v2/dataset.py`
- `src/eval/adapters/terminalbench/v2/env_single.py`
- `src/eval/adapters/terminalbench/v2/scoring.py`
- `src/eval/adapters/terminalbench/v2/version_lock.json`
- `scripts/eval_terminalbench.py`
- `scripts/terminalbench_check_drift.py`
- `scripts/terminalbench_compare_v1_v2.py`

### Shared core responsibilities
- Build canonical `TaskSpec` objects from version-specific metadata.
- Reuse existing parser/bridge/executor stack for agent interaction.
- Emit normalized `ResultRecord` and per-task trace artifacts.
- Delegate scoring to v1/v2-specific scoring modules.

### Version-specific responsibilities
- v1 adapter:
  - Parse v1 task descriptors.
  - Launch compose-like environments when needed.
  - Execute v1-style test scripts and parse v1 scorer outputs.
- v2 adapter:
  - Parse v2 task metadata (Harbor-style layout).
  - Launch single-container task runtime (or declared environment runtime).
  - Execute verifier flow and parse reward artifacts.

## Unified Result Schema and Normalization Rules

```json
{
  "schema_version": "terminalbench_adapter.v1",
  "benchmark": "terminalbench",
  "benchmark_version": "v1|v2",
  "task_id": "example_task",
  "task_variant": "base|easy|hard|null",
  "difficulty": "easy|medium|hard|null",
  "run_id": "20260226T103600Z_terminalbench_eval",
  "completed": true,
  "stop_reason": "submitted_valid|max_turns|timeout|infra_error",
  "failure": {
    "category": "none|agent|infra|verifier|schema",
    "code": "..."
  },
  "normalized": {
    "resolved": true,
    "reward_01": 1,
    "wall_time_sec": 123.4,
    "tool_calls_total": 14
  },
  "score_raw": {
    "v1": {"parser": "pytest", "exit_code": 0},
    "v2": {"reward_txt": "1", "verifier_exit_code": 0}
  },
  "artifacts": {
    "trace_path": "...",
    "logs_path": "...",
    "verifier_path": "..."
  }
}
```

Normalization rules:
- `normalized.resolved`:
  - v1 from parsed v1 scorer signal.
  - v2 from parsed verifier reward threshold.
- `normalized.reward_01`:
  - binary normalization for cross-version aggregate comparisons.
- `score_raw` retains version-native semantics for auditability.

## Execution and Scoring Flows

### v1 flow
1. Resolve pinned v1 dataset source and task set.
2. Materialize task workspace with isolated logs/test artifacts.
3. Start v1 environment runtime (compose or equivalent manager).
4. Run agent loop via existing parser + bridge + executor stack.
5. Run v1 scoring scripts/parsers; capture raw outputs.
6. Normalize result fields and persist artifacts.
7. Teardown all runtime resources and verify cleanup.

### v2 flow
1. Resolve pinned v2 task metadata source.
2. Materialize task workspace and runtime config.
3. Start v2 environment runtime (single-container default).
4. Run agent loop via existing parser + bridge + executor stack.
5. Execute v2 verifier and parse reward outputs.
6. Normalize result fields and persist artifacts.
7. Teardown runtime resources and verify cleanup.

## Cross-Version Comparability Strategy
- Comparable metrics:
  - normalized pass rate (`resolved`), normalized reward (`reward_01`), runtime and tool-use distributions.
- Not directly comparable without mapping:
  - raw parser/verifier semantics, non-binary reward detail, and potentially non-equivalent task definitions.
- Add overlap-aware comparison mode:
  - compare full-set v1 and v2 separately,
  - optionally compare only human-curated equivalent task mappings.
- Always report both:
  - normalized cross-version summary,
  - per-version raw metric summary.

## Metrics and Failure Taxonomy

### Key metrics
- `resolved_rate`, `reward_01_mean`
- `submitted_rate`, `format_valid_rate`
- `infra_failure_rate`, `verifier_failure_rate`
- `median_wall_time`, `p95_wall_time`
- `tool_calls_total`, per-tool usage rates

### Failure taxonomy
- `agent.*`
  - parse/format errors, invalid submit, max-turn exhaustion, timeout in interaction loop.
- `infra.*`
  - dataset pull/checkout failures, runtime startup failures, container/compose failures.
- `verifier.*`
  - verifier timeout/crash, missing reward artifacts, parse errors.
- `schema.*`
  - unsupported parser mode, malformed task metadata, drift-induced contract mismatch.

## Validation Matrix

### Unit
- v1 parser tests, v2 parser tests.
- normalization rule tests (v1/v2 raw -> normalized schema).
- failure classifier tests.
- lockfile/hash determinism tests.

### Integration
- v1 canary task execution + scoring.
- v2 canary task execution + verifier parsing.
- artifact path and cleanup tests.

### Drift detection
- registry/manifest hash mismatch detection.
- task metadata schema drift detection.
- runtime image digest reporting and mismatch warning/error policy.

### E2E smoke
- one v1 task and one v2 task with deterministic stub turn generator.
- expected normalized outputs and taxonomy fields asserted.

## Phased Implementation Plan

### P0: Scaffolding
- Add adapter package skeleton, result schema, lockfile format, CLI shell.
- Gate: dry-run writes valid empty result manifests and lock validation passes.

### P1: v1 adapter path
- Implement v1 loader, env manager, scoring parser, and canary test.
- Gate: v1 canary task produces stable normalized + raw outputs.

### P2: v2 adapter path
- Implement v2 loader, env manager, verifier parser, and canary test.
- Gate: v2 canary task produces stable normalized + raw outputs.

### P3: Comparability and drift hardening
- Implement cross-version comparison script + drift checks.
- Gate: end-to-end report includes full-set and overlap-aware summaries with drift status.

## Open Questions for Human Review
- For v2 source of truth, should lock target Harbor registry metadata, direct repo commits, or both?
- Should drift checks hard-fail CI or warn-only in early rollout?
- Should the first release support compose tasks in v1 immediately, or only single-container subset first?
- Do we need strict task-equivalence mapping before any v1-v2 comparison claims?
- For failures from mixed causes, should taxonomy store one primary code plus secondary list?
- Do we want to normalize non-binary v2 reward beyond `reward_01` in this PR scope?

## References
- https://raw.githubusercontent.com/laude-institute/terminal-bench/main/README.md
- https://raw.githubusercontent.com/laude-institute/terminal-bench/main/registry.json
- https://harborframework.com/docs/tasks/task-overview
- https://harborframework.com/docs/tasks/differences-from-terminal-bench
- https://tbench.ai/registry
