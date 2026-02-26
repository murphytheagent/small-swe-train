# Research: SWE-bench-Verified Adapter for `small-swe-train`

Generated: 2026-02-26 10:27 UTC
Updated: 2026-02-26 10:31 UTC (deep consult + codebase-context pass)
Status: research draft (implementation-ready plan; no code changes in this PR)

## Executive Decision Summary
- Keep the canonical internal tool contract unchanged: `bash`, `search`, `apply_patch`, `submit`.
- Keep current parser and bridge path unchanged (`turn_parser` -> `env_bridge` -> executor); build adapter around it.
- Split implementation into two strict layers: execution adapter (run + trace + patch extraction) and scoring adapter (official harness run + merge).
- Use the official SWE-bench harness as the default scoring backend; treat Harbor/TerminalBench wrappers as optional execution conveniences only.
- Extract `model_patch` from container git state (`git diff`) at episode end, not from tool arguments.
- Introduce an explicit failure taxonomy that separates model formatting/tooling failures from infrastructure failures.
- Preserve a single normalized prediction schema and derive harness JSONL from it.
- Gate rollout with deterministic artifacts and negative tests for leakage and schema drift.

## Gap Analysis Against Current Codebase

| Capability | Present now | Missing | Risk |
| --- | --- | --- | --- |
| Canonical tools + validation (`contracts.py`) | Yes | None | Low |
| Assistant turn parsing with multi-tool and submit-terminal invariants (`turn_parser.py`) | Yes | None | Low |
| Bridge execution (`env_bridge.py`) with validation and tool response blocks | Yes | None | Low |
| Docker tool executor (`docker_executor.py`) | Yes | None | Medium (runtime variability) |
| On-policy multi-turn loop + trace scaffolding (`onpolicy_collector.py`) | Yes | None | Medium |
| SWE-bench-Verified dataset adapter | No | Dataset loader + mapping contracts | High |
| Official harness scoring integration | No | Harness runner + result parser + merger | High |
| Harness-compatible prediction writer | No | Projection from normalized records | High |
| Gold-patch leakage guardrails | Partial | Strict field mapping + negative tests | Critical |
| Run-level failure taxonomy with strict categories | Partial | Enum-like codes + reporting | High |

## Adapter Architecture

### Proposed module layout
- `src/eval/adapters/swebench_verified/spec.py`
  - Dataclasses and schema constants for normalized records.
- `src/eval/adapters/swebench_verified/dataset.py`
  - SWE-bench-Verified row ingestion and safe field mapping.
- `src/eval/adapters/swebench_verified/execution.py`
  - Multi-turn attempt runner over current parser/bridge/executor stack.
- `src/eval/adapters/swebench_verified/repo_state.py`
  - Repo reset, optional `test_patch` apply, and git diff extraction.
- `src/eval/adapters/swebench_verified/harness.py`
  - Official harness invocation and result parsing.
- `src/eval/adapters/swebench_verified/report.py`
  - Summary metrics, failure classification, and artifact manifests.
- `src/eval/adapters/swebench_verified/cli.py`
  - Batch orchestration entrypoint.
- `scripts/eval_swebench_verified.py`
  - Thin script wrapper (aligned with existing `scripts/eval_swebench_lite.py` pattern).

### Data contract boundaries
- `TaskSpec` (adapter input): `instance_id`, `repo`, `base_commit`, `problem_statement`, `test_patch`, optional metadata.
- `PredictionRecord` (execution output): normalized fields for run status, patch, trace, and failure class.
- `HarnessRecord` (projection): exactly `instance_id`, `model_name_or_path`, `model_patch`.
- `ScoredRecord` (merged output): normalized record + harness verdict fields.

## Official Harness Interop Strategy

### Recommendation
Use **direct official SWE-bench harness** as the default scorer.

### Why
- It is the canonical scoring path for SWE-bench-Verified.
- It minimizes benchmark-definition drift.
- It keeps final reported numbers comparable to external SWE-bench runs.

### Wrapper stance
- Harbor/TerminalBench wrappers are acceptable for developer iteration on execution runtime.
- Final scoring pass should still use official SWE-bench harness outputs.

### Leakage safeguard (critical)
The adapter must prevent accidental application of the dataset gold patch before model action. Keep a separate `gold_patch` field and never feed it into environment-init patch application paths.

## Unified Prediction + Trace Schema

```json
{
  "schema_version": "swebench_verified.v1",
  "run_id": "20260226T103100Z_verified_eval_qwen4b",
  "instance_id": "pallets__flask-12345",
  "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
  "attempt_index": 0,
  "completed": true,
  "stop_reason": "submitted_valid",
  "failure_category": null,
  "failure_code": null,
  "wall_seconds": 412.8,
  "tool_calls_total": 19,
  "tool_validation_errors_total": 0,
  "repo": {
    "name": "pallets/flask",
    "base_commit": "abcdef123456",
    "image_name": "swebench_eval_xxx"
  },
  "patch": {
    "model_patch": "diff --git a/foo.py b/foo.py\n...",
    "changed_paths": ["foo.py"],
    "is_empty": false
  },
  "trace": {
    "path": "traces/pallets__flask-12345_attempt0.jsonl",
    "turns": 7
  },
  "harness": {
    "scored": true,
    "resolved": false,
    "report_path": "harness/run_x/report.json"
  }
}
```

Required fields:
- `schema_version`, `run_id`, `instance_id`, `model_name_or_path`, `attempt_index`
- `completed`, `stop_reason`, `failure_category`, `failure_code`
- `patch.model_patch` (may be empty string but explicitly tracked)
- `trace.path`

## Execution and Scoring Lifecycle
1. Resolve run config and output directory.
2. Load target SWE-bench-Verified instances.
3. Materialize safe task records (`gold_patch` separated from runtime patch inputs).
4. For each instance attempt:
- start container from resolved image,
- reset repo to base commit,
- apply `test_patch` policy if configured,
- run assistant multi-turn loop via existing parser/bridge/executor,
- stop on valid submit, invalid submit, timeout, max-turn, or infra failure,
- extract `git diff` as `model_patch`, persist trace.
5. Write normalized prediction JSONL.
6. Project normalized predictions into harness JSONL.
7. Invoke official harness and capture raw outputs.
8. Merge harness verdicts into normalized records.
9. Emit aggregate metrics, failure taxonomies, and per-instance manifests.

## Metrics and Failure Taxonomy

### Core metrics
- `resolved_rate`
- `submitted_rate`
- `format_valid_rate`
- `harness_scored_fraction`
- `infra_failure_rate`
- `median_wall_seconds`, `p95_wall_seconds`
- `tool_failure_rate`

### Failure category split
- `MODEL_FORMAT`
  - parse errors, unsupported tool names, arg-schema violations, invalid terminal submit.
- `MODEL_TOOL`
  - non-zero tool exits, patch apply failures, model-driven bad commands.
- `INFRA`
  - container start failures, docker exec failures, repo bootstrap failures.
- `HARNESS`
  - harness invocation failure, malformed harness inputs, harness runtime crash.

Only `INFRA` and `HARNESS` count as infrastructure-level issues. `MODEL_FORMAT` and `MODEL_TOOL` remain behavioral outcomes.

## Validation Matrix

### Unit
- Schema serialization and deterministic JSONL ordering.
- Harness projection writer emits exactly required key names.
- Failure classifier correctness.

### Integration
- Repo reset + optional `test_patch` application.
- Turn loop parity with existing `env_bridge` semantics.
- `git diff` extraction correctness across multi-file edits.

### Negative tests
- Gold-patch leakage prevention test (dataset `patch` never applied pre-run).
- Empty patch prediction handling and denominator accounting.
- Malformed submit and malformed tool JSON handling.

### E2E smoke
- 1-instance dry run with deterministic stub generator.
- 5-instance harness-scored run with artifact checks.

## Phased Implementation Plan

### P0: Contracts and CLI skeleton
- Add `spec.py`, `cli.py`, projection writer, and deterministic run manifest.
- Gate: dry-run produces normalized + harness-projection files.

### P1: Execution adapter
- Add dataset mapping, repo prep, multi-turn execution, trace writing, patch extraction.
- Gate: deterministic execution on local smoke tasks with expected stop reasons.

### P2: Harness scoring adapter
- Add harness invocation and result merge.
- Gate: scored output successfully merged with per-instance traceability.

### P3: Reliability hardening
- Add retry policy for infra-only failures, resume support, richer metrics/reporting.
- Gate: 5-10 instance pilot run with complete failure taxonomy and stable artifacts.

## Open Questions for Human Review
- Should `test_patch` always be applied in the agent runtime, or only during harness scoring?
- Single-attempt vs multi-attempt policy for each instance in v1?
- Preferred image resolution strategy: static mapping file vs harness-driven build pipeline?
- Should empty `model_patch` instances be explicitly retained as unresolved in summary denominator?
- Required strictness for format errors: immediate fail vs one repair turn allowance?
- Required trace verbosity level for long runs (full stdout/stderr vs truncated payload + external logs)?
- Is Harbor execution compatibility a hard v1 requirement or deferred to v2?

## References
- https://github.com/SWE-bench/SWE-bench
- https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
- https://www.swebench.com/evaluation
- https://www.swebench.com/SWE-bench/api/harness/
- https://harborframework.com/registry
