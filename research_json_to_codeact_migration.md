# Research: Migration from JSON Tool Calls to CodeAct-Style Markdown Actions

Generated: 2026-02-26 05:17 UTC
Updated: 2026-02-26 10:50 UTC (deep consult + codebase-context pass)
Status: research draft (implementation-ready plan; no code changes in this PR)

## 1) Executive Decision Summary
- Keep the current JSON tool-call path as canonical and production-safe during migration.
- Add CodeAct-style markdown parsing as a dual-parse path, normalized into current `ActionEnvelope`/`ToolCall` structures.
- Add minimal first-class CodeAct tools: `bash`, `python`, `apply_patch`, `submit`.
- Implement `python` as a managed, stateful kernel with strict budgets/timeouts/reset semantics.
- Preserve current validation invariants (`submit` singleton terminal, max tool calls, schema checks).
- Strengthen context safety with deterministic history-budget controls beyond current per-tool truncation.
- Ship behind feature flags with rollback to JSON-only parsing at every phase.

## 2) Target Contract Definition (CodeAct-Style Markdown + Invariants)

### Proposed CodeAct v0 format
- Optional thinking block:
  - `<think>...</think>`
- One or more tool blocks:
  - `<tool name="bash">...</tool>`
  - `<tool name="python">...</tool>`
  - `<tool name="apply_patch">...</tool>`
  - `<tool name="submit">...</tool>`

### Normalization mapping to current internal schema
- `bash` block body -> `{"tool":"bash","args":{"command":"..."}}`
- `python` block body -> `{"tool":"python","args":{"code":"..."}}`
- `apply_patch` block body -> `{"tool":"apply_patch","args":{"patch":"..."}}`
- `submit` block body -> `{"tool":"submit","args":{"final_response":"..."}}`

### Invariants (must remain true)
- At least one tool call per assistant turn.
- `submit` must be the only tool call if present.
- Max tool calls per turn must still honor runtime config.
- No free text outside `<think>`/`<tool>` blocks in CodeAct mode.
- Deterministic normalization only (no non-deterministic auto-rewriting).

## 3) Gap Analysis Against Current Codebase

| Capability | Present now | Missing | Risk |
| --- | --- | --- | --- |
| JSON tool-call parser (`turn_parser.py`) | Yes | CodeAct markdown parser | High |
| Canonical schema + validator (`contracts.py`) | Yes (`bash/search/apply_patch/submit`) | Add `python` tool schema | High |
| Bridge execution with validation (`env_bridge.py`) | Yes | Dual-parse entry + parse-format telemetry | Medium |
| Docker executor (`docker_executor.py`) | Yes (`bash/search/apply_patch/submit`) | `python` tool executor path | High |
| Deterministic tool-output truncation | Yes | propagation of truncation metadata and history budgeting | High |
| Rollout loop (`onpolicy_collector.py`) | Yes | optional python-kernel lifecycle integration + context budgeting | Medium |

Notable schema mismatch to fix during migration:
- `docker_executor` supports `bash.stdin`, but current Bash schema does not include it.

## 4) Parser Migration Architecture (Dual-Parse + Normalization)

### New files under `src/codeact/`
- `src/codeact/errors.py`
- `src/codeact/types.py`
- `src/codeact/markdown_parser.py`
- `src/codeact/dual_parser.py`

### Integration point
- Replace bridge parsing entry (`_parse_assistant_text` in `env_bridge.py`) with dual parser:
  - attempt existing JSON tool-call parse first,
  - fallback to CodeAct markdown parse,
  - normalize both to current `ActionEnvelope`.

### Mixed-format policy
- If both `<tool_call>` and `<tool name=...>` formats appear in one payload, fail fast with explicit mixed-format error.

### Validation policy
- Keep `validate_tool_call()` as single execution gate.
- Parser only parses/normalizes; validator remains source of truth for arg/tool constraints.

## 5) Executor/Runtime Migration Architecture (Including Python Tool State Model)

### New files under `src/codeact/`
- `src/codeact/executor.py` (`CompositeToolExecutor`)
- `src/codeact/python_kernel.py` (`ManagedPythonKernelExecutor`)

### Execution model
- Route `bash/search/apply_patch/submit` to existing Docker executor.
- Route `python` to managed kernel executor.

### Python tool state model
- One kernel session per attempt.
- Stateful across turns in same attempt.
- Hard controls:
  - per-call timeout,
  - max calls per attempt,
  - max cumulative output budget,
  - forced reset on threshold violation.
- Emit deterministic metadata on reset/timeouts.

### Required touched modules
- `src/schemas/contracts.py` (add `python` schema; reconcile `bash.stdin` mismatch).
- `src/verl_integration/env_bridge.py` (dual parser + format telemetry).
- `src/rollout/onpolicy_collector.py` (optional composite executor wiring behind flag).
- `src/env/docker_executor.py` (retain existing behavior; ensure compatibility with updated schema).

## 6) Context-Window and Stateful-Python Control Plan

### Existing control (keep)
- Deterministic per-tool output truncation (head/tail strategy).

### New controls
- `src/codeact/context_budget.py`:
  - deterministic history budget manager,
  - preserve most recent turns/raw outputs,
  - condense older tool outputs first,
  - explicit “budget exceeded” signal if still over budget.

### Python-specific growth controls
- truncate python stdout/stderr deterministically,
- include `kernel_reset` and `kernel_reset_reason` in tool metadata,
- expose timeout/reset events as structured tool responses.

## 7) Compatibility, Rollback, and Deployment Strategy

### Feature flags
- `action_parse_mode = json_only | dual | codeact_only`
- `action_prompt_mode = json | codeact`
- `enable_python_tool = true|false`
- `enable_history_budget = true|false`

### Rollback guarantees
- Default remains `json_only` until CodeAct stability gates are met.
- Any migration issue can revert to JSON parsing without touching executor/runtime core.
- Python tool can be disabled independently of CodeAct parsing.

### Deployment order
- parser-first dual mode,
- then optional python tool,
- then context budget manager,
- finally CodeAct-first prompting.

## 8) Telemetry and Failure Taxonomy

### Telemetry additions
- parse-format label: `json` vs `codeact`.
- parse error type/code.
- truncation events and counts.
- python kernel reset/timeout counters.

### Failure taxonomy
- `PARSE_*`
  - delimiter mismatch, invalid JSON payload, mixed format, unclosed tool tag.
- `VALIDATION_*`
  - unknown tool, missing arg, arg type/range violation.
- `EXECUTOR_*`
  - docker timeout, patch failure, python exception/timeout.
- `CONTEXT_*`
  - history budget exceeded, severe truncation events.

## 9) Validation Matrix (Unit/Integration/E2E + Adversarial)

### Unit
- CodeAct parser block extraction and invariant checks.
- dual-parser fallback and mixed-format errors.
- updated schema checks (`python`, `bash.stdin`).
- context-budget deterministic pruning tests.

### Integration
- bridge execution with CodeAct payloads.
- executor routing (`python` vs docker-backed tools).
- tool-response metadata correctness (`truncated`, reset signals).

### E2E
- collector run with CodeAct-emitting stub turn generator.
- JSON-only and dual-parse A/B stability checks.

### Adversarial
- payload containing literal `</tool>` strings in patch/code bodies.
- huge tool outputs + repeated long histories.
- infinite-loop python code (timeout/reset expected).
- prompt-injection-style fake tool tags inside tool-response text.

## 10) Phased Implementation Plan (P0-P4)

### P0: Contract scaffolding
- Add `src/codeact/` skeleton and extend schemas (`python`, `bash.stdin` policy fix).
- Gate: schema/unit tests pass with no behavior change in runtime paths.

### P1: Dual parser
- Implement CodeAct parser + dual parser integration in bridge.
- Gate: JSON goldens unchanged; CodeAct payloads parse/normalize deterministically.

### P2: Python executor
- Implement managed python kernel + composite executor.
- Gate: integration tests for state persistence, timeout, reset metadata.

### P3: Context budget controls
- Add deterministic history budget manager + richer truncation telemetry.
- Gate: long-rollout stress tests avoid uncontrolled history growth.

### P4: Prompt switch + rollout A/B
- Enable optional CodeAct-first prompting while retaining JSON fallback.
- Gate: no regression beyond agreed thresholds in format validity and terminal-submit rates.

## 11) Open Questions for Human Review
- Should `python` run host-side (simpler) or inside task containers (closer env fidelity, higher complexity)?
- Should `search` remain available but undocumented in CodeAct prompts during migration?
- Should CodeAct tool tags be line-anchored only for robustness, or allow inline closing tags?
- For oversized history, prefer strict deterministic truncation only, or optional model-based summarization later?
- Should parser-format labels become first-class rollout row fields for easier analytics?
- Should `bash.stdin` be officially supported in schema (recommended) or removed from executor behavior?

## References
- https://arxiv.org/abs/2402.01030
- https://docs.all-hands.dev/modules/usage/agents/context_condenser
- https://docs.all-hands.dev/modules/sdk/reference/workspace
- https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
