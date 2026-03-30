# Research: Migration from JSON Tool Calls to CodeAct-Style Markdown Actions

Generated: 2026-02-26 05:17 UTC
Updated: 2026-03-30 11:26 UTC (refresh against current main action surface)
Status: research draft (implementation-ready plan; no code changes in this PR)

## 1) Executive Decision Summary
- Keep the current JSON tool-call path as canonical and production-safe during migration.
- Add CodeAct-style markdown parsing as a dual-parse path, normalized into current `ActionEnvelope`/`ToolCall` structures.
- Make the first migration target the current six-tool surface: `bash`, `read`, `file_search`, `text_search`, `apply_patch`, `submit`.
- Keep `python` out of the base migration; if it is still wanted later, treat it as a follow-on feature after the six-tool path is stable.
- Preserve current validation invariants (`submit` singleton terminal, max tool calls, schema checks).
- Strengthen context safety with deterministic history-budget controls beyond current per-tool truncation.
- Ship behind feature flags with rollback to JSON-only parsing at every phase.

## 2) Target Contract Definition (CodeAct-Style Markdown + Invariants)

### Proposed CodeAct v0 format
- Optional thinking block:
  - `<think>...</think>`
- One or more tool blocks:
  - `<tool name="bash">{"command":"pytest -q","cwd":".","timeout_sec":120}</tool>`
  - `<tool name="read">{"path":"src/app.py","start_line":10,"end_line":40}</tool>`
  - `<tool name="file_search">{"query":"docker executor","root":"src","top_k":5}</tool>`
  - `<tool name="text_search">{"query":"needle","path_hint":"src","top_k":5}</tool>`
  - `<tool name="apply_patch">{"path":"src/app.py","patch":"*** Begin Patch\\n...\\n*** End Patch","description":"narrow fix"}</tool>`
  - `<tool name="submit">{"final_response":"done"}</tool>`

### Normalization mapping to current internal schema
- Parse each `<tool>` body as a JSON args object, then normalize to the current internal schema:
  - `<tool name="bash">...</tool>` -> `{"tool":"bash","args":{...}}`
  - `<tool name="read">...</tool>` -> `{"tool":"read","args":{...}}`
  - `<tool name="file_search">...</tool>` -> `{"tool":"file_search","args":{...}}`
  - `<tool name="text_search">...</tool>` -> `{"tool":"text_search","args":{...}}`
  - `<tool name="apply_patch">...</tool>` -> `{"tool":"apply_patch","args":{...}}`
  - `<tool name="submit">...</tool>` -> `{"tool":"submit","args":{...}}`
- Use uniform JSON args inside CodeAct tags so current multi-field tool schemas carry over without inventing a second mini-language for `read` / `file_search` / `text_search`.

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
| Canonical schema + validator (`contracts.py`) | Yes (`bash/read/file_search/text_search/apply_patch/submit`) | No new base-tool schema; keep CodeAct normalized onto the existing registry | Medium |
| Bridge execution with validation (`env_bridge.py`) | Yes | Dual-parse entry + parse-format telemetry | Medium |
| Docker executor (`docker_executor.py`) | Yes (`bash/read/file_search/text_search/apply_patch/submit`) | No new executor path for the base migration | Medium |
| Deterministic tool-output truncation | Yes | propagation of truncation metadata and history budgeting | High |
| Rollout loop (`onpolicy_collector.py`) | Yes | prompt-format switch + context budgeting + format analytics | Medium |

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

## 5) Executor/Runtime Integration Architecture

### Execution model
- Route `bash`, `read`, `file_search`, `text_search`, `apply_patch`, and `submit` to the existing Docker executor path.
- Keep the base migration executor-neutral: the core work is parse/render compatibility, not adding a new execution substrate.
- If `python` is ever pursued later, add it as a separate post-migration proposal with its own executor design and budget policy.

### Required touched modules
- `src/schemas/contracts.py` (reconcile `bash.stdin` mismatch and confirm CodeAct-normalized args stay aligned with the current six-tool schema).
- `src/verl_integration/env_bridge.py` (dual parser + format telemetry).
- `src/prompts/runtime_messages.py` (optional CodeAct-format contract text for the current six-tool surface).
- `src/rollout/onpolicy_collector.py` (prompt-format switch + rollout analytics behind flags).
- `src/env/docker_executor.py` (retain existing behavior; ensure compatibility with the normalized CodeAct path).

## 6) Context-Window Control Plan

### Existing control (keep)
- Deterministic per-tool output truncation (head/tail strategy).

### New controls
- `src/codeact/context_budget.py`:
  - deterministic history budget manager,
  - preserve most recent turns/raw outputs,
  - condense older tool outputs first,
  - explicit “budget exceeded” signal if still over budget.

### Out-of-scope follow-on
- Deferred with the `python` follow-on; not part of the base six-tool migration.

## 7) Compatibility, Rollback, and Deployment Strategy

### Feature flags
- `action_parse_mode = json_only | dual | codeact_only`
- `action_prompt_mode = json | codeact`
- `enable_history_budget = true|false`

### Rollback guarantees
- Default remains `json_only` until CodeAct stability gates are met.
- Any migration issue can revert to JSON parsing without touching executor/runtime core.

### Deployment order
- parser-first dual mode,
- then CodeAct-format prompting for the current six-tool surface,
- then context budget manager,
- finally CodeAct-first prompting.

## 8) Telemetry and Failure Taxonomy

### Telemetry additions
- parse-format label: `json` vs `codeact`.
- parse error type/code.
- truncation events and counts.
- per-tool format-validity counts for `bash` / `read` / `file_search` / `text_search` / `apply_patch` / `submit`.

### Failure taxonomy
- `PARSE_*`
  - delimiter mismatch, invalid JSON payload, mixed format, unclosed tool tag.
- `VALIDATION_*`
  - unknown tool, missing arg, arg type/range violation.
- `EXECUTOR_*`
  - docker timeout, patch failure, command/search/read execution failure.
- `CONTEXT_*`
  - history budget exceeded, severe truncation events.

## 9) Validation Matrix (Unit/Integration/E2E + Adversarial)

### Unit
- CodeAct parser block extraction and invariant checks.
- dual-parser fallback and mixed-format errors.
- updated schema checks for the full six-tool surface plus `bash.stdin` policy.
- context-budget deterministic pruning tests.

### Integration
- bridge execution with CodeAct payloads across `bash` / `read` / `file_search` / `text_search` / `apply_patch` / `submit`.
- tool-response metadata correctness (`truncated`, search/read metadata, submit invariants).

### E2E
- collector run with CodeAct-emitting stub turn generator.
- JSON-only and dual-parse A/B stability checks.

### Adversarial
- payload containing literal `</tool>` strings in patch/code bodies.
- huge tool outputs + repeated long histories.
- prompt-injection-style fake tool tags inside tool-response text.

## 10) Phased Implementation Plan (P0-P4)

### P0: Contract scaffolding
- Add `src/codeact/` skeleton and lock the current six-tool CodeAct normalization rules.
- Resolve the `bash.stdin` policy mismatch explicitly.
- Gate: schema/unit tests pass with no behavior change in runtime paths.

### P1: Dual parser
- Implement CodeAct parser + dual parser integration in bridge.
- Gate: JSON goldens unchanged; CodeAct payloads parse/normalize deterministically.

### P2: Prompt/render integration
- Add CodeAct-format prompt contract text for the current six-tool surface and wire feature-flagged rollout emitters.
- Gate: stub rollouts emit valid CodeAct payloads that normalize back to the existing schema.

### P3: Context budget controls
- Add deterministic history budget manager + richer truncation telemetry.
- Gate: long-rollout stress tests avoid uncontrolled history growth.

### P4: Prompt switch + rollout A/B
- Enable optional CodeAct-first prompting while retaining JSON fallback.
- Gate: no regression beyond agreed thresholds in format validity and terminal-submit rates.

## 11) Open Questions for Human Review
- Should CodeAct tool bodies use uniform JSON args for all six tools, or allow text-body sugar for simple cases like `bash` / `submit`?
- Should legacy aliases such as `search` remain parse-time compatible during migration, or should CodeAct mode require canonical `file_search` / `text_search` names only?
- Should CodeAct tool tags be line-anchored only for robustness, or allow inline closing tags?
- For oversized history, prefer strict deterministic truncation only, or optional model-based summarization later?
- Should parser-format labels become first-class rollout row fields for easier analytics?
- Should `bash.stdin` be officially supported in schema (recommended) or removed from executor behavior?

## References
- https://arxiv.org/abs/2402.01030
- https://docs.all-hands.dev/modules/usage/agents/context_condenser
- https://docs.all-hands.dev/modules/sdk/reference/workspace
- https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
