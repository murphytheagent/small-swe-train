# Research: Migration from JSON Tool Calls to CodeAct-Style Markdown Actions

Generated: 2026-02-26 05:17 UTC
Status: research draft (no implementation in this PR)

## Goal
Define a low-risk migration path from strict JSON tool envelopes to a CodeAct-style markdown action format, while preserving reliability and observability.

## Requested Target
- Move from JSON-only prompt contract to markdown-wrapped first-class tools.
- Minimal first-class tools:
  - `bash`
  - `python`
  - `apply_patch`
  - `submit`
- Rewrite parser + tool executor to support the new contract.
- Add stronger tool-response handling and context-window protection.
- Handle stateful Python kernel context growth safely.

## Source-Backed Context
- CodeAct framing reports benefits from treating actions as executable code trajectories.
  - Source: https://arxiv.org/abs/2402.01030
- Harbor Terminus2 supports both JSON and XML parsers, which is a useful precedent for dual-format migration.
  - Source: https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
- OpenHands runtime already models action primitives such as command execution, file read/write, IPython execution, and browse actions.
  - Source: https://raw.githubusercontent.com/All-Hands-AI/OpenHands/main/openhands/runtime/base.py
- OpenHands SDK workspace exposes explicit execution + git primitives (`execute_command`, `git_changes`, `git_diff`) in both local and remote modes.
  - Source: https://docs.openhands.dev/sdk/api-reference/openhands.sdk.workspace

## Proposed Markdown Contract (v0)
```markdown
<think>
...
</think>

<tool name="bash">
ls -la
pytest -q tests/test_x.py
</tool>

<tool name="python">
import json
print(json.dumps({"ok": True}))
</tool>

<tool name="apply_patch">
*** Begin Patch
...
*** End Patch
</tool>

<tool name="submit">
Final answer text here
</tool>
```

Rules:
- `submit` must be terminal and single.
- `apply_patch` must parse as valid patch format.
- `python` tool runs in a managed kernel session with bounded memory/history.

## Migration Plan
### Phase 0: dual-parse (no behavior change)
- Keep JSON as canonical execution format.
- Add markdown parser that normalizes to same internal action objects.
- Log parser success/failure by format.

### Phase 1: executor expansion
- Add `python` tool executor (stateful kernel wrapper).
- Keep existing `bash`, `apply_patch`, `submit` executors.
- Normalize all tool responses into shared feedback packet.

### Phase 2: prompt switch for selected runs
- Enable markdown-first prompting on a controlled slice.
- Keep JSON fallback parser enabled.
- Compare solve rate and parser error rate against JSON baseline.

### Phase 3: context-protection hardening
- Add aggressive trimming policy:
  - rolling tool-response summarization
  - history windowing (recent raw + compressed summary)
  - hard token budget with early summarization trigger
- Add Python-kernel guardrails:
  - per-turn output truncation
  - max object preview length
  - session reset triggers after memory/error thresholds

### Phase 4: default shift
- Promote markdown-first only after quality and safety gates pass.
- Keep JSON ingest path for backward compatibility until deprecation decision.

## Parser/Executor Rewrite Scope
### Parser rewrite
- New markdown AST parser for `<think>` and `<tool name=...>` blocks.
- Deterministic normalization to internal `ActionEnvelope`.
- Strict error classes for malformed blocks, missing tool name, multi-submit violations.

### Executor rewrite
- Tool router updates:
  - add `python` execution backend
  - enforce `submit` terminal rule centrally
- Feedback normalization updates:
  - include tool exit code, stderr excerpt, truncation flags, and retry hints.

## Robust Tool-Response Handling
For every tool call, emit:
```json
{
  "tool": "python",
  "ok": false,
  "exit_code": 1,
  "stdout": "...",
  "stderr": "...",
  "truncated": true,
  "retry_hint": "fix NameError before rerun"
}
```
Principle: model-facing failures are recoverable unless infrastructure is broken.

## Stateful Python Kernel Strategy
- Keep one kernel per task attempt for coherence.
- Track kernel state budget:
  - cumulative output tokens
  - execution count
  - exception streak
- Reset kernel when any threshold is breached and emit an explicit `kernel_reset` feedback event.
- Preserve minimal carry-over summary after reset to avoid blind restarts.

## Acceptance Criteria (for implementation PR)
- Markdown parser + JSON parser both supported and tested.
- New `python` tool execution path with deterministic guardrails.
- Stop-reason telemetry distinguishes parser errors, tool errors, and infra failures.
- Context-protection telemetry proves summarization/reset policies are active.

## Risks
- Markdown parser ambiguity can create silent mis-execution if grammar is not strict.
- Stateful `python` tool can explode context or leak hidden state across turns.
- Solve-rate regressions are likely if prompt migration is done without dual-parse fallback.

## References
- CodeAct paper:
  - https://arxiv.org/abs/2402.01030
- Harbor Terminus2 parser strategy:
  - https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
- OpenHands runtime actions:
  - https://raw.githubusercontent.com/All-Hands-AI/OpenHands/main/openhands/runtime/base.py
- OpenHands workspace API:
  - https://docs.openhands.dev/sdk/api-reference/openhands.sdk.workspace
