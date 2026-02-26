# Research: CLI Wrapper to Run Terminus2 or OpenHands with `small-swe-train` Backend

Generated: 2026-02-26 05:17 UTC
Status: research draft (no implementation in this PR)

## Goal
Define a practical path to run an external CLI/code agent runtime (Terminus2 or OpenHands) while using our trained model as the backend.

## Source-Backed Facts
- Harbor Terminus2 supports parser selection (`json` or `xml`), turn limits, summarization controls, and optional rollout-detail collection.
  - Source: https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
- Terminus2 prompt templates exist for both JSON and XML response contracts.
  - Sources:
    - https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/templates/terminus-json-plain.txt
    - https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/templates/terminus-xml-plain.txt
- OpenHands CLI supports terminal/headless/web/IDE modes and automation-friendly headless execution.
  - Source: https://raw.githubusercontent.com/OpenHands/OpenHands-CLI/main/README.md
- OpenHands SDK workspace exposes command and git primitives (`execute_command`, `git_changes`, `git_diff`) and supports local/remote workspace modes.
  - Source: https://docs.openhands.dev/sdk/api-reference/openhands.sdk.workspace
- OpenHands runtime internally routes actions like `CmdRunAction`, `FileReadAction`, `FileWriteAction`, and `IPythonRunCellAction`.
  - Source: https://raw.githubusercontent.com/All-Hands-AI/OpenHands/main/openhands/runtime/base.py

## Recommended Integration Strategy
Use a compatibility gateway that serves our trained model via an OpenAI-compatible chat endpoint, then configure Terminus2/OpenHands to call that endpoint.

Why this is the fastest path:
- avoids rewriting external runtimes
- keeps our model ownership + deployment control
- preserves runtime features from external agent frameworks (tooling, retries, UI, orchestration)

## Architecture
```text
Terminus2 or OpenHands CLI
        |
        | (chat/tool requests)
        v
Model Gateway (OpenAI-compatible API)
        |
        | (tokenizer/chat template adapter)
        v
small-swe-train model server (vLLM/TGI/custom)
```

## Wrapper Responsibilities
1. Request/response compatibility
- map incoming chat format to the model’s expected prompt template
- normalize tool-call style if runtime expects a specific schema

2. Safety and runtime controls
- enforce max output tokens
- enforce timeout and retry policy
- return structured errors for caller runtime

3. Telemetry
- include trace IDs
- capture latency + token usage
- persist raw request/response for failure replay

## Per-Runtime Notes
### Terminus2 path
- Prefer `parser_name="json"` for alignment with current internal contracts.
- Keep `collect_rollout_details=false` by default; enable only for training/eval traces.
- Tune `max_turns` + summarization settings to avoid hidden context drift.

### OpenHands path
- Start with OpenHands CLI headless mode for deterministic benchmarking.
- Bind to remote workspace when isolation is needed; local workspace for quick dev loops.
- Map model endpoint config through CLI settings so backend swap does not require code changes.

## MVP Implementation Plan
1. Build `scripts/serve_model_gateway.py` (OpenAI-compatible `/v1/chat/completions`).
2. Add one smoke script for Terminus2 backend validation.
3. Add one smoke script for OpenHands headless backend validation.
4. Add contract tests:
   - malformed request handling
   - timeout handling
   - max-token clipping
   - tool-call schema preservation

## Acceptance Criteria (for implementation PR)
- Terminus2 can complete a simple terminal task against our backend endpoint.
- OpenHands headless can complete a simple repo task against our backend endpoint.
- Unified logs capture prompt, response, stop reason, and runtime latency.
- No code changes required inside external runtime repos for basic backend swap.

## Risks
- Prompt-template mismatch can degrade tool-call correctness even when endpoint is compatible.
- Runtime-specific assumptions about tool format (XML/JSON/function-call style) can silently reduce solve rate.
- Stateful kernels (notably Python tools) can expand context rapidly; wrapper must support context-protection policies.

## References
- Harbor Terminus2 source and templates:
  - https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
  - https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/templates/terminus-json-plain.txt
  - https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/templates/terminus-xml-plain.txt
- OpenHands CLI:
  - https://raw.githubusercontent.com/OpenHands/OpenHands-CLI/main/README.md
- OpenHands SDK workspace API:
  - https://docs.openhands.dev/sdk/api-reference/openhands.sdk.workspace
- OpenHands runtime actions:
  - https://raw.githubusercontent.com/All-Hands-AI/OpenHands/main/openhands/runtime/base.py
