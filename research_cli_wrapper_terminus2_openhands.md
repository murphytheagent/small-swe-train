# Research: CLI Wrapper for Terminus2/OpenHands with `small-swe-train` Backend

Generated: 2026-02-26 05:17 UTC
Updated: 2026-02-26 10:44 UTC (deep consult + codebase-context pass)
Status: research draft (implementation-ready plan; no code changes in this PR)

## Executive Decision Summary
- Build a dedicated in-repo runtime gateway under `src/runtime_gateway/` as the interoperability boundary.
- Keep the internal action contract canonical (`bash/search/apply_patch/submit`, JSON args) and do all external mediation at the gateway boundary.
- Use OpenAI-compatible chat API at the gateway edge so Terminus2/OpenHands can integrate without patching their repos.
- Reuse existing parser/schema stack (`turn_parser`, `contracts`, `env_bridge`) for model-output validation before emitting runtime-specific responses.
- Add deterministic context-window controls in the gateway for inbound/outbound tool-output growth.
- Add strict telemetry and replay artifacts for parse/adapter/runtime failures.
- Start with minimal command-path interoperability (smoke tasks) before richer native tool mapping.

## Integration Options Matrix

| Option | Description | Effort | Reliability | Preserves canonical internal JSON | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI-compat gateway (recommended) | Our service translates external runtime expectations to/from canonical schema | Medium | High | Yes | Best isolation of drift |
| Native runtime plugin | Build runtime-specific plugins/forks | High | Medium | Yes | High maintenance burden |
| Direct prompt shim | Try to force direct prompt-level compatibility only | Low | Low | Partial | brittle and hard to audit |

## Gap Analysis Against Current Codebase

| Capability | Present now | Missing | Risk |
| --- | --- | --- | --- |
| Canonical tool schemas + validation (`contracts.py`) | Yes | None | Low |
| Canonical assistant turn parsing (`turn_parser.py`) | Yes | None | Low |
| Bridge execution + response serialization (`env_bridge.py`) | Yes | None | Low |
| OpenAI-compatible model server entry (`vllm_api_server_entry.py`) | Yes | None | Low |
| Runtime-gateway service layer | No | `src/runtime_gateway/*` | High |
| External-runtime request/response adapters | No | Terminus2/OpenHands adapter modules | High |
| Gateway-side context protection for external runtime history | Partial (only internal loop has truncation policy) | inbound guard + budget enforcement | High |
| Gateway observability/replay tooling | No | structured logs + replay bundle | High |

## Recommended Architecture

### Proposed files
- `src/runtime_gateway/app.py`
- `src/runtime_gateway/config.py`
- `src/runtime_gateway/server.py`
- `src/runtime_gateway/canonical_history.py`
- `src/runtime_gateway/canonical_prompt.py`
- `src/runtime_gateway/output_parser.py`
- `src/runtime_gateway/context_guard.py`
- `src/runtime_gateway/contract_validation.py`
- `src/runtime_gateway/adapters/base.py`
- `src/runtime_gateway/adapters/terminus2_json.py`
- `src/runtime_gateway/adapters/terminus2_xml.py`
- `src/runtime_gateway/adapters/openhands.py`
- `src/runtime_gateway/upstream/openai_client.py`
- `src/runtime_gateway/obs/logging.py`
- `src/runtime_gateway/obs/replay.py`
- `scripts/serve_runtime_gateway.py`
- `scripts/smoke_terminus2_gateway.sh`
- `scripts/smoke_openhands_gateway.sh`

### Component boundaries
- Gateway edge:
  - receives OpenAI-compatible requests from external runtimes,
  - validates request shape and limits,
  - routes through adapter mode (`terminus2_json`, `terminus2_xml`, `openhands`).
- Canonical core:
  - normalizes history/messages into canonical internal representation,
  - builds canonical prompt contract,
  - calls upstream model endpoint,
  - parses/validates returned tool calls via existing internal parser/schema.
- Adapter egress:
  - converts canonical output into runtime-specific response payloads.

## Request/Response Contract Mapping

### Internal canonical contract (unchanged)
- Tool calls remain JSON with canonical tool names and args:
  - `{"tool":"bash","args":{...}}`, `{"tool":"search","args":{...}}`, `{"tool":"apply_patch","args":{...}}`, `{"tool":"submit","args":{...}}`.
- Preserve existing invariants (`submit` terminal singleton, schema-required args, max tool-call count policy).

### Terminus2 mapping
- Inbound: OpenAI-like chat payload.
- Outbound: Terminus2 parser-compatible JSON/XML payload.
- Canonical-to-Terminus2 strategy:
  - `bash` -> command/keystroke action entries,
  - `search` -> command/keystroke action entries,
  - `apply_patch` -> patch-apply command sequence,
  - `submit` -> terminal completion mapping.

### OpenHands mapping
- Inbound: OpenAI-like chat payload and runtime tool capabilities.
- Outbound: OpenAI tool-call style responses compatible with OpenHands runtime expectations.
- P0 mapping target:
  - map canonical tool calls to command execution path first,
  - refine to richer native actions in later phase when runtime schema stability is validated.

## Tool-Format Mediation Strategy
- Internal source of truth remains canonical JSON tool schema.
- Support external formats by ingress/egress adapters only:
  - Terminus2 JSON parser format,
  - Terminus2 XML parser format,
  - OpenHands-native OpenAI tool-call response format.
- Never train/deploy against multiple internal schemas; normalize at boundaries.
- Keep conversion deterministic and log adapter decisions for replay.

## Reliability, Safety, and Security Controls

### Reliability controls
- Parse/validation fail-fast with one bounded repair retry policy.
- Hard per-request timeout and response-size limits.
- Deterministic stop-reason propagation (`parse_error`, `validation_error`, `timeout`, `upstream_error`).

### Context growth controls
- Reuse deterministic truncation policy on gateway inbound tool outputs.
- Add total-context budget guard with pre-forward pruning.
- Optional output-cache strategy for repeated large tool results (store full blob outside prompt, inject short reference).

### Security controls
- Request size limits and rate limits.
- Strict JSON schema validation on API boundary.
- Redact secrets in logs/replays.
- Gateway executes no tools itself; runtime executes tools.
- Default deployment recommendation: localhost/private network unless explicit auth/TLS hardening is configured.

## Observability and Diagnostics Plan
- Structured JSON logs:
  - request id, adapter mode, upstream latency, parse status, truncation stats.
- Replay bundle artifacts on failure:
  - sanitized request,
  - canonicalized prompt/history snapshot,
  - raw upstream response,
  - parsed envelope or parse error,
  - emitted runtime payload.
- Metrics:
  - request count by adapter mode,
  - parse failure rate,
  - truncation frequency,
  - upstream timeout/error rates,
  - p50/p95 end-to-end latency.

## Validation Matrix

### Unit
- canonical-history conversion tests.
- contract-validation tests.
- terminus2/openhands adapter mapping tests.
- context-guard determinism tests.

### Integration
- gateway -> upstream vLLM request/response conformance.
- terminus2 adapter output compatibility checks.
- openhands adapter output compatibility checks.

### Runtime smoke
- Terminus2 executes simple command task through gateway.
- OpenHands headless executes simple repository task through gateway.

### Failure injection
- malformed model output.
- oversized tool output history.
- upstream timeout and transport errors.
- adapter mapping missing/unknown tool capability.

## Phased Implementation Plan

### P0: Minimal gateway + dual smokes
- Implement service skeleton, canonical parser integration, basic adapters, and smoke scripts.
- Gate: both Terminus2 and OpenHands complete trivial smoke via gateway.

### P1: Canonical history + stronger mappings
- Add richer history normalization and improved runtime-specific tool mappings.
- Gate: multi-turn tasks pass with lower parse/adapter failure rates.

### P2: Hardening + observability
- Add replay bundles, structured metrics, and strict guardrails.
- Gate: deterministic reproduction of injected failures from replay artifacts.

### P3: Multi-backend + conformance suite
- Add interchangeable upstream backend client and full conformance regression suite.
- Gate: stable cross-runtime compatibility on maintained smoke/e2e matrix.

## Open Questions for Human Review
- For Terminus2 terminal completion, should `submit` map to empty command list or explicit no-op command?
- Which OpenHands runtime/schema version should be the pinned compatibility target?
- Should P0 include only command execution mappings, or also native file-edit/action mappings?
- Should gateway enforce strict auth/TLS in first release or remain local-only initially?
- What is the preferred policy for oversized history: strict truncation vs summarize-and-continue?
- Should context-budget limits be static or model-config-driven from runtime policy files?

## References
- https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/terminus_2.py
- https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/templates/terminus-json-plain.txt
- https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/terminus_2/templates/terminus-xml-plain.txt
- https://raw.githubusercontent.com/OpenHands/OpenHands-CLI/main/README.md
- https://raw.githubusercontent.com/All-Hands-AI/OpenHands/main/openhands/runtime/base.py
- https://docs.openhands.dev/sdk/api-reference/openhands.sdk.workspace
