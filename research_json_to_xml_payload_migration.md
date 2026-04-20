# Research: Migration from JSON Tool-Call Payloads to XML Field Payloads

Generated: 2026-04-19 20:00 UTC
Status: implemented in this branch; XML is the default assistant payload surface with dual parse for historical JSON compatibility
Related thread: 1776538929.912089

## 1) Executive Summary

- The migration target is narrow: replace the JSON object *inside* `<tool_call>...</tool_call>` with XML fields.
- The internal canonical schema stays JSON-shaped: `ToolCall`, `ActionEnvelope`, `ALLOWED_TOOLS`, `TOOL_SCHEMAS`, `validate_tool_call()`, submit-singleton semantics, and max-tool-call semantics do not change.
- Tool backends and executor semantics do not change. `ToolRequest` / `ToolResponse`, Docker execution, and feedback packets stay as they are.
- Phase 1 should not change `<tool_response>{json}</tool_response>`. That is machine feedback, not the model-authored action surface.
- Historical JSON rollout artifacts should remain valid. The migration path is dual-parse, not artifact rewrite.

Athena review conclusion: backend-unchanged is the right boundary, but “payload/parser/validator/prompt only” is too narrow for this codebase. Assistant actions are also re-rendered in preprocessing, tokenization/canonical-text generation, reward parsing, and the vLLM structured-tool-call fallback, so those surfaces must move with the format.

## 2) Exact Object Being Migrated

Current assistant action shape:

```xml
<tool_call>{"tool":"bash","args":{"command":"pytest -q","cwd":".","timeout_sec":120}}</tool_call>
```

Target assistant action shape:

```xml
<tool_call name="bash">
  <command><![CDATA[pytest -q]]></command>
  <cwd><![CDATA[.]]></cwd>
  <timeout_sec>120</timeout_sec>
</tool_call>
```

This is a surface-format migration, not a schema rewrite. The XML payload parses back into the same canonical object:

```json
{"tool":"bash","args":{"command":"pytest -q","cwd":".","timeout_sec":120}}
```

## 3) What Must Stay Fixed

### Canonical internal contract

- `ToolCall`
- `ActionEnvelope`
- `ALLOWED_TOOLS`
- `TOOL_SCHEMAS`
- `validate_tool_call()`
- `submit` must remain the only tool call in a terminal turn
- max-tool-call enforcement per turn

### Runtime / backend contract

- `ToolRequest`
- `ToolResponse`
- actual tool executors and backend semantics
- feedback-packet structure
- processed-row `tool_calls` arrays written by preprocessing

### Data compatibility

- Existing JSON assistant traces must continue to parse in dual mode.
- No historical cache/rollout rewrite should be required.

## 4) Explicit Format Contract to Add

The migration should add one explicit action-format contract to runtime defaults:

- `action_payload_format = json | xml`
- `action_parse_mode = json_only | dual | xml_only`

Conservative rollout default if a downstream consumer is not ready yet:

- `action_payload_format = json`
- `action_parse_mode = json_only`

Current branch default:

- `action_payload_format = xml`
- `action_parse_mode = dual`

Those knobs govern prompt wording, assistant-action rendering, and parse behavior. They do not affect tool backend semantics.

## 5) Recommended XML Contract

Use one intentionally boring contract:

- Outer block: `<tool_call name="bash"> ... </tool_call>`
- Direct child elements for args
- No `<args>` wrapper
- CDATA for string-valued fields so multiline `command`, `patch`, and `final_response` round-trip cleanly
- Repeated children only for real list fields
- No namespaces, DTDs, processing instructions, or external entities
- Reject duplicate scalar fields
- Reject mixed JSON/XML assistant payloads within one assistant turn

### Representative forms

`bash`

```xml
<tool_call name="bash">
  <command><![CDATA[pytest -q]]></command>
  <cwd><![CDATA[.]]></cwd>
  <timeout_sec>120</timeout_sec>
</tool_call>
```

`read`

```xml
<tool_call name="read">
  <path><![CDATA[src/app.py]]></path>
  <start_line>10</start_line>
  <end_line>40</end_line>
</tool_call>
```

`apply_patch`

```xml
<tool_call name="apply_patch">
  <path><![CDATA[src/app.py]]></path>
  <patch><![CDATA[*** Begin Patch
...
*** End Patch]]></patch>
</tool_call>
```

`submit`

```xml
<tool_call name="submit">
  <final_response><![CDATA[done]]></final_response>
  <changed_paths>
    <path><![CDATA[src/app.py]]></path>
    <path><![CDATA[tests/test_app.py]]></path>
  </changed_paths>
</tool_call>
```

## 6) Real Code Surface in `small-swe-train`

### Parser / validation entrypoints

- `src/rollout/turn_parser.py`
- shared assistant-action parse entrypoints used by:
  - `src/verl_integration/env_bridge.py`
  - `src/verl_integration/data_preprocessor.py`
  - `src/verl_integration/reward_function.py`

### Assistant-action rendering surfaces

- `src/data/tokenization.py`
- `src/verl_integration/data_preprocessor.py`
- `src/rollout/vllm_turn_generator.py`
- `src/rollout/onpolicy_collector.py`
- `src/trainer/rft_runtime.py`

### Prompt and contract surfaces

- `src/prompts/runtime_messages.py`
- `src/prompts/teacher_messages.py`
- `configs/runtime/training_policy_defaults.v1.json`

### Golden / integration test surface

- parser tests
- env bridge tests
- data preprocessor tests
- reward-function tests
- tokenization tests
- rollout/runtime tests that hard-code `<tool_call>{json}</tool_call>`

Important config caveat: the current config model assumes a literal `tool_call_start` string such as `<tool_call>`. XML payload mode will need an attribute-bearing start tag such as `<tool_call name="bash">`, so delimiter handling cannot remain a simple fixed-start-string assumption forever.

## 7) Benefits

- Less brace-heavy and quote-heavy payload text for the model to emit.
- Cleaner field boundaries for multiline `command`, `patch`, and `final_response`.
- Better compatibility with XML-oriented external adapters if that boundary still matters.

## 8) Main Risks

- literal closing-tag strings inside `command`, `patch`, or `final_response`
- `]]>` appearing inside CDATA payloads
- whitespace drift from XML pretty-printing or parser normalization
- duplicate scalar fields such as two `<command>` tags
- empty-tag semantics such as `<cwd/>` versus omitted `<cwd>`
- numeric coercion for `timeout_sec`, `start_line`, `end_line`, and `top_k`
- fake tool tags embedded inside tool outputs or copied logs
- broad blast radius from test fixtures that currently assume JSON-in-XML

## 9) Rollout Plan

### P0: Centralize parse/render with current JSON behavior

- Add one shared assistant-action parse entrypoint.
- Add one shared assistant-action render entrypoint.
- Route current JSON-emitting helper sites through those functions.
- Keep emitted JSON bytes and parser behavior unchanged.

### P1: Add explicit XML schema + dual parser / renderer

- Implement XML tool-call parser.
- Implement XML rendering path.
- Normalize parsed XML back into the existing `ToolCall` / `ActionEnvelope`.
- Add mixed-format rejection.
- Add round-trip tests for JSON and XML.

### P2: Propagate through runtime consumers

- Switch bridge, preprocessing, reward parsing, tokenization, and vLLM fallback onto the shared parse/render layer with format flags.
- Update test goldens that need format-aware expectations.

### P3: Prompt XML while keeping dual parse

- Update `runtime_messages.py` and `teacher_messages.py`.
- Keep runtime parser in dual mode so JSON traces and partial rollouts still work.

### P4: Switch defaults only after A/B is clean

Required checks before making XML the default:

- parse-validity rate
- terminal-submit rate
- no executor-regression in bridge/runtime behavior
- no evidence of degraded rollout quality relative to current JSON prompting

## 10) What This Branch Implements

This branch now implements P0 through P3:

- a dedicated XML migration note
- a shared assistant-action parse/render module
- XML parse/render support with shared canonicalization back into `ToolCall` / `ActionEnvelope`
- dual parse so historical JSON assistant traces still work
- XML-aware prompt contracts for runtime and teacher prompting
- XML-aware assistant-action rendering in tokenization, preprocessing, collector/runtime helpers, and vLLM structured-tool fallback
- centralized defaults flipped to `action_payload_format = xml` and `action_parse_mode = dual`

What is still intentionally unchanged:

- internal tool/backend schema
- validators and executor semantics
- `<tool_response>{json}</tool_response>`
- historical JSON artifacts on disk

The only remaining item from the original rollout plan is the empirical A/B gate in P4.
