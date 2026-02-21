# Tool Schema Alignment (v2)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229

## 1) Alignment strategy

We do not directly adopt trajectory tool schemas as runtime truth.

Canonical target for this project is:
- `bash`
- `search`
- `edit`
- `submit`

External trajectory schemas are adapted into this canonical set through deterministic mapping.

## 2) SWE-smith trajectory mapping

Observed tool-call names in SWE-smith `tool` split:
- `bash`
- `str_replace_editor`
- `submit`

Mapping:
- `bash` -> `bash`
- `str_replace_editor` -> `search` or `edit` based on subcommand
- `submit` -> `submit`
- legacy `answer` -> `submit` (for backward compatibility)

`str_replace_editor` subcommand map:
- `view` -> `search` (read/inspect intent)
- `create|str_replace|insert|undo_edit` -> `edit`

## 3) Message-content mapping

For each assistant turn:
- assistant thought/freeform text -> optional `<think>...</think>` segment
- one or more tool call objects -> ordered `<tool_call>{...}</tool_call>` segments

This enables training on chat-style trajectories while preserving explicit tool actions.

## 4) SWE-bench alignment note

SWE-bench issue/instance artifacts provide task/problem/eval context; they do not define a strict runtime tool-call schema. Therefore, SWE-bench is treated as task/eval source, while tool-action schema remains project-defined and adapter-backed.

## 5) Determinism requirements

Adapters must be deterministic and versioned:
- stable mapping table,
- stable argument canonicalization,
- explicit legacy alias handling (`answer` -> `submit`),
- stable ordering for multi-tool-call blocks in a turn.
