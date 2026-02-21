# Self-Containment and Canonicalization (v2)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229

## 1) Canonicalization objective

Convert raw environment/tool responses into a deterministic packet that:
- preserves actionable failure signal,
- is stable under noisy formatting changes,
- supports programmatic diagnostics for teacher-context quality.

## 2) Canonicalization pipeline

Input per turn:
- `tool`, `tool_input`, `tool_output` (`stdout`, `stderr`, `exit_code`, metadata).

Pipeline:
1. Normalize text
- remove ANSI escape/control chars,
- normalize `\r\n` to `\n`,
- collapse repeated blank lines,
- trim trailing whitespace.
2. Deterministic truncation
- apply head+tail policy after normalization (v1 default `H=768`, `T=768`).
3. Structured extraction
- `artifact_identities`: failing tests, file paths, command ids, stack signatures,
- `actionable_error_text`: main error message span (`string | null`, always present as a key),
- `localization_hints`: file:line, symbol names, failing test selectors.
4. Canonical packet build
- include normalization version and `raw_sha256` hash,
- stable key ordering.

## 3) Programmatic self-containment checks

Let:
- `A = has_failing_artifact_identity`
- `B = has_actionable_error_text`
- `C = has_localization_hint`

Rules:
- `A = (len(artifact_identities) > 0)`
- `B = actionable_error_text is non-empty after boilerplate stripping`
- `C = (len(localization_hints) > 0)`

Derived flag:
- `is_self_contained = A and B and C`

Policy note (v1.6):
- `include_student_attempt_for_teacher` remains in schema but is not derived from self-containment.
- default runtime value is `true` (always include student attempt), with field retained for future extension.

## 4) Pseudocode

```python
raw = collect_env_payload(tool_output)
normalized = normalize(raw)
truncated = head_tail(normalized, H=768, T=768)
artifact_ids = extract_artifact_ids(truncated)
error_text = extract_actionable_error(truncated)  # string or None
loc_hints = extract_localization_hints(truncated)

A = len(artifact_ids) > 0
B = bool(error_text and error_text.strip())
C = len(loc_hints) > 0

packet = {
  "canonical_feedback": {
    "normalization_version": "v1",
    "normalized_text": truncated,
    "truncated": was_truncated,
    "raw_sha256": sha256(raw),
    "artifact_identities": artifact_ids,
    "actionable_error_text": error_text,
    "localization_hints": loc_hints,
  },
  "self_containment_checks": {
    "has_failing_artifact_identity": A,
    "has_actionable_error_text": B,
    "has_localization_hint": C,
  },
  "is_self_contained": A and B and C,
  "include_student_attempt_for_teacher": True,
}
```
