# small-swe-train

Scaffold repository for a chat-style SWE training stack with RFT + step-SDPO stages.

## What is implemented
- Stable protocol types for assistant tool-call envelopes and feedback packets.
- ChatML assistant-turn parser with `<think>` and ordered `<tool_call>` support.
- Canonical feedback normalization and deterministic self-containment diagnostics.
- Deterministic adapter layer from SWE-style tool traces into canonical tools.
- Stage-aware masking policy helpers for `rft` and `step_sdpo`.
- Initial trainer/prompt/eval interface signatures.

## Layout
- `src/schemas/`: frozen JSON schema contracts + typed protocol models.
- `src/rollout/`: ChatML turn parser.
- `src/data/`: feedback canonicalizer + external tool-schema adapters.
- `src/losses/`: stage-aware action masking helpers.
- `src/teacher/`: block-structured teacher prompt builder.
- `src/trainer/`: trainer scaffold signatures.
- `tests/`: protocol stability tests.

## Quick start
```bash
python -m pytest
```

## Notes
- This commit intentionally implements interfaces and tests first, not training-loop execution.
- Design artifacts remain under `outputs/1771579678.414229/` as frozen planning context.
