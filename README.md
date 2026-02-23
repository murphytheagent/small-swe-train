# small-swe-train

Scaffold repository for a chat-style SWE training stack with RFT + step-SDPO stages.

## What is implemented
- Stable protocol types for assistant tool-call envelopes and feedback packets.
- ChatML assistant-turn parser with `<think>` and ordered `<tool_call>` support.
- Canonical feedback normalization and deterministic self-containment diagnostics.
- Deterministic adapter layer from SWE-style tool traces into canonical tools.
- Stage-aware masking policy helpers for `rft` and `step_sdpo`.
- Initial trainer/prompt/eval interface signatures.
- Dedicated `RFTTrainerScaffold` for on-policy RFT rollout/rejection/checkpoint flow, with `SDPOTrainerScaffold` delegating RFT compatibility calls.
- Optional RFT checkpoint scaffold manifests under `checkpoints/global_step_<n>/rft_step_manifest.json`.
- RFT checkpoint writes require explicit `global_step` to avoid accidental step-directory reuse.
- RFT checkpoint argument validation is fail-fast: invalid checkpoint inputs raise before rollout/training side effects.
- On-policy RFT output artifacts include `rollout_rows.jsonl` and `rollout_artifact_summary.json` (task IDs, task-image pairs, and trajectory counts) when `data.on_policy.output_dir` is set.
- Live runtime handoff orchestration is centralized in `src/trainer/rft_runtime.py` and emits `rft_runtime_manifest.json` with selected/rejected counts and rejection-reason tallies.
- Default on-policy turn generation now uses a live OpenAI-compatible vLLM endpoint (`data.on_policy.turn_generator_mode=default`), with runtime settings sourced from centralized policy + `SMALL_SWE_VLLM_*` overrides.
- RFT rejection-policy logic is centralized in `src/trainer/rft_rejection.py` with typed selection outputs/signatures.
- RFT rejection now enforces trajectory-level checks (all tool calls formatted, terminal submit present, terminal submit args valid).
- `src/verl_integration/` keeps thin compatibility wrappers for trainer-owned runtime/handoff modules.

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

Run one deterministic Step-SDPO scaffold step from JSON/JSONL rows:
```bash
python scripts/run_step_sdpo_scaffold.py \
  --input /path/to/rollout_rows.jsonl \
  --output-dir /path/to/sdpo_step_outputs
```

## Notes
- This commit intentionally implements interfaces and tests first, not training-loop execution.
- Design artifacts remain under `outputs/1771579678.414229/` as frozen planning context.
