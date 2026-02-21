# small-swe-train: Research Mode v1.6 Design Revision Packet (on `design` branch)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229
Supersedes: prior v1.5 contents at this same path

This revision addresses the 2026-02-21 06:21 UTC feedback and keeps research mode active.

## 1) Chat contract for Qwen3-4B

### 1.1 Decision
- We use ChatML-style turn framing when starting from Qwen3-4B.
- Turn boundaries are explicit:
  - `<|im_start|>{role}`
  - `<|im_end|>`

### 1.2 Intra-turn delimiters (assistant payload)
Inside one assistant turn, payload grammar is:
1. Optional thinking block:
- `<think> ... </think>`
2. One or more tool-call blocks:
- `<tool_call>{"tool":"...","args":{...}}</tool_call>`

Tool/environment feedback is returned as tool-role content, canonically wrapped for parsing as:
- `<tool_response> ... </tool_response>`

### 1.3 Turn contract notes for Qwen3 template
- Qwen3 tokenizer chat template supports function/tool call lists per assistant turn (`tool_calls`) and is compatible with the multi-call contract above.
- We keep explicit XML-like local delimiters (`<think>`, `<tool_call>`) because they simplify deterministic parsing and masking.

## 2) One tool vs multiple tools per turn

### 2.1 Research outcome
- Modern tool-capable chat contracts support multiple tool calls in one assistant turn.
- Many SWE trajectories are still single-action-per-step, but this is a data pattern, not a hard architectural limit.

### 2.2 v1 policy
- Allow `1..M` tool calls per assistant turn (`M=3` default, configurable).
- Calls are executed sequentially in listed order.
- If `submit` appears, it must be the only call in that turn (terminal).
- If a call semantically depends on previous call output, model should split across turns (preferred) to consume fresh feedback.

## 3) Tool set and terminal action

### 3.1 Canonical tools (v1.6)
- `bash`
- `search`
- `edit`
- `submit` (terminal tool)

### 3.2 Legacy alias handling
- Legacy `answer` is ingested as `submit` during canonicalization.
- Existing datasets using `submit` stay unchanged.

## 4) Default step-SDPO teacher prompt construction per turn

At turn `t`, teacher prompt is:
1. `SYSTEM_BLOCK`
- role, tool schema, delimiter contract, execution policy.
2. `TASK_BLOCK`
- issue statement, constraints, success condition.
3. `TRAJECTORY_BLOCK`
- `RECENT_RAW_BLOCK` + `COMPRESSED_MEMORY_BLOCK` + `CRITICAL_FACTS_BLOCK`.
4. `CURRENT_ATTEMPT_BLOCK`
- include student attempt by default (`include_student_attempt_for_teacher=true` in v1.6).
5. `FEEDBACK_BLOCK`
- canonicalized env feedback packet.
6. `OUTPUT_CONTRACT_BLOCK`
- optional `<think>` plus one-or-more `<tool_call>` blocks.

Main objective remains step-SDPO KL on response tokens selected by the stage mask.

## 5) Self-containment checks and canonicalization

### 5.1 Programmatic checks remain
Self-containment is still computed from canonical feedback fields:
- `has_failing_artifact_identity`
- `has_actionable_error_text`
- `has_localization_hint`

### 5.2 Policy change
- `include_student_attempt_for_teacher` is no longer derived from self-containment in v1.6.
- Default runtime value is `true` (always include) for this phase.
- Flag stays in schema for future extension/ablation.

### 5.3 Canonical feedback field requirement
- `canonical_feedback.actionable_error_text` is required as a present field (`string | null`) so derivations are auditable and recomputable.

## 6) Stage-specific token masking policy

### 6.1 RFT stage
- Mask out `<think>...</think>` tokens from supervised loss.
- Train on tool-call/argument tokens and `submit` response tokens.

### 6.2 Step-SDPO stage
- Do **not** exclude thinking tokens.
- Train thinking and tool-call tokens together under the SDPO response-token mask.
- This aligns with the requested update and PR discussion note.

## 7) Tool schema alignment with SWE-bench / SWE-smith

### 7.1 Canonical runtime schema is project-defined
We do not derive runtime schema directly from SWE-bench/SWE-smith.

### 7.2 Adapter mapping (deterministic)
- `bash` -> `bash`
- `str_replace_editor.view` -> `search`
- `str_replace_editor.create|str_replace|insert|undo_edit` -> `edit`
- `submit`/`answer` -> `submit`

### 7.3 Multi-call adaptation
- If source trajectory has one action per turn, adapter emits one `<tool_call>` block.
- Multi-action turns are supported by emitting ordered `<tool_call>` blocks in one assistant turn.

## 8) Directory layout (`src/`)

```text
small-swe-train/
  README.md
  pyproject.toml
  configs/
    model/
    runtime/
    data/
    experiments/
  data/
    manifests/
    prepared/
    cache/
  src/
    schemas/
    prompts/
    data/
    env/
    rollout/
    trainer/
    losses/
    teacher/
    metrics/
    eval/
  scripts/
    prepare_rft_data.py
    run_rft.sh
    run_sdft.sh
    run_sdpo.sh
    eval_swebench_lite.sh
  benchmarks/
    swebench_lite/
      episodes/
      metadata/
      results/
  outputs/
    <run_label>/
```

## 9) Deeper adaptation notes for `lasgroup/SDPO`

### 9.1 Reuse points retained
- `verl/trainer/ppo/ray_trainer.py` for teacher-batch assembly and reprompt path.
- `verl/workers/actor/dp_actor.py` for EMA teacher update + distillation logit path.
- `verl/trainer/ppo/core_algos.py` for `compute_self_distillation_loss` and top-k distillation.

### 9.2 Required custom changes (v1.6)
1. Parser now supports multi-tool-call turns with ordered block extraction.
2. Loss-mask builder supports stage-specific think-token behavior:
- RFT exclude think tokens,
- step-SDPO include think tokens.
3. Canonicalization requires `actionable_error_text` key presence.
4. Teacher prompt builder always includes student attempt by default; flag retained for later ablation.
5. Terminal tool semantics updated from `answer` to `submit`.

## 10) Sources checked for this revision
- Qwen3 tokenizer chat template (`tool_calls` list support):
  - https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json
- SDPO baseline code paths for distillation/teacher integration:
  - https://github.com/lasgroup/SDPO
- Thread-linked review note for schema/audit consistency:
  - https://github.com/murphytheagent/small-swe-train/pull/2#discussion_r2835868321

## 11) Status

- Research mode remains active.
- No training-loop regeneration in this update.
- Ready for your review before coding restart.
