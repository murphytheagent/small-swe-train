# small-swe-train: Research Mode v1.5 Design Revision Packet (on `design` branch)

Generated: 2026-02-21 21:37 UTC
Thread: 1771579678.414229
Supersedes: prior v1.4 contents at this same path

This revision addresses all requested updates from 2026-02-21 05:32 UTC, including the PR review note at https://github.com/murphytheagent/small-swe-train/pull/2#discussion_r2835802816.

## 1) Trajectory block policy (full history vs summarized history)

### 1.1 Decision
- We do **not** use summary-only history by default.
- We use a **hybrid bounded-context policy** inspired by the Reasoning Cache idea (arXiv:2602.03773): keep raw recent turns and compress older turns into a structured memory object.

### 1.2 Construction at turn `t`
1. `RECENT_RAW_BLOCK`: full raw turns for `t-K_raw ... t-1` (default `K_raw=6`).
2. `COMPRESSED_MEMORY_BLOCK`: deterministic summary over turns `1 ... t-K_raw-1`.
3. `CRITICAL_FACTS_BLOCK`: append-only facts that are never dropped:
- failing tests/signatures,
- file+line localization hints,
- previously attempted edit intents,
- latest known build/test status.

### 1.3 Why this over pure full trajectory
- Full `1..t-1` raw history is better semantically but unstable for token budget and training throughput.
- Summary-only is too lossy for search-heavy SWE traces.
- Hybrid keeps recent exact tool traces while bounding context growth.

## 2) Turn output contract: allow thinking + tool call, with explicit final-answer tool

### 2.1 New output grammar (chat-agent style)
Each assistant turn may emit:
1. Optional thinking block:
- `<think> ... </think>`
2. Exactly one tool call block:
- `<tool_call>{"tool":"...","args":{...}}</tool_call>`

### 2.2 Tool set (v1)
- `bash`
- `search`
- `edit`
- `answer` (terminal action)

Notes:
- `submit` from legacy trajectories is ingested as `answer` via canonicalization.
- SDPO action-token masking remains on **tool-call JSON tokens only**.
- Thinking tokens are logged and validated for delimiter balance, but excluded from SDPO action KL.

## 3) Default step-SDPO teacher prompt construction per turn

At turn `t`, teacher prompt is:
1. `SYSTEM_BLOCK`
- agent role, tool schema, delimiter contract.
2. `TASK_BLOCK`
- issue text, repo context, success condition.
3. `TRAJECTORY_BLOCK`
- `RECENT_RAW_BLOCK` + `COMPRESSED_MEMORY_BLOCK` + `CRITICAL_FACTS_BLOCK`.
4. `CURRENT_ATTEMPT_BLOCK` (conditional)
- include student attempt iff `include_student_attempt_for_teacher=true`.
5. `FEEDBACK_BLOCK`
- canonicalized env feedback packet.
6. `OUTPUT_CONTRACT_BLOCK`
- optional `<think>` plus exactly one `<tool_call>...</tool_call>` JSON object.

Main loss remains:
`L_step_sdpo = E[ sum_t sum_{n in M_t} KL( pi_theta(.|h_t,a_{t,<n}) || stopgrad(q_phi(.|h_t,f_{t+1},a_{t,<n})) ) ]`

## 4) Self-containment checks and env-response canonicalization

### 4.1 Are checks programmatic?
Yes. All three checks are computed programmatically from canonicalized feedback fields.

### 4.2 Three checks (programmatic definitions)
- `has_failing_artifact_identity`:
  - true if at least one concrete artifact id is extracted (`test_id`, `file_path`, `command_id`, `trace signature`).
- `has_actionable_error_text`:
  - true if normalized error text contains non-empty actionable failure content after boilerplate stripping.
- `has_localization_hint`:
  - true if at least one localization anchor is extracted (`file[:line]`, symbol, failing test target, stack frame anchor).

### 4.3 Canonicalization pipeline per turn
1. Ingest raw tool/env payload (`stdout`, `stderr`, exit code, metadata).
2. Normalize text:
- strip ANSI/control chars,
- normalize newlines/whitespace,
- deterministic head+tail truncation after normalization.
3. Extract structured fields:
- artifact identities,
- actionable error text,
- localization hints.
4. Build canonical packet with stable key ordering and `raw_sha256` hash.
5. Derive:
- `is_self_contained = A and B and C`
- `include_student_attempt_for_teacher = not is_self_contained`

Schema now enforces these derivations (addresses PR review comment).

## 5) Tool schema alignment with SWE-bench / SWE-smith trajectories

### 5.1 Do we derive our schema from SWE-smith?
Not directly. We keep our runtime schema as the canonical target and add deterministic adapters from SWE-smith trajectory format.

### 5.2 Mapping policy
From SWE-smith `tool` split samples:
- `bash` -> `bash`
- `str_replace_editor`:
  - `view`/path inspection operations -> `search`
  - edit operations (`create|str_replace|insert|undo_edit`) -> `edit`
- `submit` -> `answer`

Message content mapping:
- assistant freeform/thought -> optional `<think>...</think>` payload
- tool call object -> `<tool_call>{...}</tool_call>` JSON payload

## 6) Directory layout update (requested `src`)

Using `src/` (not `src/small_swe_train`):

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

## 7) Deeper adaptation plan for `lasgroup/SDPO`

### 7.1 Reuse points from SDPO codebase
- `verl/trainer/ppo/ray_trainer.py`:
  - self-distillation teacher-batch construction path (`_maybe_build_self_distillation_batch`), feedback collection path.
- `verl/workers/actor/dp_actor.py`:
  - teacher update (`ema`), logit extraction path, masked distillation integration.
- `verl/trainer/ppo/core_algos.py`:
  - `compute_self_distillation_loss` with top-k distillation support.
- `verl/trainer/config/actor/actor.yaml` + `sdpo.yaml`:
  - distillation knobs (`teacher_regularization`, `distillation_topk`, etc.).

### 7.2 Required adaptations for our SWE agent
1. Replace SDPO reprompt builder with our turn-aware prompt builder (blocks in Section 3).
2. Insert trajectory canonicalizer and self-containment derivation before teacher conditioning.
3. Swap legacy output assumption to chat contract (`<think>` + `<tool_call>`).
4. Add action-token masks that target tool-call JSON only.
5. Add tool-schema adapter for SWE-smith trajectories to our canonical action schema.
6. Replace terminal `submit` semantics with canonical `answer` tool.
7. Plug in Docker SWE env wrapper and benchmark split protocol.

### 7.3 Integration risk notes
- SDPO core can be reused; environment/prompt/schema glue is the primary custom layer.
- Biggest failure mode is schema drift between rollout parser and distillation mask; lock schema tests first.

## 8) Doc hygiene

Per request, stale round-by-round docs are removed from this run folder; current canonical artifacts are:
- this packet (`research_mode_v1_4_design_branch_packet.md`)
- `contracts_metrics_scaffold/` (updated contracts/config/metrics)
- benchmark placeholders under `benchmarks/swebench_lite/`

## 9) Status

- Research mode remains active.
- No training-loop regeneration in this update.
- Ready for your review before coding restart.
