# small-swe-train: Research Mode v1.1 (Round-2 Feedback)

Generated: 2026-02-20 23:36 UTC
Thread: 1771579678.414229

## Inputs re-read for this round
- Slack thread decisions from 2026-02-20 23:09 UTC.
- Attached markdown (`chat.md`, attachment `F0AG2NBNSS2`).
- SDPO paper (arXiv HTML v2): https://arxiv.org/html/2601.20802v2
- SDFT paper (arXiv HTML v1): https://arxiv.org/html/2601.19897v1
- Prior blueprint: `projects/small-swe-train/outputs/1771579678.414229/research_mode_blueprint.md`

## Decision reconciliation (your answers -> spec state)
1. Tool set: **locked** to minimal action set for v1.
2. Output format: **locked** to strict JSON tool-call output.
3. Pre-SDPO stabilization: **partially locked**. RFT-style format stabilization is preferred, but algorithm slot remains pluggable.
4. Teacher context includes student attempt: **conditional policy required** based on feedback self-containment.
5. Feedback length handling: **locked** to deterministic head+tail truncation.
6. Adaptation mode: **locked** to LoRA-only.
7. Stage transition: **locked at policy level** (main SDPO starts only after stable tool-call format); numeric gate still open.
8. Main optimization: **locked** to SDPO-only first.
9. Benchmark: **locked** to SWE-bench-Lite-style episodes first; no implementation yet.
10. Runtime safety: **locked baseline** to read+write+internet in sandbox; Docker isolation preferred.
11. Research-mode completion: **not complete**; another design-feedback round required before coding restart.

## Math and contract updates for v1.1

### 1) Action contract (strict JSON)
- Model output must be one JSON object per step:
- `a_t = {"tool": string, "args": object}`
- Allowed `tool` values (v1): `bash`, `search`, `edit`, `submit`.
- Non-parseable outputs are invalid and counted as format failures.

### 2) Format stage and pluggable pre-main trainer
Let `V(a_t)` be schema-validity indicator.
- Format objective remains:
- `L_format = E[-log pi_theta(a_t^* | h_t)]` on format supervision.
- Pre-main trainer slot:
- `L_pre = w_rft * L_rft + w_sdft * L_sdft`
- For this milestone, default `w_rft > 0`, `w_sdft` configurable (possibly 0).

### 3) Step-SDPO objective (main stage)
- `L_step_sdpo = E[ sum_t sum_{n in M_t} KL( pi_theta(.|h_t,a_{t,<n}) || stopgrad(q_phi(.|h_t,f_{t+1},a_{t,<n})) ) ]`
- `M_t` masks only model action tokens.
- Main stage uses SDPO-only first (no sparse-reward hybrid at start).

### 4) Conditional teacher-context rule for student attempt inclusion
SDPO paper Table 2 includes original response for re-scoring, while Section 4.6 shows adding original attempt in feedback can bias teacher and reduce exploration.
We encode this as a policy toggle:
- `include_attempt_t = 1` only if feedback packet is not self-contained for correction.
- otherwise `include_attempt_t = 0`.

Operationally:
- teacher context always includes environment output/log feedback when available;
- student attempt is appended only when a self-containment check fails.

### 5) Deterministic truncation
For long feedback `f`:
- `truncate(f; H,T)` keeps first `H` tokens and last `T` tokens.
- No summarizer in v1.

## Implementation blueprint deltas
- Placeholder benchmark structure added only (no benchmark logic):
  - `projects/small-swe-train/benchmarks/swebench_lite/episodes/`
  - `projects/small-swe-train/benchmarks/swebench_lite/metadata/`
  - `projects/small-swe-train/benchmarks/swebench_lite/results/`
- Keep research mode active; do not regenerate training code yet.

## Round-2 design questions (implementation-critical)

1. JSON schema finalization
- Do you want one unified schema for all tools (`tool`,`args`) or per-tool strict schemas with separate required keys (e.g., `bash.command`, `edit.path` + patch payload format)?

2. Invalid-output recovery policy
- On malformed JSON, should the runtime do one deterministic repair pass (parser-guided), or mark immediate failure and continue the episode with environment error feedback only?

3. Format-stability gate (numeric)
- Please set numeric thresholds for entering main SDPO, e.g.:
  - `parse_valid_rate >= ?`
  - `allowed_tool_rate >= ?`
  - `required_arg_presence >= ?`
  - measured over last `N` episodes.

4. Self-contained feedback criterion for Q4
- Which rule should decide feedback self-containment?
- Proposed default: feedback is self-contained only if it includes (a) failing artifact identity, (b) actionable error text, and (c) at least one localization hint (file/test/function).

5. Deterministic truncation constants
- Please choose `H` and `T` for head/tail windows (token counts), and whether truncation happens before or after adding system wrappers/tool metadata.

6. Pre-main trainer mix (RFT vs SDFT)
- Should v1 run as:
  - `RFT only -> SDPO`, or
  - `RFT -> short SDFT -> SDPO`, with SDFT easily disableable?

7. Teacher regularization choice in SDPO
- Select default teacher regularization for v1:
  - EMA teacher (`phi <- (1-beta)phi + beta*theta`), or
  - trust-region interpolated teacher.
- If EMA: choose `beta` range for initial runs.

8. Top-K distillation setting
- Choose fixed `K` for top-K KL approximation in v1 (paper mentions `K=100` as practical); do you want `K` fixed or schedule-based?

9. LoRA scope
- Should LoRA target all attention projections only, or attention + MLP projections?
- Also confirm base precision target (e.g., bf16 forward + 4-bit quantized base weights).

10. Sandbox execution policy detail
- For `bash`, should there be an explicit allowlist at v1 (e.g., test, grep, git diff, ls, cat) even inside Docker, or only denylist dangerous commands?

11. Research-mode exit checklist
- Please confirm exact sign-off checklist to permit coding restart:
  - locked schema,
  - locked stage gates,
  - locked teacher-context policy,
  - locked eval protocol,
  - locked safety policy,
  - approved experiment matrix.

## Recommendation before coding restart
- Freeze answers to Questions 1-11 above in one short approval message.
- After that, I will generate only contracts/metrics scaffolding first (no full training loop), then ask for another review checkpoint.
