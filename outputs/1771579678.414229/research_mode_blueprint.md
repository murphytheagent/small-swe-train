# small-swe-train: Research Mode Blueprint

Generated: 2026-02-20 10:59 UTC
Thread: 1771579678.414229

## Inputs reviewed
- Attached markdown (`chat.md`) from Slack attachment `F0AG2NBNSS2`.
- Local mirror of prior planning dialog: `projects/small-swe-train/context-tentative-plan.md`.
- SDPO paper: https://arxiv.org/html/2601.20802v2
- SDFT paper: https://arxiv.org/html/2601.19897v1
- Existing scaffold plan/code before reset: `projects/small-swe-train/codebase/`.

## Comparison: previous scaffold vs attached markdown + papers

1. What already matched
- 4-stage structure in prior scaffold matched the high-level sequence from the markdown: format bootstrapping -> SDFT -> Step-SDPO -> terminal hindsight.
- Prior scaffold already separated trajectory schema, stage runners, and stage-level metrics.

2. What was missing or weak in previous scaffold
- No formal multi-turn objective with explicit token masking over action tokens only.
- No concrete teacher regularization path (EMA or trust-region teacher).
- No top-K distillation approximation strategy.
- No paper-faithful teacher context specification (feedback payload structure and ablations).
- No rigorous teacher-ICL progression dashboard (feedback gain, random-feedback controls, copy-rate, context-usage uplift).
- No explicit evaluation protocol with public-feedback/private-eval split.
- No concrete design decisions on action grammar and environment API contracts.

3. What changes in research mode
- Freeze implementation coding.
- Lock in math-first design and experiment plan.
- Use design questions to resolve ambiguities before new code generation.

## Mathematical blueprint

### Notation
- Issue/task: `x`
- Multi-turn history before step `t`: `h_t = (o_0, a_0, ..., o_t)`
- Student action at step `t`: `a_t ~ pi_theta(. | h_t)`
- Environment feedback after `a_t`: `f_{t+1}` (tool logs, test output, traces)
- Action-token index set for step `t`: `M_t` (mask; only optimize on model action tokens)

### Stage A: Format learning objective
Train for schema-valid tool outputs before SDPO:
- `L_format = E_{(h_t, a_t^*)}[-log pi_theta(a_t^* | h_t)]`
- Validation gate: `valid_action_rate >= tau_format` before next stage.

### Stage B: SDFT warm start (demo-conditioned on-policy distillation)
Use demonstration `d_t` as privileged context for self-teacher:
- Student: `pi_theta(. | h_t)`
- Teacher: `q_phi(. | h_t, d_t)`
- On-policy reverse-KL (sequence form):
- `L_sdft = E_{a_t ~ pi_theta(.|h_t)} [ log pi_theta(a_t|h_t) - log q_phi(a_t|h_t,d_t) ]`
- Practical teacher stability: EMA teacher (`phi <- (1-beta)phi + beta theta`).

### Stage C: Multi-turn Step-SDPO
Feedback-conditioned self-distillation at each action step:
- Student token distribution: `pi_theta(. | h_t, a_{t,<n})`
- Teacher token distribution: `q_phi(. | h_t, f_{t+1}, a_{t,<n})`
- Loss over action tokens only:
- `L_step_sdpo = E_{traj} [ sum_t sum_{n in M_t} KL( pi_theta(.|h_t,a_{t,<n}) || stopgrad(q_phi(.|h_t,f_{t+1},a_{t,<n})) ) ]`

Top-K approximation (compute control):
- Distill teacher top-K logits plus one aggregated tail bucket.

Teacher regularization (choose one):
- EMA teacher (lower variance, simpler implementation), or
- Trust-region mixed teacher:
- `q_reg(y) propto exp((1-alpha) log q_ref(y) + alpha log q_cur(y))`.

### Stage D: Terminal hindsight distillation (delayed credit)
When step-local feedback is insufficient, reuse terminal feedback `F`:
- `L_hindsight = E_{traj} [ sum_{t in K_tail(traj)} KL( pi_theta(.|h_t) || stopgrad(q_phi(.|h_t,F)) ) ]`

### Optional sparse-reward hybrid
For weak initial in-context retrospection, blend SDPO with sparse reward advantage updates:
- `L_total = lambda_f * L_format + lambda_s * L_sdft + lambda_d * L_step_sdpo + lambda_h * L_hindsight + lambda_r * L_sparse_rl`

## Implementation blueprint (pre-coding spec)

1. Data and environment contracts
- Define one canonical trajectory schema with strict serialization for:
  - observations,
  - action JSON/tool-call payloads,
  - execution feedback packets,
  - hidden eval outcome.
- Enforce action grammar parser before any policy update.

2. Training phases and gates
- Phase 0 (`format`): parseability and schema validity.
- Phase 1 (`sdft`): demonstration-conditioned warm start until demo-uplift saturates.
- Phase 2 (`step-sdpo`): online feedback distillation with action-token masking.
- Phase 3 (`hindsight`): periodic delayed-credit pass on selected trajectories.

3. Teacher-ICL progression dashboard (required)
- `G_fb(k)`: repair-success(teacher with true feedback) - repair-success(student).
- `Delta_rand(k)`: success(true feedback) - success(random feedback).
- `copy_rate(k)`: fraction of teacher repairs preserving same failure signature.
- `U_ctx(k)`: context-usage KL between teacher with/without feedback.
- `pass@1` and `success@N` split to separate targeted repair vs diversity-only gains.

4. Eval split protocol
- Feedback-visible public tests during training.
- Hidden private tests for validation selection.
- Store full replay artifacts for error taxonomy analysis.

5. Failure controls
- Teacher collapse: monitor teacher-student KL floor and entropy collapse.
- Feedback leakage: random-feedback control and hidden-test firewall.
- Format regressions: hard block if parser-valid rate drops below threshold.

## Detailed design questions for human feedback

1. Action space scope
- Do you want v1 action space to stay minimal (`bash/search/edit/submit`) or include explicit actions for retrieval, test selection, and patch-apply semantics from day one?

2. Representation format
- Should the policy emit strict JSON tool calls only, or a mixed text+tool format where tool calls are extracted from tagged spans?

3. SDFT demonstration source
- Should demonstrations come from curated human traces, synthetic traces from a stronger model, or a mixed source with confidence filtering?

4. Teacher context policy in SDPO
- Should teacher input include the student's original attempted action/patch, or exclude it to reduce copying bias and enforce corrective exploration?

5. Feedback truncation policy
- For long logs, do you prefer deterministic truncation (head+tail windows) or learned/heuristic summarization before teacher conditioning?

6. Compute budget and adaptation mode
- Are we targeting LoRA/QLoRA adaptation only first, or full-parameter updates for at least one short pilot to test ceiling?

7. Stage gate thresholds
- What minimum thresholds should gate stage transitions (for example format validity, demo uplift, and feedback gain) so we avoid starting SDPO too early?

8. Hybridization strategy
- Do you want SDPO-only initially, or a scheduled SDPO+GRPO/sparse-reward blend from the first SDPO epoch to reduce weak-model instability risk?

9. Evaluation benchmark policy
- Should we anchor first on SWE-bench Lite style patch episodes, or on a custom local benchmark with tighter reproducibility and smaller repos?

10. Safety and sandbox constraints
- What execution sandbox policy should be mandatory for running generated commands during training (filesystem/network/process limits), and do we need an explicit allowlist before any autonomous tool execution?

11. Success criterion for v1 research milestone
- What exact criterion should define that research mode is complete and coding can restart: approved equations, approved stage gates, and approved experiment matrix, or do you want an additional mock dry-run spec?

## Proposed immediate next step after answers
- Freeze this blueprint as v1 spec, then generate a code skeleton only for agreed contracts and metrics (no training logic) to reduce rework risk.
