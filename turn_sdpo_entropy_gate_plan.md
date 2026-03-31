# Teacher-Entropy-Gated `turn_sdpo` Plan

Generated: 2026-03-30 03:00 UTC
Thread: 1774836140.646779
Status: planning, no implementation yet

## 1. Goal

Add a narrow, paper-grounded incorporation path for `Entropy-Aware On-Policy Distillation of Language Models`
(`arXiv:2603.07079`) into this repo's `turn_sdpo` stage.

The immediate target is not a full paper port. It is a clean first ablation:

- keep the current `turn_sdpo` training path intact;
- gate the forward-KL / JSD contribution using the reprompt teacher's token entropy;
- determine quickly whether the entropy signal is real on action-bearing SWE tokens or just prompt scaffolding noise.

## 2. Why This Is the Right Object

### 2.1 `format_rft` and `positive_rft` are not the landing zones

These stages are rollout selection plus SFT on accepted student trajectories.
They do not currently expose a token-level teacher distribution, so EOPD/Veto-style
objective changes do not attach there cleanly.

### 2.2 `turn_sdpo` already has the right loss surface

Current `turn_sdpo` already uses:

- `full_logit_distillation: true`
- fixed JSD-style forward/reverse mixing via `alpha: 0.5`
- `teacher_regularization: ema`
- `distillation_topk: 100`
- `turn_supervision_mode: current_turn`
- `legacy_distillation_gating_policy: feedback_present`

So the clean EOPD-like move here is not "add a second new objective family."
It is "make the existing forward-vs-reverse balance conditional on token entropy."

### 2.3 Paper-faithful gate: teacher entropy, not student entropy

The original EOPD objective gates on the teacher distribution's token entropy:

- `L_EOPD,t = L_OPD,t + I[H_t^te > tau] L_FKL,t`
- `H_t^te = -sum_x pi_te(x|c_t) log pi_te(x|c_t)`

So the first ablation here should use teacher-side entropy.
A student-entropy gate can exist later, but it is a different experiment.

## 3. Current Code-Grounded State

### 3.1 Existing code surfaces

- `configs/verl/sdpo_swe.yaml`
  - current self-distillation config and fixed `alpha`
- `src/verl_integration/ppo_runtime_patch.py`
  - turn-level SDPO teacher forward path, current loss assembly, and the cleanest integration point for an entropy gate
- `src/verl_integration/reprompt_adapter.py`
  - turn-level feedback-gated activation and mask construction
- `tests/test_ppo_runtime_patch.py`
  - likely first place for logic-level regressions

### 3.2 Existing constraints that should stay fixed in the first pass

- keep teacher source as the current EMA reprompt teacher
- keep `current_turn` supervision
- keep `feedback_present` gating
- keep top-`k` distillation instead of paying for full-vocab entropy immediately
- do not alter `format_rft` / `positive_rft`
- do not mix in Veto in the same PR

### 3.3 Main risk

The paper's teacher is a stronger model over the same next-token object.
This repo's teacher is an EMA/self teacher under a corrective reprompt.

That means the highest-entropy tokens may reflect:

- meaningful SWE uncertainty over tool or edit decisions, or
- reprompt scaffolding / formatting / feedback-echo noise.

The first milestone exists to separate those two cases before the repo absorbs a new loss variant.

## 4. Milestone Plan

### Milestone 1: Validate the entropy signal

Objective:

- measure teacher entropy only on already active `turn_sdpo` target tokens;
- confirm whether high-entropy mass is concentrated on action-bearing tokens or on scaffolding.

Concrete work:

- add lightweight logging in `src/verl_integration/ppo_runtime_patch.py` for:
  - active-token teacher entropy summary
  - high-entropy token fraction
  - entropy bucket counts
- if top-`k` + tail mass is already enough to estimate entropy, use that;
  do not add full-vocab teacher passes unless the approximation is clearly inadequate.

Required outputs:

- one short pilot log with entropy summaries
- one written judgment on whether the entropy signal looks usable

Go / no-go rule:

- go if high-entropy tokens frequently correspond to tool-choice / edit / action-bearing positions
- stop if they are mostly scaffolding / formatting / prompt-template artifacts

### Milestone 2: Implement EOPD-lite inside `turn_sdpo`

Objective:

- make the existing forward-KL / JSD mix conditional on teacher entropy without changing the rest of the training path.

Concrete work:

- add a small config block in `configs/verl/sdpo_swe.yaml`, for example:
  - `entropy_gate.enable`
  - `entropy_gate.source=teacher`
  - `entropy_gate.threshold`
  - `entropy_gate.threshold_mode`
  - `entropy_gate.alpha_low`
  - `entropy_gate.alpha_high`
- thread the config into `src/verl_integration/ppo_runtime_patch.py`
- on active distillation tokens:
  - use teacher entropy to decide whether to keep the baseline mix or increase the forward-KL weight
- keep all of the following fixed:
  - teacher source
  - teacher prompt construction
  - supervision scope
  - feedback gating
  - top-`k` distillation

Non-goals in this milestone:

- no student-entropy gate
- no Veto objective
- no teacher-prompt redesign
- no expansion to `format_rft` or `positive_rft`

Acceptance:

- config parses cleanly
- unit tests cover:
  - disabled behavior = exact old path
  - enabled behavior = changed weighting only on high-entropy teacher tokens
  - no unintended activation on non-distilled tokens

### Milestone 3: Pilot and decision

Objective:

- decide whether this direction is alive before opening a broader objective-design lane.

Pilot shape:

- run one small `turn_sdpo` comparison:
  - baseline fixed-`alpha` path
  - teacher-entropy-gated path
- keep the rest of the run contract matched

Minimum metrics to compare:

- resolved / pass metrics already used for `turn_sdpo`
- entropy bucket telemetry
- forward/reverse or low/high-entropy loss summaries
- prompt-level examples of where the gate activates

Success condition:

- the gated variant improves or preserves task metrics while the high-entropy gate fires on action-bearing tokens in a nontrivial fraction of active targets

Stop condition:

- no metric gain and the gate mostly lands on scaffolding / formatting entropy

## 5. Explicit Follow-Ups After This Plan

Only consider these after Milestones 1-3:

- student-entropy gate as a distinct ablation
- smooth weighting instead of a hard threshold
- Veto-style replacement objective inside `turn_sdpo`

Do not bundle any of those into the first implementation PR.

## 6. Files Expected in the First Implementation PR

- `configs/verl/sdpo_swe.yaml`
- `src/verl_integration/ppo_runtime_patch.py`
- `tests/test_ppo_runtime_patch.py`

Potentially, if needed for config plumbing only:

- `src/small_swe_runtime_patches.py`

## 7. Review Standard for This Plan PR

This planning PR is successful if a reviewer can answer all of the following without guessing:

- what exact repo object is changing
- why `turn_sdpo` is the only valid landing zone
- what the first experiment is
- what stays frozen in that first experiment
- what evidence advances the plan
- what evidence kills it quickly
