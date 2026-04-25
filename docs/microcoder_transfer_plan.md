# MicroCoder transfer plan for `small-swe-train`

Last updated: 2026-04-03 08:54 UTC.
Owner thread: Slack `1774836293.346719`.
Companion plan: `swe_smith_language_expansion_plan.md` covers non-Python SWE-smith enablement. This file is the narrower policy plan for verifier signal, data quality, and long-rollout handling.

## Goal

Adopt the parts of MicroCoder that actually match the current `small-swe-train` contract, and reject the parts that do not.

Focus areas:
- rollout verifier / outcome quality
- training data and environment quality
- handling rollouts that exceed the usable length budget

## Decision summary

- Keep the current stage split: `format_rft -> positive_rft`.
- Keep one final verifier-backed `resolved` bit for selection and supervision.
- Split infra-invalid runs away from ordinary unresolved runs. At minimum: `patch_apply_failure`, `env_init_failure`, and `verifier_crash` must stop looking like task difficulty.
- Make dataset patch meaning explicit in config. The proposed field is `patch_is_bug_introducing: true|false`.
- Add cheap deterministic row filters before rollout: duplicate-target rejection, `FAIL_TO_PASS` / `PASS_TO_PASS` overlap rejection, and dedupe on normalized problem statement + normalized target set.
- Add task-level selector smoke checks on top of the current image-level preflight.
- Do empirical difficulty banding only after that cleanup, and only on the final eligible pool that can actually reach training.
- Treat transcript-over-budget and generation-cap as different failure classes.
- Do not do transcript compaction for selected over-budget rows.
- Do not do generic masking for transcript overflow.
- Only consider MicroCoder-style masking for rows that provably hit the generation cap during rollout.

## Current code-grounded state

- `configs/data/on_policy_swe_smith.yaml` still points the default on-policy path at `SWE-bench/SWE-smith-py`.
- `src/rollout/onpolicy_collector.py::_task_patch(...)` and `_initialize_task_environment(...)` auto-apply any non-empty raw `patch` before the student acts.
- For today's default `SWE-smith-py` path, that auto-apply behavior is correct because the dataset patch is the bug-introducing setup diff.
- `src/env/task_dataset.py::_coerce_task_row(...)` currently requires non-empty `image_name`, `problem_statement`, `FAIL_TO_PASS`, and `PASS_TO_PASS`, but it does not reject duplicate targets or `FAIL_TO_PASS` / `PASS_TO_PASS` overlap.
- `src/env/preflight_onpolicy_dataset.py` currently probes only image-level prerequisites such as repo presence and `pytest` importability. It does not validate whether a task's exact selectors are valid.
- `src/trainer/rft_runtime_loop.py::filter_selected_rows_by_token_length(...)` rebuilds selected multiturn transcripts and drops rows above trainer `data.max_length` before parquet write.
- `src/trainer/rft_handoff.py::build_verl_sft_batch(...)` later truncates already-selected tensors to `rft_handoff.max_sequence_length`.
- `src/verl_integration/swe_bridge_agent_loop.py::append_response_tokens(...)` clips generation to the remaining `response_length` budget, but the runtime does not currently persist a `hit_generation_cap`-style flag.

## What transfers cleanly from MicroCoder

- Cleaner evaluator signal matters more than clever reward shaping when the current bottleneck is label contamination.
- Bad data should be filtered or tagged before it can masquerade as hard data.
- Difficulty should be measured on the cleaned, trainable pool, not on a mixture of valid rows and infrastructure failures.
- True generation-cap events deserve explicit handling rather than being lumped together with every other "too long" case.

## What should not be imported

- No transcript compaction to salvage selected rows that exceed the input budget.
- No generic masking for transcript-packaging overflow.
- No fuzzy output-equivalence evaluator for SWE patch correctness.
- No hard filter on raw number of tests by itself.
- No short-context-first curriculum as a substitute for fixing the actual long-context failure modes.

## Shared contract changes this plan assumes

These are the minimum additions needed to make the rest of the plan well-defined.

- Dataset config:
  - `patch_is_bug_introducing: true|false`
  - `verifier_kind: pytest|go_test|node_test|command` or equivalent backend-oriented enum
- Row-level validity metadata:
  - `resolved: bool`
  - `infra_invalid: bool`
  - `invalid_reason: "" | patch_apply_failure | env_init_failure | verifier_crash | selector_invalid`
- Length metadata:
  - `selected_over_budget: bool`
  - `selected_token_count: int`
  - `hit_generation_cap: bool`

The exact field names can change during implementation, but the semantic split above should not.

## Milestones

### Milestone 0: make dataset meaning and invalid-run taxonomy explicit

Delivery boundary:
- The default `SWE-smith-py` path keeps working as it does today, but patch meaning, row-static validation, and invalid-run taxonomy become explicit instead of implicit.

Implementation scope:
- Extend `configs/data/*.yaml` and the corresponding config loader in `src/config.py` with:
  - `patch_is_bug_introducing`
  - `verifier_kind`
- Set `configs/data/on_policy_swe_smith.yaml` to:
  - `patch_is_bug_introducing: true`
  - `verifier_kind: pytest`
- Update `src/env/task_dataset.py::_coerce_task_row(...)` to:
  - reject duplicate targets within `FAIL_TO_PASS`
  - reject duplicate targets within `PASS_TO_PASS`
  - reject any `FAIL_TO_PASS` / `PASS_TO_PASS` overlap
  - normalize a dedupe key from problem statement + sorted `FAIL_TO_PASS` + sorted `PASS_TO_PASS`
- Add deterministic dedupe accounting before rollout:
  - keep the first surviving row in deterministic order
  - emit filtered counts and raw task IDs for audit
- Update `src/rollout/onpolicy_collector.py` so patch application is keyed off `patch_is_bug_introducing` instead of the raw presence of `patch`.
- Split infra-invalid cases away from ordinary unresolved outcomes:
  - patch apply failed
  - environment initialization failed
  - verifier execution crashed or became unusable
- Keep one final verifier-backed `resolved` bit as the only positive supervision gate.

Why this comes first:
- The rest of the plan depends on knowing whether a row is valid data at all.
- Difficulty banding and long-rollout policy are both meaningless if the pool is still polluted by wrong patch semantics and selector mistakes.

Expected touchpoints:
- `configs/data/on_policy_swe_smith.yaml`
- `src/config.py`
- `src/env/task_dataset.py`
- `src/rollout/onpolicy_collector.py`
- `tests/test_task_dataset.py`
- `tests/test_onpolicy_collector.py`
- `tests/test_submission_verifier.py`

Acceptance gates:
- Default `SWE-smith-py` behavior is unchanged under current configs.
- A fixture with `patch_is_bug_introducing: false` proves the runtime does not auto-apply that dataset patch.
- Duplicate-target and FAIL/PASS-overlap rows are rejected deterministically with stable accounting.
- Dedupe emits both filtered counts and the affected raw task IDs.
- `positive_rft` eligibility is a strict subset of `resolved=true` and `infra_invalid=false`.
- Infra-invalid rows are excluded from later difficulty statistics.

Out of scope:
- No change to the default training mix.
- No difficulty banding yet.
- No masking yet.

### Milestone 1: add task-level selector smoke checks

Delivery boundary:
- Rows whose exact verifier selectors are broken stop being treated as hard tasks.

Implementation scope:
- Extend `src/env/preflight_onpolicy_dataset.py` so preflight has two layers:
  - `probe_image`: the current image-level environment readiness check
  - `probe_task`: a selector-validity smoke check for the task's exact targets
- For the current Python path, `probe_task` should use a cheap selector check such as `pytest --collect-only` on the exact selectors.
- Store task-level failure reasons separately from image-level failure reasons.
- Feed selector-invalid rows into the same invalid-data accounting family rather than leaving them in the rollout pool.

Why this is the next step:
- Image-level readiness only proves that the box is alive.
- It does not prove that this task's targets are real, stable, or collectable.
- Broken selectors are one of the cheapest ways bad data gets misread as difficulty.

Expected touchpoints:
- `src/env/preflight_onpolicy_dataset.py`
- `src/env/task_dataset.py`
- `tests/test_preflight_onpolicy_dataset.py`
- `tests/test_task_dataset.py`

Acceptance gates:
- Known-good selectors pass.
- Known-bad selectors fail deterministically with task-level reason codes.
- Image-level and task-level failures are reported separately.
- Selector-invalid rows never enter the pool used for difficulty accounting.

Out of scope:
- No multi-language enablement in this milestone.
- No change to training-stage objectives.

### Milestone 2: separate transcript-over-budget from generation-cap

Delivery boundary:
- The runtime has one honest input-budget story, and generation-cap events are logged as a separate class.

Implementation scope:
- Resolve a single effective keep budget for selected rows:
  - `effective_keep_budget = min(data.max_length, rft_handoff.max_sequence_length)`
- Use that budget at the selected-row filter in `src/trainer/rft_runtime_loop.py::filter_selected_rows_by_token_length(...)`.
- Keep dropping selected rows that exceed the budget before parquet write.
- Tighten `src/trainer/rft_handoff.py::build_verl_sft_batch(...)` so rows that survived selection are not silently truncated later.
  - Prefer an invariant or explicit failure over quiet downstream slicing of rows that were supposed to fit.
- Add length telemetry:
  - selected token count
  - selected-over-budget count
  - kept-row downstream truncation count
- Add explicit generation-cap telemetry in `src/verl_integration/swe_bridge_agent_loop.py`:
  - set `hit_generation_cap=true` only when `append_response_tokens(...)` actually clips a generated span because `response_length` is exhausted

Why this comes before difficulty banding:
- Difficulty should be measured on the rows that can actually survive into training.
- If banding is computed before the real keep budget is enforced, the reported difficulty distribution will not match the trained distribution.

Expected touchpoints:
- `src/trainer/rft_runtime_loop.py`
- `src/trainer/rft_handoff.py`
- `src/verl_integration/swe_bridge_agent_loop.py`
- `tests/test_rft_runtime_loop.py`
- `tests/test_rft_handoff.py`
- `tests/test_swe_bridge_agent_loop.py`

Acceptance gates:
- Selected rows that exceed the effective keep budget are dropped before parquet write.
- Rows that survive that filter never get silently truncated later in `rft_handoff.py`.
- `hit_generation_cap` is set only for true generation-budget exhaustion, not for transcript-overflow cases.
- Transcript-over-budget rows never trigger masking logic.

Out of scope:
- No compaction.
- No generic masking.

### Milestone 3: do difficulty banding on the cleaned eligible pool

Delivery boundary:
- Difficulty labels are computed on the same final pool that can actually hit training, and they are comparable within stable verifier cohorts.

Implementation scope:
- Build calibration slices only after Milestones 0-2 are landed.
- Exclude from banding:
  - infra-invalid rows
  - selector-invalid rows
  - duplicate rows removed by dedupe
  - rows dropped by the effective keep budget
- Compute bands within stable verifier cohorts keyed by `verifier_kind`, not across a mixed verifier population.
- Start with empirical resolve-rate bands rather than an LLM difficulty scorer:
  - near-impossible: `0/4`
  - learnable: `1/4` or `2/4`
  - easy: `3/4` or `4/4`
- Keep raw IDs and band assignments so later audits can prove which rows moved where.

Why this is later:
- Difficulty is an empirical property of valid rows under a real verifier, not a property of broken setup or broken selectors.
- Banding on the wrong population will produce a clean-looking but operationally false training mix.

Expected touchpoints:
- `src/env/task_dataset.py`
- `src/env/preflight_onpolicy_dataset.py`
- runtime reporting / manifests for difficulty stats
- tests covering deterministic cohort assignment and exclusion rules

Acceptance gates:
- Re-running banding on the same frozen cleaned pool produces the same assignments.
- Band counts are reported per verifier cohort.
- The banded pool matches the final training-eligible pool, not a larger pre-filter snapshot.

Out of scope:
- No LLM difficulty scorer in the first pass.
- No cross-language joint banding in a single pool.

### Milestone 4: optional masking experiment for true generation-cap events only

Delivery boundary:
- If generation-cap events are common enough to matter, run one clean ablation on those rows only.

Implementation scope:
- Use the explicit `hit_generation_cap` flag from Milestone 2.
- Compare two narrow policies on only those rows:
  - drop the chopped assistant span
  - zero or mask loss on the chopped assistant span
- Keep transcript-overflow rows completely out of this experiment.

Why this is last:
- MicroCoder-style masking is only justified once the runtime proves the problem is a true generation cap and not a packaging/input-budget issue.

Expected touchpoints:
- `src/verl_integration/swe_bridge_agent_loop.py`
- loss / mask plumbing used by the affected training stage
- tests that prove the mask can only activate on `hit_generation_cap=true`

Acceptance gates:
- The masking path cannot activate on generic long-context rows.
- The ablation reads the explicit generation-cap telemetry rather than inferring from length after the fact.

Out of scope:
- No masking for transcript-over-budget rows.
- No compaction fallback.

## Immediate delivery boundary

The first implementation wave should land:
- Milestone 0
- Milestone 1
- Milestone 2

Milestone 3 should start only after those three are in place, because the difficulty pool has to be defined by the cleaned, final-eligible data.

Milestone 4 is optional and should happen only if the new generation-cap telemetry shows that this is a real problem in practice.

## Failure modes to call out in review

- Over-broad `verifier_crash` labeling can erase real negative examples. Ordinary failed tests must stay ordinary failed tests.
- Over-aggressive dedupe can hide real environment differences if collisions appear across distinct images. Keep raw IDs for audit and verify the actual collision set before making the dedupe irreversible.
- Difficulty banding on a pre-filter or pre-length-gate population will produce a misleading training mix. The plan should be judged against the final eligible pool only.

## Human-facing summary

The clean transfer from MicroCoder is not "salvage every long rollout" and not "copy their evaluator."

It is:
- clean up invalid data before rollout,
- keep one hard correctness bit,
- stop broken selectors and infrastructure failures from masquerading as hard tasks,
- band difficulty only after cleanup,
- and treat true generation-cap events as their own problem instead of mixing them with transcript overflow.
