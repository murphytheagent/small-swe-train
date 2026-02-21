# Metric Definitions (v3)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229

## Format and action-contract metrics

- `parse_valid_rate`:
  - Fraction of assistant turns where parser extracts a valid optional `<think>` segment and valid `<tool_call>` blocks.

- `tool_call_block_presence_rate`:
  - Fraction of assistant turns containing at least one `<tool_call>...</tool_call>` block.

- `tool_call_count_valid_rate`:
  - Fraction of assistant turns where tool-call count is within configured bounds (`1..M`, default `M=3`).

- `submit_singleton_rule_rate`:
  - Fraction of turns satisfying: if `submit` appears, it is the only tool call in that turn.

- `thinking_delimiter_balance_rate`:
  - Fraction of turns where thinking delimiters are either absent or correctly balanced (`<think>...</think>`).

- `allowed_tool_rate`:
  - Fraction of parsed tool calls where `tool` is one of `bash|search|edit|submit`.

- `required_arg_presence`:
  - Fraction of parsed tool calls whose required `args` keys pass tool-specific schema checks.

## Teacher-ICL progression metrics

- `G_fb(k)`:
  - `repair_success(teacher_with_true_feedback, k) - repair_success(student, k)`.

- `Delta_rand(k)`:
  - `success(true_feedback, k) - success(random_feedback, k)`.
  - Positive gap indicates non-spurious feedback usage.

- `copy_rate(k)`:
  - Fraction of teacher repairs that preserve the same failure signature as the student attempt.

- `U_ctx(k)`:
  - KL divergence between teacher token distributions with true feedback versus without feedback.

## Training-stage health metrics

- `teacher_student_kl`:
  - Mean KL on masked response tokens between student and stop-gradient teacher.

- `teacher_entropy`:
  - Mean token entropy of teacher over masked response-token positions.

- `hindsight_gain`:
  - Success uplift from hindsight pass on tail-selected trajectories.

## Self-containment diagnostic metric

Let:
- `A = has_failing_artifact_identity`
- `B = has_actionable_error_text`
- `C = has_localization_hint`

Then:
- `is_self_contained = A and B and C`

In v1.6, this is diagnostic and does not force `include_student_attempt_for_teacher`.

## Required reporting split

All key metrics should be reported in both contexts:
- Public-feedback train/eval split.
- Hidden private-eval split.
