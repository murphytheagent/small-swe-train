# Metric Definitions (v1)

Generated: 2026-02-21 03:43 UTC
Thread: 1771579678.414229

## Format and action-contract metrics

- `parse_valid_rate`:
  - Fraction of steps in rolling window `N=200` whose model output parses as exactly one JSON object.

- `single_object_rate`:
  - Fraction of steps with exactly one action object (no concatenated or multi-action outputs).

- `allowed_tool_rate`:
  - Fraction of parsed actions where `tool` is one of `bash|search|edit|submit`.

- `required_arg_presence`:
  - Fraction of parsed actions whose required `args` keys pass tool-specific schema checks.

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
  - Mean KL on action tokens between student and stop-gradient teacher.

- `teacher_entropy`:
  - Mean token entropy of teacher over masked action-token positions.

- `hindsight_gain`:
  - Success uplift from hindsight pass on tail-selected trajectories.

## Self-containment decision metric

Let:
- `A = has_failing_artifact_identity`
- `B = has_actionable_error_text`
- `C = has_localization_hint`

Then:
- `is_self_contained = A and B and C`
- `include_student_attempt_for_teacher = not is_self_contained`

## Required reporting split

All key metrics should be reported in both contexts:
- Public-feedback train/eval split.
- Hidden private-eval split.
