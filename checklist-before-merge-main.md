# PR18 Merged Checklist: Turn-Level SDPO Alignment, Contract, Verifier, Masks, Gates

**Mode Plumbing and Turn Supervision Mapping**
- [ ] Extend `SelfDistillationConfig` compatibility patch to accept `turn_supervision_mode` and validate `next_turn|current_turn`.
- [ ] Add `actor_rollout_ref.actor.self_distillation.turn_supervision_mode` to `configs/verl/sdpo_swe.yaml` with default `next_turn` and a note that leakage-safe prompting must land before flipping to `current_turn`.
- [ ] Update `build_self_distillation_batch` to accept `turn_supervision_mode`, preserve `next_turn` behavior byte-for-byte, and implement `current_turn` logic: iterate all assistant turns, target `spans[current_turn_index]`, include turn 0 and final turn when spans exist, mark inactive only when the same-turn span is empty, and keep stable list shapes.
- [ ] Pass `turn_supervision_mode` through `ppo_runtime_patch.py` and log the selected mode at batch build time.

**Turn Supervision Invariants**
- [ ] Same-turn exactness: for `current_turn`, each active `turn_response_mask[t]` exactly matches `spans[t]` and no index is active where `_response_mask[index] == 0`.
- [ ] Coverage: turn 0 and final turn are represented whenever they have non-empty spans.
- [ ] Output alignment: `turn_teacher_prompts`, `turn_response_masks`, and `turn_distillation_mask` are aligned and each mask row width equals `len(_response_mask)`.
- [ ] Compatibility: `next_turn` mode matches pre-PR outputs.

**Teacher Contract and Leakage Safety**
- [ ] Extend `build_teacher_output_contract_block` with `supervision_mode` and add current-turn wording while keeping next-turn wording unchanged.
- [ ] Extend `TeacherPromptInputs` and `build_teacher_prompt` to use the mode-aware contract without changing block ordering or delimiters.
- [ ] Thread supervision mode into `_build_turn_prompt` and, in `current_turn`, force `current_attempt_block` to empty for leakage safety while preserving `next_turn` behavior and keeping the default leakage-safe.
- [ ] Surface contract/supervision mode knobs in the training config path and compatibility patch so mode selection is explicit.

**Contract and Leakage Invariants**
- [ ] `current_turn` prompts contain explicit same-turn objective language and no next-turn phrasing.
- [ ] `current_turn` prompts never include the target-turn assistant text.
- [ ] `next_turn` retains legacy contract wording and attempt-block behavior.

**Verifier Feedback Fusion and Gating**
- [ ] Add verifier-feedback extraction helper with stable headers; allow `verification_feedback`, `verification_error`, and pass/resolved flags; never inject `submission_final_response`.
- [ ] Add `verifier_feedback_mode` with `none|final_turn_only|all_turns` and implement the prompt injection policy.
- [ ] Add `legacy_distillation_gating_policy` with `resolved_only|feedback_present|always` and define deterministic activation behavior; `feedback_present` must activate unresolved rows with feedback.
- [ ] Ensure turn-level SWE rows remain active based on valid spans, independent of legacy fallback heuristics.
- [ ] Wire both knobs into the active training config and compatibility patch; if `configs/runtime/training_policy_defaults.v1.json` is updated, ensure runtime consumes the knobs.

**Verifier and Gating Invariants**
- [ ] When verifier metadata exists and mode is not `none`, the prompt includes the verifier header block.
- [ ] `submission_final_response` never appears in any teacher prompt.
- [ ] No policy may silently produce all-false masks when feedback exists.
- [ ] Turn-level valid spans remain active under turn-level training.

**Mask Semantics Hardening**
- [ ] In `ppo_runtime_patch.py`, decouple teacher attention-valid masks from loss-targeting masks for both row-level and turn-level teacher builders.
- [ ] Add debug metrics for teacher-attention valid-token ratio, supervised-token ratio, and invalid-overlap count.
- [ ] In `reward_adapter.py`, detect SWE rows and fail fast when `_response_mask` is missing or empty; keep non-SWE fallback behavior.
- [ ] In `_build_assistant_turn_spans`, add contiguity checks for selected generated positions and use a safe fallback behavior with diagnostics when non-contiguous.

**Mask and Preflight Invariants**
- [ ] Teacher attention-valid mask includes all response tokens needed for conditioning and is not reused from loss masks.
- [ ] No supervised token lies outside the teacher attention-valid region.
- [ ] SWE rows never pass with synthetic all-ones `_response_mask` fallback.
- [ ] Preflight checks are deterministic and yield actionable failure reasons.

**Preflight and Tiny-Run Gates**
- [ ] Add `scripts/check_sdpo_turn_integrity.py` with checks for `turn_response_mask` subset of `_response_mask`, no tool-token supervision leakage, no target-turn prompt leakage in `current_turn`, prompt/mask alignment, and truncation-rate summary; include supervision-mode and truncation-threshold flags.
- [ ] Add a tiny-run gate harness (or extend an existing scaffold) for 10-20 update steps that enforces finite loss/gradients and non-degenerate mask density.

**Go/No-Go Thresholds**
- [ ] Offline integrity gate: 0 subset violations, 0 tool-token supervision leakage events, 0 target-turn prompt leakage events in `current_turn`, 100% prompt/mask cardinality alignment, and truncation rate <= 5% unless explicitly overridden with rationale.
- [ ] Tiny-run gate: 20 update steps with no NaN/Inf loss or gradients, active-turn ratio >= 0.70 on intended turn-level setting, and non-degenerate supervised-token density for >= 95% rows.

**Tests**
- [ ] `tests/test_verl_reprompt_adapter.py`: `next_turn` compatibility, `current_turn` exact masks, inclusion of first and last turns, contiguity edge cases, verifier injection tests, gating-policy tests, leakage tests, and `submission_final_response` exclusion.
- [ ] `tests/test_teacher_messages.py`: `current_turn` has no next-turn language and `next_turn` keeps legacy language.
- [ ] `tests/test_ppo_runtime_patch.py`: attention-valid mask differs from loss mask where expected and loss-targeting masks remain unchanged.
- [ ] `tests/test_verl_reward_adapter.py`: SWE row missing `_response_mask` fails fast and non-SWE fallback remains intact.
- [ ] Script tests for `scripts/check_sdpo_turn_integrity.py`: valid fixture exits 0 and tool-token leakage or prompt leakage exits non-zero.
- [ ] Integration-style tests: reward-row to reprompt-batch path validates mask cardinality and width and no tool-token leakage; synthetic SWE sample validates verifier block placement and activation counts with no forbidden leakage.

**Rollback and Compatibility**
- [ ] Keep `turn_supervision_mode=next_turn` as the default until leakage-safe prompting is merged.
- [ ] Preserve `next_turn` behavior and allow rollback by switching config back to `next_turn`.
- [ ] Roll back verifier fusion by setting `verifier_feedback_mode=none` and conservative gating with `resolved_only`.
- [ ] Allow compatibility fallback for teacher attention masks if rollout is incremental; keep SWE strict invariants enabled.

**Merge Gates**
- [ ] Mode-plumbing tests green and same-turn invariants proven.
- [ ] Explicit `next_turn` compatibility tests pass with no regressions.
- [ ] Contract and leakage tests pass with zero target-turn prompt leakage.
- [ ] Verifier injection and gating-policy tests pass and are deterministic.
- [ ] Mask-role separation tests pass and SWE strict invariants are enforced.
- [ ] Preflight script and tiny-run gate are implemented and runnable.
- [ ] Config default remains `next_turn` in this change.

**Operator Steps Before Enabling `current_turn`**
- [ ] Merge leakage-safe prompting/contract work before flipping the mode.
- [ ] Flip config to `turn_supervision_mode=current_turn` in a dedicated change after the above merge.
- [ ] Run the preflight integrity script and tiny-run gate before expensive SDPO runs.
- [ ] Choose `verifier_feedback_mode` (`all_turns` vs `final_turn_only`) intentionally and document the choice.
