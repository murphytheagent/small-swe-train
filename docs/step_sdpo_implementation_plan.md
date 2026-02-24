# Step-SDPO Mainline Implementation Plan

- Updated: 2026-02-24 08:56 UTC
- Thread: `1771841520.464849`
- Base code: `main` @ `9bd740b`
- Scope of this PR: planning artifact only (no runtime code changes yet)

## 1. Locked Decisions

1. Train assistant-generated tokens using rollout `response_mask` only.
2. Do not add custom fine-grained step-SDPO token masking in this slice.
3. Keep upstream SDPO divergence/loss internals unchanged.
4. Integrate step-SDPO on top of `main` with explicit multi-turn + bridge loop wiring.

## 2. Deliverables and Acceptance Criteria

| ID | Deliverable | Files | Acceptance Criteria |
| --- | --- | --- | --- |
| D0 | Plan committed in PR | `docs/step_sdpo_implementation_plan.md` | Plan is reviewable in PR and defines config authority + final success gate. |
| D1 | Multi-turn/agent-loop defaults made explicit in SDPO config | `configs/verl/sdpo_swe.yaml` | Config explicitly sets: `actor_rollout_ref.rollout.multi_turn.enable`, `max_assistant_turns`, `max_user_turns`, `actor_rollout_ref.rollout.agent.default_agent_loop`, and `agent_loop_config_path` (for `swe_bridge_agent`). |
| D2 | SWE bridge agent loop integration | `configs/verl/agent_loops/swe_bridge_agent.yaml`, `src/verl_integration/swe_bridge_agent_loop.py` | Rollout executes per-turn bridge call between generation and next-turn prompt build; terminal `submit` path is deterministic even when no non-submit tool executes. |
| D3 | Trainer hook for teacher reprompt batch | `src/verl_integration/<sdpo_entry_patch>.py` (exact filename to be finalized) | `_maybe_build_self_distillation_batch` uses local SWE reprompt adapter and emits expected teacher tensors without modifying upstream loss math. |
| D4 | DataProto reward adapter in SDPO runtime path | `src/verl_integration/reward_function.py` (+ adapter glue if split) | Reward path returns expected reward tensor and feedback extras consumed by reprompt flow; submit-only terminal rows are handled deterministically (`tool_output={}`). |
| D5 | SDPO runner hygiene and authoritative entrypoint path | `scripts/run_sdpo.sh` | Script exports project `PYTHONPATH`, defaults to `python3` with fallback, and launches project-local SDPO entrypoint module so local adapters reliably load. |
| D6 | E2E monitored run from RFT checkpoint | run artifact dir under `outputs/integration/<run_label>/` | One end-to-end step-SDPO run completes from an RFT checkpoint with verified config flow, logs, and checkpointed outputs. |

## 3. Authoritative Config Flow (Required)

This is the required precedence chain for every step-SDPO run:

1. `configs/verl/sdpo_swe.yaml` is the baseline authority for SDPO defaults.
2. `scripts/run_sdpo.sh` is the only supported launcher and must set interpreter/import hygiene.
3. CLI Hydra overrides are allowed only for run-scoped values (data files, runtime limits, run directories, checkpoint path).
4. The starting model for step-SDPO must come from an RFT checkpoint path, not a base model ID.

### 3.1 Required source of checkpoint truth

Primary source:
- `outputs/rft_runtime/rft_runtime_loop_manifest.json` field `.final_model_path`.

Fallback source:
- Explicit human-provided HF checkpoint directory from a completed RFT run.

Checkpoint must resolve to a directory containing model export files (for example `config.json`, weights, tokenizer assets if applicable).

### 3.2 Required runtime overrides for e2e step-SDPO run

These overrides must be present in the launch command (either via YAML default or CLI override):

- `actor_rollout_ref.model.path=<RFT_CHECKPOINT_PATH>`
- `actor_rollout_ref.rollout.multi_turn.enable=true`
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=<N>`
- `actor_rollout_ref.rollout.multi_turn.max_user_turns=<N>`
- `actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent`
- `actor_rollout_ref.rollout.agent.agent_loop_config_path=<repo>/configs/verl/agent_loops/swe_bridge_agent.yaml`
- `data.train_files=<dataset/parquet/json path(s)>`
- `data.val_files=<dataset/parquet/json path(s)>`
- `trainer.total_training_steps=1` (first e2e gate run)
- `trainer.default_local_dir=<outputs/integration/<run_label>>`

### 3.3 Config verification commands

```bash
# 1) Resolve RFT checkpoint path from runtime manifest
RFT_MANIFEST="/path/to/rft_runtime_loop_manifest.json"
RFT_CKPT="$(jq -r '.final_model_path' "${RFT_MANIFEST}")"
test -d "${RFT_CKPT}"

# 2) Verify launcher command assembly before real run
bash scripts/run_sdpo.sh --dry-run \
  actor_rollout_ref.model.path="${RFT_CKPT}" \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$(pwd)/configs/verl/agent_loops/swe_bridge_agent.yaml" \
  trainer.total_training_steps=1
```

## 4. Final Success Gate (Authoritative)

Implementation is considered successful only when all conditions below are true in the same monitored e2e run:

1. **RFT checkpoint load confirmed**
- Startup logs/metadata show `actor_rollout_ref.model.path` equals the resolved RFT checkpoint path.

2. **Bridge multi-turn path confirmed**
- Runtime path uses `swe_bridge_agent` and executes bridge turn handling (`generate -> bridge -> append tool response -> next turn/submit`).

3. **SDPO self-distillation path confirmed**
- Training step consumes teacher tensors from reprompt hook and executes SDPO loss without fallback errors.

4. **Rollout-mask-only policy confirmed**
- Training uses rollout `response_mask`; no custom fine-mask injector is active in SDPO runtime.

5. **Step completion confirmed**
- At least one SDPO global step finishes and outputs run artifacts/checkpoint under the specified `trainer.default_local_dir`.

6. **Monitoring evidence captured**
- Run directory contains launch command, resolved config snapshot, and log summary proving items 1-5.

## 5. Monitoring Protocol for D6

For the acceptance run, record these artifacts under `outputs/integration/<run_label>/`:

- `launch_command.txt` (exact executed command)
- `resolved_runtime_values.json` (checkpoint path, loop choice, max turns, train/val files)
- `train.log` (stdout/stderr)
- `acceptance_summary.md` (pass/fail checklist for Final Success Gate)

Suggested first-pass run shape (single-step gate):

```bash
# Heavy run must be under Slurm with explicit memory.
srun --mem=384G --gres=gpu:8 --cpus-per-task=32 --time=04:00:00 bash -lc '
  set -euo pipefail
  cd /home/murphy/projects/small-swe-train-runtime
  export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
  export NPROC_PER_NODE=8

  RFT_MANIFEST=/path/to/rft_runtime_loop_manifest.json
  RFT_CKPT="$(jq -r '.final_model_path' "${RFT_MANIFEST}")"
  RUN_DIR="$(pwd)/outputs/integration/step_sdpo_e2e_from_rft_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "${RUN_DIR}"

  bash scripts/run_sdpo.sh \
    actor_rollout_ref.model.path="${RFT_CKPT}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_bridge_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$(pwd)/configs/verl/agent_loops/swe_bridge_agent.yaml" \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="${RUN_DIR}" \
    data.train_files=/path/to/train_data \
    data.val_files=/path/to/val_data \
    2>&1 | tee "${RUN_DIR}/train.log"
'
```

## 6. Implementation Order

1. D1 (config explicitness in `sdpo_swe.yaml`)
2. D2 (bridge loop class + config)
3. D3 (trainer reprompt hook)
4. D4 (reward adapter and submit terminal handling)
5. D5 (`run_sdpo.sh` hygiene + local entrypoint)
6. D6 (monitored e2e run from RFT checkpoint)

## 7. Out-of-Scope for This Slice

1. Custom fine-grained token masking beyond rollout `response_mask`.
2. Rewriting upstream SDPO divergence/policy-loss internals.
3. Multi-node scaling runs before single-step e2e gate passes.
