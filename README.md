# small-swe-train

Scaffold repository for a chat-style SWE training stack with RFT + step-SDPO stages.

## What is implemented
- Stable protocol types for assistant tool-call envelopes and feedback packets.
- ChatML assistant-turn parser with `<think>` and ordered `<tool_call>` support.
- Canonical feedback normalization and deterministic self-containment diagnostics.
- Deterministic adapter layer from SWE-style tool traces into canonical tools.
- Stage-aware masking policy helpers for `rft` and `step_sdpo`.
- Initial trainer/prompt/eval interface signatures.
- Dedicated `RFTTrainerScaffold` for on-policy RFT rollout/rejection/checkpoint flow, with `SDPOTrainerScaffold` delegating RFT compatibility calls.
- Optional RFT checkpoint scaffold manifests under `checkpoints/global_step_<n>/rft_step_manifest.json`.
- RFT checkpoint writes require explicit `global_step` to avoid accidental step-directory reuse.
- RFT checkpoint argument validation is fail-fast: invalid checkpoint inputs raise before rollout/training side effects.
- On-policy RFT output artifacts include `rollout_rows.jsonl` and `rollout_artifact_summary.json` (task IDs, task-image pairs, and trajectory counts) when `data.on_policy.output_dir` is set.
- Live runtime handoff orchestration is centralized in `src/trainer/rft_runtime.py` and emits `rft_runtime_manifest.json` with selected/rejected counts and rejection-reason tallies.
- Default on-policy turn generation now uses a live OpenAI-compatible vLLM endpoint (`data.on_policy.turn_generator_mode=default`), with runtime settings sourced from centralized policy + `SMALL_SWE_VLLM_*` overrides.
- RFT rejection-policy logic is centralized in `src/trainer/rft_rejection.py` with typed selection outputs/signatures.
- RFT rejection now enforces trajectory-level checks (all tool calls formatted, terminal submit present, terminal submit args valid).
- `scripts/run_rft.sh` now defaults to a real RFT runtime loop:
  - collect live rollouts from vLLM + Docker envs,
  - write selected trajectories to `MultiTurnSFTDataset`-compatible parquet shards,
  - train `verl.trainer.fsdp_sft_trainer` with per-step `data.train_files=<accepted_step.parquet>`,
  - detect the latest trainer checkpoint and restart vLLM on that snapshot for the next RFT step.
- vLLM launch defaults to `trainer.vllm_api_server_entry`, which delegates to the documented OpenAI server entrypoint and guards against broken external `flash_attn` wheels.
- `scripts/run_rft.sh` preserves a `RFT_RUNTIME_MODE=direct` path for proof/legacy one-shot launcher behavior.
- `src/verl_integration/` keeps thin compatibility wrappers for trainer-owned runtime/handoff modules.

## Layout
- `src/schemas/`: frozen JSON schema contracts + typed protocol models.
- `src/rollout/`: ChatML turn parser.
- `src/data/`: feedback canonicalizer + external tool-schema adapters.
- `src/losses/`: stage-aware action masking helpers.
- `src/teacher/`: block-structured teacher prompt builder.
- `src/trainer/`: trainer scaffold signatures.
- `tests/`: protocol stability tests.

## Quick start
```bash
python -m pytest
```

## Build and test

Create or refresh the training environment:
```bash
make build-train CORES=2
```

Or with `uv` directly:
```bash
MAX_JOBS=2 uv sync --python 3.13 --extra train
```

Run the regression suite:
```bash
python3 -m pytest -q
```

## Run commands

Dry-run the runtime loop launch:
```bash
NPROC_PER_NODE=8 bash scripts/run_rft.sh --dry-run trainer.total_training_steps=1
```

Run the default loop-mode RFT pipeline (collector -> rejection -> parquet handoff -> trainer -> checkpoint -> vLLM restart):
```bash
NPROC_PER_NODE=8 WANDB_MODE=offline bash scripts/run_rft.sh
```
By default, loop artifacts now write to a unique run directory:
`outputs/rft_runtime/<UTC_TIMESTAMP>_job<SLURM_JOB_ID>` (or `_pid<PID>` outside Slurm).
Set `RFT_OUTPUT_DIR` explicitly if you want a fixed path.

Run the realistic 2-step profile settings used in recent validation:
```bash
RFT_STEPS=2 \
SAMPLES_PER_TASK=8 \
RFT_TASK_BATCH_SIZE=64 \
RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT=16 \
SMALL_SWE_VLLM_MAX_TOKENS=512 \
NPROC_PER_NODE=8 \
WANDB_MODE=offline \
bash scripts/run_rft.sh
```

Run proof-mode direct launch (single-shot trainer invocation):
```bash
bash scripts/run_rft_onpolicy_rollout_proof.sh
```

Run SDPO runtime (`run_sdpo.sh`) through Slurm only:
```bash
# Required on this machine for SDPO:
# 1) put Ray temp on scratch, not /tmp
# 2) clean stale /tmp/ray/session_* only when no Ray daemons are running
if ! pgrep -fa "raylet|gcs_server|dashboard.py|runtime_env_agent" >/dev/null; then
  rm -rf /tmp/ray/session_*
fi
export RAY_TMPDIR=/data/scratch/$USER/ray_tmp/${SLURM_JOB_ID:-manual}
mkdir -p "$RAY_TMPDIR"
bash scripts/run_sdpo.sh trainer.total_training_steps=2
```
`run_sdpo.sh` defaults `TOKENIZERS_PARALLELISM=false` for Ray SDPO workers.
See `scripts/SLURM_GPU_LAUNCH.md` for full 8-GPU submit examples and SDPO-specific launcher defaults.

Rebuild flash-attn through Slurm:
```bash
bash scripts/run_flash_attn_rebuild.sh
```

## Runtime defaults

Centralized defaults are sourced from `configs/runtime/training_policy_defaults.v1.json`.
Config layout and field semantics are documented in `configs/README.md`.
For 8-GPU runs (`NPROC_PER_NODE=8`), the locked rollout defaults are:
- `rft_runtime.loop.collector_max_in_flight_tasks=32`
- `rft_runtime.vllm_parallelism.by_nproc_per_node.8.tensor_parallel_size=2`
- `rft_runtime.vllm_parallelism.by_nproc_per_node.8.data_parallel_size=4`

Run one deterministic Step-SDPO scaffold step from JSON/JSONL rows:
```bash
python scripts/run_step_sdpo_scaffold.py \
  --input /path/to/rollout_rows.jsonl \
  --output-dir /path/to/sdpo_step_outputs
```

## Notes
- End-to-end RFT runtime orchestration lives in `src/trainer/rft_runtime_loop.py`.
- Design artifacts remain under `outputs/1771579678.414229/` as frozen planning context.
