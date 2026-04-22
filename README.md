# small-swe-train

Repository for a chat-style SWE training stack with `format_rft`, optional `positive_rft`, and `turn_sdpo` stages.

Latest doc update: 2026-04-22.

## Current status
- `main` carries the merged training/runtime base, but the live review/runtime surface is still split across three open PRs:
  - PR `#33` head `47289ea` is the current `format_rft -> positive_rft` retry branch.
  - PR `#34` head `4d811e6` is the XML-default assistant-action payload branch.
  - Older PR `#30` remains parked and `DIRTY`.
- There is still no usable positive-stage checkpoint on the live PR `#33` path. The saved `2711` step-0 artifact already showed `selected_count=0` on `1,024` `combine_file` / `near_impossible` rows, and the repaired retry surface moved on to jobs `2753` / `2754` on head `47289ea` instead of the older `2750` / `2751` queue story.
- PR `#33` is still not merge-ready:
  - GitHub shows `11` unresolved non-outdated review threads.
  - `2753` is the live retry on the `2,813`-task partial labeled subset, with `2754` queued behind it.
  - maintenance is still treating the branch as blocked until that retry yields a real positive-stage artifact and the current head gets a clean detached-head review verdict.
- PR `#34` still has a separate runtime proof surface:
  - isolated scratch worktree `/data/scratch/murphy/projects/worktrees/small-swe-train-pr34-xml`
  - shared `.venv` reuse
  - isolated Slurm proof job `2752`
- PR `#34` is also still not merge-ready:
  - GitHub shows `1` unresolved non-outdated review thread.
  - the last full `python3 -m pytest -q` maintenance audit still failed during collection with a circular import through `prompts.runtime_messages`, `rollout.action_format`, `env.container_pool`, and `task_dataset`.

Canonical staged pipeline:
- `format_rft`
- `positive_rft` as an optional follow-up stage
- `turn_sdpo`

## What is implemented
- Stable protocol types for assistant tool-call envelopes and feedback packets.
- ChatML assistant-turn parser with `<think>` and ordered `<tool_call>` support.
- Canonical feedback normalization and deterministic self-containment diagnostics.
- Deterministic adapter layer from SWE-style tool traces into canonical tools.
- Assistant-turn supervision helpers used by the `format_rft` and `turn_sdpo` paths.
- Initial trainer/prompt/eval interface signatures.
- On-policy RFT output artifacts include `rollout_rows.jsonl` and `rollout_artifact_summary.json` (task IDs, task-image pairs, and trajectory counts) when `data.on_policy.output_dir` is set.
- Live on-policy handoff is coordinated by `src/trainer/rft_handoff.py`, summarized in `src/trainer/rft_runtime.py`, and persisted by `src/trainer/rft_runtime_loop.py` in `rft_runtime_loop_manifest.json`.
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
- `src/trainer/`: RFT runtime loop + handoff utilities.
- `tests/`: protocol stability tests.

## Quick start
```bash
MAX_JOBS=2 uv sync --python 3.13 --extra train
uv run python -m pytest
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
uv run python -m pytest -q
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

Run `turn_sdpo` runtime (`run_sdpo.sh`) through Slurm only:
```bash
# Required on this machine for SDPO:
# 1) put Ray temp on scratch, not /tmp
# 2) optional: clean stale user-owned sessions under that scratch dir
export RAY_TMPDIR=/data/scratch/$USER/ray_tmp/${SLURM_JOB_ID:-manual}
mkdir -p "$RAY_TMPDIR"
if ! pgrep -u "$(id -u)" -fa "raylet|gcs_server|dashboard.py|runtime_env_agent" >/dev/null; then
  find "$RAY_TMPDIR" -mindepth 1 -maxdepth 1 -type d -name 'session_*' -exec rm -rf {} + 2>/dev/null || true
fi
bash scripts/run_sdpo.sh trainer.total_training_steps=2
```
`run_sdpo.sh` defaults `TOKENIZERS_PARALLELISM=false` for Ray `turn_sdpo` workers.
`run_sdpo.sh` also prints a launch summary and watchdog heartbeats to stdout, and warns if trainer logs stay unchanged for too long.
Useful knobs when checking for stalls:
- `SDPO_MONITOR_INTERVAL_SEC` (default `120`)
- `SDPO_STALL_WARN_SEC` (default `900`)
- `SDPO_MONITOR_ENABLE=0` to disable the watchdog
- `SDPO_TRAINER_LOG_PATH=/path/to/trainer.log` to control the mirrored trainer log path
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

## Notes
- End-to-end RFT runtime orchestration lives in `src/trainer/rft_runtime_loop.py`.
- Design and implementation history remains in `design.md` and `IMPLEMENTATION_BLUEPRINT.md`.
- SWE-smith language-expansion plan lives in `swe_smith_language_expansion_plan.md`.
