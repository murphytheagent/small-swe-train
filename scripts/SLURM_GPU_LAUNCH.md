# GPU Script Launch Guide (Slurm)

This note covers GPU-using scripts under `scripts/` and the resource requests that match each launcher.

## Quick Resource Matrix

| Script | GPUs | CPUs | Memory | Time | Why |
| --- | --- | --- | --- | --- | --- |
| `run_rft.sh` | Variable (recommend 8) | `8 x GPUs` | `64G x GPUs` | `24:00:00` | RFT loop uses `NPROC_PER_NODE`; scales by GPU count. |
| `run_rft_onpolicy_rollout_proof.sh` | Variable (recommend 8) | `8 x GPUs` | `64G x GPUs` | `08:00:00` | Direct proof path, also keyed by GPU count. |
| `run_sdpo.sh` | **8 required** | 64 | 512G | `24:00:00` | `configs/verl/sdpo_swe.yaml` is 8-GPU tuned (`fsdp_size=8`, rollout TP/DP layout). |
| `run_sdft.sh` | **8 required** | 64 | 512G | `24:00:00` | Same base config and parallel layout as SDPO. |
| `run_teacher_reprompt_pilot_slurm.sh` | Variable (recommend 8) | `8 x GPUs` | `32G x GPUs` | `12:00:00` | Pilot ablations run with vLLM tensor parallel; set `PILOT_VLLM_TP_SIZE == requested GPUs`. |
| `run_flash_attn_rebuild.sh` | 0 (default) | 8 | 128G | `03:00:00` | Build job defaults to no GPU request; override if your site requires one. |

## Shared Setup

Run from repo root:

```bash
cd /home/zhiwang/small-swe-train
mkdir -p "$PWD/outputs/slurm"
```

All example jobs export:

```bash
PYTHON_BIN=$PWD/.venv/bin/python
```

so Slurm runs use the project virtualenv interpreter.

W&B run naming:
- `run_rft.sh`, `run_sdpo.sh`, and `run_sdft.sh` use `EXPERIMENT` as the W&B run name.
- `run_rft_onpolicy_rollout_proof.sh` uses `WANDB_RUN_NAME`.
- To map runs back to Slurm jobs, set names with `\$SLURM_JOB_ID` in every `--wrap`.
- `run_rft.sh` loop mode now defaults to one outer RFT W&B run (`SMALL_SWE_RFT_LOOP_WANDB_ENABLE=1`) and keeps inner SFT W&B disabled (`SMALL_SWE_RFT_INNER_TRAINER_WANDB_ENABLE=0`), while still surfacing inner `train/loss` and `val/loss` as `rft/inner_*` metrics.
- To restore old behavior (a separate inner SFT W&B run per RFT step), set `SMALL_SWE_RFT_INNER_TRAINER_WANDB_ENABLE=1`.

## 1) `scripts/run_rft.sh` (variable GPU count)

Choose requested GPU count first, and pass the same count to `NPROC_PER_NODE`.
By default, `run_rft.sh` writes artifacts under a unique directory:
`outputs/rft_runtime/<UTC_TIMESTAMP>_job<SLURM_JOB_ID>`.

```bash
GPUS=8
CPUS=$((GPUS * 16))
MEM="$((GPUS * 48))G"

sbatch \
  --partition=gpu \
  --nodes=1 \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time=6:00:00 \
  --job-name=small-swe-rft \
  --output="$PWD/outputs/slurm/%x-%j.out" \
  --error="$PWD/outputs/slurm/%x-%j.err" \
  --wrap "cd $PWD && export PYTHON_BIN=$PWD/.venv/bin/python && export WANDB_MODE=offline && export NPROC_PER_NODE=${GPUS} && export EXPERIMENT=small-swe-rft_job\$SLURM_JOB_ID_\$(date -u +%Y%m%dT%H%M%SZ) && bash scripts/run_rft.sh"
```

To keep runtime artifacts under the Slurm tree, add:
`export RFT_OUTPUT_ROOT=$PWD/outputs/slurm/rft_runtime` inside `--wrap`.

Dry-run:

```bash
NPROC_PER_NODE=8 bash scripts/run_rft.sh --dry-run trainer.total_training_steps=1
```

Config note:
- `run_rft.sh` sets the training model via `model.partial_pretrain` (`--initial-model`).
- Do not pass `actor_rollout_ref.model.path=...` with `rft_swe`; that key is not defined in the RFT config schema.
- Thinking mode is disabled via `++data.apply_chat_template_kwargs.enable_thinking=false` (safe whether the key exists or not).
- No launch-command change is required for single-run RFT W&B logging; it is the default.

## 2) `scripts/run_rft_onpolicy_rollout_proof.sh` (variable GPU count)

Use the same resource formula and pass the exact GPU request to `ON_POLICY_PROOF_NPROC_PER_NODE`.

```bash
GPUS=8
CPUS=$((GPUS * 8))
MEM="$((GPUS * 64))G"

sbatch \
  --partition=gpu \
  --nodes=1 \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time=08:00:00 \
  --job-name=small-swe-rft-proof \
  --output="$PWD/outputs/slurm/%x-%j.out" \
  --error="$PWD/outputs/slurm/%x-%j.err" \
  --wrap "cd $PWD && export PYTHON_BIN=$PWD/.venv/bin/python && export WANDB_MODE=offline && export ON_POLICY_PROOF_NPROC_PER_NODE=${GPUS} && export WANDB_RUN_NAME=rft-proof_job\$SLURM_JOB_ID_\$(date -u +%Y%m%dT%H%M%SZ) && bash scripts/run_rft_onpolicy_rollout_proof.sh"
```

Dry-run:

```bash
bash scripts/run_rft_onpolicy_rollout_proof.sh --dry-run
```

## 3) `scripts/run_sdpo.sh` (8 GPUs required)
quick math (verl `0.7.0.dev`, legacy FSDP worker path):
- `ppo_mini_batch_size` is a prompt-level knob that gets normalized to per-GPU rollout units:
  `normalized_ppo_mini_batch_size_per_gpu = ppo_mini_batch_size * vllm.n / dp_world_size`.
- `ppo_micro_batch_size_per_gpu` is the per-GPU rollout micro-batch size for each fwd/bwd step (`use_dynamic_bsz=false` path).
- `ppo_epochs` is the number of passes over mini-batches per update step.
- `vllm.n` is the number of rollouts per prompt.
- `train_batch_size` is the number of prompts in one update step.

example (valid numbers):
- `ppo_mini_batch_size` = 16
- `ppo_micro_batch_size_per_gpu` = 2
- `ppo_epochs` = 1
- `vllm.n` = 16
- `train_batch_size` = 128
assume world size is 8:
- one update contains `128 * 16 = 2048` rollouts.
- each GPU sees `2048 / 8 = 256` rollouts.
- normalized mini-batch per GPU is `16 * 16 / 8 = 32` rollouts.
- one update has `256 / 32 = 8` mini-steps.
- each mini-step has `32 / 2 = 16` micro-steps per GPU.
In summary, one update is 8 mini-steps, each mini-step is 16 micro-steps, each micro-step is 2 rollouts per GPU.
The VRAM peak comes from the largest micro-step; with fixed micro-batching it scales roughly with `ppo_micro_batch_size_per_gpu * sequence_length`.
Finally, `ppo_epochs = 1` means one pass over those mini-steps per update.

Note: `train_batch_size` must be `>= ppo_mini_batch_size`. For example, `train_batch_size=128` with `ppo_mini_batch_size=256` fails validation.

Length knobs in `configs/verl/sdpo_swe.yaml`:
- `max_model_len`: total rollout context cap per sequence.
- `data.max_prompt_length`: prompt-context cap before rollout.
- `data.max_response_length`: generated-token cap during rollout.
- rollout `prompt_length` / `response_length` map directly to those `data.max_*` values.
- keep `max_model_len >= max_prompt_length + max_response_length`.


`run_sdpo.sh` now auto-resolves two inputs before launching trainer:
- RFT checkpoint path (wired into `actor_rollout_ref.model.path` in `sdpo_swe`) from:
  - `SDPO_RFT_CHECKPOINT` (highest priority), or
  - `SDPO_RFT_MANIFEST`, or
  - latest `outputs/rft_runtime/*/rft_runtime_loop_manifest.json` (`final_model_path` fallback keys).
- SDPO prompt parquet overrides:
  - if `data.train_files`/`data.val_files` are not passed, it resolves preloaded
    parquet paths from `data/sdpo_task_cache` by default.
- Ray CPU budget (`ray_kwargs.ray_init.num_cpus`) from:
  - explicit Hydra override (highest priority), or
  - `SDPO_RAY_NUM_CPUS`, or
  - `SLURM_CPUS_PER_TASK`, or
  - `SLURM_CPUS_PER_GPU * visible GPUs`, or
  - CPU affinity fallback (`os.sched_getaffinity(0)`).
- Ray GPU visibility defaults to no-set mode for SDPO:
  - `run_sdpo.sh` exports `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` by default
    so worker local-rank device binding is deterministic.
  - Set `SDPO_RAY_FORCE_NOSET_VISIBLE_DEVICES=0` to disable this behavior.
- Tokenizer parallelism defaults to safe mode for SDPO:
  - `run_sdpo.sh` now exports `TOKENIZERS_PARALLELISM=false` by default.
  - This avoids forked-worker tokenizer deadlocks in Ray agent-loop workers.

`RAY_TMPDIR` is required for SDPO on this node:
- Without it, Ray defaults to `/tmp/ray`, and this host has frequent `/tmp` disk pressure.
- Typical failure symptom when `/tmp/ray` fills: `ray.exceptions.ActorDiedError` with worker
  connection error / EOF details.
- Set `RAY_TMPDIR` to a high-capacity scratch path (for this machine: `/data/scratch/$USER/ray_tmp/$SLURM_JOB_ID`).
- Optional cleanup before launch (only when no Ray jobs are running):
  - `if ! pgrep -u "$(id -u)" -fa "raylet|gcs_server|dashboard.py|runtime_env_agent" >/dev/null; then find "$RAY_TMPDIR" -mindepth 1 -maxdepth 1 -type d -name 'session_*' -exec rm -rf {} + 2>/dev/null || true; fi`

If no checkpoint can be resolved, the launcher exits early. If no data overrides are passed,
the launcher expects those parquet files to already exist; `run_sdpo.sh` does not preload/build them.

Common environment knobs:
- `SDPO_TASK_CACHE_DIR` (default: `$PWD/data/sdpo_task_cache`)
- `SDPO_PRELOADED_TASK_PARQUET=/path/file.parquet` to use one file for both train/val
- `SDPO_ROLLOUT_ONLY_E2E=1` to auto-set `trainer.test_freq=0` and `trainer.val_before_train=false`
- `SDPO_RAY_NUM_CPUS=<N>` to pin Ray CPU count when cluster Slurm env vars are non-standard
- `RAY_TMPDIR=/data/scratch/$USER/ray_tmp/$SLURM_JOB_ID` to keep Ray temp/spill files off `/tmp`
- Watchdog / stall visibility:
  - `SDPO_MONITOR_INTERVAL_SEC=120` heartbeat period in seconds
  - `SDPO_STALL_WARN_SEC=900` warn when trainer log is unchanged for this long
  - `SDPO_MONITOR_ENABLE=0` disable watchdog
  - `SDPO_TRAINER_LOG_PATH=/path/to/trainer.log` set trainer log mirror file

Watchdog log lines in Slurm output look like:
- `run_sdpo.sh watchdog: ts=... trainer_pid=... proc=... idle_log_sec=...`
- `run_sdpo.sh watchdog WARN: no trainer log updates for ...; job may be stalled.`

For stable Slurm behavior, keep CPU requests proportional to GPU requests (same CPUs per GPU).
Example: `GPUS=8`, `CPUS_PER_GPU=8`, `--cpus-per-task=$((GPUS * CPUS_PER_GPU))`.

One-time preload (manual, outside `run_sdpo.sh`):

```bash
PYTHONPATH=src ./.venv/bin/python -m env.preload_sdpo_dataset \
  --data-config-name on_policy_swe_smith \
  --cache-dir data/sdpo_task_cache \
  --emit-split \
  --emit-hydra-overrides \
  --force-refresh
```

Example submit (auto-checkpoint + default data cache paths):

```bash
export RAY_TMPDIR=/data/scratch/$USER/ray_tmp/${SLURM_JOB_ID:-manual}
mkdir -p "$RAY_TMPDIR"
if ! pgrep -u "$(id -u)" -fa "raylet|gcs_server|dashboard.py|runtime_env_agent" >/dev/null; then
  find "$RAY_TMPDIR" -mindepth 1 -maxdepth 1 -type d -name 'session_*' -exec rm -rf {} + 2>/dev/null || true
fi

sbatch \
  --partition=gpu \
  --nodes=1 \
  --gres=gpu:8 \
  --cpus-per-task=32 \
  --mem=512G \
  --time=12:00:00 \
  --job-name=small-swe-sdpo \
  --output="$PWD/outputs/slurm/%x-%j.out" \
  --error="$PWD/outputs/slurm/%x-%j.err" \
  --wrap "cd $PWD \
    && export PYTHON_BIN=$PWD/.venv/bin/python \
    && export WANDB_MODE=offline \
    && export EXPERIMENT=small-swe-sdpo_job\$SLURM_JOB_ID_\$(date -u +%Y%m%dT%H%M%SZ) \
    && export RAY_TMPDIR=/data/scratch/\$USER/ray_tmp/\$SLURM_JOB_ID \
    && mkdir -p \$RAY_TMPDIR \
    && bash scripts/run_sdpo.sh trainer.total_training_steps=2"
```

Example submit (pin explicit RFT manifest + keep cached parquet):

```bash
RFT_MANIFEST="$PWD/outputs/rft_runtime/<run>/rft_runtime_loop_manifest.json"

sbatch \
  --partition=gpu \
  --nodes=1 \
  --gres=gpu:8 \
  --cpus-per-task=64 \
  --mem=512G \
  --time=24:00:00 \
  --job-name=small-swe-sdpo \
  --output="$PWD/outputs/slurm/%x-%j.out" \
  --error="$PWD/outputs/slurm/%x-%j.err" \
  --wrap "cd $PWD \
    && export PYTHON_BIN=$PWD/.venv/bin/python \
    && export WANDB_MODE=offline \
    && export EXPERIMENT=small-swe-sdpo_job\$SLURM_JOB_ID_\$(date -u +%Y%m%dT%H%M%SZ) \
    && export RAY_TMPDIR=/data/scratch/\$USER/ray_tmp/\$SLURM_JOB_ID \
    && mkdir -p \$RAY_TMPDIR \
    && export SDPO_RFT_MANIFEST=${RFT_MANIFEST} \
    && bash scripts/run_sdpo.sh trainer.total_training_steps=1"
```

Dry-run (prints resolved command including auto-overrides; does not start Ray):

```bash
bash scripts/run_sdpo.sh --dry-run trainer.total_training_steps=1
```

## 4) `scripts/run_teacher_reprompt_pilot_slurm.sh` (variable GPU count)

This launcher starts a local vLLM OpenAI endpoint and runs
`scripts/run_teacher_reprompt_pilot.py` against it. By default it maps
`--tensor-parallel-size` to the Slurm GPU allocation (`SLURM_GPUS_ON_NODE`), so
requesting `--gres=gpu:x` yields `tensor-parallel-size=x`. For ablations, set
`PILOT_VLLM_TP_SIZE=${GPUS}` explicitly.
If `PILOT_TEACHER_TURN_INDEX=-1` is provided, the launcher auto-normalizes to
`--teacher-reprompt-turn-index-mode dynamic_middle`.

RFT checkpoint selection is resolved via `run_teacher_reprompt_pilot.py` and
applied everywhere in the launcher (`--model`, `--served-model-name`, and pilot
`--rft-checkpoint`), overriding `PILOT_MODEL_PATH`/`PILOT_SERVED_MODEL` when set:
- `--load-latest-rft-checkpoint` discovers the newest
  `rft_runtime_loop_manifest.json` from `outputs/slurm/rft_runtime/*` (or
  `outputs/rft_runtime/*`) and uses the resolved checkpoint as the vLLM model.
- `--rft-manifest <path>` uses an explicit manifest.
- `--rft-checkpoint <path-or-model-id>` directly overrides the model name.
- If none of those flags are passed, the launcher falls back to
  `PILOT_MODEL_PATH` (default `/data/scratch/$USER/models/Qwen3-4B-Instruct-2507`).

Example submit:

```bash
GPUS=2
CPUS=$((GPUS * 16))
MEM="$((GPUS * 48))G"

sbatch \
  --partition=gpu \
  --nodes=1 \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time=12:00:00 \
  --job-name=teacher-reprompt-pilot \
  --output="$PWD/outputs/slurm/%x-%j.out" \
  --error="$PWD/outputs/slurm/%x-%j.err" \
  --wrap "cd $PWD \
    && export PYTHON_BIN=$PWD/.venv/bin/python \
    && export PILOT_VLLM_TP_SIZE=${GPUS} \
    && export PILOT_TEACHER_TURN_INDEX_MODE=dynamic_middle \
    && export PILOT_TEACHER_TURN_INDEX=-1 \
    && export PILOT_TURN_SUPERVISION_MODE=current_turn \
    && export PILOT_VERIFIER_FEEDBACK_MODE=all_turns \
    && bash scripts/run_teacher_reprompt_pilot_slurm.sh --load-latest-rft-checkpoint"
```

Dry-run:

```bash
PILOT_VLLM_TP_SIZE=8 \
PILOT_TEACHER_TURN_INDEX_MODE=dynamic_middle \
PILOT_TEACHER_TURN_INDEX=-1 \
bash scripts/run_teacher_reprompt_pilot_slurm.sh --dry-run
```

Explicit manifest example (pin a specific run):

```bash
bash scripts/run_teacher_reprompt_pilot_slurm.sh \
  --rft-manifest "$PWD/outputs/slurm/rft_runtime/20260305T120000Z_job123456/rft_runtime_loop_manifest.json"
```

Latest manifest auto-discovery example:

```bash
bash scripts/run_teacher_reprompt_pilot_slurm.sh --load-latest-rft-checkpoint
```

## 5) `scripts/run_flash_attn_rebuild.sh` (resources handled by script)

This launcher already calls `sbatch` with:
- `FLASH_ATTN_BUILD_GRES` default `0` (no `--gres` added)
- `FLASH_ATTN_BUILD_CPUS_PER_TASK` default `8`
- `FLASH_ATTN_BUILD_MEM` default `128G`
- `FLASH_ATTN_BUILD_TIME` default `03:00:00`

Run from login/head node:

```bash
bash scripts/run_flash_attn_rebuild.sh
```

Override resources if needed:

```bash
FLASH_ATTN_BUILD_PARTITION=gpu \
FLASH_ATTN_BUILD_GRES=0 \
FLASH_ATTN_BUILD_CPUS_PER_TASK=8 \
FLASH_ATTN_BUILD_MEM=128G \
FLASH_ATTN_BUILD_TIME=03:00:00 \
bash scripts/run_flash_attn_rebuild.sh
```

If your cluster requires an explicit GPU request for CUDA extension builds:

```bash
FLASH_ATTN_BUILD_GRES=gpu:1 bash scripts/run_flash_attn_rebuild.sh
```

Dry-run:

```bash
bash scripts/run_flash_attn_rebuild.sh --dry-run
```

## Notes

- `eval_swebench_lite.py` and `eval_swebench_lite.sh` are not GPU launch scripts.
- If your cluster policy differs, keep the GPU count contract intact:
  - `run_rft.sh`: `NPROC_PER_NODE == requested GPUs`
  - `run_rft_onpolicy_rollout_proof.sh`: `ON_POLICY_PROOF_NPROC_PER_NODE == requested GPUs`
  - `run_teacher_reprompt_pilot_slurm.sh`: `PILOT_VLLM_TP_SIZE == requested GPUs`
