# GPU Script Launch Guide (Slurm)

This note covers GPU-using scripts under `scripts/` and the resource requests that match each launcher.

## Quick Resource Matrix

| Script | GPUs | CPUs | Memory | Time | Why |
| --- | --- | --- | --- | --- | --- |
| `run_rft.sh` | Variable (recommend 8) | `8 x GPUs` | `64G x GPUs` | `24:00:00` | RFT loop uses `NPROC_PER_NODE`; scales by GPU count. |
| `run_rft_onpolicy_rollout_proof.sh` | Variable (recommend 8) | `8 x GPUs` | `64G x GPUs` | `08:00:00` | Direct proof path, also keyed by GPU count. |
| `run_sdpo.sh` | **8 required** | 64 | 512G | `24:00:00` | `configs/verl/sdpo_swe.yaml` is 8-GPU tuned (`fsdp_size=8`, rollout TP/DP layout). |
| `run_sdft.sh` | **8 required** | 64 | 512G | `24:00:00` | Same base config and parallel layout as SDPO. |
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
  --wrap "cd $PWD && export PYTHON_BIN=$PWD/.venv/bin/python && export WANDB_MODE=offline && export NPROC_PER_NODE=${GPUS} && bash scripts/run_rft.sh"
```

To keep runtime artifacts under the Slurm tree, add:
`export RFT_OUTPUT_ROOT=$PWD/outputs/slurm/rft_runtime` inside `--wrap`.

Dry-run:

```bash
NPROC_PER_NODE=8 bash scripts/run_rft.sh --dry-run trainer.total_training_steps=1
```

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
  --wrap "cd $PWD && export PYTHON_BIN=$PWD/.venv/bin/python && export WANDB_MODE=offline && export ON_POLICY_PROOF_NPROC_PER_NODE=${GPUS} && bash scripts/run_rft_onpolicy_rollout_proof.sh"
```

Dry-run:

```bash
bash scripts/run_rft_onpolicy_rollout_proof.sh --dry-run
```

## 3) `scripts/run_sdpo.sh` (8 GPUs required)

`run_sdpo.sh` now auto-resolves two inputs before launching trainer:
- RFT checkpoint path (`actor_rollout_ref.model.path`) from:
  - `SDPO_RFT_CHECKPOINT` (highest priority), or
  - `SDPO_RFT_MANIFEST`, or
  - latest `outputs/rft_runtime/*/rft_runtime_loop_manifest.json` (`final_model_path` fallback keys).
- SDPO prompt parquet overrides:
  - if `data.train_files`/`data.val_files` are not passed, it resolves deterministic split
    parquet paths in `data/sdpo_task_cache` by default.

If no checkpoint can be resolved, the launcher exits early. If no data overrides are passed,
the launcher expects those parquet files to already exist; `run_sdpo.sh` does not preload/build them.

Common environment knobs:
- `SDPO_TASK_CACHE_DIR` (default: `$PWD/data/sdpo_task_cache`)
- `SDPO_DATA_CONFIG_NAME` (default: `on_policy_swe_smith`)
- `SDPO_EVAL_SPLIT_FRACTION` / `SDPO_EVAL_MIN_ROWS` (affect default split file path resolution)
- `SDPO_PRELOADED_TASK_PARQUET=/path/file.parquet` to use one file for both train/val
- `SDPO_ROLLOUT_ONLY_E2E=1` to auto-set `trainer.test_freq=0` and `trainer.val_before_train=false`

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
    && export SDPO_ROLLOUT_ONLY_E2E=1 \
    && bash scripts/run_sdpo.sh trainer.total_training_steps=1"
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
    && export SDPO_RFT_MANIFEST=${RFT_MANIFEST} \
    && bash scripts/run_sdpo.sh trainer.total_training_steps=1"
```

Dry-run (prints resolved command including auto-overrides):

```bash
bash scripts/run_sdpo.sh --dry-run trainer.total_training_steps=1
```

## 4) `scripts/run_sdft.sh` (8 GPUs required)

```bash
sbatch \
  --partition=gpu \
  --nodes=1 \
  --gres=gpu:8 \
  --cpus-per-task=64 \
  --mem=512G \
  --time=24:00:00 \
  --job-name=small-swe-sdft \
  --output="$PWD/outputs/slurm/%x-%j.out" \
  --error="$PWD/outputs/slurm/%x-%j.err" \
  --wrap "cd $PWD && export PYTHON_BIN=$PWD/.venv/bin/python && bash scripts/run_sdft.sh <hydra-overrides>"
```

Dry-run:

```bash
bash scripts/run_sdft.sh --dry-run <hydra-overrides>
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

- `run_step_sdpo_scaffold.py`, `eval_swebench_lite.py`, and `eval_swebench_lite.sh` are not GPU launch scripts.
- If your cluster policy differs, keep the GPU count contract intact:
  - `run_rft.sh`: `NPROC_PER_NODE == requested GPUs`
  - `run_rft_onpolicy_rollout_proof.sh`: `ON_POLICY_PROOF_NPROC_PER_NODE == requested GPUs`
