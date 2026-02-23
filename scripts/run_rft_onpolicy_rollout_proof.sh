#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEPS="${ON_POLICY_PROOF_STEPS:-1}"
NPROC_PER_NODE="${ON_POLICY_PROOF_NPROC_PER_NODE:-1}"
TASK_BATCH_SIZE="${ON_POLICY_TASK_BATCH_SIZE:-${NPROC_PER_NODE}}"
ATTEMPTS_PER_TASK="${ON_POLICY_ATTEMPTS_PER_TASK:-2}"
ENV_POOL_SIZE="${ON_POLICY_ENV_POOL_SIZE:-${TASK_BATCH_SIZE}}"
MAX_TURNS_PER_ATTEMPT="${ON_POLICY_MAX_TURNS_PER_ATTEMPT:-5}"
TRAIN_BATCH_SIZE="${ON_POLICY_TRAIN_BATCH_SIZE:-${NPROC_PER_NODE}}"
MICRO_BATCH_SIZE_PER_GPU="${ON_POLICY_MICRO_BATCH_SIZE_PER_GPU:-1}"
MODEL_PATH="${ON_POLICY_PROOF_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
PROOF_OUTPUT_DIR="${ON_POLICY_PROOF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/integration/rft_onpolicy_rollout_train_step}"
RFT_ATTN_IMPLEMENTATION="${RFT_ATTN_IMPLEMENTATION:-eager}"
RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE:-verl_integration.fsdp_sft_trainer_entry}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-small-swe-rft}"
WANDB_GROUP="${WANDB_GROUP:-rft-onpolicy-proof}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-rft-proof-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  SMALL_SWE_RFT_ATTN_IMPL="${RFT_ATTN_IMPLEMENTATION}" \
  RFT_RUNTIME_MODE="direct" \
  RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE}" \
  WANDB_MODE="${WANDB_MODE}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" "${SCRIPT_DIR}/run_rft.sh" \
    --dry-run \
    model.partial_pretrain="${MODEL_PATH}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${STEPS}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.default_local_dir="${PROOF_OUTPUT_DIR}/checkpoints" \
    trainer.logger=[console,wandb] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.group_name="${WANDB_GROUP}" \
    trainer.experiment_name="${WANDB_RUN_NAME}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.on_policy.enabled=true \
    data.on_policy.data_config_name=on_policy_swe_smith \
    data.on_policy.turn_generator_mode=proof_tool_chain \
    data.on_policy.total_steps="${STEPS}" \
    data.on_policy.output_dir="${PROOF_OUTPUT_DIR}" \
    +data.on_policy.runtime_overrides.task_batch_size="${TASK_BATCH_SIZE}" \
    +data.on_policy.runtime_overrides.attempts_per_task="${ATTEMPTS_PER_TASK}" \
    +data.on_policy.runtime_overrides.env_pool_size="${ENV_POOL_SIZE}" \
    +data.on_policy.runtime_overrides.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
    "$@"
  exit 0
fi

SMALL_SWE_RFT_ATTN_IMPL="${RFT_ATTN_IMPLEMENTATION}" \
RFT_RUNTIME_MODE="direct" \
RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE}" \
WANDB_MODE="${WANDB_MODE}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" "${SCRIPT_DIR}/run_rft.sh" \
  model.partial_pretrain="${MODEL_PATH}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${STEPS}" \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.default_local_dir="${PROOF_OUTPUT_DIR}/checkpoints" \
  trainer.logger=[console,wandb] \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.group_name="${WANDB_GROUP}" \
  trainer.experiment_name="${WANDB_RUN_NAME}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  data.on_policy.enabled=true \
  data.on_policy.data_config_name=on_policy_swe_smith \
  data.on_policy.turn_generator_mode=proof_tool_chain \
  data.on_policy.total_steps="${STEPS}" \
  data.on_policy.output_dir="${PROOF_OUTPUT_DIR}" \
  +data.on_policy.runtime_overrides.task_batch_size="${TASK_BATCH_SIZE}" \
  +data.on_policy.runtime_overrides.attempts_per_task="${ATTEMPTS_PER_TASK}" \
  +data.on_policy.runtime_overrides.env_pool_size="${ENV_POOL_SIZE}" \
  +data.on_policy.runtime_overrides.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
  "$@"
