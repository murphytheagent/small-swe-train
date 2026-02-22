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
TASK_BATCH_SIZE="${ON_POLICY_TASK_BATCH_SIZE:-2}"
ATTEMPTS_PER_TASK="${ON_POLICY_ATTEMPTS_PER_TASK:-2}"
ENV_POOL_SIZE="${ON_POLICY_ENV_POOL_SIZE:-2}"
MAX_TURNS_PER_ATTEMPT="${ON_POLICY_MAX_TURNS_PER_ATTEMPT:-5}"
NPROC_PER_NODE="${ON_POLICY_PROOF_NPROC_PER_NODE:-1}"
MODEL_PATH="${ON_POLICY_PROOF_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
PROOF_OUTPUT_DIR="${ON_POLICY_PROOF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/integration/rft_onpolicy_rollout_train_step}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE}" "${SCRIPT_DIR}/run_rft.sh" \
    --dry-run \
    model.partial_pretrain="${MODEL_PATH}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${STEPS}" \
    trainer.logger=[console] \
    data.train_batch_size=1 \
    data.micro_batch_size_per_gpu=1 \
    data.on_policy.enabled=true \
    data.on_policy.data_config_name=on_policy_swe_smith \
    data.on_policy.turn_generator_mode=proof_tool_chain \
    data.on_policy.total_steps=1 \
    data.on_policy.output_dir="${PROOF_OUTPUT_DIR}" \
    +data.on_policy.runtime_overrides.task_batch_size="${TASK_BATCH_SIZE}" \
    +data.on_policy.runtime_overrides.attempts_per_task="${ATTEMPTS_PER_TASK}" \
    +data.on_policy.runtime_overrides.env_pool_size="${ENV_POOL_SIZE}" \
    +data.on_policy.runtime_overrides.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
    "$@"
  exit 0
fi

NPROC_PER_NODE="${NPROC_PER_NODE}" "${SCRIPT_DIR}/run_rft.sh" \
  model.partial_pretrain="${MODEL_PATH}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${STEPS}" \
  trainer.logger=[console] \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.on_policy.enabled=true \
  data.on_policy.data_config_name=on_policy_swe_smith \
  data.on_policy.turn_generator_mode=proof_tool_chain \
  data.on_policy.total_steps=1 \
  data.on_policy.output_dir="${PROOF_OUTPUT_DIR}" \
  +data.on_policy.runtime_overrides.task_batch_size="${TASK_BATCH_SIZE}" \
  +data.on_policy.runtime_overrides.attempts_per_task="${ATTEMPTS_PER_TASK}" \
  +data.on_policy.runtime_overrides.env_pool_size="${ENV_POOL_SIZE}" \
  +data.on_policy.runtime_overrides.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
  "$@"
