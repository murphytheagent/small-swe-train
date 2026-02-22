#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${ON_POLICY_PROOF_STEPS:-3}"
TASK_BATCH_SIZE="${ON_POLICY_TASK_BATCH_SIZE:-2}"
ATTEMPTS_PER_TASK="${ON_POLICY_ATTEMPTS_PER_TASK:-2}"
ENV_POOL_SIZE="${ON_POLICY_ENV_POOL_SIZE:-2}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${SCRIPT_DIR}/run_rft.sh" \
    --dry-run \
    on_policy.enabled=true \
    on_policy.rollout_only=true \
    on_policy.data_config_name=on_policy_swe_smith \
    on_policy.task_batch_size="${TASK_BATCH_SIZE}" \
    on_policy.attempts_per_task="${ATTEMPTS_PER_TASK}" \
    on_policy.env_pool_size="${ENV_POOL_SIZE}" \
    trainer.total_training_steps="${STEPS}" \
    "$@"
  exit 0
fi

"${SCRIPT_DIR}/run_rft.sh" \
  on_policy.enabled=true \
  on_policy.rollout_only=true \
  on_policy.data_config_name=on_policy_swe_smith \
  on_policy.task_batch_size="${TASK_BATCH_SIZE}" \
  on_policy.attempts_per_task="${ATTEMPTS_PER_TASK}" \
  on_policy.env_pool_size="${ENV_POOL_SIZE}" \
  trainer.total_training_steps="${STEPS}" \
  "$@"
