#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

_is_executable_cmd() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]]
    return
  fi
  command -v "${candidate}" >/dev/null 2>&1
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if _is_executable_cmd "${VENV_PYTHON}"; then
    PYTHON_BIN="${VENV_PYTHON}"
  elif _is_executable_cmd python3; then
    PYTHON_BIN="python3"
  elif _is_executable_cmd python; then
    PYTHON_BIN="python"
  else
    echo "No Python interpreter found. Expected ${VENV_PYTHON} or python3/python in PATH."
    exit 1
  fi
elif ! _is_executable_cmd "${PYTHON_BIN}"; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 1
fi

_detect_available_gpu_count() {
  local detected
  detected="$(
    "${PYTHON_BIN}" - <<'PY'
try:
    import torch
except Exception:
    print(0)
else:
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    print(count if count > 0 else 0)
PY
  )"
  if [[ "${detected}" =~ ^[0-9]+$ ]] && (( detected > 0 )); then
    printf '%s\n' "${detected}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
    if [[ "${detected}" =~ ^[0-9]+$ ]] && (( detected > 0 )); then
      printf '%s\n' "${detected}"
      return
    fi
  fi

  printf '1\n'
}

REQUESTED_NPROC_PER_NODE="${ON_POLICY_PROOF_NPROC_PER_NODE:-${NPROC_PER_NODE:-}}"
if [[ -z "${REQUESTED_NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="$(_detect_available_gpu_count)"
else
  NPROC_PER_NODE="${REQUESTED_NPROC_PER_NODE}"
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Resolved NPROC_PER_NODE must be a positive integer (got: ${NPROC_PER_NODE})."
  exit 1
fi

STEPS="${ON_POLICY_PROOF_STEPS:-1}"
TASK_BATCH_SIZE="${ON_POLICY_TASK_BATCH_SIZE:-${NPROC_PER_NODE}}"
ATTEMPTS_PER_TASK="${ON_POLICY_ATTEMPTS_PER_TASK:-2}"
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
  PYTHON_BIN="${PYTHON_BIN}" \
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
    ++data.on_policy.total_steps="${STEPS}" \
    data.on_policy.output_dir="${PROOF_OUTPUT_DIR}" \
    +data.on_policy.runtime_overrides.task_batch_size="${TASK_BATCH_SIZE}" \
    +data.on_policy.runtime_overrides.attempts_per_task="${ATTEMPTS_PER_TASK}" \
    +data.on_policy.runtime_overrides.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
    "$@"
  exit 0
fi

SMALL_SWE_RFT_ATTN_IMPL="${RFT_ATTN_IMPLEMENTATION}" \
RFT_RUNTIME_MODE="direct" \
RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE}" \
WANDB_MODE="${WANDB_MODE}" \
PYTHON_BIN="${PYTHON_BIN}" \
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
  ++data.on_policy.total_steps="${STEPS}" \
  data.on_policy.output_dir="${PROOF_OUTPUT_DIR}" \
  +data.on_policy.runtime_overrides.task_batch_size="${TASK_BATCH_SIZE}" \
  +data.on_policy.runtime_overrides.attempts_per_task="${ATTEMPTS_PER_TASK}" \
  +data.on_policy.runtime_overrides.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
  "$@"
