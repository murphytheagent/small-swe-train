#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/verl"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Neither python3 nor python is available in PATH."
    exit 1
  fi
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE:-verl.trainer.fsdp_sft_trainer}"
RFT_TASK_NAME="${RFT_TASK_NAME:-small-swe-rft}"

_load_rft_runtime_defaults() {
  "${PYTHON_BIN}" - <<'PY'
from config import DEFAULT_TRAINING_MODEL_NAME, rft_runtime_defaults

runtime = rft_runtime_defaults()
loop = runtime.get("loop", {})
vllm = runtime.get("vllm", {})

def _positive_int(value, fallback):
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 1:
        return value
    return fallback

def _finite_float(value, fallback):
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    return fallback

steps = _positive_int(loop.get("steps"), 1)
samples_per_task = _positive_int(loop.get("samples_per_task"), 1)
task_batch_size = _positive_int(loop.get("task_batch_size"), 1)
sft_num_epoch_per_batch = _positive_int(loop.get("sft_num_epoch_per_batch"), 1)

base_url = vllm.get("base_url")
if not isinstance(base_url, str) or not base_url.strip():
    base_url = "http://127.0.0.1:8000/v1"
model_name = vllm.get("model_name")
if not isinstance(model_name, str) or not model_name.strip():
    model_name = DEFAULT_TRAINING_MODEL_NAME
request_timeout_sec = _positive_int(vllm.get("request_timeout_sec"), 90)
max_tokens = _positive_int(vllm.get("max_tokens"), 1024)
temperature = _finite_float(vllm.get("temperature"), 0.0)
top_p = _finite_float(vllm.get("top_p"), 1.0)

print(
    steps,
    samples_per_task,
    task_batch_size,
    sft_num_epoch_per_batch,
    base_url.strip(),
    model_name.strip(),
    request_timeout_sec,
    max_tokens,
    temperature,
    top_p,
)
PY
}

RFT_DEFAULTS="$(_load_rft_runtime_defaults)"
read -r DEFAULT_RFT_STEPS DEFAULT_SAMPLES_PER_TASK DEFAULT_RFT_TASK_BATCH_SIZE DEFAULT_RFT_SFT_NUM_EPOCH_PER_BATCH DEFAULT_VLLM_BASE_URL DEFAULT_VLLM_MODEL DEFAULT_VLLM_REQUEST_TIMEOUT DEFAULT_VLLM_MAX_TOKENS DEFAULT_VLLM_TEMPERATURE DEFAULT_VLLM_TOP_P <<<"${RFT_DEFAULTS}"

RFT_STEPS="${RFT_STEPS:-${DEFAULT_RFT_STEPS}}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-${DEFAULT_SAMPLES_PER_TASK}}"
RFT_TASK_BATCH_SIZE="${RFT_TASK_BATCH_SIZE:-${DEFAULT_RFT_TASK_BATCH_SIZE}}"
RFT_SFT_NUM_EPOCH_PER_BATCH="${RFT_SFT_NUM_EPOCH_PER_BATCH:-${DEFAULT_RFT_SFT_NUM_EPOCH_PER_BATCH}}"
RFT_BATCH_SIZE="${RFT_BATCH_SIZE:-$((SAMPLES_PER_TASK * RFT_TASK_BATCH_SIZE))}"
RFT_TRAIN_BATCH_SIZE="${RFT_TRAIN_BATCH_SIZE:-${RFT_BATCH_SIZE}}"

export SMALL_SWE_VLLM_BASE_URL="${SMALL_SWE_VLLM_BASE_URL:-${DEFAULT_VLLM_BASE_URL}}"
export SMALL_SWE_VLLM_MODEL="${SMALL_SWE_VLLM_MODEL:-${DEFAULT_VLLM_MODEL}}"
export SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC="${SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC:-${DEFAULT_VLLM_REQUEST_TIMEOUT}}"
export SMALL_SWE_VLLM_MAX_TOKENS="${SMALL_SWE_VLLM_MAX_TOKENS:-${DEFAULT_VLLM_MAX_TOKENS}}"
export SMALL_SWE_VLLM_TEMPERATURE="${SMALL_SWE_VLLM_TEMPERATURE:-${DEFAULT_VLLM_TEMPERATURE}}"
export SMALL_SWE_VLLM_TOP_P="${SMALL_SWE_VLLM_TOP_P:-${DEFAULT_VLLM_TOP_P}}"

CMD=(
  torchrun
  --standalone
  --nnodes "${NNODES}"
  --nproc_per_node "${NPROC_PER_NODE}"
  -m "${RFT_TRAINER_MODULE}"
  --config-name rft_swe
  --config-dir "${CONFIG_DIR}"
  trainer.total_epochs="${RFT_SFT_NUM_EPOCH_PER_BATCH}"
  trainer.total_training_steps="${RFT_STEPS}"
  data.train_batch_size="${RFT_TRAIN_BATCH_SIZE}"
  data.on_policy.total_steps="${RFT_STEPS}"
  +data.on_policy.runtime_overrides.task_batch_size="${RFT_TASK_BATCH_SIZE}"
  +data.on_policy.runtime_overrides.attempts_per_task="${SAMPLES_PER_TASK}"
  +data.on_policy.runtime_overrides.env_pool_size="${RFT_TASK_BATCH_SIZE}"
  "$@"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if ! "${PYTHON_BIN}" -c "import verl" >/dev/null 2>&1; then
  echo "verl is not installed. Install SDPO/verl and retry."
  echo "  pip install -e \".[train]\""
  exit 1
fi

export TASK="${TASK:-${RFT_TASK_NAME}}"
"${CMD[@]}"
