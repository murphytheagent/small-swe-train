#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
RUNTIME_USER="${USER:-}"
if [[ -z "${RUNTIME_USER}" ]]; then
  RUNTIME_USER="$(id -un 2>/dev/null || true)"
fi
if [[ -z "${RUNTIME_USER}" ]]; then
  RUNTIME_USER="unknown"
fi

MODEL_PATH="${PILOT_MODEL_PATH:-/data/scratch/${RUNTIME_USER}/models/Qwen3-4B-Instruct-2507}"
SERVED_MODEL="${PILOT_SERVED_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter at ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
  fi
fi

DEFAULT_CACHE_BASE="/data/users/${RUNTIME_USER}/cache"
if [[ ! -d "/data/users/${RUNTIME_USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${RUNTIME_USER}/cache"
fi
export HF_HOME="${HF_HOME:-${DEFAULT_CACHE_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEFAULT_CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${DEFAULT_CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DEFAULT_CACHE_BASE}}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"
fi

SLURM_GPU_COUNT_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ "${SLURM_GPU_COUNT_RAW}" =~ ^[0-9]+$ ]]; then
  DEFAULT_TP_SIZE="${SLURM_GPU_COUNT_RAW}"
elif [[ "${SLURM_GPU_COUNT_RAW}" =~ ([0-9]+) ]]; then
  DEFAULT_TP_SIZE="${BASH_REMATCH[1]}"
else
  DEFAULT_TP_SIZE="1"
fi
if [[ "${DEFAULT_TP_SIZE}" -lt 1 ]]; then
  DEFAULT_TP_SIZE="1"
fi
TP_SIZE="${PILOT_VLLM_TP_SIZE:-${DEFAULT_TP_SIZE}}"
MAX_IN_FLIGHT_TASKS="${PILOT_MAX_IN_FLIGHT_TASKS:-$(( TP_SIZE * 8 ))}"

export SMALL_SWE_VLLM_BASE_URL="${SMALL_SWE_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export SMALL_SWE_VLLM_MODEL="${SMALL_SWE_VLLM_MODEL:-${SERVED_MODEL}}"
export SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC="${SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC:-300}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

VLLM_LOG_DIR="${PROJECT_ROOT}/outputs/slurm"
mkdir -p "${VLLM_LOG_DIR}"
VLLM_LOG="${VLLM_LOG_DIR}/teacher-pilot-vllm-${SLURM_JOB_ID:-manual}.log"

VLLM_CMD=(
  "${PYTHON_BIN}" -m trainer.vllm_api_server_entry
  --host 127.0.0.1
  --port 8000
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${PILOT_GPU_MEMORY_UTILIZATION:-0.90}"
  --max-model-len "${PILOT_MAX_MODEL_LEN:-32768}"
)
if [[ -n "${PILOT_VLLM_KV_CACHE_MEMORY_BYTES:-}" ]]; then
  VLLM_CMD+=(--kv-cache-memory-bytes "${PILOT_VLLM_KV_CACHE_MEMORY_BYTES}")
fi
if [[ -n "${PILOT_VLLM_NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
  VLLM_CMD+=(--num-gpu-blocks-override "${PILOT_VLLM_NUM_GPU_BLOCKS_OVERRIDE}")
fi

OUTPUT_DIR="${PROJECT_ROOT}/outputs/teacher_reprompt_pilot/job${SLURM_JOB_ID:-manual}"
mkdir -p "${OUTPUT_DIR}"

PILOT_CMD=(
  "${PYTHON_BIN}" scripts/run_teacher_reprompt_pilot.py
  --output-dir "${OUTPUT_DIR}"
  --task-batch-size "${PILOT_TASK_BATCH_SIZE:-128}"
  --attempts-per-task "${PILOT_ATTEMPTS_PER_TASK:-8}"
  --max-in-flight-tasks "${MAX_IN_FLIGHT_TASKS}"
  --teacher-reprompt-turn-index "${PILOT_TEACHER_TURN_INDEX:-1}"
  --turn-supervision-mode "${PILOT_TURN_SUPERVISION_MODE:-current_turn}"
  --verifier-feedback-mode "${PILOT_VERIFIER_FEEDBACK_MODE:-all_turns}"
  --max-reprompt-len "${PILOT_MAX_REPROMPT_LEN:-12288}"
  --num-recent-raw-blocks "${PILOT_NUM_RECENT_RAW_BLOCKS:-3}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${VLLM_CMD[@]}"; printf '\n'
  printf '%q ' "${PILOT_CMD[@]}"; printf '\n'
  exit 0
fi

cd "${PROJECT_ROOT}"
"${VLLM_CMD[@]}" >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${PYTHON_BIN}" - <<'PY'
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

url = "http://127.0.0.1:8000/v1/models"
deadline = time.time() + 600
while time.time() < deadline:
    try:
        with urlopen(url, timeout=5) as response:
            if response.status == 200:
                sys.exit(0)
    except (URLError, HTTPError):
        pass
    time.sleep(2)
print("Timed out waiting for local vLLM server readiness", file=sys.stderr)
sys.exit(1)
PY

"${PILOT_CMD[@]}" "$@"
