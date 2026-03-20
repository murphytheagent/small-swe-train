#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
LOAD_LATEST_RFT_CHECKPOINT=0
RFT_MANIFEST_PATH=""
RFT_CHECKPOINT_OVERRIDE=""
PILOT_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --load-latest-rft-checkpoint)
      LOAD_LATEST_RFT_CHECKPOINT=1
      shift
      ;;
    --rft-manifest)
      if [[ $# -lt 2 ]]; then
        echo "--rft-manifest requires a path argument." >&2
        exit 1
      fi
      RFT_MANIFEST_PATH="$2"
      shift 2
      ;;
    --rft-manifest=*)
      RFT_MANIFEST_PATH="${1#*=}"
      shift
      ;;
    --rft-checkpoint)
      if [[ $# -lt 2 ]]; then
        echo "--rft-checkpoint requires a checkpoint path argument." >&2
        exit 1
      fi
      RFT_CHECKPOINT_OVERRIDE="$2"
      shift 2
      ;;
    --rft-checkpoint=*)
      RFT_CHECKPOINT_OVERRIDE="${1#*=}"
      shift
      ;;
    *)
      PILOT_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

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

is_huggingface_repo_id() {
  local value="$1"
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

validate_model_reference() {
  local value="$1"
  local allow_hf_repo_id="${2:-0}"
  if [[ -d "${value}" ]]; then
    return 0
  fi
  if [[ "${allow_hf_repo_id}" == "1" ]] && is_huggingface_repo_id "${value}"; then
    return 0
  fi
  if [[ "${allow_hf_repo_id}" == "1" ]]; then
    echo "Model reference must be an existing local directory or a Hugging Face repo id: ${value}" >&2
  else
    echo "Checkpoint path does not exist: ${value}" >&2
  fi
  return 1
}

PYTHON_RESOLVER_BIN="${PYTHON_BIN}"
if [[ ! -x "${PYTHON_RESOLVER_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_RESOLVER_BIN="$(command -v python3)"
  fi
fi

RESOLVED_RFT_CHECKPOINT=""
if [[ -n "${RFT_CHECKPOINT_OVERRIDE}" || -n "${RFT_MANIFEST_PATH}" || "${LOAD_LATEST_RFT_CHECKPOINT}" -eq 1 ]]; then
  RESOLVE_RFT_CMD=(
    "${PYTHON_RESOLVER_BIN}" "${SCRIPT_DIR}/run_teacher_reprompt_pilot.py"
    --print-resolved-rft-checkpoint
  )
  if [[ -n "${RFT_CHECKPOINT_OVERRIDE}" ]]; then
    RESOLVE_RFT_CMD+=(--rft-checkpoint "${RFT_CHECKPOINT_OVERRIDE}")
  fi
  if [[ -n "${RFT_MANIFEST_PATH}" ]]; then
    RESOLVE_RFT_CMD+=(--rft-manifest "${RFT_MANIFEST_PATH}")
  fi
  if [[ "${LOAD_LATEST_RFT_CHECKPOINT}" -eq 1 ]]; then
    RESOLVE_RFT_CMD+=(--load-latest-rft-checkpoint)
  fi
  if ! RESOLVED_RFT_CHECKPOINT="$("${RESOLVE_RFT_CMD[@]}")"; then
    echo "Unable to resolve pilot RFT checkpoint via run_teacher_reprompt_pilot.py." >&2
    exit 1
  fi
  RESOLVED_RFT_CHECKPOINT="$(printf '%s' "${RESOLVED_RFT_CHECKPOINT}" | tr -d '\r')"
  if [[ -z "${RESOLVED_RFT_CHECKPOINT}" ]]; then
    echo "Resolved empty pilot RFT checkpoint." >&2
    exit 1
  fi
fi

MODEL_PATH="${PILOT_MODEL_PATH:-/data/scratch/${RUNTIME_USER}/models/Qwen3-4B-Instruct-2507}"
SERVED_MODEL="${PILOT_SERVED_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
if [[ -n "${RESOLVED_RFT_CHECKPOINT}" ]]; then
  MODEL_PATH="${RESOLVED_RFT_CHECKPOINT}"
  SERVED_MODEL="${RESOLVED_RFT_CHECKPOINT}"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter at ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ -n "${RESOLVED_RFT_CHECKPOINT}" ]]; then
    if ! validate_model_reference "${MODEL_PATH}" 0; then
      exit 1
    fi
  elif ! validate_model_reference "${MODEL_PATH}" 1; then
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
MAX_IN_FLIGHT_TASKS="${PILOT_MAX_IN_FLIGHT_TASKS:-$(( TP_SIZE * 16 ))}"

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

TURN_INDEX_MODE_RAW="${PILOT_TEACHER_TURN_INDEX_MODE:-dynamic_middle}"
TURN_INDEX_MODE="$(printf '%s' "${TURN_INDEX_MODE_RAW}" | tr '[:upper:]' '[:lower:]')"
TURN_INDEX_VALUE="${PILOT_TEACHER_TURN_INDEX:-}"

case "${TURN_INDEX_MODE}" in
  fixed|dynamic_middle) ;;
  *)
    echo "Invalid PILOT_TEACHER_TURN_INDEX_MODE=${TURN_INDEX_MODE_RAW}. Supported: fixed, dynamic_middle." >&2
    exit 1
    ;;
esac

if [[ -z "${TURN_INDEX_VALUE}" ]]; then
  if [[ "${TURN_INDEX_MODE}" == "dynamic_middle" ]]; then
    TURN_INDEX_VALUE="-1"
  else
    TURN_INDEX_VALUE="1"
  fi
fi

if [[ "${TURN_INDEX_MODE}" == "fixed" ]]; then
  if [[ "${TURN_INDEX_VALUE}" == "-1" ]]; then
    echo "PILOT_TEACHER_TURN_INDEX=-1 is dynamic-middle sentinel; switching mode to dynamic_middle." >&2
    TURN_INDEX_MODE="dynamic_middle"
  elif [[ "${TURN_INDEX_VALUE}" =~ ^- ]]; then
    echo "Invalid PILOT_TEACHER_TURN_INDEX=${TURN_INDEX_VALUE} for fixed mode; must be >= 0." >&2
    exit 1
  fi
fi

if [[ "${TURN_INDEX_MODE}" == "dynamic_middle" && "${TURN_INDEX_VALUE}" != "-1" ]]; then
  echo "dynamic_middle mode requires PILOT_TEACHER_TURN_INDEX=-1; overriding provided value ${TURN_INDEX_VALUE}." >&2
  TURN_INDEX_VALUE="-1"
fi

PILOT_CMD=(
  "${PYTHON_BIN}" scripts/run_teacher_reprompt_pilot.py
  --output-dir "${OUTPUT_DIR}"
  --task-batch-size "${PILOT_TASK_BATCH_SIZE:-1024}"
  --attempts-per-task "${PILOT_ATTEMPTS_PER_TASK:-4}"
  --max-in-flight-tasks "${MAX_IN_FLIGHT_TASKS}"
  --teacher-reprompt-turn-index "${TURN_INDEX_VALUE}"
  --teacher-reprompt-turn-index-mode "${TURN_INDEX_MODE}"
  --turn-supervision-mode "${PILOT_TURN_SUPERVISION_MODE:-current_turn}"
  --verifier-feedback-mode "${PILOT_VERIFIER_FEEDBACK_MODE:-all_turns}"
  --max-reprompt-len "${PILOT_MAX_REPROMPT_LEN:-16384}"
  --num-recent-raw-blocks "${PILOT_NUM_RECENT_RAW_BLOCKS:-3}"
)
if [[ -n "${RESOLVED_RFT_CHECKPOINT}" ]]; then
  PILOT_CMD+=(--rft-checkpoint "${RESOLVED_RFT_CHECKPOINT}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${VLLM_CMD[@]}"; printf '\n'
  printf '%q ' "${PILOT_CMD[@]}" "${PILOT_EXTRA_ARGS[@]}"; printf '\n'
  exit 0
fi

export SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE="${SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE:-1}"
export SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES="${SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES:-onpolicy-task sdpo-swe-bridge}"
bash "${SCRIPT_DIR}/preflight_sweep_stale_docker_containers.sh"

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

READY=0
for _ in $(seq 1 300); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Local vLLM server process exited before becoming ready. Last vLLM log lines:" >&2
    tail -n 120 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 5 "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "Timed out waiting for local vLLM server readiness. Last vLLM log lines:" >&2
  tail -n 120 "${VLLM_LOG}" >&2 || true
  exit 1
fi

"${PILOT_CMD[@]}" "${PILOT_EXTRA_ARGS[@]}"
