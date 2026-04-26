#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
PROBE_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      PROBE_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-}"
if [[ -z "${PROJECT_ROOT}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/src" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]] && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

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
  if [[ -d "${value}" ]]; then
    return 0
  fi
  if is_huggingface_repo_id "${value}"; then
    return 0
  fi
  echo "Probe model must be an existing local directory or a Hugging Face repo id: ${value}" >&2
  return 1
}

INITIAL_MODEL="${PROBE_INITIAL_MODEL:-}"
if [[ -z "${INITIAL_MODEL}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    INITIAL_MODEL="Qwen/Qwen3-4B-Instruct-2507"
  else
    echo "Set PROBE_INITIAL_MODEL to the format-stage checkpoint or model to probe." >&2
    exit 1
  fi
fi
SERVED_MODEL="${PROBE_SERVED_MODEL:-${INITIAL_MODEL}}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter at ${PYTHON_BIN}" >&2
    exit 1
  fi
  if ! validate_model_reference "${INITIAL_MODEL}"; then
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
SLURM_TYPED_GPU_COUNT_RE=':([0-9]+)(\([^)]*\))?$'
if [[ "${SLURM_GPU_COUNT_RAW}" =~ ^[0-9]+$ ]]; then
  VISIBLE_GPU_COUNT="${SLURM_GPU_COUNT_RAW}"
elif [[ "${SLURM_GPU_COUNT_RAW}" =~ ${SLURM_TYPED_GPU_COUNT_RE} ]]; then
  VISIBLE_GPU_COUNT="${BASH_REMATCH[1]}"
elif [[ "${SLURM_GPU_COUNT_RAW}" =~ ([0-9]+) ]]; then
  VISIBLE_GPU_COUNT="${BASH_REMATCH[1]}"
else
  VISIBLE_GPU_COUNT="1"
fi
if [[ "${VISIBLE_GPU_COUNT}" -lt 1 ]]; then
  VISIBLE_GPU_COUNT="1"
fi

SLURM_CPU_COUNT_RAW="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-}}"
if [[ "${SLURM_CPU_COUNT_RAW}" =~ ^[0-9]+$ ]]; then
  VISIBLE_CPU_COUNT="${SLURM_CPU_COUNT_RAW}"
elif [[ "${SLURM_CPU_COUNT_RAW}" =~ ([0-9]+) ]]; then
  VISIBLE_CPU_COUNT="${BASH_REMATCH[1]}"
else
  VISIBLE_CPU_COUNT="64"
fi
if [[ "${VISIBLE_CPU_COUNT}" -lt 1 ]]; then
  VISIBLE_CPU_COUNT="64"
fi

DP_SIZE="${PROBE_VLLM_DP_SIZE:-}"
if [[ -n "${DP_SIZE}" ]]; then
  if ! [[ "${DP_SIZE}" =~ ^[0-9]+$ ]] || [[ "${DP_SIZE}" -lt 1 ]]; then
    echo "PROBE_VLLM_DP_SIZE must be an integer >= 1." >&2
    exit 1
  fi
fi

TP_SIZE="${PROBE_VLLM_TP_SIZE:-}"
if [[ -n "${TP_SIZE}" ]]; then
  if ! [[ "${TP_SIZE}" =~ ^[0-9]+$ ]] || [[ "${TP_SIZE}" -lt 1 ]]; then
    echo "PROBE_VLLM_TP_SIZE must be an integer >= 1." >&2
    exit 1
  fi
elif [[ -n "${DP_SIZE}" ]]; then
  if (( VISIBLE_GPU_COUNT % DP_SIZE != 0 )); then
    echo "Visible GPU count ${VISIBLE_GPU_COUNT} is not divisible by PROBE_VLLM_DP_SIZE=${DP_SIZE}; set PROBE_VLLM_TP_SIZE explicitly." >&2
    exit 1
  fi
  TP_SIZE="$(( VISIBLE_GPU_COUNT / DP_SIZE ))"
else
  TP_SIZE="${VISIBLE_GPU_COUNT}"
fi

if (( TP_SIZE > VISIBLE_GPU_COUNT )); then
  echo "Requested PROBE_VLLM_TP_SIZE=${TP_SIZE} exceeds visible GPU count ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

if [[ -n "${DP_SIZE}" ]] && (( TP_SIZE * DP_SIZE > VISIBLE_GPU_COUNT )); then
  echo "Requested vLLM topology TP=${TP_SIZE}, DP=${DP_SIZE} exceeds visible GPU count ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

DEFAULT_PORT="${PROBE_VLLM_PORT:-}"
if [[ -z "${DEFAULT_PORT}" ]]; then
  if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    DEFAULT_PORT="$(( 18000 + (SLURM_JOB_ID % 1000) ))"
  else
    DEFAULT_PORT="18080"
  fi
fi

RAW_BASE_URL="${SMALL_SWE_VLLM_BASE_URL:-http://127.0.0.1:${DEFAULT_PORT}/v1}"
export SMALL_SWE_VLLM_MODEL="${SMALL_SWE_VLLM_MODEL:-${SERVED_MODEL}}"
export SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC="${SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC:-300}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
NORMALIZED_BASE_URL="$("${PYTHON_BIN}" - "${RAW_BASE_URL}" <<'PY'
from __future__ import annotations

import sys
from urllib.parse import urlparse

raw_value = sys.argv[1].strip()
parsed = urlparse(raw_value)
if parsed.scheme != "http" or not parsed.netloc:
    raise SystemExit(f"Invalid SMALL_SWE_VLLM_BASE_URL: {raw_value!r}")
host = (parsed.hostname or "").strip().lower()
if host not in {"127.0.0.1", "localhost", "::1"}:
    raise SystemExit(
        "SMALL_SWE_VLLM_BASE_URL must point at a local loopback host because "
        "run_onpolicy_difficulty_probe_slurm.sh always starts its own local vLLM server."
    )
path = parsed.path.rstrip("/")
if path in {"", "/v1"}:
    normalized_path = "/v1"
elif path == "/v1/models":
    normalized_path = "/v1"
elif path == "/v1/chat/completions":
    normalized_path = "/v1"
elif path == "/chat/completions":
    normalized_path = "/v1"
elif path == "/models":
    normalized_path = "/v1"
else:
    raise SystemExit(
        "SMALL_SWE_VLLM_BASE_URL must be empty or point at the local /v1, /v1/models, "
        "or /v1/chat/completions surface exposed by the spawned server."
    )
print(parsed._replace(path=normalized_path, params="", query="", fragment="").geturl())
PY
)"
export SMALL_SWE_VLLM_BASE_URL="${NORMALIZED_BASE_URL}"
PORT="$("${PYTHON_BIN}" - "${SMALL_SWE_VLLM_BASE_URL}" <<'PY'
from __future__ import annotations

import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1].strip())
if parsed.port is None:
    raise SystemExit("SMALL_SWE_VLLM_BASE_URL must include an explicit port.")
print(parsed.port)
PY
)"

VLLM_LOG_DIR="${PROJECT_ROOT}/outputs/slurm"
mkdir -p "${VLLM_LOG_DIR}"
VLLM_LOG="${VLLM_LOG_DIR}/difficulty-probe-vllm-${SLURM_JOB_ID:-manual}.log"

VLLM_CMD=(
  "${PYTHON_BIN}" -m trainer.vllm_api_server_entry
  --host 127.0.0.1
  --port "${PORT}"
  --model "${INITIAL_MODEL}"
  --served-model-name "${SERVED_MODEL}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${PROBE_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  --max-model-len "${PROBE_VLLM_MAX_MODEL_LEN:-32768}"
)
if [[ -n "${DP_SIZE}" ]]; then
  VLLM_CMD+=(--data-parallel-size "${DP_SIZE}")
fi
if [[ -n "${PROBE_VLLM_KV_CACHE_MEMORY_BYTES:-}" ]]; then
  VLLM_CMD+=(--kv-cache-memory-bytes "${PROBE_VLLM_KV_CACHE_MEMORY_BYTES}")
fi
if [[ -n "${PROBE_VLLM_NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
  VLLM_CMD+=(--num-gpu-blocks-override "${PROBE_VLLM_NUM_GPU_BLOCKS_OVERRIDE}")
fi

CACHE_DIR="${PROBE_CACHE_DIR:-${PROJECT_ROOT}/data/on_policy_difficulty_band_cache}"
CPU_SCALED_ACTIVE_TASKS="$(( VISIBLE_CPU_COUNT * 2 ))"
if [[ "${CPU_SCALED_ACTIVE_TASKS}" -lt 128 ]]; then
  CPU_SCALED_ACTIVE_TASKS=128
fi
GPU_SCALED_ACTIVE_TASKS="$(( VISIBLE_GPU_COUNT * 32 ))"
if [[ "${GPU_SCALED_ACTIVE_TASKS}" -lt 128 ]]; then
  GPU_SCALED_ACTIVE_TASKS=128
fi
DEFAULT_PROBE_ACTIVE_TASKS="${CPU_SCALED_ACTIVE_TASKS}"
if [[ "${GPU_SCALED_ACTIVE_TASKS}" -lt "${DEFAULT_PROBE_ACTIVE_TASKS}" ]]; then
  DEFAULT_PROBE_ACTIVE_TASKS="${GPU_SCALED_ACTIVE_TASKS}"
fi
if [[ "${DEFAULT_PROBE_ACTIVE_TASKS}" -lt 1 ]]; then
  DEFAULT_PROBE_ACTIVE_TASKS=1
fi
DEFAULT_PROBE_ENV_POOL_SIZE="${PROBE_ENV_POOL_SIZE:-${DEFAULT_PROBE_ACTIVE_TASKS}}"
DEFAULT_PROBE_MAX_IN_FLIGHT_TASKS="${PROBE_MAX_IN_FLIGHT_TASKS:-${DEFAULT_PROBE_ENV_POOL_SIZE}}"
DEFAULT_PROBE_TASK_BATCH_SIZE="${PROBE_TASK_BATCH_SIZE:-$(( DEFAULT_PROBE_MAX_IN_FLIGHT_TASKS * 8 ))}"
PROBE_CMD=(
  "${PYTHON_BIN}" -m env.preload_onpolicy_difficulty_bands
  --data-config-name "${PROBE_DATA_CONFIG_NAME:-on_policy_swe_smith}"
  --initial-model "${INITIAL_MODEL}"
  --cache-dir "${CACHE_DIR}"
  --probe-label "${PROBE_LABEL:-positive_rft_probe}"
  --turn-generator-mode "${PROBE_TURN_GENERATOR_MODE:-default}"
  --stage-name "${PROBE_STAGE_NAME:-positive_rft}"
  --task-partition "${PROBE_TASK_PARTITION:-all}"
  --attempts-per-task "${PROBE_ATTEMPTS_PER_TASK:-4}"
  --start-task-index "${PROBE_START_TASK_INDEX:-0}"
  --task-batch-size "${DEFAULT_PROBE_TASK_BATCH_SIZE}"
  --env-pool-size "${DEFAULT_PROBE_ENV_POOL_SIZE}"
  --max-in-flight-tasks "${DEFAULT_PROBE_MAX_IN_FLIGHT_TASKS}"
)
if [[ -n "${PROBE_TASK_LIMIT:-}" ]]; then
  PROBE_CMD+=(--task-limit "${PROBE_TASK_LIMIT}")
fi
if [[ -n "${PROBE_TASK_EVAL_SPLIT_FRACTION:-}" ]]; then
  PROBE_CMD+=(--eval-split-fraction "${PROBE_TASK_EVAL_SPLIT_FRACTION}")
fi
if [[ -n "${PROBE_TASK_EVAL_MIN_ROWS:-}" ]]; then
  PROBE_CMD+=(--min-eval-rows "${PROBE_TASK_EVAL_MIN_ROWS}")
fi
if [[ "${PROBE_FORCE_REFRESH:-0}" == "1" ]]; then
  PROBE_CMD+=(--force-refresh)
fi

FULL_PROBE_CMD=("${PROBE_CMD[@]}")
if (( ${#PROBE_EXTRA_ARGS[@]} > 0 )); then
  FULL_PROBE_CMD+=("${PROBE_EXTRA_ARGS[@]}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${VLLM_CMD[@]}"; printf '\n'
  printf '%q ' "${FULL_PROBE_CMD[@]}"; printf '\n'
  exit 0
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE="${SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE:-1}"
export SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES="${SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES:-onpolicy-task}"
bash "${SCRIPT_DIR}/preflight_sweep_stale_docker_containers.sh"

cd "${PROJECT_ROOT}"
"${VLLM_CMD[@]}" >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

READY_TIMEOUT_SEC="${PROBE_VLLM_READY_TIMEOUT_SEC:-300}"
READY_STATUS=1
READY_DEADLINE="$(( SECONDS + READY_TIMEOUT_SEC ))"

vllm_is_running() {
  local process_state
  process_state="$(ps -o stat= -p "${VLLM_PID}" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -z "${process_state}" ]]; then
    return 1
  fi
  if [[ "${process_state:0:1}" == "Z" ]]; then
    return 1
  fi
  return 0
}

while (( SECONDS < READY_DEADLINE )); do
  if ! vllm_is_running; then
    READY_STATUS=2
    break
  fi
  if "${PYTHON_BIN}" - "${SMALL_SWE_VLLM_BASE_URL}" <<'PY'
from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen

base_url = sys.argv[1].rstrip("/")
endpoint = base_url + "/models"

try:
    with urlopen(Request(endpoint, method="GET"), timeout=1) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if isinstance(payload, dict) else 1)
PY
  then
    READY_STATUS=0
    break
  fi
  sleep 1
done
if [[ "${READY_STATUS}" -ne 0 ]]; then
  if [[ "${READY_STATUS}" -eq 2 ]]; then
    echo "vLLM process exited before readiness probe succeeded." >&2
  else
  echo "Timed out waiting for vLLM readiness at ${SMALL_SWE_VLLM_BASE_URL}/models" >&2
  fi
  tail -n 120 "${VLLM_LOG}" >&2 || true
  exit 1
fi

CACHE_PATH="$("${FULL_PROBE_CMD[@]}")"
CACHE_PATH="$(printf '%s' "${CACHE_PATH}" | tail -n 1 | tr -d '\r')"
if [[ -z "${CACHE_PATH}" ]]; then
  echo "Difficulty-band probe did not print a cache path." >&2
  exit 1
fi
if [[ ! -f "${CACHE_PATH}" ]]; then
  echo "Difficulty-band probe reported missing cache file: ${CACHE_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${CACHE_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
payload = json.loads(cache_path.read_text(encoding="utf-8"))
records = payload.get("records")
task_count = payload.get("task_count")
if not isinstance(records, list):
    raise SystemExit("Difficulty-band cache is missing a records list.")
if not isinstance(task_count, int):
    raise SystemExit("Difficulty-band cache is missing integer task_count.")
if len(records) != task_count:
    raise SystemExit(
        f"Difficulty-band cache count mismatch: task_count={task_count} records={len(records)}."
    )
task_ids = [str(record.get("task_id", "")).strip() for record in records]
if any(not task_id for task_id in task_ids):
    raise SystemExit("Difficulty-band cache contains an empty task_id.")
if len(set(task_ids)) != len(task_ids):
    raise SystemExit("Difficulty-band cache contains duplicate task_id entries.")
PY

printf '%s\n' "${CACHE_PATH}"
