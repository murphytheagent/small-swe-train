#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARTITION="${FLASH_ATTN_BUILD_PARTITION:-gpu}"
GRES="${FLASH_ATTN_BUILD_GRES:-gpu:1}"
CPUS_PER_TASK="${FLASH_ATTN_BUILD_CPUS_PER_TASK:-8}"
MEMORY="${FLASH_ATTN_BUILD_MEM:-128G}"
TIME_LIMIT="${FLASH_ATTN_BUILD_TIME:-03:00:00}"
MAX_JOBS="${FLASH_ATTN_BUILD_MAX_JOBS:-8}"
UV_BIN="${UV_BIN:-uv}"
VENV_PYTHON="${VENV_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
FLASH_ATTN_PACKAGE="${FLASH_ATTN_PACKAGE:-flash-attn}"
RUN_LABEL="${FLASH_ATTN_BUILD_RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)_flash_attn_rebuild}"
LOG_DIR="${FLASH_ATTN_BUILD_LOG_DIR:-${PROJECT_ROOT}/outputs/flash_attn_rebuild/${RUN_LABEL}}"

WRAP_CMD=(
  "set -euo pipefail"
  "cd ${PROJECT_ROOT}"
  "make rebuild-flash-attn CORES=${MAX_JOBS} UV=${UV_BIN} VENV_PYTHON=${VENV_PYTHON} FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} FLASH_ATTN_PACKAGE=${FLASH_ATTN_PACKAGE}"
)

SBATCH_CMD=(
  sbatch
  --parsable
  --partition "${PARTITION}"
  --gres "${GRES}"
  --cpus-per-task "${CPUS_PER_TASK}"
  --mem "${MEMORY}"
  --time "${TIME_LIMIT}"
  --job-name flash-attn-rebuild
  --output "${LOG_DIR}/slurm-%j.out"
  --error "${LOG_DIR}/slurm-%j.err"
  --wrap "$(IFS='; '; echo "${WRAP_CMD[*]}")"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${LOG_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available in PATH."
  exit 1
fi

JOB_ID="$("${SBATCH_CMD[@]}")"
echo "Submitted flash-attn rebuild job: ${JOB_ID}"
echo "Logs: ${LOG_DIR}/slurm-${JOB_ID}.out"
