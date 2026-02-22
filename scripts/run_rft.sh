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
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE:-verl.trainer.fsdp_sft_trainer}"
RFT_TASK_NAME="${RFT_TASK_NAME:-small-swe-rft}"
CMD=(
  torchrun
  --standalone
  --nnodes "${NNODES}"
  --nproc_per_node "${NPROC_PER_NODE}"
  -m "${RFT_TRAINER_MODULE}"
  --config-name rft_swe
  --config-dir "${CONFIG_DIR}"
  "$@"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if ! python -c "import verl" >/dev/null 2>&1; then
  echo "verl is not installed. Install SDPO/verl and retry."
  echo "  pip install -e \".[train]\""
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TASK="${TASK:-${RFT_TASK_NAME}}"
"${CMD[@]}"
