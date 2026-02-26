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

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
SDPO_TRAINER_MODULE="${SDPO_TRAINER_MODULE:-verl_integration.main_ppo_entry}"
export SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH="${SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH:-1}"
SDPO_ROLLOUT_ONLY_E2E="${SDPO_ROLLOUT_ONLY_E2E:-0}"

_has_val_files_override() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      data.val_files=*|+data.val_files=*|~data.val_files|\\~data.val_files)
        return 0
        ;;
    esac
  done
  return 1
}

ROLLOUT_ONLY_OVERRIDES=()
if [[ "${SDPO_ROLLOUT_ONLY_E2E}" == "1" ]] && ! _has_val_files_override "$@"; then
  # RL e2e mode should consume rollout-generated data only.
  ROLLOUT_ONLY_OVERRIDES+=("data.val_files=[]")
fi

CMD=(
  "${PYTHON_BIN}" -m "${SDPO_TRAINER_MODULE}"
  --config-name sdpo_swe
  --config-dir "${CONFIG_DIR}"
)

if [[ "${#ROLLOUT_ONLY_OVERRIDES[@]}" -gt 0 ]]; then
  CMD+=("${ROLLOUT_ONLY_OVERRIDES[@]}")
fi

CMD+=("$@")

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

"${CMD[@]}"
