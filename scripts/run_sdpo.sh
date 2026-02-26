#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
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
SDPO_DATA_CONFIG_NAME="${SDPO_DATA_CONFIG_NAME:-on_policy_swe_smith}"
SDPO_TASK_CACHE_DIR="${SDPO_TASK_CACHE_DIR:-${PROJECT_ROOT}/data/sdpo_task_cache}"
SDPO_EVAL_SPLIT_FRACTION="${SDPO_EVAL_SPLIT_FRACTION:-}"
SDPO_EVAL_MIN_ROWS="${SDPO_EVAL_MIN_ROWS:-}"
SDPO_PRELOADED_TASK_PARQUET="${SDPO_PRELOADED_TASK_PARQUET:-}"
SDPO_RFT_CHECKPOINT="${SDPO_RFT_CHECKPOINT:-${RFT_CKPT:-}}"
SDPO_RFT_MANIFEST="${SDPO_RFT_MANIFEST:-${RFT_MANIFEST:-}}"
SDPO_TASK_NAME="${SDPO_TASK_NAME:-small-swe-sdpo}"
export EXPERIMENT="${EXPERIMENT:-${SDPO_TASK_NAME}}"
export TASK="${TASK:-${SDPO_TASK_NAME}}"

_has_override_with_prefix() {
  local prefix="$1"
  shift
  local arg
  for arg in "$@"; do
    case "${arg}" in
      "${prefix}"=*|+"${prefix}"=*)
        return 0
        ;;
    esac
  done
  return 1
}

_has_override_for_key() {
  local key="$1"
  shift
  local arg
  for arg in "$@"; do
    case "${arg}" in
      "${key}"=*|+"${key}"=*|~"${key}")
        return 0
        ;;
    esac
  done
  return 1
}

_extract_override_value() {
  local prefix="$1"
  shift
  local arg
  for arg in "$@"; do
    case "${arg}" in
      "${prefix}"=*|+"${prefix}"=*)
        printf '%s' "${arg#*=}"
        return 0
        ;;
    esac
  done
  return 1
}

_discover_latest_rft_manifest() {
  local latest_manifest=""
  local candidate
  shopt -s nullglob
  for candidate in "${PROJECT_ROOT}"/outputs/rft_runtime/*/rft_runtime_loop_manifest.json; do
    if [[ -z "${latest_manifest}" || "${candidate}" -nt "${latest_manifest}" ]]; then
      latest_manifest="${candidate}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${latest_manifest}"
}

_checkpoint_from_manifest() {
  local manifest_path="$1"
  "${PYTHON_BIN}" - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
for key in ("final_model_path", "latest_vllm_checkpoint", "latest_hf_checkpoint"):
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        print(value.strip())
        raise SystemExit(0)
raise SystemExit(1)
PY
}

_resolve_sdpo_rft_checkpoint() {
  if [[ -n "${SDPO_RFT_CHECKPOINT}" ]]; then
    printf '%s' "${SDPO_RFT_CHECKPOINT}"
    return 0
  fi

  local manifest_path="${SDPO_RFT_MANIFEST}"
  if [[ -z "${manifest_path}" ]]; then
    manifest_path="$(_discover_latest_rft_manifest)"
  fi
  if [[ -z "${manifest_path}" ]]; then
    return 1
  fi
  if [[ ! -f "${manifest_path}" ]]; then
    echo "SDPO_RFT_MANIFEST does not exist: ${manifest_path}" >&2
    return 1
  fi
  _checkpoint_from_manifest "${manifest_path}"
}

_resolve_sdpo_dataset_overrides() {
  "${PYTHON_BIN}" - "${PROJECT_ROOT}" "${SDPO_DATA_CONFIG_NAME}" "${SDPO_TASK_CACHE_DIR}" "${SDPO_EVAL_SPLIT_FRACTION}" "${SDPO_EVAL_MIN_ROWS}" <<'PY'
import sys
from pathlib import Path
from typing import Any, Mapping

project_root = Path(sys.argv[1])
data_config_name = str(sys.argv[2]).strip()
cache_dir = str(sys.argv[3]).strip()
eval_split_fraction_raw = str(sys.argv[4]).strip()
eval_min_rows_raw = str(sys.argv[5]).strip()

sys.path.insert(0, str(project_root / "src"))

from config import rft_runtime_defaults, resolve_on_policy_settings
from env.task_dataset import resolve_sdpo_task_split_cache_paths

def _coerce_eval_split_fraction(value: Any, *, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        candidate = float(value)
        if 0.0 <= candidate < 1.0:
            return candidate
    return fallback

def _coerce_eval_min_rows(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 0:
        return int(value)
    return fallback

defaults = rft_runtime_defaults()
loop_defaults = defaults.get("loop") if isinstance(defaults, Mapping) else None
fallback_eval_split_fraction = 0.1
fallback_eval_min_rows = 1
if isinstance(loop_defaults, Mapping):
    fallback_eval_split_fraction = _coerce_eval_split_fraction(
        loop_defaults.get("eval_split_fraction"),
        fallback=fallback_eval_split_fraction,
    )
    fallback_eval_min_rows = _coerce_eval_min_rows(
        loop_defaults.get("eval_min_rows"),
        fallback=fallback_eval_min_rows,
    )

eval_split_fraction = (
    float(eval_split_fraction_raw)
    if eval_split_fraction_raw
    else fallback_eval_split_fraction
)
eval_min_rows = int(eval_min_rows_raw) if eval_min_rows_raw else fallback_eval_min_rows

settings = resolve_on_policy_settings(data_config_name=data_config_name)
train_path, val_path = resolve_sdpo_task_split_cache_paths(
    config=settings.data,
    cache_dir=cache_dir,
    eval_split_fraction=eval_split_fraction,
    min_eval_rows=eval_min_rows,
)
print(f"data.train_files={train_path}")
print(f"data.val_files={val_path}")
PY
}

AUTO_OVERRIDES=()

if ! _has_override_for_key "data.apply_chat_template_kwargs.enable_thinking" "$@"; then
  AUTO_OVERRIDES+=("~data.apply_chat_template_kwargs.enable_thinking")
fi

if ! _has_override_with_prefix "actor_rollout_ref.model.path" "$@"; then
  if ! SDPO_RFT_CHECKPOINT="$(_resolve_sdpo_rft_checkpoint)"; then
    echo "Unable to resolve SDPO RFT checkpoint. Set SDPO_RFT_CHECKPOINT or SDPO_RFT_MANIFEST (or pass actor_rollout_ref.model.path=...)."
    exit 1
  fi
  if [[ "${DRY_RUN}" -ne 1 && ! -e "${SDPO_RFT_CHECKPOINT}" ]]; then
    echo "Resolved SDPO RFT checkpoint does not exist: ${SDPO_RFT_CHECKPOINT}"
    exit 1
  fi
  AUTO_OVERRIDES+=("actor_rollout_ref.model.path=${SDPO_RFT_CHECKPOINT}")
fi

TRAIN_FILES_OVERRIDE="$(_extract_override_value "data.train_files" "$@" || true)"
VAL_FILES_OVERRIDE="$(_extract_override_value "data.val_files" "$@" || true)"

if [[ -n "${TRAIN_FILES_OVERRIDE}" && -z "${VAL_FILES_OVERRIDE}" ]]; then
  AUTO_OVERRIDES+=("data.val_files=${TRAIN_FILES_OVERRIDE}")
elif [[ -z "${TRAIN_FILES_OVERRIDE}" && -n "${VAL_FILES_OVERRIDE}" ]]; then
  AUTO_OVERRIDES+=("data.train_files=${VAL_FILES_OVERRIDE}")
elif [[ -z "${TRAIN_FILES_OVERRIDE}" && -z "${VAL_FILES_OVERRIDE}" ]]; then
  if [[ -n "${SDPO_PRELOADED_TASK_PARQUET}" ]]; then
    SDPO_DATA_OVERRIDES=(
      "data.train_files=${SDPO_PRELOADED_TASK_PARQUET}"
      "data.val_files=${SDPO_PRELOADED_TASK_PARQUET}"
    )
  else
    mapfile -t SDPO_DATA_OVERRIDES < <(_resolve_sdpo_dataset_overrides)
  fi
  if [[ "${#SDPO_DATA_OVERRIDES[@]}" -lt 2 ]]; then
    echo "Failed to resolve SDPO data.train_files/data.val_files overrides."
    exit 1
  fi
  for override in "${SDPO_DATA_OVERRIDES[@]}"; do
    if [[ "${override}" != data.train_files=* && "${override}" != data.val_files=* ]]; then
      echo "Unexpected SDPO data override: ${override}"
      exit 1
    fi
    value="${override#*=}"
    if [[ "${DRY_RUN}" -ne 1 && ! -f "${value}" ]]; then
      echo "Resolved SDPO parquet does not exist: ${value}"
      echo "Preload dataset artifacts in advance or pass explicit data.train_files/data.val_files overrides."
      exit 1
    fi
    AUTO_OVERRIDES+=("${override}")
  done
fi

ROLLOUT_ONLY_OVERRIDES=()
if [[ "${SDPO_ROLLOUT_ONLY_E2E}" == "1" ]]; then
  # RL e2e mode should use rollout-generated training data only; disable
  # validation rollouts while preserving config compatibility.
  if ! _has_override_with_prefix "trainer.test_freq" "$@"; then
    ROLLOUT_ONLY_OVERRIDES+=("trainer.test_freq=0")
  fi
  if ! _has_override_with_prefix "trainer.val_before_train" "$@"; then
    ROLLOUT_ONLY_OVERRIDES+=("trainer.val_before_train=false")
  fi
fi

CMD=(
  "${PYTHON_BIN}" -m "${SDPO_TRAINER_MODULE}"
  --config-name sdpo_swe
  --config-dir "${CONFIG_DIR}"
)

if [[ "${#AUTO_OVERRIDES[@]}" -gt 0 ]]; then
  CMD+=("${AUTO_OVERRIDES[@]}")
fi

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
