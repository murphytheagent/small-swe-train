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
DEFAULT_SDPO_TASK_CACHE_DIR="$("${PYTHON_BIN}" - "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
sys.path.insert(0, str(project_root / "src"))

from runtime_paths import resolve_sdpo_task_cache_dir

print(resolve_sdpo_task_cache_dir(project_root=project_root))
PY
)"
SDPO_TRAINER_MODULE="${SDPO_TRAINER_MODULE:-verl_integration.main_ppo_entry}"
export SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH="${SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH:-1}"
# Prevent tokenizer-thread deadlocks in forked Ray worker processes.
# verl's PPO runtime sets TOKENIZERS_PARALLELISM=true by default unless this is preset.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Optional torch-c-dlpack JIT extension is noisy when unavailable and not required for correctness.
export TVM_FFI_DISABLE_TORCH_C_DLPACK="${TVM_FFI_DISABLE_TORCH_C_DLPACK:-1}"
SDPO_ROLLOUT_ONLY_E2E="${SDPO_ROLLOUT_ONLY_E2E:-0}"
SDPO_TASK_CACHE_DIR="${SDPO_TASK_CACHE_DIR:-${DEFAULT_SDPO_TASK_CACHE_DIR}}"
SDPO_PRELOADED_TASK_PARQUET="${SDPO_PRELOADED_TASK_PARQUET:-}"
SDPO_RFT_CHECKPOINT="${SDPO_RFT_CHECKPOINT:-${RFT_CKPT:-}}"
SDPO_RFT_MANIFEST="${SDPO_RFT_MANIFEST:-${RFT_MANIFEST:-}}"
SDPO_TASK_NAME="${SDPO_TASK_NAME:-small-swe-sdpo}"
SDPO_RUN_TIMESTAMP="${SDPO_RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  DEFAULT_SDPO_RUN_LABEL="${SDPO_RUN_TIMESTAMP}_job${SLURM_JOB_ID}"
else
  DEFAULT_SDPO_RUN_LABEL="${SDPO_RUN_TIMESTAMP}_pid$$"
fi
SDPO_RUN_LABEL="${SDPO_RUN_LABEL:-${DEFAULT_SDPO_RUN_LABEL}}"
export SDPO_RUN_LABEL
export EXPERIMENT="${EXPERIMENT:-${SDPO_TASK_NAME}_${SDPO_RUN_LABEL}}"
export TASK="${TASK:-${SDPO_TASK_NAME}}"
SDPO_CLEANUP_ON_EXIT="${SDPO_CLEANUP_ON_EXIT:-1}"
SDPO_CLEANUP_GRACE_SEC="${SDPO_CLEANUP_GRACE_SEC:-5}"
SDPO_CONTAINER_CLEANUP_ENABLE="${SDPO_CONTAINER_CLEANUP_ENABLE:-1}"
SDPO_CONTAINER_NAME_PREFIX="${SDPO_CONTAINER_NAME_PREFIX:-sdpo-swe-bridge}"
SDPO_MONITOR_ENABLE="${SDPO_MONITOR_ENABLE:-1}"
SDPO_MONITOR_INTERVAL_SEC="${SDPO_MONITOR_INTERVAL_SEC:-120}"
SDPO_STALL_WARN_SEC="${SDPO_STALL_WARN_SEC:-900}"
SDPO_MONITOR_GPU_SNAPSHOT="${SDPO_MONITOR_GPU_SNAPSHOT:-1}"
SDPO_MONITOR_LOG_DIR="${SDPO_MONITOR_LOG_DIR:-${PROJECT_ROOT}/outputs/slurm/sdpo_monitor}"
SDPO_TRAINER_LOG_PATH="${SDPO_TRAINER_LOG_PATH:-}"
_SDPO_CLEANUP_COMPLETED=0
_SDPO_MONITOR_PID=""
_SDPO_TRAINER_PID=""

_resolve_slurm_job_id() {
  local job_id="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
  if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${job_id}"
    return 0
  fi
  return 1
}

_collect_slurm_job_ray_pids() {
  local job_id="$1"
  if [[ -z "${job_id}" ]] || ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi

  # Include Ray core daemons/workers plus vLLM engine/resource-tracker children
  # that can outlive the trainer when a Slurm job fails mid-shutdown.
  local process_pattern
  process_pattern="ray::|raylet|gcs_server|dashboard|runtime_env_agent|default_worker|plasma_store|VLLM::EngineCore|multiprocessing\\.resource_tracker"

  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    [[ "${pid}" != "$$" ]] || continue
    [[ -r "/proc/${pid}/environ" ]] || continue
    if tr '\0' '\n' <"/proc/${pid}/environ" \
      | grep -Eq "^SLURM_JOB_ID=${job_id}$|^SLURM_JOBID=${job_id}$"; then
      printf '%s\n' "${pid}"
    fi
  done < <(pgrep -u "$(id -u)" -f "${process_pattern}" || true)
}

_cleanup_slurm_job_ray_processes() {
  local job_id="$1"
  [[ -n "${job_id}" ]] || return 0

  local -a pids=()
  local -a still_running=()
  local pid
  mapfile -t pids < <(_collect_slurm_job_ray_pids "${job_id}")
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "run_sdpo.sh cleanup: sending SIGTERM to ${#pids[@]} runtime process(es) for SLURM job ${job_id}."
  kill "${pids[@]}" 2>/dev/null || true
  sleep "${SDPO_CLEANUP_GRACE_SEC}"

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=("${pid}")
    fi
  done

  if [[ "${#still_running[@]}" -gt 0 ]]; then
    echo "run_sdpo.sh cleanup: force-killing ${#still_running[@]} lingering process(es)."
    kill -9 "${still_running[@]}" 2>/dev/null || true
  fi
}

_collect_sdpo_bridge_container_ids() {
  local job_id="$1"
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi

  local -a docker_cmd=(
    docker ps -aq
    --filter "label=small_swe.managed=1"
    --filter "label=small_swe.pool_name=${SDPO_CONTAINER_NAME_PREFIX}"
  )
  if [[ -n "${job_id}" ]]; then
    docker_cmd+=(--filter "label=small_swe.slurm_job_id=${job_id}")
  elif [[ -n "${SDPO_RUN_LABEL:-}" ]]; then
    docker_cmd+=(--filter "label=small_swe.run_label=${SDPO_RUN_LABEL}")
  fi
  "${docker_cmd[@]}" 2>/dev/null || true
}

_cleanup_sdpo_bridge_containers() {
  local job_id="$1"
  if [[ "${SDPO_CONTAINER_CLEANUP_ENABLE}" != "1" ]]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi

  local -a container_ids=()
  mapfile -t container_ids < <(_collect_sdpo_bridge_container_ids "${job_id}")
  if [[ "${#container_ids[@]}" -eq 0 ]]; then
    # Fallback for older runs that did not attach cleanup labels.
    mapfile -t container_ids < <(docker ps -aq --filter "name=${SDPO_CONTAINER_NAME_PREFIX}-" 2>/dev/null || true)
  fi
  if [[ "${#container_ids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "run_sdpo.sh cleanup: removing ${#container_ids[@]} ${SDPO_CONTAINER_NAME_PREFIX} container(s)."
  docker rm -f "${container_ids[@]}" >/dev/null 2>&1 || true
}

_cleanup_sdpo_runtime_once() {
  if [[ "${_SDPO_CLEANUP_COMPLETED}" == "1" ]]; then
    return 0
  fi
  _SDPO_CLEANUP_COMPLETED=1

  if [[ "${DRY_RUN}" == "1" ]] || [[ "${SDPO_CLEANUP_ON_EXIT}" != "1" ]]; then
    return 0
  fi

  local slurm_job_id=""
  slurm_job_id="$(_resolve_slurm_job_id || true)"
  if [[ -n "${slurm_job_id}" ]]; then
    _cleanup_slurm_job_ray_processes "${slurm_job_id}"
  fi
  _cleanup_sdpo_bridge_containers "${slurm_job_id}"
}

_resolve_file_mtime_epoch() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf '0'
    return 0
  fi

  local mtime
  if mtime="$(stat -c %Y "${path}" 2>/dev/null)"; then
    printf '%s' "${mtime}"
    return 0
  fi
  if mtime="$(stat -f %m "${path}" 2>/dev/null)"; then
    printf '%s' "${mtime}"
    return 0
  fi

  printf '0'
}

_format_sdpo_process_snapshot() {
  local pid="$1"
  local snapshot
  snapshot="$(
    ps -p "${pid}" -o stat=,etimes=,%cpu=,rss=,comm= 2>/dev/null \
      | tr -s ' ' \
      | sed 's/^ //'
  )"
  if [[ -z "${snapshot}" ]]; then
    printf 'proc=unavailable'
    return 0
  fi
  printf 'proc=%s' "${snapshot}"
}

_format_sdpo_gpu_snapshot() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local snapshot
  snapshot="$(
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | tr '\n' ';' \
      | sed 's/;*$//'
  )"
  if [[ -n "${snapshot}" ]]; then
    printf '%s' "${snapshot}"
  fi
}

_start_sdpo_watchdog() {
  local trainer_pid="$1"
  local trainer_log_path="$2"
  local interval_sec="$3"
  local stall_warn_sec="$4"
  local monitor_gpu="$5"

  (
    set +e

    local last_log_epoch
    local last_warn_epoch
    local now_epoch
    local now_iso
    local log_mtime
    local idle_sec
    local process_snapshot
    local gpu_snapshot

    last_log_epoch="$(_resolve_file_mtime_epoch "${trainer_log_path}")"
    if ! [[ "${last_log_epoch}" =~ ^[0-9]+$ ]] || [[ "${last_log_epoch}" == "0" ]]; then
      last_log_epoch="$(date +%s)"
    fi
    last_warn_epoch=0

    while kill -0 "${trainer_pid}" 2>/dev/null; do
      sleep "${interval_sec}"
      if ! kill -0 "${trainer_pid}" 2>/dev/null; then
        break
      fi

      now_epoch="$(date +%s)"
      now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      log_mtime="$(_resolve_file_mtime_epoch "${trainer_log_path}")"
      if [[ "${log_mtime}" =~ ^[0-9]+$ ]] && (( log_mtime > last_log_epoch )); then
        last_log_epoch="${log_mtime}"
      fi
      idle_sec=$(( now_epoch - last_log_epoch ))
      process_snapshot="$(_format_sdpo_process_snapshot "${trainer_pid}")"

      echo "run_sdpo.sh watchdog: ts=${now_iso} trainer_pid=${trainer_pid} ${process_snapshot} idle_log_sec=${idle_sec}"
      if [[ "${monitor_gpu}" == "1" ]]; then
        gpu_snapshot="$(_format_sdpo_gpu_snapshot)"
        if [[ -n "${gpu_snapshot}" ]]; then
          echo "run_sdpo.sh watchdog: gpu=${gpu_snapshot}"
        fi
      fi
      if (( idle_sec >= stall_warn_sec )) && (( now_epoch - last_warn_epoch >= interval_sec )); then
        echo "run_sdpo.sh watchdog WARN: no trainer log updates for ${idle_sec}s (threshold=${stall_warn_sec}s); job may be stalled."
        last_warn_epoch="${now_epoch}"
      fi
    done
  ) &
  _SDPO_MONITOR_PID="$!"
}

_stop_sdpo_watchdog() {
  local monitor_pid="${_SDPO_MONITOR_PID:-}"
  if [[ -z "${monitor_pid}" ]]; then
    return 0
  fi
  if kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  _SDPO_MONITOR_PID=""
}

_terminate_sdpo_trainer_if_running() {
  local trainer_pid="${_SDPO_TRAINER_PID:-}"
  if [[ -z "${trainer_pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${trainer_pid}" 2>/dev/null; then
    return 0
  fi
  kill "${trainer_pid}" 2>/dev/null || true
  sleep 2
  if kill -0 "${trainer_pid}" 2>/dev/null; then
    kill -9 "${trainer_pid}" 2>/dev/null || true
  fi
}

_on_sdpo_exit() {
  local exit_code=$?
  _stop_sdpo_watchdog
  _cleanup_sdpo_runtime_once
  return "${exit_code}"
}

_on_sdpo_int() {
  trap - EXIT INT TERM
  _stop_sdpo_watchdog
  _terminate_sdpo_trainer_if_running
  _cleanup_sdpo_runtime_once
  exit 130
}

_on_sdpo_term() {
  trap - EXIT INT TERM
  _stop_sdpo_watchdog
  _terminate_sdpo_trainer_if_running
  _cleanup_sdpo_runtime_once
  exit 143
}

trap _on_sdpo_exit EXIT
trap _on_sdpo_int INT
trap _on_sdpo_term TERM

# Some cluster environments export ROCR/HIP selectors even on CUDA nodes.
# verl workers reject simultaneous ROCR + CUDA/HIP visibility variables.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}" ]]; then
  unset ROCR_VISIBLE_DEVICES
fi

# Ray+verl SDPO workers rely on explicit local-rank device binding.
# Keep CUDA visibility under launcher control by default so runtime patches can
# assign `torch.cuda.set_device(LOCAL_RANK)` deterministically per worker.
SDPO_RAY_FORCE_NOSET_VISIBLE_DEVICES="${SDPO_RAY_FORCE_NOSET_VISIBLE_DEVICES:-1}"
if [[ "${SDPO_RAY_FORCE_NOSET_VISIBLE_DEVICES}" == "1" ]]; then
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
fi

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

_count_visible_gpus() {
  local devices="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}"
  if [[ -n "${devices}" ]]; then
    IFS=',' read -r -a _gpu_items <<< "${devices}"
    printf '%d' "${#_gpu_items[@]}"
    return 0
  fi

  local slurm_gpus="${SLURM_GPUS_ON_NODE:-}"
  if [[ -z "${slurm_gpus}" ]]; then
    slurm_gpus="${SLURM_JOB_GPUS:-}"
  fi
  if [[ -n "${slurm_gpus}" ]]; then
    if [[ "${slurm_gpus}" =~ ^[0-9]+$ ]]; then
      printf '%s' "${slurm_gpus}"
      return 0
    fi
    if [[ "${slurm_gpus}" =~ ([0-9]+) ]]; then
      printf '%s' "${BASH_REMATCH[1]}"
      return 0
    fi
  fi

  printf '0'
}

_resolve_sdpo_ray_num_cpus() {
  if [[ -n "${SDPO_RAY_NUM_CPUS:-}" ]]; then
    printf '%s' "${SDPO_RAY_NUM_CPUS}"
    return 0
  fi

  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && [[ "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${SLURM_CPUS_PER_TASK}"
    return 0
  fi

  if [[ -n "${SLURM_CPUS_PER_GPU:-}" ]] && [[ "${SLURM_CPUS_PER_GPU}" =~ ^[0-9]+$ ]]; then
    local gpu_count
    gpu_count="$(_count_visible_gpus)"
    if [[ "${gpu_count}" =~ ^[0-9]+$ ]] && [[ "${gpu_count}" -gt 0 ]]; then
      printf '%d' "$((SLURM_CPUS_PER_GPU * gpu_count))"
      return 0
    fi
  fi

  if [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
    if [[ "${SLURM_JOB_CPUS_PER_NODE}" =~ ^([0-9]+) ]]; then
      printf '%s' "${BASH_REMATCH[1]}"
      return 0
    fi
  fi

  "${PYTHON_BIN}" - <<'PY'
import multiprocessing
import os

try:
    affinity_count = len(os.sched_getaffinity(0))
except Exception:
    affinity_count = 0

if affinity_count > 0:
    print(affinity_count)
else:
    print(multiprocessing.cpu_count())
PY
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

_resolve_sdpo_dataset_overrides_from_cache() {
  "${PYTHON_BIN}" - "${SDPO_TASK_CACHE_DIR}" "${DRY_RUN}" <<'PY'
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1])
dry_run = str(sys.argv[2]).strip() == "1"

def _emit(train_path: Path, val_path: Path) -> None:
    print(f"data.train_files={train_path}")
    print(f"data.val_files={val_path}")

def _canonical_paths() -> tuple[Path, Path]:
    return cache_dir / "train.parquet", cache_dir / "val.parquet"

if cache_dir.is_dir():
    # Preferred fixed filenames for turn-level SDPO preload output.
    preferred_pairs = (
        (cache_dir / "train.parquet", cache_dir / "val.parquet"),
        (cache_dir / "turn_sdpo_train.parquet", cache_dir / "turn_sdpo_val.parquet"),
    )
    for train_path, val_path in preferred_pairs:
        if train_path.is_file() and val_path.is_file():
            _emit(train_path, val_path)
            raise SystemExit(0)

    # Backward compatibility for deterministic split-cache naming.
    val_by_prefix: dict[str, Path] = {}
    for val_path in cache_dir.glob("*_val.parquet"):
        if not val_path.is_file() or not val_path.name.endswith("_val.parquet"):
            continue
        prefix = val_path.name[: -len("_val.parquet")]
        current = val_by_prefix.get(prefix)
        if current is None or val_path.stat().st_mtime > current.stat().st_mtime:
            val_by_prefix[prefix] = val_path

    train_candidates = sorted(
        (
            train_path
            for train_path in cache_dir.glob("*_train.parquet")
            if train_path.is_file() and train_path.name.endswith("_train.parquet")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for train_path in train_candidates:
        prefix = train_path.name[: -len("_train.parquet")]
        val_path = val_by_prefix.get(prefix)
        if val_path is not None:
            _emit(train_path, val_path)
            raise SystemExit(0)

    all_parquet_files = sorted(
        (path for path in cache_dir.glob("*.parquet") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if len(all_parquet_files) == 1:
        only_path = all_parquet_files[0]
        _emit(only_path, only_path)
        raise SystemExit(0)

if dry_run:
    train_path, val_path = _canonical_paths()
    _emit(train_path, val_path)
    raise SystemExit(0)

if not cache_dir.is_dir():
    raise SystemExit(
        f"SDPO task cache directory does not exist: {cache_dir}. "
        "Run the preload script first or pass explicit data.train_files/data.val_files overrides."
    )

raise SystemExit(
    "Unable to resolve preloaded SDPO parquet files from "
    f"{cache_dir}. Expected either train.parquet+val.parquet, "
    "turn_sdpo_train.parquet+turn_sdpo_val.parquet, or a matching *_train/_val pair."
)
PY
}

_validate_sdpo_parquet_schema() {
  local parquet_path="$1"
  "${PYTHON_BIN}" - "${parquet_path}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(f"Unable to validate parquet schema for {path}: pyarrow unavailable ({exc})")

schema = pq.read_schema(path)
columns = {str(name) for name in schema.names}
required = {"prompt", "task_id", "image_name", "data_source", "reward_model"}
missing = sorted(required - columns)
if missing:
    raise SystemExit(
        f"Parquet schema missing required columns for SDPO runtime: {', '.join(missing)} (file={path})"
    )
PY
}

AUTO_OVERRIDES=()

if ! _has_override_with_prefix "ray_kwargs.ray_init.num_cpus" "$@"; then
  SDPO_RAY_NUM_CPUS_VALUE="$(_resolve_sdpo_ray_num_cpus)"
  if [[ -n "${SDPO_RAY_NUM_CPUS_VALUE}" ]]; then
    AUTO_OVERRIDES+=("ray_kwargs.ray_init.num_cpus=${SDPO_RAY_NUM_CPUS_VALUE}")
  fi
fi

if ! _has_override_with_prefix "data.apply_chat_template_kwargs.enable_thinking" "$@"; then
  AUTO_OVERRIDES+=("++data.apply_chat_template_kwargs.enable_thinking=false")
fi

if ! _has_override_with_prefix "data.filter_overlong_prompts" "$@"; then
  AUTO_OVERRIDES+=("data.filter_overlong_prompts=false")
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
    mapfile -t SDPO_DATA_OVERRIDES < <(_resolve_sdpo_dataset_overrides_from_cache)
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
    if [[ "${DRY_RUN}" -ne 1 ]]; then
      if ! _validate_sdpo_parquet_schema "${value}"; then
        echo "Resolved SDPO parquet is incompatible with current runtime: ${value}"
        echo "Regenerate cache with force refresh (for example: python -m env.preload_sdpo_dataset --emit-split --force-refresh --cache-dir data/sdpo_task_cache)."
        exit 1
      fi
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

if [[ "${SDPO_MONITOR_ENABLE}" != "0" && "${SDPO_MONITOR_ENABLE}" != "1" ]]; then
  echo "SDPO_MONITOR_ENABLE must be 0 or 1 (got: ${SDPO_MONITOR_ENABLE})."
  exit 1
fi
if ! [[ "${SDPO_MONITOR_INTERVAL_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SDPO_MONITOR_INTERVAL_SEC must be a positive integer (got: ${SDPO_MONITOR_INTERVAL_SEC})."
  exit 1
fi
if ! [[ "${SDPO_STALL_WARN_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SDPO_STALL_WARN_SEC must be a positive integer (got: ${SDPO_STALL_WARN_SEC})."
  exit 1
fi
if [[ "${SDPO_MONITOR_GPU_SNAPSHOT}" != "0" && "${SDPO_MONITOR_GPU_SNAPSHOT}" != "1" ]]; then
  echo "SDPO_MONITOR_GPU_SNAPSHOT must be 0 or 1 (got: ${SDPO_MONITOR_GPU_SNAPSHOT})."
  exit 1
fi
if (( SDPO_STALL_WARN_SEC < SDPO_MONITOR_INTERVAL_SEC )); then
  SDPO_STALL_WARN_SEC="${SDPO_MONITOR_INTERVAL_SEC}"
fi

if [[ -z "${SDPO_TRAINER_LOG_PATH}" ]]; then
  SDPO_TRAINER_LOG_PATH="${SDPO_MONITOR_LOG_DIR}/${SDPO_TASK_NAME}_${SDPO_RUN_LABEL}.trainer.log"
fi

RESOLVED_TOTAL_STEPS="$(_extract_override_value "trainer.total_training_steps" "${CMD[@]}" || true)"
RESOLVED_CHECKPOINT="$(_extract_override_value "actor_rollout_ref.model.path" "${CMD[@]}" || true)"
RESOLVED_TRAIN_FILES="$(_extract_override_value "data.train_files" "${CMD[@]}" || true)"
RESOLVED_VAL_FILES="$(_extract_override_value "data.val_files" "${CMD[@]}" || true)"

echo "run_sdpo.sh launch: task=${TASK} experiment=${EXPERIMENT} slurm_job_id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "run_sdpo.sh launch: trainer_module=${SDPO_TRAINER_MODULE} total_steps=${RESOLVED_TOTAL_STEPS:-<config-default>}"
echo "run_sdpo.sh launch: checkpoint=${RESOLVED_CHECKPOINT:-<config-default>}"
echo "run_sdpo.sh launch: train_files=${RESOLVED_TRAIN_FILES:-<config-default>} val_files=${RESOLVED_VAL_FILES:-<config-default>}"

if [[ "${SDPO_MONITOR_ENABLE}" == "1" ]]; then
  mkdir -p "$(dirname "${SDPO_TRAINER_LOG_PATH}")"
  : > "${SDPO_TRAINER_LOG_PATH}"
  echo "run_sdpo.sh watchdog: enabled interval=${SDPO_MONITOR_INTERVAL_SEC}s stall_warn=${SDPO_STALL_WARN_SEC}s trainer_log=${SDPO_TRAINER_LOG_PATH}"
  set +e
  "${CMD[@]}" > >(tee -a "${SDPO_TRAINER_LOG_PATH}") 2>&1 &
  _SDPO_TRAINER_PID="$!"
  _start_sdpo_watchdog "${_SDPO_TRAINER_PID}" "${SDPO_TRAINER_LOG_PATH}" "${SDPO_MONITOR_INTERVAL_SEC}" "${SDPO_STALL_WARN_SEC}" "${SDPO_MONITOR_GPU_SNAPSHOT}"
  wait "${_SDPO_TRAINER_PID}"
  TRAINER_EXIT_CODE=$?
  set -e
  _SDPO_TRAINER_PID=""
  _stop_sdpo_watchdog
else
  echo "run_sdpo.sh watchdog: disabled (SDPO_MONITOR_ENABLE=0)"
  set +e
  "${CMD[@]}"
  TRAINER_EXIT_CODE=$?
  set -e
fi

exit "${TRAINER_EXIT_CODE}"
