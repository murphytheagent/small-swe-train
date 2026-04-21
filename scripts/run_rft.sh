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
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

_is_executable_cmd() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]]
    return
  fi
  command -v "${candidate}" >/dev/null 2>&1
}

_to_lower_ascii() {
  local value="${1:-}"
  printf '%s' "${value}" | tr '[:upper:]' '[:lower:]'
}

_run_small_swe_preflight_container_sweep() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  export SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE="${SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE:-1}"
  export SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES="${SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES:-onpolicy-task sdpo-swe-bridge}"
  bash "${SCRIPT_DIR}/preflight_sweep_stale_docker_containers.sh"
}

RFT_PROC_ROOT="${RFT_PROC_ROOT:-/proc}"
RFT_CLEANUP_ON_EXIT="${RFT_CLEANUP_ON_EXIT:-1}"
RFT_CLEANUP_DRAIN_SEC="${RFT_CLEANUP_DRAIN_SEC:-30}"
RFT_CLEANUP_GRACE_SEC="${RFT_CLEANUP_GRACE_SEC:-5}"
_RFT_CLEANUP_COMPLETED=0

_resolve_slurm_job_id() {
  local job_id="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
  if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${job_id}"
    return 0
  fi
  return 1
}

_pid_matches_slurm_job() {
  local pid="$1"
  local job_id="$2"
  [[ -n "${job_id}" ]] || return 1
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  local environ_path="${RFT_PROC_ROOT}/${pid}/environ"
  [[ -r "${environ_path}" ]] || return 1
  tr '\0' '\n' <"${environ_path}" | grep -Eq "^SLURM_JOB_ID=${job_id}$|^SLURM_JOBID=${job_id}$"
}

_collect_rft_job_runtime_pids() {
  local job_id="$1"
  if [[ -z "${job_id}" ]] || ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi

  local process_pattern
  process_pattern="trainer\\.rft_runtime_loop|trainer\\.vllm_api_server_entry|vllm\\.entrypoints\\.openai\\.api_server|VLLM::EngineCore|torch\\.distributed\\.run|verl_integration\\.fsdp_sft_trainer_entry|multiprocessing\\.resource_tracker"

  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    [[ "${pid}" != "$$" ]] || continue
    if _pid_matches_slurm_job "${pid}" "${job_id}"; then
      printf '%s\n' "${pid}"
    fi
  done < <(pgrep -u "$(id -u)" -f "${process_pattern}" || true)
}

_collect_live_slurm_job_pids() {
  local job_id="$1"
  shift || true
  local pid
  for pid in "$@"; do
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" 2>/dev/null && _pid_matches_slurm_job "${pid}" "${job_id}"; then
      printf '%s\n' "${pid}"
    fi
  done
}

_cleanup_rft_runtime_processes() {
  local job_id="$1"
  [[ -n "${job_id}" ]] || return 0

  local -a pids=()
  local -a pids_after_drain=()
  local -a still_running=()
  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    pids+=("${pid}")
  done < <(_collect_rft_job_runtime_pids "${job_id}")
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  local drain_sec=0
  if [[ "${RFT_CLEANUP_DRAIN_SEC}" =~ ^[0-9]+$ ]]; then
    drain_sec=$((10#${RFT_CLEANUP_DRAIN_SEC}))
  fi
  if (( drain_sec > 0 )); then
    local drain_deadline_epoch
    local now_epoch
    drain_deadline_epoch=$(( $(date +%s) + drain_sec ))
    pids_after_drain=("${pids[@]}")
    while [[ "${#pids_after_drain[@]}" -gt 0 ]]; do
      now_epoch="$(date +%s)"
      if [[ "${now_epoch}" -ge "${drain_deadline_epoch}" ]]; then
        break
      fi
      local -a next_still_running=()
      for pid in "${pids_after_drain[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          next_still_running+=("${pid}")
        fi
      done
      pids_after_drain=("${next_still_running[@]}")
      if [[ "${#pids_after_drain[@]}" -eq 0 ]]; then
        break
      fi
      sleep 1
    done
    if [[ "${#pids_after_drain[@]}" -eq 0 ]]; then
      echo "run_rft.sh cleanup: all runtime processes exited during ${drain_sec}s drain window for SLURM job ${job_id}."
      return 0
    fi
    pids=("${pids_after_drain[@]}")
  fi

  local -a verified_pids=()
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    verified_pids+=("${pid}")
  done < <(_collect_live_slurm_job_pids "${job_id}" "${pids[@]}")
  if [[ "${#verified_pids[@]}" -eq 0 ]]; then
    echo "run_rft.sh cleanup: no matching runtime processes remained after drain for SLURM job ${job_id}."
    return 0
  fi
  pids=("${verified_pids[@]}")

  echo "run_rft.sh cleanup: sending SIGTERM to ${#pids[@]} runtime process(es) for SLURM job ${job_id}."
  kill "${pids[@]}" 2>/dev/null || true
  sleep "${RFT_CLEANUP_GRACE_SEC}"

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    still_running+=("${pid}")
  done < <(_collect_live_slurm_job_pids "${job_id}" "${pids[@]}")

  if [[ "${#still_running[@]}" -gt 0 ]]; then
    echo "run_rft.sh cleanup: force-killing ${#still_running[@]} lingering process(es)."
    kill -9 "${still_running[@]}" 2>/dev/null || true
  fi
}

_cleanup_rft_runtime_once() {
  if [[ "${_RFT_CLEANUP_COMPLETED}" == "1" ]]; then
    return 0
  fi
  _RFT_CLEANUP_COMPLETED=1

  if [[ "${DRY_RUN}" == "1" ]] || [[ "${RFT_CLEANUP_ON_EXIT}" != "1" ]]; then
    return 0
  fi

  local slurm_job_id=""
  slurm_job_id="$(_resolve_slurm_job_id || true)"
  if [[ -n "${slurm_job_id}" ]]; then
    _cleanup_rft_runtime_processes "${slurm_job_id}"
  fi
}

_on_rft_exit() {
  local exit_code=$?
  _cleanup_rft_runtime_once
  return "${exit_code}"
}

_on_rft_int() {
  trap - EXIT INT TERM
  _cleanup_rft_runtime_once
  exit 130
}

_on_rft_term() {
  trap - EXIT INT TERM
  _cleanup_rft_runtime_once
  exit 143
}

trap _on_rft_exit EXIT
trap _on_rft_int INT
trap _on_rft_term TERM

_check_managed_vllm_bind_target_available() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if [[ "${RFT_RUNTIME_MODE}" != "loop" ]]; then
    return 0
  fi
  if [[ "${RFT_MANAGE_VLLM}" == "0" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${SMALL_SWE_VLLM_BASE_URL}" <<'PY'
import errno
import socket
import sys
from urllib.parse import urlsplit

url = sys.argv[1]
parsed = urlsplit(url)
if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
    print(
        "SMALL_SWE_VLLM_BASE_URL must include scheme, host, and port "
        f"(got {url!r}).",
        file=sys.stderr,
    )
    sys.exit(2)

host = parsed.hostname
port = parsed.port
try:
    addr_infos = socket.getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
except socket.gaierror as exc:
    print(
        f"Unable to resolve SMALL_SWE_VLLM_BASE_URL host {host!r}: {exc}.",
        file=sys.stderr,
    )
    sys.exit(2)

seen: set[tuple[int, object]] = set()
bind_errors: list[str] = []
address_in_use = False

for family, socktype, proto, _canonname, sockaddr in addr_infos:
    key = (family, sockaddr)
    if key in seen:
        continue
    seen.add(key)
    sock = socket.socket(family, socktype, proto)
    try:
        sock.bind(sockaddr)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            address_in_use = True
        elif exc.errno not in {errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT}:
            bind_errors.append(f"{sockaddr}: {exc}")
    else:
        sys.exit(0)
    finally:
        sock.close()

if address_in_use:
    print(
        "Managed vLLM launch target is already in use at "
        f"{host}:{port}. Choose a free SMALL_SWE_VLLM_BASE_URL port, or set "
        "RFT_MANAGE_VLLM=0 if you intend to reuse an existing vLLM server.",
        file=sys.stderr,
    )
    sys.exit(1)

if bind_errors:
    print(
        "Unable to validate managed vLLM launch target "
        f"{host}:{port}: {'; '.join(bind_errors)}",
        file=sys.stderr,
    )
    sys.exit(2)

print(
    f"Unable to validate managed vLLM launch target {host}:{port}.",
    file=sys.stderr,
)
sys.exit(2)
PY
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

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(_detect_available_gpu_count)"
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer (got: ${NPROC_PER_NODE})."
  exit 1
fi
export NPROC_PER_NODE
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  OMP_NUM_THREADS=1
  if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
    OMP_NUM_THREADS="$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))"
    if (( OMP_NUM_THREADS < 1 )); then
      OMP_NUM_THREADS=1
    fi
  fi
  export OMP_NUM_THREADS
fi
NNODES="${NNODES:-1}"
# Grounded defaults:
# - verl SFT trainer entrypoint: https://github.com/lasgroup/SDPO/blob/main/verl/trainer/fsdp_sft_trainer.py
# - local wrapper keeps verl behavior but guards flash-attn portability:
#   src/verl_integration/fsdp_sft_trainer_entry.py
# - vLLM OpenAI server entrypoint: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
# - Wrapper module delegates to vLLM entrypoint with flash-attn ABI guard:
#   src/trainer/vllm_api_server_entry.py
RFT_TRAINER_MODULE="${RFT_TRAINER_MODULE:-verl_integration.fsdp_sft_trainer_entry}"
RFT_TASK_NAME="${RFT_TASK_NAME:-small-swe-rft}"
RFT_RUNTIME_MODE="${RFT_RUNTIME_MODE:-loop}"
RFT_MANAGE_VLLM="${RFT_MANAGE_VLLM:-1}"
RFT_VLLM_LAUNCH_MODULE="${RFT_VLLM_LAUNCH_MODULE:-trainer.vllm_api_server_entry}"
RFT_VLLM_READY_TIMEOUT_SEC="${RFT_VLLM_READY_TIMEOUT_SEC:-180}"
RFT_VLLM_STOP_TIMEOUT_SEC="${RFT_VLLM_STOP_TIMEOUT_SEC:-30}"
RFT_VLLM_EXTRA_ARGS="${RFT_VLLM_EXTRA_ARGS:-}"
RFT_VLLM_GPU_MEMORY_UTILIZATION="${RFT_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
RFT_VLLM_TP_SIZE="${RFT_VLLM_TP_SIZE:-}"
RFT_VLLM_DP_SIZE="${RFT_VLLM_DP_SIZE:-}"
RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS="${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS:-}"
RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT="${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT:-}"
RFT_EVAL_SPLIT_FRACTION="${RFT_EVAL_SPLIT_FRACTION:-}"
RFT_EVAL_MIN_ROWS="${RFT_EVAL_MIN_ROWS:-}"
RFT_STAGE_NAME="${RFT_STAGE_NAME:-format_rft}"
RFT_DATA_CONFIG_NAME="${RFT_DATA_CONFIG_NAME:-on_policy_swe_smith}"
RFT_TURN_GENERATOR_MODE="${RFT_TURN_GENERATOR_MODE:-default}"

_load_rft_runtime_defaults() {
  "${PYTHON_BIN}" - <<'PY'
import os
from collections.abc import Mapping

from config import (
    adaptation_defaults,
    DEFAULT_TRAINING_MODEL_NAME,
    on_policy_runtime_defaults,
    rft_handoff_defaults,
    resolve_rft_collector_max_in_flight_default,
    resolve_rft_vllm_parallel_defaults,
    rft_runtime_defaults,
)

runtime = rft_runtime_defaults()
loop = runtime.get("loop")
if not isinstance(loop, Mapping):
    raise ValueError("`rft_runtime.loop` must be configured as a mapping.")
vllm = runtime.get("vllm")
if not isinstance(vllm, Mapping):
    raise ValueError("`rft_runtime.vllm` must be configured as a mapping.")
on_policy = on_policy_runtime_defaults()
if not isinstance(on_policy, Mapping):
    raise ValueError("`on_policy` must be configured as a mapping.")
handoff = rft_handoff_defaults()
if not isinstance(handoff, Mapping):
    raise ValueError("`rft_handoff` must be configured as a mapping.")
adaptation = adaptation_defaults()
if not isinstance(adaptation, Mapping):
    raise ValueError("`adaptation` must be configured as a mapping.")


def _required_positive_int(value, *, label):
    if isinstance(value, bool):
        raise ValueError(f"`{label}` must be an integer >= 1.")
    if isinstance(value, int) and value >= 1:
        return value
    raise ValueError(f"`{label}` must be an integer >= 1.")


def _required_number(value, *, label):
    if isinstance(value, bool):
        raise ValueError(f"`{label}` must be a finite number.")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"`{label}` must be a finite number.")


def _required_non_empty_str(value, *, label):
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    raise ValueError(f"`{label}` must be a non-empty string.")


def _parse_positive_int(value, fallback):
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 1 else fallback


steps = _required_positive_int(loop.get("steps"), label="rft_runtime.loop.steps")
samples_per_task = _required_positive_int(
    loop.get("samples_per_task"),
    label="rft_runtime.loop.samples_per_task",
)
task_batch_size = _required_positive_int(
    loop.get("task_batch_size"),
    label="rft_runtime.loop.task_batch_size",
)
collector_max_in_flight_tasks = resolve_rft_collector_max_in_flight_default(
    task_batch_size=task_batch_size
)
sft_num_epoch_per_batch = _required_positive_int(
    loop.get("sft_num_epoch_per_batch"),
    label="rft_runtime.loop.sft_num_epoch_per_batch",
)
checkpoint_keep_last = _required_positive_int(
    loop.get("checkpoint_keep_last"),
    label="rft_runtime.loop.checkpoint_keep_last",
)
train_batch_size = _required_positive_int(
    loop.get("train_batch_size"),
    label="rft_runtime.loop.train_batch_size",
)
eval_split_fraction = _required_number(
    loop.get("eval_split_fraction", 0.1),
    label="rft_runtime.loop.eval_split_fraction",
)
if eval_split_fraction < 0.0 or eval_split_fraction >= 1.0:
    raise ValueError("`rft_runtime.loop.eval_split_fraction` must satisfy 0.0 <= value < 1.0.")
eval_min_rows = _required_positive_int(
    loop.get("eval_min_rows", 1),
    label="rft_runtime.loop.eval_min_rows",
)
nproc_per_node = _parse_positive_int(os.environ.get("NPROC_PER_NODE"), 1)
default_tp, default_dp = resolve_rft_vllm_parallel_defaults(nproc_per_node=nproc_per_node)

base_url = _required_non_empty_str(
    vllm.get("base_url"),
    label="rft_runtime.vllm.base_url",
)
model_name = _required_non_empty_str(
    vllm.get("model_name", DEFAULT_TRAINING_MODEL_NAME),
    label="rft_runtime.vllm.model_name",
)
request_timeout_sec = _required_positive_int(
    vllm.get("request_timeout_sec"),
    label="rft_runtime.vllm.request_timeout_sec",
)
max_tokens = _required_positive_int(
    vllm.get("max_tokens"),
    label="rft_runtime.vllm.max_tokens",
)
temperature = _required_number(
    vllm.get("temperature"),
    label="rft_runtime.vllm.temperature",
)
top_p = _required_number(
    vllm.get("top_p"),
    label="rft_runtime.vllm.top_p",
)
default_max_turns_per_attempt = _required_positive_int(
    on_policy.get("max_turns_per_attempt"),
    label="on_policy.max_turns_per_attempt",
)
default_max_sequence_length = _required_positive_int(
    handoff.get("max_sequence_length"),
    label="rft_handoff.max_sequence_length",
)
adaptation_mode = _required_non_empty_str(
    adaptation.get("mode"),
    label="adaptation.mode",
).lower()
if adaptation_mode != "lora":
    raise ValueError(
        "run_rft.sh currently supports only adaptation.mode='lora'. "
        f"Got {adaptation_mode!r}."
    )

target_modules_raw = adaptation.get("target_modules")
if not isinstance(target_modules_raw, list) or not target_modules_raw:
    raise ValueError("`adaptation.target_modules` must be a non-empty list of strings.")
target_modules: list[str] = []
for item in target_modules_raw:
    if not isinstance(item, str) or not item.strip():
        raise ValueError("`adaptation.target_modules` must contain non-empty strings.")
    target_modules.append(item.strip())
lora_rank = _required_positive_int(
    adaptation.get("lora_rank", 16),
    label="adaptation.lora_rank",
)
lora_alpha = _required_positive_int(
    adaptation.get("lora_alpha", 32),
    label="adaptation.lora_alpha",
)

compute_precision = _required_non_empty_str(
    adaptation.get("compute_precision"),
    label="adaptation.compute_precision",
).lower()
if compute_precision not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
    raise ValueError(
        "run_rft.sh supports adaptation.compute_precision in "
        "{bf16,bfloat16,fp16,float16,fp32,float32}. "
        f"Got {compute_precision!r}."
    )

print(
    steps,
    samples_per_task,
    task_batch_size,
    collector_max_in_flight_tasks,
    sft_num_epoch_per_batch,
    checkpoint_keep_last,
    train_batch_size,
    eval_split_fraction,
    eval_min_rows,
    default_tp,
    default_dp,
    base_url,
    model_name,
    request_timeout_sec,
    max_tokens,
    temperature,
    top_p,
    default_max_turns_per_attempt,
    default_max_sequence_length,
    adaptation_mode,
    compute_precision,
    lora_rank,
    lora_alpha,
    ",".join(target_modules),
)
PY
}

_resolve_direct_mode_partial_rollout_probe_validation_surface() {
  PARTIAL_ROLLOUT_PROBE_VALIDATION_DATA_CONFIG_NAME="${RFT_DATA_CONFIG_NAME}"
  PARTIAL_ROLLOUT_PROBE_VALIDATION_EVAL_SPLIT_FRACTION="${RFT_EVAL_SPLIT_FRACTION}"
  local override normalized_override override_key override_value
  for override in "$@"; do
    normalized_override="${override}"
    while [[ "${normalized_override}" == +* ]]; do
      normalized_override="${normalized_override#+}"
    done
    override_key="${normalized_override%%=*}"
    override_value="${normalized_override#*=}"
    case "${override_key}" in
      data.on_policy.data_config_name)
        PARTIAL_ROLLOUT_PROBE_VALIDATION_DATA_CONFIG_NAME="${override_value}"
        ;;
      data.on_policy.task_eval_split_fraction)
        PARTIAL_ROLLOUT_PROBE_VALIDATION_EVAL_SPLIT_FRACTION="${override_value}"
        ;;
    esac
  done
}

_validate_partial_rollout_probe_partition_surface() {
  local data_config_name="${1}"
  local eval_split_fraction="${2}"
  "${PYTHON_BIN}" - "${data_config_name}" "${eval_split_fraction}" <<'PY'
import sys

from config import resolve_on_policy_settings
from env.task_dataset import validate_partial_rollout_probe_partition_request

data_config_name = sys.argv[1].strip()
eval_split_fraction = float(sys.argv[2])
if eval_split_fraction <= 0.0:
    raise SystemExit(0)

settings = resolve_on_policy_settings(data_config_name=data_config_name)
validate_partial_rollout_probe_partition_request(
    settings.data,
    task_partition="train",
)
validate_partial_rollout_probe_partition_request(
    settings.data,
    task_partition="eval",
)
PY
}

RFT_DEFAULTS="$(_load_rft_runtime_defaults)"
read -r DEFAULT_RFT_STEPS DEFAULT_SAMPLES_PER_TASK DEFAULT_RFT_TASK_BATCH_SIZE DEFAULT_RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS DEFAULT_RFT_SFT_NUM_EPOCH_PER_BATCH DEFAULT_RFT_CHECKPOINT_KEEP_LAST DEFAULT_RFT_TRAIN_BATCH_SIZE DEFAULT_RFT_EVAL_SPLIT_FRACTION DEFAULT_RFT_EVAL_MIN_ROWS DEFAULT_VLLM_TP_SIZE DEFAULT_VLLM_DP_SIZE DEFAULT_VLLM_BASE_URL DEFAULT_VLLM_MODEL DEFAULT_VLLM_REQUEST_TIMEOUT DEFAULT_VLLM_MAX_TOKENS DEFAULT_VLLM_TEMPERATURE DEFAULT_VLLM_TOP_P DEFAULT_ON_POLICY_MAX_TURNS_PER_ATTEMPT DEFAULT_RFT_MAX_SEQUENCE_LENGTH DEFAULT_ADAPTATION_MODE DEFAULT_ADAPTATION_COMPUTE_PRECISION DEFAULT_LORA_RANK DEFAULT_LORA_ALPHA DEFAULT_LORA_TARGET_MODULES <<<"${RFT_DEFAULTS}"

RFT_STEPS="${RFT_STEPS:-${DEFAULT_RFT_STEPS}}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-${DEFAULT_SAMPLES_PER_TASK}}"
RFT_TASK_BATCH_SIZE="${RFT_TASK_BATCH_SIZE:-${DEFAULT_RFT_TASK_BATCH_SIZE}}"
RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS="${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS:-${DEFAULT_RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS}}"
RFT_SFT_NUM_EPOCH_PER_BATCH="${RFT_SFT_NUM_EPOCH_PER_BATCH:-${DEFAULT_RFT_SFT_NUM_EPOCH_PER_BATCH}}"
RFT_CHECKPOINT_KEEP_LAST="${RFT_CHECKPOINT_KEEP_LAST:-${DEFAULT_RFT_CHECKPOINT_KEEP_LAST}}"
RFT_BATCH_SIZE="${RFT_BATCH_SIZE:-$((SAMPLES_PER_TASK * RFT_TASK_BATCH_SIZE))}"
RFT_TRAIN_BATCH_SIZE="${RFT_TRAIN_BATCH_SIZE:-${DEFAULT_RFT_TRAIN_BATCH_SIZE}}"
RFT_EVAL_SPLIT_FRACTION="${RFT_EVAL_SPLIT_FRACTION:-${DEFAULT_RFT_EVAL_SPLIT_FRACTION}}"
RFT_EVAL_MIN_ROWS="${RFT_EVAL_MIN_ROWS:-${DEFAULT_RFT_EVAL_MIN_ROWS}}"
RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT="${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT:-${DEFAULT_ON_POLICY_MAX_TURNS_PER_ATTEMPT}}"
RFT_MAX_SEQUENCE_LENGTH="${RFT_MAX_SEQUENCE_LENGTH:-${DEFAULT_RFT_MAX_SEQUENCE_LENGTH}}"
RFT_ADAPTATION_MODE="${RFT_ADAPTATION_MODE:-${DEFAULT_ADAPTATION_MODE}}"
RFT_COMPUTE_PRECISION="${RFT_COMPUTE_PRECISION:-${DEFAULT_ADAPTATION_COMPUTE_PRECISION}}"
RFT_LORA_RANK="${RFT_LORA_RANK:-${DEFAULT_LORA_RANK}}"
RFT_LORA_ALPHA="${RFT_LORA_ALPHA:-${DEFAULT_LORA_ALPHA}}"
RFT_LORA_TARGET_MODULES="${RFT_LORA_TARGET_MODULES:-${DEFAULT_LORA_TARGET_MODULES}}"
RFT_OUTPUT_ROOT="${RFT_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/rft_runtime}"
RFT_RUN_TIMESTAMP="${RFT_RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  DEFAULT_RFT_RUN_LABEL="${RFT_RUN_TIMESTAMP}_job${SLURM_JOB_ID}"
else
  DEFAULT_RFT_RUN_LABEL="${RFT_RUN_TIMESTAMP}_pid$$"
fi
RFT_RUN_LABEL="${RFT_RUN_LABEL:-${DEFAULT_RFT_RUN_LABEL}}"
RFT_OUTPUT_DIR="${RFT_OUTPUT_DIR:-${RFT_OUTPUT_ROOT}/${RFT_RUN_LABEL}}"
RFT_INITIAL_MODEL="${RFT_INITIAL_MODEL:-${DEFAULT_VLLM_MODEL}}"
RFT_STAGE_NAME="$(_to_lower_ascii "${RFT_STAGE_NAME}")"
case "${RFT_STAGE_NAME}" in
  format|format_rft)
    RFT_STAGE_NAME="format_rft"
    RFT_SELECTION_REQUIRE_TERMINAL="true"
    RFT_SELECTION_REQUIRE_FORMAT_VALID="true"
    RFT_SELECTION_REQUIRE_RESOLVED="false"
    RFT_SELECTION_REJECT_ON_INVALID_FINAL_SUBMIT="true"
    RFT_VERIFY_SUBMISSIONS="false"
    ;;
  positive|positive_rft)
    RFT_STAGE_NAME="positive_rft"
    RFT_SELECTION_REQUIRE_TERMINAL="false"
    RFT_SELECTION_REQUIRE_FORMAT_VALID="false"
    RFT_SELECTION_REQUIRE_RESOLVED="true"
    RFT_SELECTION_REJECT_ON_INVALID_FINAL_SUBMIT="false"
    RFT_VERIFY_SUBMISSIONS="true"
    ;;
  *)
    echo "Unsupported RFT_STAGE_NAME=${RFT_STAGE_NAME}. Expected format_rft or positive_rft."
    exit 1
    ;;
esac

if [[ "${RFT_ADAPTATION_MODE}" != "lora" ]]; then
  echo "run_rft.sh currently supports only RFT_ADAPTATION_MODE=lora (resolved: ${RFT_ADAPTATION_MODE})."
  exit 1
fi
RFT_LORA_TARGET_MODULES_HYDRA="[${RFT_LORA_TARGET_MODULES}]"
if ! [[ "${RFT_LORA_RANK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RFT_LORA_RANK must be a positive integer (got: ${RFT_LORA_RANK})."
  exit 1
fi
if ! [[ "${RFT_LORA_ALPHA}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RFT_LORA_ALPHA must be a positive integer (got: ${RFT_LORA_ALPHA})."
  exit 1
fi
RFT_COMPUTE_PRECISION="$(_to_lower_ascii "${RFT_COMPUTE_PRECISION}")"
case "${RFT_COMPUTE_PRECISION}" in
  bf16|bfloat16)
    RFT_MODEL_DTYPE="bf16"
    ;;
  fp16|float16)
    RFT_MODEL_DTYPE="fp16"
    ;;
  fp32|float32)
    RFT_MODEL_DTYPE="fp32"
    ;;
  *)
    echo "Unsupported RFT_COMPUTE_PRECISION=${RFT_COMPUTE_PRECISION}. Expected bf16/bfloat16, fp16/float16, or fp32/float32."
    exit 1
    ;;
esac

RFT_VLLM_TP_SIZE="${RFT_VLLM_TP_SIZE:-${DEFAULT_VLLM_TP_SIZE}}"
if [[ -z "${RFT_VLLM_DP_SIZE}" ]]; then
  RFT_VLLM_DP_SIZE="${DEFAULT_VLLM_DP_SIZE}"
  if (( NPROC_PER_NODE % RFT_VLLM_TP_SIZE != 0 )); then
    RFT_VLLM_DP_SIZE="1"
  else
    MAX_RFT_VLLM_DP_SIZE="$(( NPROC_PER_NODE / RFT_VLLM_TP_SIZE ))"
    if (( RFT_VLLM_DP_SIZE < 1 )); then
      RFT_VLLM_DP_SIZE="1"
    elif (( RFT_VLLM_DP_SIZE > MAX_RFT_VLLM_DP_SIZE )); then
      RFT_VLLM_DP_SIZE="${MAX_RFT_VLLM_DP_SIZE}"
    fi
  fi
fi

if [[ -z "${RFT_VLLM_EXTRA_ARGS}" ]]; then
  RFT_VLLM_EXTRA_ARGS="--tensor-parallel-size ${RFT_VLLM_TP_SIZE}"
  if (( RFT_VLLM_DP_SIZE > 1 )); then
    RFT_VLLM_EXTRA_ARGS="${RFT_VLLM_EXTRA_ARGS} --data-parallel-size ${RFT_VLLM_DP_SIZE}"
  fi
  RFT_VLLM_EXTRA_ARGS="${RFT_VLLM_EXTRA_ARGS} --gpu-memory-utilization ${RFT_VLLM_GPU_MEMORY_UTILIZATION}"
fi

export SMALL_SWE_VLLM_BASE_URL="${SMALL_SWE_VLLM_BASE_URL:-${DEFAULT_VLLM_BASE_URL}}"
export SMALL_SWE_VLLM_MODEL="${SMALL_SWE_VLLM_MODEL:-${DEFAULT_VLLM_MODEL}}"
export SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC="${SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC:-${DEFAULT_VLLM_REQUEST_TIMEOUT}}"
export SMALL_SWE_VLLM_MAX_TOKENS="${SMALL_SWE_VLLM_MAX_TOKENS:-${DEFAULT_VLLM_MAX_TOKENS}}"
export SMALL_SWE_VLLM_TEMPERATURE="${SMALL_SWE_VLLM_TEMPERATURE:-${DEFAULT_VLLM_TEMPERATURE}}"
export SMALL_SWE_VLLM_TOP_P="${SMALL_SWE_VLLM_TOP_P:-${DEFAULT_VLLM_TOP_P}}"
export SMALL_SWE_RFT_MODEL_DTYPE="${SMALL_SWE_RFT_MODEL_DTYPE:-${RFT_MODEL_DTYPE}}"
export EXPERIMENT="${EXPERIMENT:-${RFT_TASK_NAME}}"
export SMALL_SWE_RFT_LOOP_WANDB_ENABLE="${SMALL_SWE_RFT_LOOP_WANDB_ENABLE:-1}"

PARTIAL_ROLLOUT_PROBE_VALIDATION_DATA_CONFIG_NAME="${RFT_DATA_CONFIG_NAME}"
PARTIAL_ROLLOUT_PROBE_VALIDATION_EVAL_SPLIT_FRACTION="${RFT_EVAL_SPLIT_FRACTION}"
if [[ "${RFT_RUNTIME_MODE}" == "direct" ]]; then
  _resolve_direct_mode_partial_rollout_probe_validation_surface "$@"
fi

if ! _validate_partial_rollout_probe_partition_surface \
  "${PARTIAL_ROLLOUT_PROBE_VALIDATION_DATA_CONFIG_NAME}" \
  "${PARTIAL_ROLLOUT_PROBE_VALIDATION_EVAL_SPLIT_FRACTION}"; then
  exit 1
fi

if [[ "${RFT_RUNTIME_MODE}" == "direct" ]]; then
  CMD=(
    "${PYTHON_BIN}"
    -m
    torch.distributed.run
    --standalone
    --nnodes "${NNODES}"
    --nproc_per_node "${NPROC_PER_NODE}"
    -m "${RFT_TRAINER_MODULE}"
    --config-name rft_swe
    --config-dir "${CONFIG_DIR}"
    "++data.apply_chat_template_kwargs.enable_thinking=false"
    max_model_len="${RFT_MAX_SEQUENCE_LENGTH}"
    trainer.total_epochs="${RFT_SFT_NUM_EPOCH_PER_BATCH}"
    trainer.total_training_steps="${RFT_STEPS}"
    data.train_batch_size="${RFT_TRAIN_BATCH_SIZE}"
    model.partial_pretrain="${RFT_INITIAL_MODEL}"
    model.fsdp_config.model_dtype="${RFT_MODEL_DTYPE}"
    model.lora_rank="${RFT_LORA_RANK}"
    model.lora_alpha="${RFT_LORA_ALPHA}"
    model.target_modules="${RFT_LORA_TARGET_MODULES_HYDRA}"
    actor_rollout_ref.model.lora.enable=true
    actor_rollout_ref.model.lora.rank="${RFT_LORA_RANK}"
    actor_rollout_ref.model.lora.alpha="${RFT_LORA_ALPHA}"
    actor_rollout_ref.model.lora.target_modules="${RFT_LORA_TARGET_MODULES_HYDRA}"
    ++data.on_policy.total_steps="${RFT_STEPS}"
    +data.on_policy.stage_name="${RFT_STAGE_NAME}"
    +data.on_policy.task_eval_split_fraction="${RFT_EVAL_SPLIT_FRACTION}"
    +data.on_policy.task_eval_min_rows="${RFT_EVAL_MIN_ROWS}"
    +data.on_policy.runtime_overrides.task_batch_size="${RFT_TASK_BATCH_SIZE}"
    +data.on_policy.runtime_overrides.attempts_per_task="${SAMPLES_PER_TASK}"
    +data.on_policy.runtime_overrides.max_in_flight_tasks="${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS}"
    +data.on_policy.runtime_overrides.max_turns_per_attempt="${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT}"
    +data.on_policy.runtime_overrides.verify_submissions="${RFT_VERIFY_SUBMISSIONS}"
    +data.on_policy.rft_handoff_overrides.selection.require_terminal="${RFT_SELECTION_REQUIRE_TERMINAL}"
    +data.on_policy.rft_handoff_overrides.selection.require_format_valid="${RFT_SELECTION_REQUIRE_FORMAT_VALID}"
    +data.on_policy.rft_handoff_overrides.selection.require_resolved="${RFT_SELECTION_REQUIRE_RESOLVED}"
    +data.on_policy.rft_handoff_overrides.selection.reject_on_invalid_final_submit="${RFT_SELECTION_REJECT_ON_INVALID_FINAL_SUBMIT}"
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

  _run_small_swe_preflight_container_sweep
  export TASK="${TASK:-${RFT_TASK_NAME}}"
  "${CMD[@]}"
  exit 0
fi

LOOP_CMD=(
  "${PYTHON_BIN}"
  -m trainer.rft_runtime_loop
  --project-root "${PROJECT_ROOT}"
  --config-dir "${CONFIG_DIR}"
  --config-name rft_swe
  --trainer-module "${RFT_TRAINER_MODULE}"
  --python-bin "${PYTHON_BIN}"
  --nnodes "${NNODES}"
  --nproc-per-node "${NPROC_PER_NODE}"
  --rft-steps "${RFT_STEPS}"
  --samples-per-task "${SAMPLES_PER_TASK}"
  --task-batch-size "${RFT_TASK_BATCH_SIZE}"
  --sft-num-epoch-per-batch "${RFT_SFT_NUM_EPOCH_PER_BATCH}"
  --checkpoint-keep-last "${RFT_CHECKPOINT_KEEP_LAST}"
  --train-batch-size "${RFT_TRAIN_BATCH_SIZE}"
  --eval-split-fraction "${RFT_EVAL_SPLIT_FRACTION}"
  --eval-min-rows "${RFT_EVAL_MIN_ROWS}"
  --stage-name "${RFT_STAGE_NAME}"
  --output-dir "${RFT_OUTPUT_DIR}"
  --data-config-name "${RFT_DATA_CONFIG_NAME}"
  --turn-generator-mode "${RFT_TURN_GENERATOR_MODE}"
  --initial-model "${RFT_INITIAL_MODEL}"
  --vllm-base-url "${SMALL_SWE_VLLM_BASE_URL}"
  --vllm-served-model "${SMALL_SWE_VLLM_MODEL}"
  --vllm-launch-module "${RFT_VLLM_LAUNCH_MODULE}"
  --vllm-ready-timeout-sec "${RFT_VLLM_READY_TIMEOUT_SEC}"
  --vllm-stop-timeout-sec "${RFT_VLLM_STOP_TIMEOUT_SEC}"
  --vllm-extra-args "${RFT_VLLM_EXTRA_ARGS}"
  --trainer-override "max_model_len=${RFT_MAX_SEQUENCE_LENGTH}"
  --trainer-override "model.fsdp_config.model_dtype=${RFT_MODEL_DTYPE}"
  --trainer-override "model.lora_rank=${RFT_LORA_RANK}"
  --trainer-override "model.lora_alpha=${RFT_LORA_ALPHA}"
  --trainer-override "model.target_modules=${RFT_LORA_TARGET_MODULES_HYDRA}"
  --trainer-override "actor_rollout_ref.model.lora.enable=true"
  --trainer-override "actor_rollout_ref.model.lora.rank=${RFT_LORA_RANK}"
  --trainer-override "actor_rollout_ref.model.lora.alpha=${RFT_LORA_ALPHA}"
  --trainer-override "actor_rollout_ref.model.lora.target_modules=${RFT_LORA_TARGET_MODULES_HYDRA}"
  --trainer-override "++data.apply_chat_template_kwargs.enable_thinking=false"
)

if [[ -n "${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS}" ]]; then
  LOOP_CMD+=(--collector-max-in-flight-tasks "${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS}")
fi
if [[ -n "${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT}" ]]; then
  LOOP_CMD+=(--collector-max-turns-per-attempt "${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT}")
fi

if [[ "${RFT_MANAGE_VLLM}" == "0" ]]; then
  LOOP_CMD+=(--skip-vllm-management)
fi

for override in "$@"; do
  LOOP_CMD+=(--trainer-override "${override}")
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  LOOP_CMD+=(--dry-run)
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if ! "${PYTHON_BIN}" -c "import verl" >/dev/null 2>&1; then
    echo "verl is not installed. Install SDPO/verl and retry."
    echo "  pip install -e \".[train]\""
    exit 1
  fi
fi

_check_managed_vllm_bind_target_available
_run_small_swe_preflight_container_sweep
export TASK="${TASK:-${RFT_TASK_NAME}}"
"${LOOP_CMD[@]}"
