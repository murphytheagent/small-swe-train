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
RFT_VLLM_TP_SIZE="${RFT_VLLM_TP_SIZE:-}"
RFT_VLLM_DP_SIZE="${RFT_VLLM_DP_SIZE:-}"
RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS="${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS:-}"
RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT="${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT:-}"
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

compute_precision = _required_non_empty_str(
    adaptation.get("compute_precision"),
    label="adaptation.compute_precision",
).lower()
if compute_precision not in {"bf16", "bfloat16", "fp16", "float16"}:
    raise ValueError(
        "run_rft.sh supports adaptation.compute_precision in {bf16,bfloat16,fp16,float16}. "
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
    ",".join(target_modules),
)
PY
}

RFT_DEFAULTS="$(_load_rft_runtime_defaults)"
read -r DEFAULT_RFT_STEPS DEFAULT_SAMPLES_PER_TASK DEFAULT_RFT_TASK_BATCH_SIZE DEFAULT_RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS DEFAULT_RFT_SFT_NUM_EPOCH_PER_BATCH DEFAULT_RFT_CHECKPOINT_KEEP_LAST DEFAULT_RFT_TRAIN_BATCH_SIZE DEFAULT_VLLM_TP_SIZE DEFAULT_VLLM_DP_SIZE DEFAULT_VLLM_BASE_URL DEFAULT_VLLM_MODEL DEFAULT_VLLM_REQUEST_TIMEOUT DEFAULT_VLLM_MAX_TOKENS DEFAULT_VLLM_TEMPERATURE DEFAULT_VLLM_TOP_P DEFAULT_ON_POLICY_MAX_TURNS_PER_ATTEMPT DEFAULT_RFT_MAX_SEQUENCE_LENGTH DEFAULT_ADAPTATION_MODE DEFAULT_LORA_TARGET_MODULES <<<"${RFT_DEFAULTS}"

RFT_STEPS="${RFT_STEPS:-${DEFAULT_RFT_STEPS}}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-${DEFAULT_SAMPLES_PER_TASK}}"
RFT_TASK_BATCH_SIZE="${RFT_TASK_BATCH_SIZE:-${DEFAULT_RFT_TASK_BATCH_SIZE}}"
RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS="${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS:-${DEFAULT_RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS}}"
RFT_SFT_NUM_EPOCH_PER_BATCH="${RFT_SFT_NUM_EPOCH_PER_BATCH:-${DEFAULT_RFT_SFT_NUM_EPOCH_PER_BATCH}}"
RFT_CHECKPOINT_KEEP_LAST="${RFT_CHECKPOINT_KEEP_LAST:-${DEFAULT_RFT_CHECKPOINT_KEEP_LAST}}"
RFT_BATCH_SIZE="${RFT_BATCH_SIZE:-$((SAMPLES_PER_TASK * RFT_TASK_BATCH_SIZE))}"
RFT_TRAIN_BATCH_SIZE="${RFT_TRAIN_BATCH_SIZE:-${DEFAULT_RFT_TRAIN_BATCH_SIZE}}"
RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT="${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT:-${DEFAULT_ON_POLICY_MAX_TURNS_PER_ATTEMPT}}"
RFT_MAX_SEQUENCE_LENGTH="${RFT_MAX_SEQUENCE_LENGTH:-${DEFAULT_RFT_MAX_SEQUENCE_LENGTH}}"
RFT_ADAPTATION_MODE="${RFT_ADAPTATION_MODE:-${DEFAULT_ADAPTATION_MODE}}"
RFT_LORA_TARGET_MODULES="${RFT_LORA_TARGET_MODULES:-${DEFAULT_LORA_TARGET_MODULES}}"
RFT_OUTPUT_DIR="${RFT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/rft_runtime}"
RFT_INITIAL_MODEL="${RFT_INITIAL_MODEL:-${DEFAULT_VLLM_MODEL}}"

if [[ "${RFT_ADAPTATION_MODE}" != "lora" ]]; then
  echo "run_rft.sh currently supports only RFT_ADAPTATION_MODE=lora (resolved: ${RFT_ADAPTATION_MODE})."
  exit 1
fi
RFT_LORA_TARGET_MODULES_HYDRA="[${RFT_LORA_TARGET_MODULES}]"

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
fi

export SMALL_SWE_VLLM_BASE_URL="${SMALL_SWE_VLLM_BASE_URL:-${DEFAULT_VLLM_BASE_URL}}"
export SMALL_SWE_VLLM_MODEL="${SMALL_SWE_VLLM_MODEL:-${DEFAULT_VLLM_MODEL}}"
export SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC="${SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC:-${DEFAULT_VLLM_REQUEST_TIMEOUT}}"
export SMALL_SWE_VLLM_MAX_TOKENS="${SMALL_SWE_VLLM_MAX_TOKENS:-${DEFAULT_VLLM_MAX_TOKENS}}"
export SMALL_SWE_VLLM_TEMPERATURE="${SMALL_SWE_VLLM_TEMPERATURE:-${DEFAULT_VLLM_TEMPERATURE}}"
export SMALL_SWE_VLLM_TOP_P="${SMALL_SWE_VLLM_TOP_P:-${DEFAULT_VLLM_TOP_P}}"
export EXPERIMENT="${EXPERIMENT:-${RFT_TASK_NAME}}"

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
    +max_model_len="${RFT_MAX_SEQUENCE_LENGTH}"
    data.max_length="${RFT_MAX_SEQUENCE_LENGTH}"
    trainer.total_epochs="${RFT_SFT_NUM_EPOCH_PER_BATCH}"
    trainer.total_training_steps="${RFT_STEPS}"
    data.train_batch_size="${RFT_TRAIN_BATCH_SIZE}"
    actor_rollout_ref.model.path="${RFT_INITIAL_MODEL}"
    model.partial_pretrain="${RFT_INITIAL_MODEL}"
    actor_rollout_ref.model.lora.enable=true
    actor_rollout_ref.model.lora.target_modules="${RFT_LORA_TARGET_MODULES_HYDRA}"
    ++data.on_policy.total_steps="${RFT_STEPS}"
    +data.on_policy.runtime_overrides.task_batch_size="${RFT_TASK_BATCH_SIZE}"
    +data.on_policy.runtime_overrides.attempts_per_task="${SAMPLES_PER_TASK}"
    +data.on_policy.runtime_overrides.max_in_flight_tasks="${RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS}"
    +data.on_policy.runtime_overrides.max_turns_per_attempt="${RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT}"
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
  --trainer-override "+max_model_len=${RFT_MAX_SEQUENCE_LENGTH}"
  --trainer-override "data.max_length=${RFT_MAX_SEQUENCE_LENGTH}"
  --trainer-override "actor_rollout_ref.model.path=${RFT_INITIAL_MODEL}"
  --trainer-override "actor_rollout_ref.model.lora.enable=true"
  --trainer-override "actor_rollout_ref.model.lora.target_modules=${RFT_LORA_TARGET_MODULES_HYDRA}"
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

export TASK="${TASK:-${RFT_TASK_NAME}}"
"${LOOP_CMD[@]}"
