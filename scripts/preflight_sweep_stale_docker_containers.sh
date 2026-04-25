#!/usr/bin/env bash
set -euo pipefail

if [[ "${SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE:-1}" != "1" ]]; then
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  exit 0
fi
if ! command -v squeue >/dev/null 2>&1; then
  exit 0
fi

RUNTIME_USER="${USER:-${LOGNAME:-}}"
if [[ -z "${RUNTIME_USER}" ]]; then
  exit 0
fi

POOL_NAMES="${SMALL_SWE_PREFLIGHT_CONTAINER_POOL_NAMES:-onpolicy-task sdpo-swe-bridge}"
if [[ -z "${POOL_NAMES}" ]]; then
  exit 0
fi

ACTIVE_JOB_IDS="$(
  squeue -h -o "%i" 2>/dev/null || true
)"
SEEN_CONTAINER_IDS=""
stale_container_ids=()
stale_job_ids=()
stale_pool_names=()

for pool_name in ${POOL_NAMES}; do
  [[ -n "${pool_name}" ]] || continue
  while IFS=' ' read -r container_id job_id; do
    [[ -n "${container_id}" ]] || continue
    [[ -n "${job_id}" ]] || continue
    if printf '%s\n' "${ACTIVE_JOB_IDS}" | grep -Fqx -- "${job_id}"; then
      continue
    fi
    if printf '%s\n' "${SEEN_CONTAINER_IDS}" | grep -Fqx -- "${container_id}"; then
      continue
    fi
    SEEN_CONTAINER_IDS="${SEEN_CONTAINER_IDS}"$'\n'"${container_id}"
    stale_container_ids+=("${container_id}")
    stale_job_ids+=("${job_id}")
    stale_pool_names+=("${pool_name}")
  done < <(
    docker ps -a \
      --filter "label=small_swe.managed=1" \
      --filter "label=small_swe.user=${RUNTIME_USER}" \
      --filter "label=small_swe.pool_name=${pool_name}" \
      --format '{{.ID}} {{.Label "small_swe.slurm_job_id"}}' 2>/dev/null || true
  )
done

if [[ "${#stale_container_ids[@]}" -eq 0 ]]; then
  exit 0
fi

stale_job_summary="$(
  printf '%s\n' "${stale_job_ids[@]}" \
    | sort -u \
    | tr '\n' ' ' \
    | sed 's/ $//'
)"
stale_pool_summary="$(
  printf '%s\n' "${stale_pool_names[@]}" \
    | sort -u \
    | tr '\n' ' ' \
    | sed 's/ $//'
)"

echo "preflight_sweep_stale_docker_containers.sh: removing ${#stale_container_ids[@]} stale container(s) for user ${RUNTIME_USER} from pool(s) ${stale_pool_summary} and finished Slurm job(s): ${stale_job_summary}."
docker rm -f "${stale_container_ids[@]}" >/dev/null 2>&1 || true
