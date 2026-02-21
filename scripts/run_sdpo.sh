#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs/verl" && pwd)"
CMD=(
  python -m verl.trainer.main_ppo
  --config-name sdpo_swe
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

"${CMD[@]}"
