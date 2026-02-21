#!/usr/bin/env bash
set -euo pipefail

if ! python -c "import verl" >/dev/null 2>&1; then
  echo "verl is not installed. Install SDPO/verl and retry."
  echo "  pip install -e \".[train]\""
  exit 1
fi

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs/verl" && pwd)"
python -m verl.trainer.main_ppo \
  --config-name rft_swe \
  --config-dir "${CONFIG_DIR}" \
  "$@"
