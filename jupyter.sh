#!/usr/bin/env bash
# Start JupyterLab from Vegeta's virtual environment in notebooks/ (no activation needed).
# Extra arguments go to `jupyter lab`, e.g. ./jupyter.sh --port 8890
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VEGETA_VENV:-$ROOT/.venv}"
if [ ! -x "$VENV/bin/jupyter" ]; then
  echo "No JupyterLab in $VENV — run ./install_local.sh first." >&2
  exit 1
fi
cd "$ROOT/notebooks"
exec "$VENV/bin/jupyter" lab "$@"
