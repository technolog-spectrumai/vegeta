#!/usr/bin/env bash
# Cotesia and Vespula: CAD, BOM, yx2 export, CFD, FEA, stability. Run  labs/aero/sims/wasp_run.sh --help  for how.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/wasp_run.py" "$@"
