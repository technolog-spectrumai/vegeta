#!/usr/bin/env bash
# Transonic carbon prop + fan (and notebook 25's subsonic ones): fea. Help: scripts/transonic/fea.sh --help
#   scripts/transonic/fea.sh -j 8                                    all four parts, 8 cores
#   scripts/transonic/fea.sh -j 8 --parts transonic_prop transonic_fan
# Output: output/transonic/runs/fea_<time>_<host>.yaml (+ PNG charts). Needs CalculiX + gmsh (FEA, vib), OpenFOAM v2412 (CFD).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export PYTHONPATH="$ROOT/notebooks/designs:$(ls -d "$ROOT"/vegeta-*/src | tr '\n' ':')${PYTHONPATH:-}"
cd "$ROOT" && exec "$PY" "$HERE/transonic_runs.py" fea "$@"
