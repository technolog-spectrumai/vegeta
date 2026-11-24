#!/usr/bin/env bash
# Transonic carbon fan, all stages (design, CAD, FEA, vibration, CFD) with one progress bar. Help: --help
#   scripts/transonic/transonic_fan.sh -j 12                  everything
#   scripts/transonic/transonic_fan.sh -j 12 --skip cfd       no OpenFOAM;  --compare adds the old 90 mm EDF
# Output: output/transonic/runs/fan_<time>_<host>.yaml (+ the stage YAMLs and PNGs). Needs CalculiX + gmsh, OpenFOAM v2412.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export PYTHONPATH="$ROOT/notebooks/designs:$(ls -d "$ROOT"/vegeta-*/src | tr '\n' ':')${PYTHONPATH:-}"
cd "$ROOT" && exec "$PY" "$HERE/transonic_fan.py" "$@"
