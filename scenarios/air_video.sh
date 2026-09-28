#!/usr/bin/env bash
# Whole fixed-wing aircraft with both propellers (rotor disks) in air -> CFD -> particle movie.
# Output: scenarios/output/air/ (air.mp4, summary.json, case/, cad/). See scenarios/scenarios.md.
#   scenarios/air_video.sh            1 core
#   scenarios/air_video.sh -j 8       8 cores for the solver (MPI)
#   scenarios/air_video.sh --dry-run  geometry + prepared case + the commands; no OpenFOAM
# Python: $VEGETA_PYTHON, else .venv/bin/python, else python3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/run_scenario.py" air "$@"
