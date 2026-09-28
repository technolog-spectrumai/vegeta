#!/usr/bin/env bash
# Submarine with its propeller (rotor disk), submerged -> CFD -> particle movie.
# Output: scenarios/output/sub/ (sub.mp4, summary.json, case/, cad/). See scenarios/scenarios.md.
#   scenarios/sub_video.sh            1 core
#   scenarios/sub_video.sh -j 8       8 cores for the solver (MPI)
#   scenarios/sub_video.sh --dry-run  geometry + prepared case + the commands; no OpenFOAM
# Python: $VEGETA_PYTHON, else .venv/bin/python, else python3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/run_scenario.py" sub "$@"
