#!/usr/bin/env bash
# ASTOR-V / ASTOR-BOOST (the variable-sweep, net-dropping ASTOR), phase: stability.
# Run  labs/aero/sims/astor_stability.sh --help  for how;  --dry-run  lists what it would do;  --design dart  runs the first ASTOR's.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/astor_v_run.py" stability "$@"
