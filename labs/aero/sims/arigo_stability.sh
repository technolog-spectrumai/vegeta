#!/usr/bin/env bash
# ARIGO PHASE 2: performance + stability (no CFD). Run  labs/aero/sims/arigo_stability.sh --help  for how.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
# one maths thread per process (numpy BLAS would start one per core in every worker and freeze the machine)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}" VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/arigo_stability.py" "$@"
