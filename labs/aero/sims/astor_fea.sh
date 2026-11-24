#!/usr/bin/env bash
# HAWK 2 (ASTOR-TWIN): the wing spar at 8 g in CalculiX.
#   labs/aero/sims/astor_fea.sh -j 8         # mesh and solve
#   labs/aero/sims/astor_fea.sh --help | --dry-run
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
# one maths thread per process (numpy BLAS would start one per core in every worker)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/astor_fea.py" "$@"
