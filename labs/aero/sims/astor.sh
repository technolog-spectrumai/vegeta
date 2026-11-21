#!/usr/bin/env bash
# ASTOR-V / ASTOR-BOOST: every simulation (CFD, stability, net drop) under one progress bar.
#   labs/aero/sims/astor.sh -j 12            # everything
#   labs/aero/sims/astor.sh -j 12 --cfd-only | --stability-only | --netdrop-only | --skip-cfd
#   labs/aero/sims/astor.sh --help           # all options;  --dry-run  lists what it would do
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
# one maths thread per process (numpy BLAS would start one per core in every worker)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/astor_v_run.py" "$@"
