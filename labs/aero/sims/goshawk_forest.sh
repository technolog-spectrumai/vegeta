#!/usr/bin/env bash
# GOSHAWK through a wood: the flights of every stand density and speed under one progress bar.
#   labs/aero/sims/goshawk_forest.sh -j 12   # 240 flights
#   labs/aero/sims/goshawk_forest.sh --help | --dry-run
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
# one maths thread per process (numpy BLAS would start one per core in every worker)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/goshawk_forest.py" "$@"
