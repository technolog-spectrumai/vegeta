#!/usr/bin/env bash
# RAPTORS exploration RUN 2 (around run 1's winners): the product lines BUTEO, ARPELLA, ASTOR, SPARFELL around ARIGO-1H, with the launch-success and
# operating-wind models. No CFD. Restartable. Run  labs/aero/sims/raptors_explore2.sh --help  for how.
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
exec nice -n 5 "$PY" "$HERE/raptors_explore2.py" "$@"
