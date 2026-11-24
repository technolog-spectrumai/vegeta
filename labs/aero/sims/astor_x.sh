#!/usr/bin/env bash
# ASTOR-X: CFD, stability and FEA under one progress bar; one timestamped YAML per phase in sims/output/runs.
#   labs/aero/sims/astor_x.sh -j 12            # everything: CFD, then stability, then the FEA
#   labs/aero/sims/astor_x.sh -j 12 --cfd-only | --stability-only | --fea-only | --skip-cfd | --skip-fea
#   labs/aero/sims/astor_x.sh --dry-run        # list what would run;  --help  all options;  --fresh  forget saved results
# Restartable: solved CFD cases, stability jobs and FEA cases are kept and read back.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
# one maths thread per process (numpy BLAS would start one per core in every worker)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/astor_v_run.py" --design x "$@"
