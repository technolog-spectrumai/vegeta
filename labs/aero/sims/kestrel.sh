#!/usr/bin/env bash
# Kestrel-4Q-2N (the net launcher in the pod, labs/aero/designs/kestrel_netter.py): every simulation under one progress bar.
#   labs/aero/sims/kestrel.sh -j 8                    # CFD (stall, sweep, cruise, transitions), then FEA, then the stability checks
#   labs/aero/sims/kestrel.sh -j 8 --cfd              # one phase: --cfd | --fea | --stability (any combination, in that order)
#   labs/aero/sims/kestrel.sh -j 8 --cfd --quality fine
#   labs/aero/sims/kestrel.sh --dry-run               # list the cases and jobs, run nothing
#   labs/aero/sims/kestrel.sh --help                  # all options (kestrel_run.py)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$ROOT"
exec nice -n 5 "$PY" "$HERE/kestrel_run.py" --python "$PY" "$@"
