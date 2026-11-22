#!/usr/bin/env bash
# ASTOR-T's win rate against quadcopters across the design grid (see astor_win.py --help).
#   labs/aero/sims/astor_win.sh -j 14 [--sorties N] [--only-baseline] [--fresh] [--dry-run]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec nice -n 5 "$PY" "$HERE/astor_win.py" "$@"
