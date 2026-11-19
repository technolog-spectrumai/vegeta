#!/usr/bin/env bash
# ARIGO-1 CFD stall sweeps (OpenFOAM via Aeromant). Output: labs/aero/sims/output/arigo1_cfd.yaml -> paste it back.
#   labs/aero/sims/arigo1.sh --list                               the cases, nothing run
#   labs/aero/sims/arigo1.sh --processors 8                       3 wings, pusher, 11 angles (33 cases)
#   labs/aero/sims/arigo1.sh --variants hybrid --propulsion pusher tractor twin --processors 8
# Python: $VEGETA_PYTHON, else .venv/bin/python, else python3. Needs OpenFOAM (source its bashrc first).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/arigo1_cfd.py" "$@"
