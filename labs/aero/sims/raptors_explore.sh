#!/usr/bin/env bash
# RAPTORS exploration: the product lines BUTEO, ARPELLA, ASTOR, SPARFELL around ARIGO-1H, with the launch-success and
# operating-wind models. No CFD. Restartable. Run  labs/aero/sims/raptors_explore.sh --help  for how.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/raptors_explore.py" "$@"
