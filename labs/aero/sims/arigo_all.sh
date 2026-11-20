#!/usr/bin/env bash
# The ARIGO family's analyses with one progress bar, one result file per run and charts.
# Run  labs/aero/sims/arigo_all.sh --help  for what it does and how to run it.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/arigo_all.py" "$@"
