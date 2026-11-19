#!/usr/bin/env bash
# ARIGO-1 stability tests in MuJoCo -> labs/aero/sims/output/arigo1_mujoco.yaml (paste it back)
#   labs/aero/sims/arigo1_mujoco.sh --help
#   labs/aero/sims/arigo1_mujoco.sh                                 3 wings x 3 propulsions x 2 post-stall bounds (~2 min)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi

exec "$PY" "$HERE/arigo1_mujoco.py" "$@"
