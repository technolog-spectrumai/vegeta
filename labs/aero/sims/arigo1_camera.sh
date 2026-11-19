#!/usr/bin/env bash
# ARIGO-1: propeller vs camera — synced capture, two-frame fusion, motion blur, rolling shutter (seconds).
# Output: labs/aero/sims/output/arigo1_camera.yaml (paste back) + arigo1_camera.png
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/arigo1_camera.py" "$@"
