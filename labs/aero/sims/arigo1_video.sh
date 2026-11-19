#!/usr/bin/env bash
# ARIGO-1 manoeuvre videos in MuJoCo -> labs/aero/sims/output/arigo1_video/*.mp4 + arigo1_video.yaml (paste the yaml back)
#   labs/aero/sims/arigo1_video.sh --help
#   labs/aero/sims/arigo1_video.sh                                  hybrid pusher, all six manoeuvres
#   labs/aero/sims/arigo1_video.sh --variants goshawk zero hybrid --propulsion pusher tractor twin
# Rendering: MUJOCO_GL=egl (default here; GPU), or MUJOCO_GL=osmesa (software). Needs ffmpeg.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$PY" "$HERE/arigo1_video.py" "$@"
