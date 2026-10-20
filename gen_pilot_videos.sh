#!/usr/bin/env bash
# Thin wrapper: runs gen_pilot_videos.py with the repository's python (see that file for the options).
cd "$(dirname "$0")" || exit 2
PY="${PYTHON:-}"; [ -z "$PY" ] && [ -x .venv/bin/python ] && PY=.venv/bin/python; [ -z "$PY" ] && PY="$(command -v python3 || command -v python)"
echo "gen_pilot_videos.sh: $PY gen_pilot_videos.py $*"
exec "$PY" gen_pilot_videos.py "$@"
