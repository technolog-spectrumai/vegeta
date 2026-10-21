#!/usr/bin/env bash
# Thin wrapper: runs tune_autopilot.py with the repository's python (see that file for the options).
cd "$(dirname "$0")/../.." || exit 1          # the Piccolo repository (its .venv)
PY="${PYTHON:-}"; [ -z "$PY" ] && [ -x .venv/bin/python ] && PY=.venv/bin/python; [ -z "$PY" ] && PY="$(command -v python3 || command -v python)"
echo "tune_autopilot.sh: $PY races/scripts/tune_autopilot.py $*"
exec "$PY" races/scripts/tune_autopilot.py "$@"
