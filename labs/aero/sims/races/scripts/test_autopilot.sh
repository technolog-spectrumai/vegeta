#!/usr/bin/env bash
# Thin wrapper: runs test_autopilot.py with the repository's python (see that file for the options).
cd "$(dirname "$0")/../.." || exit 2          # the Piccolo repository (its .venv)
PY="${PYTHON:-}"; [ -z "$PY" ] && [ -x .venv/bin/python ] && PY=.venv/bin/python; [ -z "$PY" ] && PY="$(command -v python3 || command -v python)"
echo "test_autopilot.sh: $PY races/scripts/test_autopilot.py $*"
exec "$PY" races/scripts/test_autopilot.py "$@"
