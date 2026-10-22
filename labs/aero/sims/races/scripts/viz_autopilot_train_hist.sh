#!/usr/bin/env bash
# Thin wrapper: runs viz_autopilot_train_hist.py with the repository's python (see that file for the options).
cd "$(dirname "$0")/../.." || exit 2          # the Piccolo repository (its .venv)
PY="${PYTHON:-}"; [ -z "$PY" ] && [ -x .venv/bin/python ] && PY=.venv/bin/python; [ -z "$PY" ] && PY="$(command -v python3 || command -v python)"
echo "viz_autopilot_train_hist.sh: $PY races/scripts/viz_autopilot_train_hist.py $*"
exec "$PY" races/scripts/viz_autopilot_train_hist.py "$@"
