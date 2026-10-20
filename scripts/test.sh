#!/usr/bin/env bash
# Ogma's tests from the source tree: no install, no network. Extra arguments go to pytest.
#   --smoke   the same suite, timed
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ "${1:-}" = "--smoke" ]; then
  shift
  TIMEFORMAT='ogma smoke: %R s'
  time "$PY" -m pytest -q -p no:cacheprovider "$@"
  exit
fi
exec "$PY" -m pytest -q -p no:cacheprovider "$@"
