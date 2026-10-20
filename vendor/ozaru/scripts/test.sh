#!/usr/bin/env bash
# Ozaru's tests from the source tree: no install, no network. Extra arguments go to pytest.
#   --smoke   the same suite, timed (Ozaru's whole suite is smoke-sized: about a second)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ "${1:-}" = "--smoke" ]; then
  shift
  TIMEFORMAT='ozaru smoke: %R s'
  time "$PY" -m pytest -q -p no:cacheprovider "$@"
  exit
fi
if "$PY" -c "import pytest" 2>/dev/null; then exec "$PY" -m pytest -q "$@"; fi
exec pytest -q "$@"
