#!/usr/bin/env bash
# Yax's tests from the source tree: no install, no network. Extra arguments go to pytest.
#   ./test.sh --smoke   the whole suite, timed (it is smoke-sized)
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ "${1:-}" = "--smoke" ] && shift
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TIMEFORMAT='yax tests: %R s'
time "$PY" -m pytest -q -p no:cacheprovider "$@"
