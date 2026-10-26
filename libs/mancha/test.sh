#!/usr/bin/env bash
# Mancha's tests from the source tree: no install, no network. Extra arguments go to pytest.
#   --interpret   run the Triton tests on the CPU through Triton's interpreter (slow; no GPU needed)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ "${1:-}" = "--interpret" ]; then
  shift
  export TRITON_INTERPRET=1
fi
exec "$PY" -m pytest -q -p no:cacheprovider "$@"
