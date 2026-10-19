#!/usr/bin/env bash
# Ozaru's tests from the source tree: no install, no network. Extra arguments go to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if "$PY" -c "import pytest" 2>/dev/null; then exec "$PY" -m pytest -q "$@"; fi
exec pytest -q "$@"
