#!/usr/bin/env bash
# Run each package's tests on its own. Extra args go to pytest, e.g. -m "not slow".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
status=0
for pkg in dedalus talos aeromant mellonia; do
  [ -d "$ROOT/packages/$pkg" ] || continue
  echo "=== $pkg ==="
  (cd "$ROOT/packages/$pkg" && python3 -m pytest -q "$@") || status=1
done
exit $status
