#!/usr/bin/env bash
# Run the tests of each part on its own: the tools (vegeta-cli/tests/<tool>) and vegeta-core.
# Extra args go to pytest, e.g. -m "not slow".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
status=0
for part in cli dedalus talos aeromant mellonia; do
  [ -d "$ROOT/vegeta-cli/tests/$part" ] || continue
  echo "=== $part ==="
  (cd "$ROOT/vegeta-cli" && python3 -m pytest -q "tests/$part" "$@") || status=1
done
if [ -d "$ROOT/vegeta-core/tests" ]; then
  echo "=== core ==="
  (cd "$ROOT/vegeta-core" && python3 -m pytest -q "$@") || status=1
fi
exit $status
