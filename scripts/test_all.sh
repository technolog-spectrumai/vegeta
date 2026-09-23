#!/usr/bin/env bash
# Run the tests of each part on its own: the tools (vegeta-cli/tests/<tool>) and vegeta-core.
# Extra args go to pytest, e.g. -m "not slow".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
status=0
for part in cli dedalus talos aeromant mellonia boreas chronos; do
  [ -d "$ROOT/vegeta-cli/tests/$part" ] || continue
  echo "=== $part ==="
  (cd "$ROOT/vegeta-cli" && python3 -m pytest -q "tests/$part" "$@") || status=1
done
for pkg in vegeta-core vegeta-ai; do
  [ -d "$ROOT/$pkg/tests" ] || continue
  echo "=== ${pkg#vegeta-} ==="
  (cd "$ROOT/$pkg" && python3 -m pytest -q "$@") || status=1
done
exit $status
