#!/usr/bin/env bash
# Run the tests of each tool on its own (vegeta-cli/tests/<tool>). Extra args go to pytest,
# e.g. -m "not slow".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
status=0
cd "$ROOT/vegeta-cli"
for part in cli dedalus talos aeromant mellonia; do
  [ -d "tests/$part" ] || continue
  echo "=== $part ==="
  python3 -m pytest -q "tests/$part" "$@" || status=1
done
exit $status
