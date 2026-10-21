#!/usr/bin/env bash
# Run the tests of each part on its own: the tools (libs/vegeta-cli/tests/<tool>), vegeta-core, vegeta-ai, vegeta-fidia, Ozaru (vendor/ozaru: Porunga, Mimezia, Yanus), Yax (vendor/yax), Ogma (libs/ogma/),
# components and assemblies (geometry only).
# Extra args go to pytest, e.g. -m "not slow".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
status=0
for part in cli dedalus talos aeromant mellonia boreas chronos chiron atropos zephyr; do
  [ -d "$ROOT/libs/vegeta-cli/tests/$part" ] || continue
  echo "=== $part ==="
  (cd "$ROOT/libs/vegeta-cli" && python3 -m pytest -q "tests/$part" "$@") || status=1
done
for pkg in vegeta-core vegeta-ai vegeta-fidia vendor/ozaru vendor/yax ogma; do
  [ -d "$ROOT/$pkg/tests" ] || continue
  echo "=== ${pkg#vegeta-} ==="
  (cd "$ROOT/$pkg" && python3 -m pytest -q "$@") || status=1
done
for pkg in components assemblies; do
  [ -d "$ROOT/$pkg/tests" ] || continue
  echo "=== $pkg ==="
  (cd "$ROOT" && python3 -m pytest -q "$pkg/tests" "$@") || status=1
done
exit $status
