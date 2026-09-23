#!/usr/bin/env bash
# Execute every notebook headlessly (outputs go to a temp dir; committed notebooks stay clean).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
status=0
cd "$ROOT/notebooks"
for nb in "${@:-*.ipynb}"; do
  for f in $nb; do
    echo "=== $f ==="
    jupyter nbconvert --to notebook --execute "$f" --output-dir "$OUT" --ExecutePreprocessor.timeout=1800 \
      >/dev/null 2>"$OUT/$f.err" && echo ok || { status=1; tail -20 "$OUT/$f.err"; }
  done
done
exit $status
