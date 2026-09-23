#!/usr/bin/env bash
# Execute every notebook headlessly (outputs go to a temp dir; committed notebooks stay clean).
# Uses the Python that runs this script (kernel python3), e.g. after `source .venv/bin/activate`.
# VEGETA_SKIP_OPENFOAM=1 makes the notebooks that offer it skip their OpenFOAM cells (no OpenFOAM on this machine).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
status=0
cd "$ROOT/notebooks"
for nb in "${@:-*.ipynb}"; do
  for f in $nb; do
    echo "=== $f ==="
    PYVISTA_JUPYTER_BACKEND=static PYVISTA_OFF_SCREEN=true jupyter nbconvert --to notebook --execute "$f" --output-dir "$OUT" --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
      >/dev/null 2>"$OUT/$f.err" && echo ok || { status=1; tail -20 "$OUT/$f.err"; }
  done
done
exit $status
