#!/usr/bin/env bash
# Run notebooks end to end in DEBUG mode (FEA and CFD mocked by vegeta.mock) to test them before a full run:
#   scripts/debug_notebooks.sh                      # every notebook
#   scripts/debug_notebooks.sh 08_quadcopter 12     # those whose name starts with these
#   -j N  notebooks side by side (default 2)
# Each runs exactly as  VEGETA_DEBUG=1 jupyter nbconvert --to notebook --execute X.ipynb --output X-run.ipynb
# in notebooks/ (the X-run.ipynb outputs are gitignored); a failure prints the end of its error.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS=2
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in -j) JOBS="$2"; shift 2 ;; -h|--help) sed -n '2,8p' "$0"; exit 0 ;; *) ARGS+=("$1"); shift ;; esac
done
cd "$ROOT/notebooks"
LIST=()
for nb in *.ipynb; do
  case "$nb" in *-run.ipynb) continue ;; esac
  if [ ${#ARGS[@]} -eq 0 ]; then LIST+=("${nb%.ipynb}"); continue; fi
  for a in "${ARGS[@]}"; do case "$nb" in "$a"*) LIST+=("${nb%.ipynb}") ;; esac; done
done
LOGS="$(mktemp -d)"
# one virtual display for all notebooks (pyvista renders need one; one xvfb-run per notebook races when run in parallel)
if [ -z "${DISPLAY:-}" ] && command -v Xvfb >/dev/null; then
  for n in $(seq 90 140); do [ -e "/tmp/.X$n-lock" ] || { DNUM=$n; break; }; done
  Xvfb ":$DNUM" -screen 0 1280x1024x24 >/dev/null 2>&1 &
  XVFB_PID=$!
  trap 'kill $XVFB_PID 2>/dev/null' EXIT
  export DISPLAY=":$DNUM"
  sleep 1
fi
export VEGETA_DEBUG=1 PYVISTA_OFF_SCREEN=true PYVISTA_JUPYTER_BACKEND=static MPLBACKEND=Agg
unset VEGETA_SKIP_OPENFOAM
run_one() {
  local nb="$1" t0=$SECONDS
  local cmd=(jupyter nbconvert --to notebook --execute "$nb.ipynb" --output "$nb-run.ipynb" --ExecutePreprocessor.timeout=3600)
  if "${cmd[@]}" >"$LOGS/$nb.log" 2>&1; then
    printf 'ok      %-40s %5s s\n' "$nb" $((SECONDS - t0))
  else
    printf 'FAILED  %-40s %5s s\n' "$nb" $((SECONDS - t0)); grep -E "Error|error" "$LOGS/$nb.log" | tail -4 | sed 's/^/        /'
  fi
}
export -f run_one; export LOGS
printf '%s\n' "${LIST[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} | tee "$LOGS/summary.txt"
grep -q FAILED "$LOGS/summary.txt" && { echo "logs: $LOGS"; exit 1; } || exit 0
