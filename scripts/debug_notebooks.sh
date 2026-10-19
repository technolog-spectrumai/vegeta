#!/usr/bin/env bash
# Run notebooks end to end with the mocks on (DEBUG; FEA, CFD and slicing mocked by vegeta.mock) — never a real solver:
#   scripts/debug_notebooks.sh                      # every notebook but the slow ones (SLOW below)
#   scripts/debug_notebooks.sh --all                # the slow ones too (MuJoCo missions and movies: up to an hour each)
#   scripts/debug_notebooks.sh 08_quadcopter 43     # those whose name starts with these
#   -j N  notebooks side by side (default 2);  -t S  per-cell time limit in seconds (default 1200)
# Notebooks live in notebooks/{devices,parts,scenarios,demo}/; each runs in its own folder, its executed copy goes to
# a temporary log folder. Tracked files a notebook writes (designs/data/*.json, ...) are put back afterwards.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS=2; CELL=1200; ALL=0
SLOW="08b_ 17_ 20_ 21_ 22_ 23_ 30_ 31_ 32_ 33_ 35_ 37_ 07_ai_ 15_fidia"   # MuJoCo missions/movies; 07/15 need an API key
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in -j) JOBS="$2"; shift 2 ;; -t) CELL="$2"; shift 2 ;; --all) ALL=1; shift ;; -h|--help) sed -n '2,8p' "$0"; exit 0 ;; *) ARGS+=("$1"); shift ;; esac
done
LIST=()
for path in "$ROOT"/notebooks/{devices,parts,scenarios,demo}/*.ipynb; do
  nb="$(basename "$path" .ipynb)"; case "$nb" in *-run) continue ;; esac
  if [ ${#ARGS[@]} -eq 0 ]; then
    skip=0; [ $ALL -eq 0 ] && for s in $SLOW; do case "$nb" in "$s"*) skip=1 ;; esac; done
    [ $skip -eq 1 ] && { echo "skip    $nb (slow; --all runs it)"; continue; }
    LIST+=("$path"); continue
  fi
  for a in "${ARGS[@]}"; do case "$nb" in "$a"*) LIST+=("$path") ;; esac; done
done
LOGS="$(mktemp -d)"
SNAP="$LOGS/tracked"; mkdir -p "$SNAP"
git -C "$ROOT" ls-files -z notebooks designs | grep -zv '\.ipynb$' | (cd "$ROOT" && xargs -0 -r cp --parents -t "$SNAP" 2>/dev/null)
restore() {
  (cd "$SNAP" && find . -type f -print0) | while IFS= read -r -d '' f; do
    cmp -s "$SNAP/$f" "$ROOT/$f" || { cp "$SNAP/$f" "$ROOT/$f"; echo "restored ${f#./} (written by a mocked run)"; }
  done
}
if [ -z "${DISPLAY:-}" ] && command -v Xvfb >/dev/null; then       # one virtual display for pyvista renders
  for n in $(seq 90 140); do [ -e "/tmp/.X$n-lock" ] || { DNUM=$n; break; }; done
  Xvfb ":$DNUM" -screen 0 1280x1024x24 >/dev/null 2>&1 &
  XVFB_PID=$!; export DISPLAY=":$DNUM"; sleep 1
fi
trap 'kill ${XVFB_PID:-} 2>/dev/null; restore' EXIT
export VEGETA_DEBUG=1 PYVISTA_OFF_SCREEN=true PYVISTA_JUPYTER_BACKEND=static MPLBACKEND=Agg PORUNGA_DIR="$LOGS/porunga"
run_one() {
  local path="$1" t0=$SECONDS; local nb; nb="$(basename "$path" .ipynb)"
  if (cd "$(dirname "$path")" && jupyter nbconvert --to notebook --execute "$nb.ipynb" --output-dir "$LOGS" --output "$nb-run.ipynb" \
        --ExecutePreprocessor.timeout="$CELL") >"$LOGS/$nb.log" 2>&1; then
    printf 'ok      %-40s %5s s\n' "$nb" $((SECONDS - t0))
  else
    printf 'FAILED  %-40s %5s s\n' "$nb" $((SECONDS - t0)); grep -E "Error|error" "$LOGS/$nb.log" | grep -v "^\s*File" | tail -3 | sed 's/^/        /'
  fi
}
export -f run_one; export LOGS CELL
printf '%s\n' "${LIST[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} | tee "$LOGS/summary.txt"
echo "logs and executed notebooks: $LOGS"
grep -q FAILED "$LOGS/summary.txt" && exit 1 || exit 0
