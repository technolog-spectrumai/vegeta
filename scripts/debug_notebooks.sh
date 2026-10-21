#!/usr/bin/env bash
# Run notebooks end to end with the mocks on (DEBUG; FEA, CFD and slicing mocked by vegeta.mock; VEGETA_NO_SOLVERS=1
# refuses to start CalculiX, OpenFOAM, PrusaSlicer or XFOIL; the AI demos use scripted answers, never the API):
#   scripts/debug_notebooks.sh                      # every notebook but the slow ones (SLOW below)
#   scripts/debug_notebooks.sh --all                # the slow ones too (MuJoCo missions and movies: up to an hour each)
#   scripts/debug_notebooks.sh --include-experiments   # also labs/*/exp/ (Kestrel, the microjet: experiments)
#   scripts/debug_notebooks.sh 08_quadcopter 43     # those whose name starts with these
#   -j N  notebooks side by side (default 2);  -t S  per-cell time limit in seconds (default 1200)
# Notebooks live in labs/<lab>/studies/ (and exp/); each runs in its own folder, its executed copy goes to
# a temporary log folder. Tracked files a notebook writes (labs/*/designs/data/*.json, ...) are put back afterwards.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS=2; CELL=1200; ALL=0; EXP=0
SLOW="08b_ 17_ 20_ 21_ 22_ 23_ 30_ 31_ 32_ 33_ 35_ 37_ 07_ai_ 15_fidia"   # MuJoCo missions/movies; 07/15 need an API key
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in -j) JOBS="$2"; shift 2 ;; -t) CELL="$2"; shift 2 ;; --all) ALL=1; shift ;; --include-experiments) EXP=1; shift ;; -h|--help) sed -n '2,8p' "$0"; exit 0 ;; *) ARGS+=("$1"); shift ;; esac
done
LIST=()
FOLDERS=(studies); [ $EXP -eq 1 ] && FOLDERS+=(exp)     # exp/: limbo (Kestrel experiments), only when asked
for path in $(for f in "${FOLDERS[@]}"; do ls "$ROOT"/labs/*/"$f"/*.ipynb 2>/dev/null; done); do
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
git -C "$ROOT" ls-files -z labs | grep -zv '\.ipynb$' | (cd "$ROOT" && xargs -0 -r cp --parents -t "$SNAP" 2>/dev/null)
restore() {
  (cd "$SNAP" && find . -type f -print0) | while IFS= read -r -d '' f; do
    cmp -s "$SNAP/$f" "$ROOT/$f" || { cp "$SNAP/$f" "$ROOT/$f"; echo "restored ${f#./} (written by a mocked run)"; }
  done
}
if [ -z "${DISPLAY:-}" ] && command -v Xvfb >/dev/null; then       # one virtual display for pyvista renders
  for n in $(seq 90 140); do [ -e "/tmp/.X$n-lock" ] || { DNUM=$n; break; }; done
  Xvfb ":$DNUM" -screen 0 1280x1024x24 >/dev/null 2>&1 &
  XVFB_PID=$!; export DISPLAY=":$DNUM"
  for _ in $(seq 50); do [ -e "/tmp/.X11-unix/X$DNUM" ] && break; sleep 0.2; done   # wait until the display answers
  sleep 0.5
fi
[ "$JOBS" -gt 4 ] && JOBS=4      # more simultaneous kernel starts race on ports and fail in seconds
trap 'kill ${XVFB_PID:-} 2>/dev/null; restore' EXIT
export VEGETA_DEBUG=1 VEGETA_NO_SOLVERS=1 PYVISTA_OFF_SCREEN=true PYVISTA_JUPYTER_BACKEND=static MPLBACKEND=Agg PORUNGA_DIR="$LOGS/porunga"
# Notebooks name either the "vegeta" or the "python3" kernel; a machine usually registers only one of them. Run every
# notebook on whichever is installed (vegeta first) instead of failing in three seconds on "No such kernel".
KERNEL=""; for k in vegeta python3; do jupyter kernelspec list 2>/dev/null | grep -qE "^\s*$k\s" && { KERNEL="$k"; break; }; done
exec_nb() {
  (cd "$(dirname "$1")" && jupyter nbconvert --to notebook --execute "$2.ipynb" --output-dir "$LOGS" --output "$2-run.ipynb" \
     --ExecutePreprocessor.timeout="$CELL" ${KERNEL:+--ExecutePreprocessor.kernel_name="$KERNEL"}) >"$LOGS/$2.log" 2>&1
}
run_one() {
  local path="$1" t0=$SECONDS note=""; local nb; nb="$(basename "$path" .ipynb)"
  exec_nb "$path" "$nb"; local rc=$?
  if [ $rc -ne 0 ] && [ $((SECONDS - t0)) -lt 15 ]; then     # failed before any cell could run: a start-up failure, retry once
    cp "$LOGS/$nb.log" "$LOGS/$nb.first.log"; sleep 2; t0=$SECONDS; note=" (retried after a start-up failure)"
    exec_nb "$path" "$nb"; rc=$?
  fi
  if [ $rc -eq 0 ]; then
    printf 'ok      %-40s %5s s%s\n' "$nb" $((SECONDS - t0)) "$note"
  else
    printf 'FAILED  %-40s %5s s%s\n' "$nb" $((SECONDS - t0)) "$note"; tail -15 "$LOGS/$nb.log" | sed 's/^/        /'
  fi
}
export -f exec_nb run_one; export LOGS CELL KERNEL
printf '%s\n' "${LIST[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} | tee "$LOGS/summary.txt"
echo "logs and executed notebooks: $LOGS"
grep -q FAILED "$LOGS/summary.txt" && exit 1 || exit 0
