#!/usr/bin/env bash
# Kestrel-4Q-2N (the net launcher in the pod, labs/aero/designs/kestrel_netter.py): every simulation from one script.
#
#   labs/aero/sims/kestrel.sh -j 8                 # CFD (stall, sweep, cruise, transitions), then FEA, then stability, on 8 cores
#   labs/aero/sims/kestrel.sh --stability          # the reduced-order checks only (no solver, ~5 min)
#   labs/aero/sims/kestrel.sh --fea                # the frame and the spar–post joint (CalculiX)        ~1 h on 4 threads
#   labs/aero/sims/kestrel.sh --cfd -j 8           # the airframe campaigns (OpenFOAM)                  screening: hours
#   labs/aero/sims/kestrel.sh --cfd --quality fine # the fine meshes (a day)
#   labs/aero/sims/kestrel.sh --dry-run            # print the commands, run nothing
#   labs/aero/sims/kestrel.sh --help
#
# Options: -j N      cores: MPI ranks per CFD case and CalculiX threads (default 4); -n N / --threads N override one of them
#          --quality screening|fine (CFD, default screening)  --campaigns "stall sweep cruise transitions" (CFD, in order)
#          --matrix  the stability phase also flies the back transition's remedy matrix (~15 min)
#          --variant 4Q-2N (default) | 4Q-2 | 4Q-3 | 4Q-1     --python PATH   the interpreter (default .venv/bin/python)
#
# The CFD and FEA go through the side-labs chapter scripts, so every run lands under side_labs/runs/kestrel/<chapter>/
# with its manifest, log and summary, and is restartable:  .venv/bin/python labs/aero/campaigns/side_labs/kestrel/airframe_cfd.py --resume <run dir>
# (`wf runs`, `wf status`, `wf tail` list and follow them). The chapters that need the sweep (cruise, transitions)
# take the latest completed sweep run of the same variant through the Yanus gate — run the sweep first (the default order).
# The stability phase writes labs/aero/sims/output/kestrel_netter/ (stability.json, the CSVs, the transition figures).
# Each phase's console output is also kept in labs/aero/sims/output/kestrel_netter/logs/.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SIDE="$ROOT/labs/aero/campaigns/side_labs"
PY="${VEGETA_PYTHON:-}"
VARIANT="4Q-2N"; QUALITY="screening"; JOBS=4; PROCS=""; THREADS=""; DRY=0; MATRIX=0
CAMPAIGNS="stall sweep cruise transitions"
DO_STAB=0; DO_FEA=0; DO_CFD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stability) DO_STAB=1; shift ;;
    --fea) DO_FEA=1; shift ;;
    --cfd) DO_CFD=1; shift ;;
    --matrix) MATRIX=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --campaigns) CAMPAIGNS="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    -j|--jobs|--cores) JOBS="$2"; shift 2 ;;
    -n|--processors) PROCS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [ $DO_STAB -eq 0 ] && [ $DO_FEA -eq 0 ] && [ $DO_CFD -eq 0 ]; then DO_STAB=1; DO_FEA=1; DO_CFD=1; fi
PROCS="${PROCS:-$JOBS}"; THREADS="${THREADS:-$JOBS}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
OUT="$HERE/output/kestrel_netter"; LOGS="$OUT/logs"
mkdir -p "$LOGS"
export PYTHONPATH="$SIDE:$ROOT/labs/aero/designs:$ROOT/labs/aero/exp${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run() {                                   # run <name> <command...>: timed, logged, never stops the script
  local name="$1"; shift
  local log="$LOGS/$(date +%Y%m%d-%H%M%S)_$name.log"
  echo "== $name"
  echo "   $*"
  if [ $DRY -eq 1 ]; then return 0; fi
  local t0=$SECONDS
  if "$@" 2>&1 | tee "$log"; then
    echo "   ok ($((SECONDS - t0)) s) — $log"; RESULTS+=("ok      $name  $((SECONDS - t0)) s")
  else
    echo "   FAILED ($((SECONDS - t0)) s) — $log"; RESULTS+=("FAILED  $name  $((SECONDS - t0)) s"); FAILED=1
  fi
}
RESULTS=(); FAILED=0
echo "Kestrel-$VARIANT — stability $DO_STAB, FEA $DO_FEA, CFD $DO_CFD ($QUALITY: $CAMPAIGNS); $PROCS MPI ranks per CFD case, $THREADS CalculiX threads; python $PY"
cd "$ROOT"

# 1. CFD: the bare airframe at the stall angles first (CL max: the wing area with the launcher hangs on it), the sweep
#    (the model polar's reference), then the blown cases that take the sweep (the cruise and the transition points)
if [ $DO_CFD -eq 1 ]; then
  for c in $CAMPAIGNS; do
    run "cfd-$c-$QUALITY" "$PY" "$SIDE/kestrel/airframe_cfd.py" --variant "$VARIANT" --chapter "$c" --quality "$QUALITY" --processors "$PROCS"
  done
fi
# 2. FEA: the carbon frame (posts) under the 4 g pull-up, the 75 % wing, the heading manoeuvre, the modes; the spar–post joint
if [ $DO_FEA -eq 1 ]; then
  run fea-frame-and-joint "$PY" "$SIDE/kestrel/structure_fea.py" --variant "$VARIANT" --chapter all --threads "$THREADS"
fi

# 3. the reduced-order checks: no solver; the drive table is built with Boreas on the first run (~40 s)
if [ $DO_STAB -eq 1 ]; then
  if [ "$VARIANT" = "4Q-2N" ]; then
    stab=("$PY" "$HERE/kestrel_netter_stability.py"); [ $MATRIX -eq 1 ] && stab+=(--matrix)
    run stability "${stab[@]}"
  else
    echo "== stability: the reduced-order checks are written for 4Q-2N (kestrel_netter_stability.py); for $VARIANT run notebook 43 (kestrel_study)"
  fi
fi


echo
echo "==== Kestrel-$VARIANT ===="
for r in "${RESULTS[@]}"; do echo "  $r"; done
[ $DRY -eq 1 ] && echo "  (dry run: nothing was executed)"
echo "  runs: side_labs/runs/kestrel/<chapter>/   stability: $OUT/   logs: $LOGS/"
exit $FAILED
