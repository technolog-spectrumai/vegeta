#!/usr/bin/env bash
# ASTOR-V / ASTOR-BOOST: everything in one go — CFD, then stability, then the net drop.
#
#   labs/aero/sims/astor_all.sh -j 12                 # all three phases, 12 cores
#   labs/aero/sims/astor_all.sh -j 12 --skip-cfd      # stability and net drop only (on the lattice model, or the CFD already installed)
#   labs/aero/sims/astor_all.sh -j 12 --only netdrop  # one phase: cfd | stability | netdrop
#   labs/aero/sims/astor_all.sh --dry-run             # list what every phase would do, run nothing
#   labs/aero/sims/astor_all.sh -j 12 --design boost --sorties 40
#
# -j N / --cpu N   CPU cores (default 4). CFD: OpenFOAM's MPI processes. Stability and net drop: worker processes,
#                  lowered if the machine cannot carry them (cores - 2; ~1 GB of free RAM each); --force-cpu insists.
# Every other option is passed to the phases (see  astor_cfd.sh --help): --design v|boost|dart, --fresh, --sorties N,
# --latency S, --noise M, --full-cfd, --no-install, --dry-run.
#
# Each phase is restartable (finished cases and jobs are kept) and has its own progress bar; each writes one
# timestamped file in labs/aero/sims/output/runs/. A phase that fails does not stop the next: if the CFD cannot run
# (no OpenFOAM), stability and the net drop still run on the lattice model. The end lists what ran and how it ended.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CPU=4; SKIP_CFD=0; ONLY=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--cpu) CPU="$2"; shift ;;
    --skip-cfd) SKIP_CFD=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
case "$ONLY" in ""|cfd|stability|netdrop) ;; *) echo "--only takes cfd, stability or netdrop"; exit 2 ;; esac
PHASES=(cfd stability netdrop)
[ "$SKIP_CFD" = 1 ] && PHASES=(stability netdrop)
[ -n "$ONLY" ] && PHASES=("$ONLY")
T0=$(date +%s); SUMMARY=(); N=0
for ph in "${PHASES[@]}"; do
  N=$((N + 1))
  echo
  echo "================ ASTOR ${N}/${#PHASES[@]}: ${ph}  (-j ${CPU}) ================"
  t=$(date +%s)
  "$HERE/astor_${ph}.sh" -j "$CPU" ${ARGS[@]+"${ARGS[@]}"}
  rc=$?
  SUMMARY+=("$(printf '%-10s %s  (%d min)' "$ph" "$([ $rc = 0 ] && echo ok || echo "ended with code $rc")" $(( ($(date +%s) - t + 30) / 60 )))")
done
echo
echo "================ ASTOR: done in $(( ($(date +%s) - T0 + 30) / 60 )) min ================"
printf '  %s\n' "${SUMMARY[@]}"
echo "  results: $HERE/output/runs/astor_v_*.yaml  (paste the newest of each phase back)"
