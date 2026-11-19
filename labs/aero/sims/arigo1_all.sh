#!/usr/bin/env bash
# ARIGO-1: everything, all 3 wings x 3 propulsions. Paste back labs/aero/sims/output/arigo1_{cfd,mujoco,video}.yaml
#   labs/aero/sims/arigo1_all.sh            8 CFD processors (default)
#   labs/aero/sims/arigo1_all.sh 16         16 CFD processors
#   SKIP_CFD=1 / SKIP_VIDEO=1 labs/aero/sims/arigo1_all.sh   leave a stage out
# Order: MuJoCo stability (fast) -> videos -> CFD (slow, restartable). One stage failing does not stop the others.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NP="${1:-8}"
ALL=(--variants goshawk zero hybrid --propulsion pusher tractor twin)
status=0
echo "== stage 1/3: MuJoCo stability (~1 min)";  "$HERE/arigo1_mujoco.sh" "${ALL[@]}" || status=1
if [ -z "${SKIP_VIDEO:-}" ]; then echo "== stage 2/3: manoeuvre videos (54; minutes, depends on the GPU)"; "$HERE/arigo1_video.sh" "${ALL[@]}" || status=1; fi
if [ -z "${SKIP_CFD:-}" ]; then echo "== stage 3/3: CFD, 99 cases on $NP processors (hours; each line shows the time left)"; "$HERE/arigo1.sh" "${ALL[@]}" --processors "$NP" || status=1; fi
echo "== done (status $status). Paste back:"
ls -1 "$HERE/output"/arigo1_*.yaml 2>/dev/null
exit $status
