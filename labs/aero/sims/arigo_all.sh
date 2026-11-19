#!/usr/bin/env bash
# The whole ARIGO family, every analysis, restartable:
#   ARIGO-0  nose tractor, camera through the disc (+ its chin-strut version for comparison)   3 wings
#   ARIGO-1  pusher                                                                              3 wings
#   ARIGO-2  two folding wing tractors                                                           3 wings
#   ARIGO-20 two ducted fans over the rear fuselage, H tail
#
#   labs/aero/sims/arigo_all.sh                      8 CFD processors, all four aircraft
#   labs/aero/sims/arigo_all.sh 16                   16 CFD processors
#   labs/aero/sims/arigo_all.sh 8 ARIGO-0 ARIGO-20   only these
#   SKIP_CFD=1 / SKIP_VIDEO=1 / FRESH=1 labs/aero/sims/arigo_all.sh
#
# Stop it any time (Ctrl-C) and run it again: each step reads its saved YAML and goes on where it stopped (a CFD
# case already solved comes back from its OpenFOAM folder in seconds). Every line shows the time left of its step.
# Order: the fast steps for all aircraft first (stability, camera), then the videos, then the CFD (hours).
# Paste back the YAML files listed at the end (labs/aero/sims/output/).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
NP="${1:-8}"; shift || true
WANT=("$@"); [ ${#WANT[@]} -eq 0 ] && WANT=(ARIGO-0 ARIGO-1 ARIGO-2 ARIGO-20)
has() { local x; for x in "${WANT[@]}"; do [ "${x^^}" = "$1" ] && return 0; done; return 1; }
F=(); [ -n "${FRESH:-}" ] && F=(--fresh)
W=(--variants goshawk zero hybrid)
status=0; T0=$SECONDS
step() { echo; echo "======== [$(( (SECONDS - T0) / 60 )) min] $*"; }
run() { "$@" || { echo "   (step failed; going on)"; status=1; }; }

# ---------------------------------------------------------------- 1. stability (MuJoCo, minutes)
has ARIGO-0  && { step "ARIGO-0 stability";  run "$PY" "$HERE/arigo1_mujoco.py" "${W[@]}" --propulsion tractor_lowcam tractor --name arigo0 "${F[@]}"; }
has ARIGO-1  && { step "ARIGO-1 stability";  run "$PY" "$HERE/arigo1_mujoco.py" "${W[@]}" --propulsion pusher --name arigo1 "${F[@]}"; }
has ARIGO-2  && { step "ARIGO-2 stability + engine out"; run "$PY" "$HERE/arigo2_mujoco.py" "${F[@]}"; }
has ARIGO-20 && { step "ARIGO-20 stability + fan out + power pitch"; run "$PY" "$HERE/arigo20_mujoco.py" "${F[@]}"; }
has ARIGO-0  && { step "ARIGO-0 propeller in the camera's view"; run "$PY" "$HERE/arigo1_camera.py" --layouts tractor_lowcam tractor pusher; }

# ---------------------------------------------------------------- 2. manoeuvre videos (OpenGL + ffmpeg)
if [ -z "${SKIP_VIDEO:-}" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  has ARIGO-0  && { step "ARIGO-0 videos (36)";  run "$PY" "$HERE/arigo1_video.py" "${W[@]}" --propulsion tractor_lowcam tractor --name arigo0 "${F[@]}"; }
  has ARIGO-1  && { step "ARIGO-1 videos (18)";  run "$PY" "$HERE/arigo1_video.py" "${W[@]}" --propulsion pusher --name arigo1 "${F[@]}"; }
  has ARIGO-2  && { step "ARIGO-2 videos (21)";  run "$PY" "$HERE/arigo2_video.py" "${F[@]}"; }
  has ARIGO-20 && { step "ARIGO-20 videos (6)";  run "$PY" "$HERE/arigo20_video.py" "${F[@]}"; }
fi

# ---------------------------------------------------------------- 3. CFD (OpenFOAM, hours)
if [ -z "${SKIP_CFD:-}" ]; then
  has ARIGO-1  && { step "ARIGO-1 CFD (33 cases)";  run "$PY" "$HERE/arigo1_cfd.py" "${W[@]}" --propulsion pusher --name arigo1 --processors "$NP" "${F[@]}"; }
  has ARIGO-0  && { step "ARIGO-0 CFD (66 cases)";  run "$PY" "$HERE/arigo1_cfd.py" "${W[@]}" --propulsion tractor_lowcam tractor --name arigo0 --processors "$NP" "${F[@]}"; }
  has ARIGO-20 && { step "ARIGO-20 CFD (11 cases)"; run "$PY" "$HERE/arigo20_cfd.py" --processors "$NP" "${F[@]}"; }
  has ARIGO-2  && { step "ARIGO-2 CFD (48 cases: power off and on)"; run "$PY" "$HERE/arigo2_cfd.py" --processors "$NP" "${F[@]}"; }
fi

step "done in $(( (SECONDS - T0) / 60 )) min (status $status). Paste back:"
ls -1 "$HERE/output"/arigo0_*.yaml "$HERE/output"/arigo1_*.yaml "$HERE/output"/arigo2_*.yaml "$HERE/output"/arigo20_*.yaml 2>/dev/null
exit $status
