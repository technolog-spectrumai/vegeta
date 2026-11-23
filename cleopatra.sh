#!/usr/bin/env bash
# Cleopatra v2 (the three-segment, twelve-legged Myropod, pipe version) in MuJoCo.
#
#   ./cleopatra.sh mission                    the video: through an air duct into the bandits' datacenter, between the
#                                             cables and racks to a server, a pendrive in, a copy, and out again
#   ./cleopatra.sh mission --text             the same with captions
#   ./cleopatra.sh mission --width 960 --height 540 --fps 24 --out my.mp4      smaller and quicker
#
# Output: runs/cleopatra/cleopatra_mission.mp4 (1280 × 720, 30 fps; about 20–40 min on a laptop). Uses the project's
# virtual environment (.venv, or $VEGETA_VENV): ./install_local.sh sets it up; needs numpy, mujoco, cadquery, Pillow
# and ffmpeg (else the frames are left as JPEGs).
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")"

VENV="${VEGETA_VENV:-$PWD/.venv}"
if [ -x "$VENV/bin/python" ]; then
  py="$VENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="$(command -v python3)"
  echo "cleopatra.sh: no virtual environment at $VENV — using $py (run ./install_local.sh for the full setup)" >&2
else
  echo "cleopatra.sh: no Python found — run ./install_local.sh first (or export VEGETA_VENV=/path/to/venv)" >&2
  exit 1
fi
export VEGETA_ROOT="$PWD"
export PYTHONPATH="$PWD/libs/vegeta-cli/src:$PWD/libs/labkit/src:$PWD/libs/ogma/src:$PWD/vendor/yax/src${PYTHONPATH:+:$PYTHONPATH}"

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,12p' "$0"
  exit 0
fi
case "$1" in
  mission)
    shift
    exec "$py" labs/legged/sims/cleopatra_mission.py "$@" ;;
  *)
    echo "cleopatra.sh: unknown command '$1' (mission)" >&2
    exit 2 ;;
esac
