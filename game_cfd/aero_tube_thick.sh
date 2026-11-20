#!/usr/bin/env bash
# Game CFD (approximate, coarse): see game_cfd/tube_exit.py. Writes game_cfd/output/aero_tube_thick.yaml.
# Usage: ./aero_tube_thick.sh [--dry] [--processors N] [--out file.yaml] [--case slow|subsonic_fast|near_sonic]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${VEGETA_VENV:-$HERE/../.venv}/bin/activate"
cd "$HERE"
exec python tube_exit.py thick "$@"
