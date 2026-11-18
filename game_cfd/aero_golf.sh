#!/usr/bin/env bash
# Game CFD (approximate, coarse): see game_cfd/ball_aero.py. Writes game_cfd/output/aero_golf.yaml.
# Usage: ./aero_golf.sh [--dry] [--processors N] [--out file.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${VEGETA_VENV:-$HERE/../.venv}/bin/activate"
cd "$HERE"
exec python ball_aero.py golf "$@"
