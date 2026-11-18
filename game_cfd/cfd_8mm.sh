#!/usr/bin/env bash
# Game CFD for the recoilless 8 mm emitter (approximate, coarse): see game_cfd/cfd_8mm.py.
# Writes game_cfd/output/cfd_8mm.yaml.
# Usage: ./cfd_8mm.sh [--only drag|tube] [--processors N] [--travel MM] [--dry] [--out file.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${VEGETA_VENV:-$HERE/../.venv}/bin/activate"
cd "$HERE"
exec python cfd_8mm.py "$@"
