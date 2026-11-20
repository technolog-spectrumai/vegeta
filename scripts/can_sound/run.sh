#!/usr/bin/env bash
# Shared runner for sound_can1.sh / sound_can5.sh: the Pekari recoilless 8 mm cannon's firing sound from CFD
# (labs/aero/exp/bearing_sound_cfd.py). Settings by environment variable: PROCS (cores, 4), DIST (listener m, 10),
# ANGLE (deg from the muzzle direction, 90), INTERVAL (s between balls, 0.12), OUT (folder, ./pekari_sounds),
# VEGETA_VENV (the Python environment). Extra arguments go to the Python script (--write, --post, --fine, --far ...).
set -euo pipefail
SHOTS="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/activate.sh" || { echo "no Python environment: run $REPO/install_local.sh once, or set VEGETA_VENV" >&2; exit 1; }
export PYTHONPATH="$REPO/labs/aero/exp:$REPO/labs/aero/designs:${PYTHONPATH:-}"
for d in "$REPO"/libs/*/src; do PYTHONPATH="$d:$PYTHONPATH"; done
OUT="${OUT:-$PWD/pekari_sounds}"
echo "Pekari | $SHOTS shot(s) | listener ${DIST:-10} m at ${ANGLE:-90} deg | ${PROCS:-4} cores | out $OUT"
exec python "$REPO/labs/aero/exp/bearing_sound_cfd.py" --shots "$SHOTS" --interval "${INTERVAL:-0.12}" \
  --distance "${DIST:-10}" --angle "${ANGLE:-90}" --procs "${PROCS:-4}" --out "$OUT" "$@"
