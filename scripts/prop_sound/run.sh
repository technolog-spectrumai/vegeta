#!/usr/bin/env bash
# Shared runner for sound4.sh / sound6.sh: finds (or makes) a dev_arigo checkout, its Python environment, then runs
# prop_sound.py. Settings by environment variable: SPEEDS (km/h, e.g. "0 20 40 60"), QUALITY (smoke|screening|fine),
# PROCS (cores), OUT (folder), VEGETA_ARIGO (a dev_arigo checkout), VEGETA_VENV (its virtual environment).
set -euo pipefail
MACHINE="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
if [ -n "${VEGETA_ARIGO:-}" ]; then ARIGO="$VEGETA_ARIGO"
elif [ -d "$REPO/labs/aero/designs" ]; then ARIGO="$REPO"
else
  ARIGO="$REPO/_arigo"
  if [ ! -d "$ARIGO/labs/aero/designs" ]; then
    echo "fetching dev_arigo (the designs the sounds are made from) into $ARIGO"
    git -C "$REPO" fetch origin dev_arigo
    git -C "$REPO" worktree add --detach "$ARIGO" origin/dev_arigo
  fi
fi
export VEGETA_VENV="${VEGETA_VENV:-$( [ -d "$ARIGO/.venv" ] && echo "$ARIGO/.venv" || echo "$REPO/.venv" )}"
# shellcheck disable=SC1091
source "$ARIGO/activate.sh" || { echo "no Python environment: run $ARIGO/install_local.sh once, or set VEGETA_VENV" >&2; exit 1; }
export PYTHONPATH="$ARIGO/labs/aero/designs:${PYTHONPATH:-}"
for d in "$ARIGO"/libs/*/src; do PYTHONPATH="$d:$PYTHONPATH"; done
SPEEDS_CSV="$(echo "$SPEEDS" | tr ' ' ',')"
OUT="${OUT:-$PWD/prop_sounds}"
echo "machine $MACHINE | speeds $SPEEDS km/h | quality ${QUALITY:-screening} | ${PROCS:-4} cores | out $OUT"
exec python "$HERE/prop_sound.py" "$MACHINE" --speeds "$SPEEDS_CSV" --quality "${QUALITY:-screening}" --procs "${PROCS:-4}" --out "$OUT" "$@"
