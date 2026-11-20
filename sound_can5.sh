#!/usr/bin/env bash
# Pekari recoilless cannon: 5 balls one after another (INTERVAL s apart, default 0.12), sound from CFD; one CFD case per
# shot (the plenum sags between balls), added at their firing times -> pekari_sounds/pekari_5shots_10m_90deg.wav
#   ./sound_can5.sh                  # PROCS DIST ANGLE INTERVAL as in sound_can1.sh;  --write / --post likewise
exec "$(cd "$(dirname "$0")" && pwd)/scripts/can_sound/run.sh" 5 "$@"
