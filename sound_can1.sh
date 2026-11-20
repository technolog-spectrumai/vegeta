#!/usr/bin/env bash
# Pekari recoilless cannon (8 mm steel ball, 280 m/s): one shot, sound from CFD (rhoCentralFoam, ~1.1 M cells, ~26 ms).
#   ./sound_can1.sh                  # write + run the CFD + WAV -> pekari_sounds/pekari_1shot_10m_90deg.wav
#   PROCS=16 DIST=5 ANGLE=150 ./sound_can1.sh
#   ./sound_can1.sh --write          # case only;  ./sound_can1.sh --post   # WAV from a finished run
exec "$(cd "$(dirname "$0")" && pwd)/scripts/can_sound/run.sh" 1 "$@"
