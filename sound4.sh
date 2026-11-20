#!/usr/bin/env bash
# Quadcopter (Drongo, 4 x 5x4.3 tri-blade) propeller sound from CFD at several speeds:
#   ./sound4.sh                      # 0 20 40 60 80 km/h, screening CFD, 4 cores, into ./prop_sounds
#   SPEEDS="0 30 60" PROCS=8 ./sound4.sh
#   ./sound4.sh --no-cfd             # dry run without OpenFOAM (blade-element theory)
# Files: drongo_quad_<kmh>kmh_1prop.wav and drongo_quad_<kmh>kmh_4props.wav, plus drongo_quad_sound.csv/json.
SPEEDS="${SPEEDS:-0 20 40 60 80}" exec "$(cd "$(dirname "$0")" && pwd)/scripts/prop_sound/run.sh" quad "$@"
