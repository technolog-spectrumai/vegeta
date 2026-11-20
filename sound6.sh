#!/usr/bin/env bash
# Hexacopter (Drako, 6 x T-Motor 30x10.5) propeller sound from CFD at several speeds:
#   ./sound6.sh                      # 0 15 30 45 60 km/h, screening CFD, 4 cores, into ./prop_sounds
#   SPEEDS="0 30 60" PROCS=8 ./sound6.sh
#   ./sound6.sh --no-cfd             # dry run without OpenFOAM (blade-element theory)
# Files: drako_hexa_<kmh>kmh_1prop.wav and drako_hexa_<kmh>kmh_6props.wav, plus drako_hexa_sound.csv/json.
SPEEDS="${SPEEDS:-0 15 30 45 60}" exec "$(cd "$(dirname "$0")" && pwd)/scripts/prop_sound/run.sh" hexa "$@"
