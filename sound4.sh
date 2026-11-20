#!/usr/bin/env bash
# Drongo quadcopter, 4 rotors - sound from CFD, through labs/aero/sims/sound_eval.sh (structured YAML + charts + WAVs in labs/aero/sims/output).
#   ./sound4.sh -j 8            8 cores;   -j 12 -p 3   three CFD cases at once on 4 cores each
#   ./sound4.sh --no-cfd        dry run;   ./sound4.sh --help  everything else
exec "$(cd "$(dirname "$0")" && pwd)/labs/aero/sims/sound_eval.sh" quad "$@"
