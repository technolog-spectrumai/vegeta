#!/usr/bin/env bash
# Drako hexacopter, 6 rotors - sound from CFD, through labs/aero/sims/sound_eval.sh (structured YAML + charts + WAVs in labs/aero/sims/output).
#   ./sound6.sh -j 8            8 cores;   -j 12 -p 3   three CFD cases at once on 4 cores each
#   ./sound6.sh --no-cfd        dry run;   ./sound6.sh --help  everything else
exec "$(cd "$(dirname "$0")" && pwd)/labs/aero/sims/sound_eval.sh" hexa "$@"
