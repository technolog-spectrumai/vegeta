#!/usr/bin/env bash
# Pekari cannon, one shot - sound from CFD, through labs/aero/sims/sound_eval.sh (structured YAML + charts + WAVs in labs/aero/sims/output).
#   ./sound_can1.sh -j 8            8 cores;   -j 12 -p 3   three CFD cases at once on 4 cores each
#   ./sound_can1.sh --no-cfd        dry run;   ./sound_can1.sh --help  everything else
exec "$(cd "$(dirname "$0")" && pwd)/labs/aero/sims/sound_eval.sh" can1 "$@"
