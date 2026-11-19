#!/usr/bin/env bash
# ARIGO-1 only: see arigo_all.sh (restartable). Usage: labs/aero/sims/arigo1_all.sh [processors]
exec "$(cd "$(dirname "$0")" && pwd)/arigo_all.sh" "${1:-8}" ARIGO-1
