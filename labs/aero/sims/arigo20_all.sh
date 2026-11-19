#!/usr/bin/env bash
# ARIGO-20 only: see arigo_all.sh (restartable). Usage: labs/aero/sims/arigo20_all.sh [processors]
exec "$(cd "$(dirname "$0")" && pwd)/arigo_all.sh" "${1:-8}" ARIGO-20
