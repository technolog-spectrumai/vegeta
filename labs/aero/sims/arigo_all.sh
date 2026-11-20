#!/usr/bin/env bash
# The whole ARIGO family, every analysis, restartable, one global progress bar (arigo_all.py):
#   ARIGO-0  nose tractor, camera through the disc (+ its chin-strut version)   3 wings
#   ARIGO-1  pusher                                                              3 wings
#   ARIGO-2  two folding wing tractors                                           3 wings
#   ARIGO-20 two ducted fans, H tail
#
#   labs/aero/sims/arigo_all.sh                         8 CFD processors, all four aircraft
#   labs/aero/sims/arigo_all.sh -j 16 ARIGO-0 ARIGO-20  16 cores for the CFD (or --cpu 16), only these aircraft
#   labs/aero/sims/arigo_all.sh --stability-only        MuJoCo stability + camera only (minutes)
#   labs/aero/sims/arigo_all.sh --full-cfd -j 16          CFD at fine quality (converged; ~5-8 min a case, a day or more
#                                                       for all 158) into arigo*_fine_cfd.yaml; screening files kept
#   labs/aero/sims/arigo_all.sh --skip-video / --skip-cfd / --fresh   (or SKIP_VIDEO=1, SKIP_CFD=1, FRESH=1)
#
# Stop it any time (Ctrl-C) and run again: every step goes on where it stopped. The runners' own output is in
# labs/aero/sims/output/arigo_all.log (tail -f it for detail). Paste back the YAML files listed at the end.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/arigo_all.py" "$@"
