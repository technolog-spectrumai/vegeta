#!/usr/bin/env bash
# Re-solve the goshawk CFD cases that ran on the old geometry (ARIGO-1 pusher, ARIGO-0 chin-strut tractor, ARIGO-2).
#   labs/aero/sims/goshawk_rerun.sh        12 processors
#   labs/aero/sims/goshawk_rerun.sh 8      8 processors
# Restartable: stop any time and run again. New case folders carry a _g<hash> tag (the new geometry).
# Paste back: labs/aero/sims/output/arigo1_cfd.yaml arigo0_cfd.yaml arigo2_cfd.yaml
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
NP="${1:-12}"
status=0
echo "== 1/3 ARIGO-1 goshawk pusher (11 cases)"
"$PY" "$HERE/arigo1_cfd.py" --variants goshawk --propulsion pusher --name arigo1 --processors "$NP" || status=1
echo "== 2/3 ARIGO-0 goshawk chin-strut tractor (11 cases)"
"$PY" "$HERE/arigo1_cfd.py" --variants goshawk --propulsion tractor --name arigo0 --processors "$NP" || status=1
echo "== 3/3 ARIGO-2 goshawk (11 power off + 5 power on)"
"$PY" "$HERE/arigo2_cfd.py" --wings goshawk --processors "$NP" || status=1
echo "== done (status $status). Paste back:"
ls -1 "$HERE/output"/arigo1_cfd.yaml "$HERE/output"/arigo0_cfd.yaml "$HERE/output"/arigo2_cfd.yaml 2>/dev/null
exit $status
