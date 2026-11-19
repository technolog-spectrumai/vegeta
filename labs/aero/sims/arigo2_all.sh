#!/usr/bin/env bash
# ARIGO-2 (two folding wing tractors): everything, all three wings. Paste back labs/aero/sims/output/arigo2_{cfd,mujoco,video}.yaml
#   labs/aero/sims/arigo2_all.sh            8 CFD processors
#   labs/aero/sims/arigo2_all.sh 16         16 CFD processors
#   SKIP_CFD=1 / SKIP_VIDEO=1 labs/aero/sims/arigo2_all.sh   leave a stage out;  FRESH=1 ... start over
# Stop any time (Ctrl-C) and run again: every stage resumes where it stopped. Each line shows the time left.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
NP="${1:-8}"
F=(); [ -n "${FRESH:-}" ] && F=(--fresh)
status=0
echo "== stage 1/3: MuJoCo stability + engine out, 6 runs (~1 min)"
"$PY" "$HERE/arigo2_mujoco.py" "${F[@]}" || status=1
if [ -z "${SKIP_VIDEO:-}" ]; then
  echo "== stage 2/3: manoeuvre videos, 21 (3 wings x 7)"
  MUJOCO_GL="${MUJOCO_GL:-egl}" "$PY" "$HERE/arigo2_video.py" "${F[@]}" || status=1
fi
if [ -z "${SKIP_CFD:-}" ]; then
  echo "== stage 3/3: CFD, 48 cases (33 power off + 15 power on) on $NP processors (hours)"
  "$PY" "$HERE/arigo2_cfd.py" --processors "$NP" "${F[@]}" || status=1
fi
echo "== done (status $status). Paste back:"
ls -1 "$HERE/output"/arigo2_*.yaml 2>/dev/null
exit $status
