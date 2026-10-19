#!/usr/bin/env bash
# Tune the MuJoCo autopilot of the reference quad per race course (todo R.2). Real MuJoCo, no mocks; no solver.
#   ./tune_autopilot.sh                 # all four courses, the full grid (72 laps each; roughly 1–2 h in total)
#   ./tune_autopilot.sh --quick         # 8 laps per course, to see it work
#   ./tune_autopilot.sh --courses hangar quarry
# Results: races/runs/tune/<course>.csv (every try) and races/runs/tune/best.json (the winning settings per course).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$ROOT/activate.sh" ] && source "$ROOT/activate.sh" >/dev/null
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONPATH="$ROOT:$ROOT/designs:${PYTHONPATH:-}"
cd "$ROOT"
python -m races.tune "$@" 2>&1 | tee "races/runs/tune_$(date +%Y%m%d_%H%M%S).log"
