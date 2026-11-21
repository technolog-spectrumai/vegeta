#!/usr/bin/env bash
# The spider simulations (ARACHNE, SALTY, the actuation study), run in parallel on your CPU cores; results as YAML.
#
#   ./spider.sh list                                   the simulations and their case counts
#   ./spider.sh run all -j 8                           everything, on 8 cores (default: all cores but one)
#   ./spider.sh run all -j 8 --quick                   a small grid first: checks the setup in a few minutes
#   ./spider.sh run arachne-climb salty-jump -j 6      some of them
#   ./spider.sh run all -j 8 --out runs/spider/mine    choose the output folder (default runs/spider/<date-time>)
#
# Simulations: arachne-climb (MuJoCo dynamics on a wall), arachne-statics (adhesion margins, Monte Carlo),
# arachne-gaits (kinematic gait checks), spider-actuation (electric vs fluid joints over robot mass),
# salty-jump (SALTY's hydraulic catapult). Each writes <out>/<name>.yaml (summary + every case); <out>/index.yaml
# lists them. Uses the project's virtual environment (.venv, or $VEGETA_VENV): ./install_local.sh sets it up; the
# simulations need numpy and mujoco (pip install mujoco) — nothing else.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")"

VENV="${VEGETA_VENV:-$PWD/.venv}"
if [ -x "$VENV/bin/python" ]; then
  py="$VENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="$(command -v python3)"
  echo "spider.sh: no virtual environment at $VENV — using $py (run ./install_local.sh for the full setup)" >&2
else
  echo "spider.sh: no Python found — run ./install_local.sh first (or export VEGETA_VENV=/path/to/venv)" >&2
  exit 1
fi
export VEGETA_ROOT="$PWD"
export PYTHONPATH="$PWD/libs/vegeta-cli/src:$PWD/libs/labkit/src:$PWD/libs/ogma/src:$PWD/vendor/yax/src${PYTHONPATH:+:$PYTHONPATH}"
# one thread per worker: -j sets the parallelism
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,15p' "$0"
  exit 0
fi
missing=$("$py" - <<'PY'
import importlib.util
print(" ".join(m for m in ("numpy", "mujoco") if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$missing" ]; then
  echo "spider.sh: missing Python packages: $missing  →  $py -m pip install $missing" >&2
  exit 1
fi
exec "$py" labs/legged/sims/spider_sims.py "$@"
