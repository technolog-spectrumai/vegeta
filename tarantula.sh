#!/usr/bin/env bash
# TARANTULA's gait and stability simulations (MuJoCo + kinematics + statics), run in parallel on your CPU cores;
# results as YAML.
#
#   ./tarantula.sh list                                 the simulations and their case counts
#   ./tarantula.sh run all -j 8 --quick                 a small grid first: checks the setup in a minute or two
#   ./tarantula.sh run all -j 8                         everything, on 8 cores (default: all cores but one)
#   ./tarantula.sh run walk slope -j 6                  some of them (the "tarantula-" prefix is optional)
#   ./tarantula.sh run all -j 8 --out runs/tarantula/mine   choose the output folder
#   ./tarantula.sh video                                a MuJoCo video: harpoon, down and up the wall on the dragline
#   ./tarantula.sh mission                              a MuJoCo short: off a roof on the dragline, a gift for a bandit
#
# Simulations: gaits (kinematic checks: IK, joint limits, static stability, tip-over angles), walk (flat ground:
# gait × posture × payload × friction), slope (inclines 0–40° up, across and down), rough (rough ground 10–35 mm RMS
# and steps 30–90 mm), push (push recovery 10–80 N), faults (a limp or locked leg), climb (45–90° with pads, spines
# or magnets), attach (does it need adhesives: the statics of five feet on ten surfaces).
# Output: <out>/tarantula-<name>.yaml (meta, summary, every case) and <out>/index.yaml; default
# runs/tarantula/<date-time>/. Uses the project's virtual environment (.venv, or $VEGETA_VENV): ./install_local.sh
# sets it up; needs numpy and mujoco (pip install mujoco).
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")"

VENV="${VEGETA_VENV:-$PWD/.venv}"
if [ -x "$VENV/bin/python" ]; then
  py="$VENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="$(command -v python3)"
  echo "tarantula.sh: no virtual environment at $VENV — using $py (run ./install_local.sh for the full setup)" >&2
else
  echo "tarantula.sh: no Python found — run ./install_local.sh first (or export VEGETA_VENV=/path/to/venv)" >&2
  exit 1
fi
export VEGETA_ROOT="$PWD"
export PYTHONPATH="$PWD/libs/vegeta-cli/src:$PWD/libs/labkit/src:$PWD/libs/ogma/src:$PWD/vendor/yax/src${PYTHONPATH:+:$PYTHONPATH}"
# one thread per worker: -j sets the parallelism
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,20p' "$0"
  exit 0
fi
missing=$("$py" - <<'PY'
import importlib.util
print(" ".join(m for m in ("numpy", "mujoco") if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$missing" ]; then
  echo "tarantula.sh: missing Python packages: $missing  →  $py -m pip install $missing" >&2
  exit 1
fi
runner=labs/legged/sims/spider_sims.py
case "$1" in
  list)
    exec "$py" "$runner" list tarantula ;;
  video)
    shift
    exec "$py" labs/legged/sims/tarantula_video.py "$@" ;;
  mission)
    shift
    exec "$py" labs/legged/sims/tarantula_mission.py "$@" ;;
  run)
    shift
    args=()
    names=0
    while [ $# -gt 0 ]; do
      case "$1" in
        -j|--jobs|--out) args+=("$1" "$2"); shift 2 ;;
        -*) args+=("$1"); shift ;;
        all|tarantula) args+=("tarantula"); names=1; shift ;;
        tarantula-*) args+=("$1"); names=1; shift ;;
        *) args+=("tarantula-$1"); names=1; shift ;;
      esac
    done
    [ "$names" -eq 1 ] || args=("tarantula" "${args[@]}")
    exec "$py" "$runner" run "${args[@]}" ;;
  *)
    echo "tarantula.sh: unknown command '$1' (list | run | video | mission)" >&2
    exit 2 ;;
esac
