#!/usr/bin/env bash
# Install everything the Vegeta packages need on Ubuntu (tested on 24.04; 22.04 should work):
#   - first, a Python virtual environment (default: <repo>/.venv)
#   - system tools: CalculiX (ccx), PrusaSlicer, Gmsh/CadQuery runtime libraries, xvfb
#   - OpenFOAM v2412 from conda-forge (the Ubuntu 'openfoam' package cannot run forceCoeffs)
#   - into the virtual environment: vegeta-cli (editable: vegeta.dedalus/talos/aeromant/mellonia)
#     plus JupyterLab and test tools, registered as the Jupyter kernel "Python (vegeta)"
# Then it checks every component (./test.sh --quick); ./test.sh runs a full working check.
#
# Usage: ./install_local.sh [--venv DIR] [--python EXE] [--system-python] [--skip-system] [--no-openfoam] [--openfoam-prefix DIR]
#   --python EXE   interpreter for the venv (default: Ubuntu's /usr/bin/python3, not a conda one)
#   --skip-system  skip apt and OpenFOAM installation (tools already installed or no sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
USE_VENV=1
PYTHON_EXE=""
WITH_OPENFOAM=1
WITH_SYSTEM=1
FOAM_PREFIX=/opt/foam
MAMBA_DIR=/opt/micromamba
MICROMAMBA_URL=https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --python) PYTHON_EXE="$2"; shift 2 ;;
    --system-python) USE_VENV=0; shift ;;
    --no-openfoam) WITH_OPENFOAM=0; shift ;;
    --skip-system) WITH_SYSTEM=0; shift ;;
    --openfoam-prefix) FOAM_PREFIX="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"
step() { printf '\n==> %s\n' "$*"; }

if [ -r /etc/os-release ]; then
  . /etc/os-release
  [ "${ID:-}" = "ubuntu" ] || echo "warning: this script targets Ubuntu, found ${ID:-unknown}" >&2
fi

# Use Ubuntu's Python by default: an active conda env (e.g. "(base)") may provide a Python version
# for which CadQuery/OCP wheels do not exist yet.
if [ -z "$PYTHON_EXE" ]; then
  if [ -x /usr/bin/python3 ]; then PYTHON_EXE=/usr/bin/python3; else PYTHON_EXE="$(command -v python3)"; fi
fi
[ -n "${CONDA_PREFIX:-}" ] && echo "note: conda environment '$CONDA_PREFIX' is active; the venv is built from $PYTHON_EXE instead"

step "Virtual environment"
if [ "$USE_VENV" -eq 1 ]; then
  if ! "$PYTHON_EXE" -c "import venv, ensurepip" >/dev/null 2>&1; then
    [ "$WITH_SYSTEM" -eq 1 ] || { echo "python3-venv is missing; install it or drop --skip-system" >&2; exit 1; }
    $SUDO apt-get update
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip
  fi
  # An existing venv built from another Python (e.g. conda's) is rebuilt from $PYTHON_EXE.
  want="$("$PYTHON_EXE" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [ -x "$VENV/bin/python" ]; then
    have="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo broken)"
    if [ "$have" != "$want" ]; then
      echo "existing venv uses Python $have, rebuilding it with Python $want"
      rm -rf "$VENV"
    fi
  fi
  "$PYTHON_EXE" -m venv "$VENV"
  PY="$VENV/bin/python"
  echo "virtual environment: $VENV (activate with: source $VENV/bin/activate)"
else
  PY="$PYTHON_EXE"
  echo "using $PY (system Python; Ubuntu may refuse pip installs here — prefer the default venv)"
fi

if [ "$WITH_SYSTEM" -eq 1 ]; then
  step "System packages (apt)"
  $SUDO apt-get update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    curl bzip2 ca-certificates git \
    calculix-ccx prusa-slicer xvfb \
    libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libxext6 libxi6

  if [ "$WITH_OPENFOAM" -eq 1 ]; then
    step "OpenFOAM v2412 (conda-forge) in $FOAM_PREFIX"
    if [ -x "$FOAM_PREFIX/bin/simpleFoam" ]; then
      echo "already installed"
    else
      $SUDO mkdir -p "$MAMBA_DIR"
      curl -Ls "$MICROMAMBA_URL" | $SUDO tar -xj -C "$MAMBA_DIR" bin/micromamba
      $SUDO env MAMBA_ROOT_PREFIX="$MAMBA_DIR/root" "$MAMBA_DIR/bin/micromamba" create -y -p "$FOAM_PREFIX" \
        -c conda-forge openfoam=2412
    fi
  fi
else
  step "Skipping system packages and OpenFOAM (--skip-system)"
fi

step "Python packages (vegeta-cli, JupyterLab)"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install \
  -e "$ROOT/vegeta-cli[pandas,test]" \
  jupyterlab ipywidgets nbconvert nbformat ipykernel
"$PY" -m ipykernel install --user --name vegeta --display-name "Python (vegeta)"

step "Checks"
status=0
"$ROOT/test.sh" --quick --python "$PY" || status=1

if [ $status -eq 0 ]; then
  printf '\nAll dependencies installed. Next:\n'
  echo "  ./jupyter.sh             # JupyterLab from the venv; open notebooks/00_smoke_test.ipynb"
  [ "$USE_VENV" -eq 1 ] && echo "  source $VENV/bin/activate   # for the vegeta command and scripts"
  echo "  ./test.sh                 # installation check with a short working run"
  echo "  scripts/test_all.sh      # full test suite"
  echo "  scripts/demo_cli.sh      # CAD -> FEA -> print -> CFD with the CLIs"
else
  printf '\nSome checks failed (see MISSING above).\n' >&2
fi
exit $status
