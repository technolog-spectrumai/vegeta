#!/usr/bin/env bash
# Install everything the Vegeta packages need on Ubuntu (tested on 24.04; 22.04 should work):
#   - system tools: CalculiX (ccx), PrusaSlicer, Gmsh/CadQuery runtime libraries, xvfb
#   - OpenFOAM v2412 from conda-forge (the Ubuntu 'openfoam' package cannot run forceCoeffs)
#   - a Python virtual environment with vegeta-cli (editable: vegeta.dedalus/talos/aeromant/mellonia)
#     plus Jupyter and test tools
# Then it checks that every tool is usable.
#
# Usage: scripts/install_ubuntu.sh [--venv DIR] [--system-python] [--no-openfoam] [--openfoam-prefix DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
USE_VENV=1
WITH_OPENFOAM=1
FOAM_PREFIX=/opt/foam
MAMBA_DIR=/opt/micromamba
MICROMAMBA_URL=https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --system-python) USE_VENV=0; shift ;;
    --no-openfoam) WITH_OPENFOAM=0; shift ;;
    --openfoam-prefix) FOAM_PREFIX="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
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

step "Python environment"
if [ "$USE_VENV" -eq 1 ]; then
  python3 -m venv "$VENV"
  PY="$VENV/bin/python"
  echo "virtual environment: $VENV (activate with: source $VENV/bin/activate)"
else
  PY="$(command -v python3)"
  echo "using $PY (system Python; Ubuntu may refuse pip installs here — prefer the default venv)"
fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install \
  -e "$ROOT/vegeta-cli[pandas,test]" \
  jupyterlab nbconvert nbformat ipykernel

step "Checks"
status=0
check() {  # check <label> <command...>
  local label="$1"; shift
  if out="$("$@" 2>&1)"; then printf '  ok       %-12s %s\n' "$label" "$(echo "$out" | head -1)"
  else printf '  MISSING  %-12s %s\n' "$label" "$(echo "$out" | tail -1)"; status=1; fi
}
check cadquery   "$PY" -c "import cadquery; print(cadquery.__version__)"
check gmsh       "$PY" -c "import gmsh; print(gmsh.__version__)"
check calculix   bash -c "ccx -v | grep -i version"
check prusaslicer bash -c "prusa-slicer --help | head -1"
if [ "$WITH_OPENFOAM" -eq 1 ]; then
  check openfoam "$PY" -c "
import os, subprocess
from vegeta import aeromant
e = aeromant.OpenFOAMEnvironment.conda('$FOAM_PREFIX')
r = subprocess.run(e.command(['simpleFoam', '-help']), env={**os.environ, **e.env}, capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-300:]
print('simpleFoam in $FOAM_PREFIX; use aeromant.OpenFOAMEnvironment.conda(\'$FOAM_PREFIX\')')"
fi
for tool in dedalus talos aeromant mellonia; do
  check "$tool" "$PY" -c "from vegeta import $tool; print('vegeta.$tool', $tool.__version__)"
done
check vegeta "$(dirname "$PY")/vegeta" --version

if [ $status -eq 0 ]; then
  printf '\nAll dependencies installed. Next:\n'
  [ "$USE_VENV" -eq 1 ] && echo "  source $VENV/bin/activate"
  echo "  scripts/test_all.sh      # tests"
  echo "  scripts/demo_cli.sh      # CAD -> FEA -> print -> CFD with the CLIs"
else
  printf '\nSome checks failed (see MISSING above).\n' >&2
fi
exit $status
