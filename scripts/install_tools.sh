#!/usr/bin/env bash
# Install the external engineering tools used by the Vegeta packages (Ubuntu 24.04).
set -euo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  calculix-ccx prusa-slicer \
  libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1
python3 -m pip install cadquery gmsh numpy matplotlib tqdm pandas pytest nbconvert nbformat ipykernel
# OpenFOAM: the Ubuntu 'openfoam' package (v1912) cannot run function objects; use conda-forge.
if [ ! -x /opt/foam/bin/simpleFoam ]; then
  mkdir -p /opt/mm
  curl -Ls https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2 | tar -xj -C /opt/mm bin/micromamba
  MAMBA_ROOT_PREFIX=/opt/mm/root /opt/mm/bin/micromamba create -y -p /opt/foam -c conda-forge openfoam=2412
fi
echo "OpenFOAM: aeromant.OpenFOAMEnvironment.conda('/opt/foam') (also found by detect())"
