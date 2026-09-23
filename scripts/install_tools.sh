#!/usr/bin/env bash
# Install the external engineering tools used by the Vegeta packages (Ubuntu 24.04).
set -euo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  calculix-ccx prusa-slicer openfoam \
  libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1
python3 -m pip install cadquery gmsh numpy matplotlib tqdm pandas pytest nbconvert nbformat ipykernel
