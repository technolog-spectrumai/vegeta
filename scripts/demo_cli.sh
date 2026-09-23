#!/usr/bin/env bash
# Demonstrates composition through files using only the four CLIs. Each command is an explicit,
# separate step; stop anywhere. Outputs go to runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
EX=examples/cli

# 1. Dedalus: geometry -> STEP/STL
dedalus generate dedalus.examples:Bracket -o runs/bracket --png
dedalus generate dedalus.examples:StreamlinedBody -o runs/body --formats stl --stl-tolerance 0.05

# 2. Talos consumes the STEP
talos inspect runs/bracket/Bracket.step --units mm-N-MPa
talos mesh  $EX/bracket_fea.py -w runs/bracket_fea -q
talos solve $EX/bracket_fea.py -w runs/bracket_fea -q --png

# 3. Mellonia consumes the STL (orientation chosen here: as modelled)
mellonia slice runs/bracket/Bracket.stl -s $EX/print_settings.py:FINE -o runs/bracket_print --png

# 4. Aeromant consumes the other STL
aeromant prepare $EX/body_cfd.py --overwrite
aeromant run $EX/body_cfd.py -q
aeromant results runs/body_cfd --png
