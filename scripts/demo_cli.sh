#!/usr/bin/env bash
# Demonstrates composition through files using only the vegeta command (`vegeta <tool>` = `<tool>`). Each command is an explicit,
# separate step; stop anywhere. Outputs go to runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
EX=examples/cli

# 1. Dedalus: geometry -> STEP/STL
vegeta dedalus generate vegeta.dedalus.examples:Bracket -o runs/bracket --png
vegeta dedalus generate vegeta.dedalus.examples:StreamlinedBody -o runs/body --formats stl --stl-tolerance 0.05

# 2. Talos consumes the STEP
vegeta talos inspect runs/bracket/Bracket.step --units mm-N-MPa
vegeta talos mesh  $EX/bracket_fea.py -w runs/bracket_fea -q
vegeta talos solve $EX/bracket_fea.py -w runs/bracket_fea -q --png

# 3. Mellonia consumes the STL (orientation chosen here: as modelled)
vegeta mellonia slice runs/bracket/Bracket.stl -s $EX/print_settings.py:FINE -o runs/bracket_print --png

# 4. Aeromant consumes the other STL
vegeta aeromant prepare $EX/body_cfd.py --overwrite
vegeta aeromant run $EX/body_cfd.py -q
vegeta aeromant results runs/body_cfd --png
