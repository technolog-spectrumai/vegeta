# Composition through files

The four packages never import each other. They compose through standard files:

```
Dedalus ──STEP──▶ Talos      (any STEP from any CAD program works)
        ──STL───▶ Aeromant   (any STL; units stated explicitly)
        ──STL───▶ Mellonia   (any STL)
```

This loose coupling is intentional: each tool can be replaced, run on files from elsewhere, or
skipped. Nothing triggers anything else — the engineer runs each step.

## From the command line
`scripts/demo_cli.sh` runs a complete example with the CLIs only (inputs in `examples/cli/`):

| step | command | reads | writes (in `runs/`) |
|------|---------|-------|---------------------|
| CAD | `dedalus generate dedalus.examples:Bracket -o runs/bracket --png` | Python design | `Bracket.step`, `Bracket.stl`, views PNG, `summary.json` |
| inspect | `talos inspect runs/bracket/Bracket.step --units mm-N-MPa` | STEP | surface table (choose regions) |
| mesh | `talos mesh examples/cli/bracket_fea.py -w runs/bracket_fea` | STEP + model | `mesh.msh`, `gmsh.log` |
| FEA | `talos solve examples/cli/bracket_fea.py -w runs/bracket_fea --png` | mesh + model | `model.inp/.frd/.dat`, logs, PNGs |
| print | `mellonia slice runs/bracket/Bracket.stl -s examples/cli/print_settings.py:FINE -o runs/bracket_print` | STL + settings | G-code, `effective_config.ini` |
| CFD | `aeromant prepare examples/cli/body_cfd.py` then `aeromant run ...` | STL + template + values | full OpenFOAM case, `log.*` |
| results | `aeromant results runs/body_cfd --png` / `talos results runs/bracket_fea` | existing outputs | nothing new is run |

Every command prints a readable summary, `--json` gives the machine-readable one, and the exit code
is 0 for success, 1 for a failed run and 2 for invalid input.

## From Python / Jupyter
The same steps are plain function calls; see the per-package notebooks in `notebooks/`.
```python
g = dedalus.examples.Bracket().generate(thickness=8)
res = g.export("runs/bracket_t8")                          # STEP + STL
model = talos.StructuralModel(geometry=res.artifacts["step"], ...)
model.mesh("runs/bracket_t8_fea"); model.solve("runs/bracket_t8_fea")
mellonia.slice_stl(res.artifacts["stl"], settings, mellonia.Orientation(), "runs/bracket_t8_print")
```
