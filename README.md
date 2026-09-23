# Vegeta

A Python-first, human-in-the-loop engineering workbench built from small,
independent engineering packages:

| Package | Job | Wraps | Consumes → Produces |
|---------|-----|-------|---------------------|
| **Dedalus** | parametric CAD | CadQuery | Python design → STEP, STL, measurements |
| **Talos** | linear static FEA | Gmsh + CalculiX | STEP + explicit config → mesh, `.inp`, `.frd`/`.dat`, summary |
| **Aeromant** | aerodynamics / CFD | OpenFOAM | STL + template case + explicit values → case, logs, Cd/Cl/Cm |
| **Mellonia** | 3D-print manufacturability | PrusaSlicer | STL + explicit print settings → G-code, time, material |
| **Vegeta** | workbench (later) | the four above | revisions, artifacts, comparison |

Dedalus designs. Talos tests structures. Aeromant tests aerodynamics.
Mellonia tests manufacturability. Vegeta organizes the engineer's work.

## Principles
- The four engineering packages are **standalone**: none imports another or Vegeta.
- Each is a plain Python library, usable from a script or Jupyter, with a thin CLI on top.
  No server, database or background service. A GUI comes later, on top of the same API.
- Configuration happens in **Python** (dataclasses / keyword arguments), not config files.
- Packages compose through **files**: STEP, STL, meshes, CalculiX decks, OpenFOAM cases, G-code.
- Native solver artifacts are always kept; `summary.json` is a convenience, not a replacement.
- Nothing is guessed: supports, loads, materials, CFD values and print settings are explicit.
- Nothing runs automatically: every analysis is started by the engineer.

See [`docs/philosophy.md`](docs/philosophy.md), [`docs/result-shape.md`](docs/result-shape.md)
and [`docs/installation.md`](docs/installation.md). Progress is tracked in [`todo.md`](todo.md).

## Layout
```
packages/<name>/   pyproject.toml, src/<name>/, tests/     (dedalus, talos, aeromant, mellonia)
notebooks/         one notebook per package
docs/              philosophy, result shape, installation, per-package guides
scripts/           install_tools.sh, test_all.sh
```

## Quick start
```bash
scripts/install_tools.sh            # pip + apt dependencies (Ubuntu)
pip install -e packages/dedalus -e packages/talos -e packages/aeromant -e packages/mellonia
scripts/test_all.sh                 # each package tested on its own
```

## Licence
MIT (see `LICENSE`). External tools keep their own licences, see `THIRD_PARTY_LICENSES.md`.
