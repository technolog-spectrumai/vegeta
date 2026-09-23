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

## Status
Stages 1–5 are implemented: the four standalone packages, each with a Python API, a CLI, tests
validated against known solutions, documentation and a notebook. Vegeta itself (revisions,
comparison, experiments, a GUI later) comes next — see [`todo.md`](todo.md).

| package | CLI | validation | guide | notebook |
|---------|-----|------------|-------|----------|
| Dedalus | `dedalus params/generate/measure` | exact box/hole volumes, STEP round trip | [docs/dedalus.md](docs/dedalus.md) | `01_dedalus_cad` |
| Talos | `talos inspect/mesh/solve/results` | cantilever vs beam theory (−0.6 %), bar F/A, gravity | [docs/talos.md](docs/talos.md) | `02_talos_fea` |
| Aeromant | `aeromant templates/prepare/run/results` | sphere Re=100 Cd 1.100 vs 1.092 | [docs/aeromant.md](docs/aeromant.md) | `03_aeromant_cfd` |
| Mellonia | `mellonia slice/parse` | layer counts, solid cube volume (+0.9 %) | [docs/mellonia.md](docs/mellonia.md) | `04_mellonia_print` |

More: [philosophy](docs/philosophy.md) · [result shape](docs/result-shape.md) ·
[installation](docs/installation.md) · [composition through files](docs/composition.md).

## Layout
```
packages/<name>/   pyproject.toml, src/<name>/, tests/     (dedalus, talos, aeromant, mellonia)
notebooks/         one notebook per package
docs/              philosophy, result shape, installation, composition, per-package guides
examples/cli/      input files for the CLI demo (Talos model, Aeromant case, Mellonia settings)
scripts/           install_tools.sh, test_all.sh, run_notebooks.sh, demo_cli.sh
```

## Quick start
```bash
scripts/install_tools.sh            # pip + apt dependencies (Ubuntu) and OpenFOAM from conda-forge
pip install -e packages/dedalus -e packages/talos -e packages/aeromant -e packages/mellonia
scripts/test_all.sh                 # each package tested on its own
scripts/run_notebooks.sh            # execute the notebooks headlessly
scripts/demo_cli.sh                 # CAD -> FEA -> print -> CFD using only the CLIs
```

## Licence
MIT (see `LICENSE`). External tools keep their own licences, see `THIRD_PARTY_LICENSES.md`.
