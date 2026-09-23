<p align="center"><img src="logo_with_text.png" alt="Vegeta logo" width="320"></p>

# Vegeta

A Python-first, human-in-the-loop engineering workbench built from small,
independent engineering tools, shipped together as one installable package, **`vegeta-cli`**
(`from vegeta import dedalus, talos, aeromant, mellonia`):

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
- The four engineering tools are **independent** subpackages (`vegeta.dedalus`, ...): none imports
  another or any other part of Vegeta (enforced by tests). `vegeta` is a namespace package so more
  distributions can join it later.
- Each is a plain Python library, usable from a script or Jupyter, with a thin CLI on top
  (`vegeta <tool> ...`, or the shortcuts `dedalus`, `talos`, `aeromant`, `mellonia`).
  No server, database or background service. A GUI comes later, on top of the same API.
- Configuration happens in **Python** (dataclasses / keyword arguments), not config files.
- Packages compose through **files**: STEP, STL, meshes, CalculiX decks, OpenFOAM cases, G-code.
- Native solver artifacts are always kept; `summary.json` is a convenience, not a replacement.
- Nothing is guessed: supports, loads, materials, CFD values and print settings are explicit.
- Nothing runs automatically: every analysis is started by the engineer.

## Status
Stages 1–5 are implemented: the four independent tools, each with a Python API, a CLI, tests
validated against known solutions, documentation and a notebook. Vegeta itself (revisions,
comparison, experiments, a GUI later) comes next — see [`todo.md`](todo.md).

| package | CLI | validation | guide | notebook |
|---------|-----|------------|-------|----------|
| Dedalus | `vegeta dedalus params/generate/measure` | exact box/hole volumes, STEP round trip | [docs/dedalus.md](docs/dedalus.md) | `01_dedalus_cad` |
| Talos | `vegeta talos inspect/mesh/solve/results` | cantilever vs beam theory (−0.6 %), bar F/A, gravity | [docs/talos.md](docs/talos.md) | `02_talos_fea` |
| Aeromant | `vegeta aeromant templates/prepare/run/results` | sphere Re=100 Cd 1.100 vs 1.092 | [docs/aeromant.md](docs/aeromant.md) | `03_aeromant_cfd` |
| Mellonia | `vegeta mellonia slice/parse` | layer counts, solid cube volume (+0.9 %) | [docs/mellonia.md](docs/mellonia.md) | `04_mellonia_print` |

More: [philosophy](docs/philosophy.md) · [result shape](docs/result-shape.md) ·
[installation](docs/installation.md) · [composition through files](docs/composition.md).

## Layout
```
vegeta-cli/        the vegeta-cli package: src/vegeta/{cli,dedalus,talos,aeromant,mellonia}, tests/<tool>/
notebooks/         one notebook per package
docs/              philosophy, result shape, installation, composition, per-package guides
examples/cli/      input files for the CLI demo (Talos model, Aeromant case, Mellonia settings)
scripts/           install_ubuntu.sh, test_all.sh, run_notebooks.sh, demo_cli.sh
```

## Quick start
```bash
scripts/install_ubuntu.sh           # apt tools, OpenFOAM (conda-forge), .venv with vegeta-cli
source .venv/bin/activate
scripts/test_all.sh                 # each package tested on its own
scripts/run_notebooks.sh            # execute the notebooks headlessly
scripts/demo_cli.sh                 # CAD -> FEA -> print -> CFD using only the CLIs
```

## Licence
Proprietary — **internal use only** (see `LICENSE`). External tools keep their own licences, see
`THIRD_PARTY_LICENSES.md`.
