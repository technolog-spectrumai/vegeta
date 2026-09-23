# Vegeta — TODO

Working order: prototype libraries first (Python API), test each through its CLI, integrate a GUI later.
Rules: configuration through the Python API (no YAML/JSON input config unless an external tool
forces it), standalone packages never import each other or Vegeta, every stage ends with tests + docs.

Legend: `[ ]` open · `[x]` done · `[-]` deferred

## Stage 1 — Common philosophy
- [x] 1.1 Repository layout (`vegeta-cli/`, `notebooks/`, `docs/`, `scripts/`), README, this todo
- [x] 1.2 Per-package `_process.py`: `run_command` recording command, cwd, return code, duration, stdout, stderr, log file
- [x] 1.3 Per-package `result.py`: common result shape (status, metrics, artifacts, messages, duration, execution, metadata)
- [x] 1.4 Error policy: invalid explicit config → `ValueError` at construction; execution failures → `failed` result
- [x] 1.5 `docs/philosophy.md`, `docs/result-shape.md`, `docs/installation.md`, `install_local.sh`, `scripts/test_all.sh`

## Stage 2 — Dedalus (CadQuery CAD)
- [x] 2.1 Package skeleton (`pyproject.toml`, deps: cadquery, numpy, matplotlib, tqdm; optional pandas)
- [x] 2.2 `Parameter` and parameter-set validation
- [x] 2.3 `Design` (subclass + decorator), `generate(**overrides)`, source identity hash
- [x] 2.4 `Geometry`: measurements (bbox, dimensions, volume, area, center of mass, validity)
- [x] 2.5 STEP/STL export, `GenerationResult`, `summary.json`
- [x] 2.6 Jupyter display via CadQuery's own visualisation; `plot_*` helpers (matplotlib)
- [x] 2.7 Example designs: `CantileverBeam`, `Bracket`, `StreamlinedBody`, `Cube`
- [x] 2.8 CLI: `dedalus generate`, `dedalus measure` (`--json`)
- [x] 2.9 Tests (unit + CadQuery integration + CLI), `docs/dedalus.md`, `notebooks/01_dedalus_cad.ipynb`

## Stage 3 — Talos (Gmsh + CalculiX, linear static)
- [x] 3.1 Package skeleton (deps: gmsh, numpy, matplotlib, tqdm)
- [x] 3.2 Units (`mm-N-MPa`, `m-N-Pa`) and `Material` (explicit E, ν, optional density, yield)
- [x] 3.3 `inspect_step`: surface/volume table for region selection (any STEP)
- [x] 3.4 Regions: `Surfaces`, `SurfacesInBox`, `SurfacesOnPlane` (empty selection is an error)
- [x] 3.5 Supports and loads: `FixedSupport`, `Displacement`, `Force`, `Pressure`, `Acceleration`
- [x] 3.6 Meshing with Gmsh (`MeshSettings`), `mesh.msh`, CalculiX `.inp` writer
- [x] 3.7 `StructuralModel.mesh()` / `.solve()` (separate explicit steps), ccx runner
- [x] 3.8 FRD/DAT parsers, `StructuralResult` (max displacement, von Mises, reactions, safety factor only with yield)
- [x] 3.9 CLI: `talos inspect`, `talos mesh`, `talos solve`
- [x] 3.10 Tests: unit (writers/parsers), integration (STEP→Gmsh, cantilever vs beam theory, axial bar, reactions), CLI; `docs/talos.md`, `notebooks/02_talos_fea.ipynb`

## Stage 4 — Aeromant (OpenFOAM, template cases)
- [x] 4.1 Package skeleton (deps: numpy, matplotlib, tqdm)
- [x] 4.2 STL utilities (read/write, bbox, explicit unit scaling)
- [x] 4.3 `TemplateSpec` (Python) + template cases: `laminar_external`, `rans_ksst_external`
- [x] 4.4 `CFDCase`: prepare (copy, render placeholders, insert geometry, domain check)
- [x] 4.5 Pipeline runner: blockMesh, features, snappyHexMesh, checkMesh, solver; `OpenFOAMEnvironment` (openfoam.com and openfoam.org case files)
- [x] 4.6 Results: forceCoeffs parsing (Cd/Cl/Cm), checkMesh summary, residual/coefficient history plots
- [x] 4.7 CLI: `aeromant templates`, `prepare`, `run`, `results`
- [x] 4.8 Tests: unit (rendering, STL, parsers), integration (small known case, skip without OpenFOAM), CLI; `docs/aeromant.md`, `notebooks/03_aeromant_cfd.ipynb`

## Stage 5 — Mellonia (PrusaSlicer)
- [x] 5.1 Package skeleton (deps: numpy, matplotlib, tqdm)
- [x] 5.2 `PrintSettings` (Python dicts → ini artifacts; `from_ini` for existing exports), example settings
- [x] 5.3 `Orientation` (engineer-chosen rotations, no optimisation)
- [x] 5.4 `slice_stl` via PrusaSlicer CLI, G-code preserved
- [x] 5.5 G-code parser: print time, filament (mm/cm³/g), cost, layer count, max Z
- [x] 5.6 CLI: `mellonia slice`, `mellonia parse`
- [x] 5.7 Tests: unit (parser, settings), integration (20 mm cube layer count/height, skip without PrusaSlicer), CLI; `docs/mellonia.md`, `notebooks/04_mellonia_print.ipynb`

## Stage 6 — Consistent interfaces
- [x] 6.1 Result-shape conformance test in every package against `docs/result-shape.md`
- [x] 6.2 Import-isolation test (no cross-package imports)
- [x] 6.3 File-based composition documented (STEP → Talos, STL → Aeromant/Mellonia)
- [x] 6.4 One installable distribution `vegeta-cli`: `vegeta.dedalus/talos/aeromant/mellonia` in a `vegeta` namespace, `vegeta <tool>` command plus shortcuts

## Stage 7 — Jupyter as first-class interface
- [x] 7.1 Per-package notebooks executed headlessly in tests (done per stage above)
- [x] 7.2 Combined workflow notebook (CAD → inspect → FEA → CFD → slice, every step user-initiated)
- [x] 7.3 Interactive visualisation (pyvista/trame) per package; product notebooks `08_quadcopter`, `09_fixed_wing_drone`

## Stage 8 — Vegeta Core (`vegeta-core`, branch `dev_core`)
- [x] 8.1 Workspaces, designs, revisions (immutable), evaluations, artifacts
- [x] 8.2 Branching from revisions, human labels (preferred/rejected/reference/unclassified)
- [x] 8.3 Reproducibility metadata (parameters, configs, tool versions, commands, timestamps)
- [x] 8.4 Invokes the four packages exactly as a Python user would

## Stage 9 — Open-loop workflow
- [x] 9.1 No automatic chaining (CAD ↛ FEA ↛ CFD); missing analyses shown as NOT RUN
- [x] 9.2 Vegeta CLI for the workflow (design → generate → inspect → choose → run → compare)

## Stage 10 — Vegeta Studio (deferred; CLI first, GUI integration later)
- [-] 10.1 Thin GUI over Vegeta Core and the four packages; no engineering logic unavailable from Python

## Stage 11 — Comparison and experiments (deferred)
- [-] 11.1 Compare revisions and existing results without generating missing analyses
- [-] 11.2 Explicit parameter sweeps (not an optimizer), full reproducibility records

## Stage 12 — AI assistance (deferred)
- [x] 12.1 `vegeta-ai`: `Proposer` protocol, Claude adapter (`ClaudeConfig`: api key, model, effort), `DesignSession` — proposals built, measured, diffed, explicitly accepted/rejected
- [ ] 12.2 OpenAI adapter; parameter-only proposals against `vegeta.core` criteria (see `plan.md` step 3)

## Stage 13 — Validation (continuous)
- [x] 13.1 Unit tests in every package
- [x] 13.2 Integration tests with numerical validation; clean skips when tools are missing
- [ ] 13.3 After every stage: run tests, fix regressions, update/remove documentation (ongoing; done for stages 1–9)
