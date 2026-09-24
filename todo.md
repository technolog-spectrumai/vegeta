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
- [x] 4.9 `test_openfoam.sh`: per-flavour check of both templates on the installations of the machine (the `org/` case files are validated this way on an openfoam.org machine, not in CI)

## Stage 5 — Mellonia (PrusaSlicer)
- [x] 5.1 Package skeleton (deps: numpy, matplotlib, tqdm)
- [x] 5.2 `PrintSettings` (Python dicts → ini artifacts; `from_ini` for existing exports), example settings
- [x] 5.3 `Orientation` (engineer-chosen rotations, no optimisation)
- [x] 5.4 `slice_stl` via PrusaSlicer CLI, G-code preserved
- [x] 5.5 G-code parser: print time, filament (mm/cm³/g), cost, layer count, max Z
- [x] 5.6 CLI: `mellonia slice`, `mellonia parse`
- [x] 5.7 Tests: unit (parser, settings), integration (20 mm cube layer count/height, skip without PrusaSlicer), CLI; `docs/mellonia.md`, `notebooks/04_mellonia_print.ipynb`

## Stage 5b — Boreas (propeller/rotor BEMT, motor, battery)
- [x] 5b.1 `vegeta.boreas`: `Propeller`, `Airfoil`, `solve`/`rpm_for_thrust`, `Motor`/`Battery`/`Propulsion`, `excitations`, JSON `export`; CLI `boreas point|for-thrust|map`; tests vs scaling laws and an APC static point
- [x] 5b.2 Dedalus `Propeller` example design; propeller parts of `08_quadcopter`, `09_fixed_wing_drone` (formerly notebooks 11, 12) (export JSON + CAD for the mission/fatigue work)
- [x] 5b.3 `boreas.noise`: Gutin tonal harmonics, broadband allowance, cavitation number
- [x] 4.10 Aeromant `rotor_mrf` / `rotor_mrf_static` (MRF propeller, forces → thrust/torque/power/Ct/Cp/η/FM, both dialects); CFD sections in notebooks 11/12 (`VEGETA_SKIP_OPENFOAM=1` skips them) — live validation on a machine with OpenFOAM: `test_openfoam.sh` + the notebook cells

## Stage 5c — Chronos (missions, vibration, cyclic loads, life)
- [x] 5c.1 Talos: `PointMass`, `solve_modes` (CalculiX `*FREQUENCY`, validated vs beam theory and Rayleigh), `read_frd_steps`, `viz.plot_mode`
- [x] 5c.2 Talos: `FatigueCurve`, `assess_fatigue` (unit stress fields × spectrum, Basquin + Goodman + Miner, hotspot map `viz.plot_damage`)
- [x] 5c.3 `vegeta.chronos`: `Mission`/`Segment`/`Excitation`, `Structure` (DAF, margins, Campbell), ASTM rainflow, `build_spectrum` (JSON hand-off), `SNCurve`/`hotspot_damage`, `simulate_life`; CLI `chronos spectrum|life`; tests
- [x] 5c.4 Life parts of `08_quadcopter`, `09_fixed_wing_drone` (formerly notebooks 13, 14): three missions each, vibration → spectra → FEA fatigue → static re-check → fleet life; design comparison

## Stage 5d — Land vehicles (branch `dev_land`)
- [x] 5d.1 `designs/rover.py` (tub chassis, trailing arms, wheels; `part` = rover/chassis/arm) and `11_rover_mechanics` (formerly 15): ISO 8608 terrains + rocks + drop → quarter-car wheel forces → static strength over the operating cases (rocky-field peak, 30° hill climbing with weight transfer and traction limits, mud with sinkage and stall torque, 6 kg payload with a loaded quarter-car re-run) on the arm (FEA), the chassis (FEA: torsion, payload, loaded climb) and the pins (by hand); arm modes and a speed–frequency diagram, rainflow spectra per terrain, fatigue and life, printed arm
- [ ] 5d.2 Motor torque reaction and cornering loads on the arm; chassis drop case with the battery; measured spring/tyre rates

## Stage 5e — Boats and submarines (branch `dev_sea`)
- [x] 5e.1 `designs/survey_boat.py` (lofted hard-chine hull, shell, deck, transom bracket; `part` = boat/hull_solid/hull_shell/bracket) and `12_boat_at_sea` (formerly 16), Part 1: hydrostatics and GZ from the mesh, ITTC resistance + double-body RANS check, Boreas propeller in water, hull/bracket FEA (hydrostatic, slamming, thrust, wave slap), bracket modes, three sea states → spectra → fatigue → life, printed bracket
- [x] 5e.3 `12_boat_at_sea` Part 2 (formerly `17_water_propeller`), plus a flow video, a blade-stress video and one bracket + blade frequency diagram: Boreas in sea water (η/Ct vs J, cavitation number vs rpm, inception rpm), blade CAD, guarded `rotor_mrf` CFD check at cruise, one-blade FEA at bollard pull and its modes vs blade-pass frequency, Gutin tones + broadband in dB re 1 µPa, JSON export
- [x] 5e.4 `designs/submarine.py` (Myring body, sail, fins; `part` = vehicle/body/pressure_hull with truncated ellipsoidal caps) and `13_submarine` (formerly 18): mass budget → buoyancy, trim lead position, BG; ITTC + form factor resistance, guarded `rans_ksst_external` check (nose upstream, L/4 reference length); Boreas thruster, top speed, endurance/range vs speed; pressure hull FEA at 200 m vs thin-shell theory, yield and Windenburg–Trilling collapse; three dive profiles → pressure spectra → damage per dive; the propeller (rotor CFD check with a flow video, Gutin noise and cavitation at two depths, blade FEA with a stress video, wet modes and Campbell); a dive simulation (speed vs depth, pressure and hull stress vs depth, propulsion loss → speed decay and buoyant ascent); JSON export
- [ ] 5e.2 Free-surface resistance (needs an interFoam template), trim and sinkage at speed, seakeeping (heave/pitch RAOs)

## Stage 6 — Consistent interfaces
- [x] 6.1 Result-shape conformance test in every package against `docs/result-shape.md`
- [x] 6.2 Import-isolation test (no cross-package imports)
- [x] 6.3 File-based composition documented (STEP → Talos, STL → Aeromant/Mellonia)
- [x] 6.4 One installable distribution `vegeta-cli`: `vegeta.dedalus/talos/aeromant/mellonia` in a `vegeta` namespace, `vegeta <tool>` command plus shortcuts

## Stage 7 — Jupyter as first-class interface
- [x] 7.1 Per-package notebooks executed headlessly in tests (done per stage above)
- [x] 7.2 Combined workflow notebook (CAD → inspect → FEA → CFD → slice, every step user-initiated)
- [x] 7.3 Interactive visualisation (pyvista/trame) per package; product notebooks `08_quadcopter`, `09_fixed_wing_drone`
- [x] 7.4 Notebooks merged by machine (one per product); noise sections (Gutin + broadband) and frequency diagrams everywhere; `talos.viz.animate` (rotating stressed blade, load ramp) and `aeromant.viz.animate_particles` (tracers in the converged rotor flow) write MP4 with OpenCV

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
- [x] 12.3 `Campaign`: bounded agentic parameter search on `vegeta.core` revisions (criteria, objective, budget, approval policy, STOP file, resumable record); `notebooks/10_agentic_design.ipynb`

## Stage 13 — Validation (continuous)
- [x] 13.1 Unit tests in every package
- [x] 13.2 Integration tests with numerical validation; clean skips when tools are missing
- [ ] 13.3 After every stage: run tests, fix regressions, update/remove documentation (ongoing; done for stages 1–9)
