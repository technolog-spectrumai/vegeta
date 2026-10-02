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
- [x] 5b.2 Dedalus `Propeller` example design; propeller parts of `08_quadcopter`, `09a_fixed_wing_design` (formerly notebooks 11, 12) (export JSON + CAD for the mission/fatigue work)
- [x] 5b.3 `boreas.noise`: Gutin tonal harmonics, broadband allowance, cavitation number
- [x] 4.10 Aeromant `rotor_mrf` / `rotor_mrf_static` (MRF propeller, forces → thrust/torque/power/Ct/Cp/η/FM, both dialects); CFD sections in notebooks 11/12 (`VEGETA_SKIP_OPENFOAM=1` skips them) — live validation on a machine with OpenFOAM: `test_openfoam.sh` + the notebook cells
- [x] 4.11 Aeromant `aircraft_rotor_disks`: a whole aircraft with two propellers as `rotorDisk` blade-element sources (Boreas blade + polar tables), both dialects; `09a_fixed_wing_design` Part 3 — live validation on a machine with OpenFOAM
- [x] 4.13 Parallel CFD runs: `CFDCase.run(processors=N)` (decomposePar, mpirun, reconstructPar); scenarios: `scenarios/air_video.sh [-j N]` (preset aircraft + rotor disks → CFD → particle movie), plan in `scenarios/scenarios.md`
- [x] 4.14 Scenarios for the submarine (`sub_video.sh`, template `hull_rotor_disk`) and the boat (`boat_video.sh`, double body, no free surface)
- [ ] 4.15 Scenarios read the design JSON exported by the notebooks instead of presets; free-surface boat scenario (interFoam)
- [ ] 4.12 Full realistic simulation of an aircraft with its propellers: resolved rotating blades in the whole-aircraft mesh (several MRF zones, then sliding mesh / AMI for the unsteady blade passing), instead of rotor disks — some day

## Stage 5c — Chronos (missions, vibration, cyclic loads, life)
- [x] 5c.1 Talos: `PointMass`, `solve_modes` (CalculiX `*FREQUENCY`, validated vs beam theory and Rayleigh), `read_frd_steps`, `viz.plot_mode`
- [x] 5c.2 Talos: `FatigueCurve`, `assess_fatigue` (unit stress fields × spectrum, Basquin + Goodman + Miner, hotspot map `viz.plot_damage`)
- [x] 5c.3 `vegeta.chronos`: `Mission`/`Segment`/`Excitation`, `Structure` (DAF, margins, Campbell), ASTM rainflow, `build_spectrum` (JSON hand-off), `SNCurve`/`hotspot_damage`, `simulate_life`; CLI `chronos spectrum|life`; tests
- [x] 5c.4 Life parts of `08_quadcopter`, `09b_fixed_wing_durability` (formerly notebooks 13, 14): three missions each, vibration → spectra → FEA fatigue → static re-check → fleet life; design comparison

## Stage 5d — Land vehicles (branch `dev_land`)
- [x] 5d.1 `designs/rover.py` (tub chassis, trailing arms, wheels; `part` = rover/chassis/arm) and `11_rover_mechanics` (formerly 15): ISO 8608 terrains + rocks + drop → quarter-car wheel forces → static strength over the operating cases (rocky-field peak, 30° hill climbing with weight transfer and traction limits, mud with sinkage and stall torque, 6 kg payload with a loaded quarter-car re-run) on the arm (FEA), the chassis (FEA: torsion, payload, loaded climb) and the pins (by hand); arm modes and a speed–frequency diagram, rainflow spectra per terrain, fatigue and life, printed arm
- [ ] 5d.2 Motor torque reaction and cornering loads on the arm; chassis drop case with the battery; measured spring/tyre rates

## Stage 5e — Boats and submarines (branch `dev_sea`)
- [x] 5e.1 `designs/survey_boat.py` (lofted hard-chine hull, shell, deck, transom bracket; `part` = boat/hull_solid/hull_shell/bracket) and `12_boat_at_sea` (formerly 16), Part 1: hydrostatics and GZ from the mesh, ITTC resistance + double-body RANS check, Boreas propeller in water, hull/bracket FEA (hydrostatic, slamming, thrust, wave slap), bracket modes, three sea states → spectra → fatigue → life, printed bracket
- [x] 5e.3 `12_boat_at_sea` Part 2 (formerly `17_water_propeller`), plus a flow video, a blade-stress video and one bracket + blade frequency diagram: Boreas in sea water (η/Ct vs J, cavitation number vs rpm, inception rpm), blade CAD, guarded `rotor_mrf` CFD check at cruise, one-blade FEA at bollard pull and its modes vs blade-pass frequency, Gutin tones + broadband in dB re 1 µPa, JSON export
- [x] 5e.4 `designs/submarine.py` (Myring body, sail, fins; `part` = vehicle/body/pressure_hull with truncated ellipsoidal caps) and `13_submarine` (formerly 18): mass budget → buoyancy, trim lead position, BG; ITTC + form factor resistance, guarded `rans_ksst_external` check (nose upstream, L/4 reference length); Boreas thruster, top speed, endurance/range vs speed; pressure hull FEA at 200 m vs thin-shell theory, yield and Windenburg–Trilling collapse; three dive profiles → pressure spectra → damage per dive; the propeller (rotor CFD check with a flow video, Gutin noise and cavitation at two depths, blade FEA with a stress video, wet modes and Campbell); a dive simulation (speed vs depth, pressure and hull stress vs depth, propulsion loss → speed decay and buoyant ascent); JSON export
- [x] 5e.5 `14_submarine_propeller`: `DESIGN` dict → Boreas + CAD; operating points; pressure vs speed and depth (suction peak, cavitation inception); fin-wake model or hull CFD wake (`boreas.wake.sampled_wake`); guarded `rotor_mrf` at three points (thrust/torque/η vs BEMT, blade surface pressure vs relative speed); 3D streamlines + tracer particles (slipstream model or CFD field); OpenCV movies per operating point joined into one; blade FEA held at the bore, weak points; load harmonics (`boreas.wake.load_harmonics`), frequency diagram vs blade and hull modes, blade response and Goodman fatigue at the weak points; steady + wake tones + broadband; rule-based design proposals with repair rounds, evaluated and ranked (safe first); section 11: quieting measures tried on the recommendation and kept or dropped (skew in the CAD and `load_harmonics(skew_deg=)`, anti-singing edge vs trailing-edge shedding, material damping, larger/slower, fins upstream)
- [ ] 5e.2 Free-surface resistance (needs an interFoam template), trim and sinkage at speed, seakeeping (heave/pitch RAOs)

## Stage 5f — Air couriers (branch `dev_velutina`)
- [x] 5f.1 `designs/velutina.py` (slim-body quadrotor: capsule nose with handle, pusher arms, fins, parachute bay; `part`, `angle_of_attack_deg`), `designs/velutina_flight.py` (terrain, ISA, wind + gusts, reduced 6-DOF flight model with set-down and hand-over, parachute estimate, movie) and `24_velutina`: mass budget, drag areas by hand + actuator-disk slipstream, Aeromant screening (fast/full presets), Boreas at the depot and the site, arm/capsule/shell FEA, modes and Campbell, the mission with precision statistics and energy, Chronos fatigue, printing, movie, JSON; `scenarios/velutina_mission.py`, `user_tests.sh velutina`; `core.Revision.run_cfd(processors=)`
- [ ] 5f.2 Parachute in CFD (opening, canopy drag, the body in the canopy's wake); the bridle anchored to the arm frame and a reefed canopy (the shell fails the opening shock in 24 §5); a drop test case
- [ ] 5f.3 Propeller guard rings for the hand-over mode (a hand near the rotors): CAD, mass, their drag in the CFD
- [ ] 5f.4 Four rotor disks in the whole-aircraft CFD (`aircraft_rotor_disks` has two): the slipstream over the arms and fins instead of the analytic increment
- [ ] 5f.5 **Velutina v2 — infrastructure inspection**: an onshore wind turbine, flying along a blade with a camera to film it for cracks (a camera nose instead of the capsule, a blade-following guidance mode, the turbine and its wake in the flight model's scene, station keeping in the tower's wind shadow)
- [ ] 5f.6 Coupon tests for LW-PLA and PETG-CF replacing the assumed material values; the FULL CFD preset run on a workstation and its numbers recorded

## Stage 5g — Air propellers (branch `dev_crazy_prop`)
- [x] 5g.1 `designs/air_propeller.py` (pod + pylon, tractor/pusher, movie outline, pylon wake models, sound synthesis) and
  `25_air_propeller`: tractor then pusher, 2 and 3 blades — BEMT efficiency, wake and blade-load harmonics, CFD (isolated
  `rotor_mrf`, installed `rotor_mrf_installed`, pod alone), 3D particles and a ball movie of the four configurations, blade FEA
  with `talos.Centrifugal`, Campbell, fatigue, noise (`boreas.wake.rotating_tones`, `boreas.vortex_noise`, dB(A)), WAV files;
  efficiency against rpm in six panels (alone / tractor / pusher × 2 / 3 blades; `boreas.wake.effective_inflow`, `installation_wake`,
  `installation_drag`; the CFD as an rpm sweep with the pod-alone drag for the net efficiency)
- [ ] 5g.2 Run `AIR_PROP_CFD=full` on a workstation and record the installed efficiencies and the CFD wake; mesh study on one case
- [ ] 5g.3 Unsteady installed CFD (sliding mesh / AMI) for the blade-passing loads directly, against the quasi-steady + Sears model
- [ ] 5g.4 Thickness noise and forward-flight Doppler in the tones; a static (take-off) wake model for the pusher (the induced inflow past the pylon)

## Stage 6 — Consistent interfaces
- [x] 6.1 Result-shape conformance test in every package against `docs/result-shape.md`
- [x] 6.2 Import-isolation test (no cross-package imports)
- [x] 6.3 File-based composition documented (STEP → Talos, STL → Aeromant/Mellonia)
- [x] 6.4 One installable distribution `vegeta-cli`: `vegeta.dedalus/talos/aeromant/mellonia` in a `vegeta` namespace, `vegeta <tool>` command plus shortcuts

## Stage 7 — Jupyter as first-class interface
- [x] 7.1 Per-package notebooks executed headlessly in tests (done per stage above)
- [x] 7.2 Combined workflow notebook (CAD → inspect → FEA → CFD → slice, every step user-initiated)
- [x] 7.3 Interactive visualisation (pyvista/trame) per package; product notebooks `08_quadcopter`, `09a_fixed_wing_design` + `09b_fixed_wing_durability`
- [x] 7.4 Notebooks merged by machine (one per product); noise sections (Gutin + broadband) and frequency diagrams everywhere; `talos.viz.animate` (rotating stressed blade, load ramp) and `aeromant.viz.animate_particles` (tracers in the converged rotor flow) write MP4 with OpenCV
- [x] 7.5 `09b_fixed_wing_durability` Part 2: the whole fuselage (hollow printed shell + tail, `part="fuselage"`) over a long full-battery survey — modes and propeller lines, unit cases (inertia, tail, fin), gust mission, rainflow spectrum, fatigue and life; `09a_fixed_wing_design` Part 3: the whole aircraft in flight with its propellers (rotor disks). Notebook 09 split into 09a (design and performance) and 09b (durability) with a JSON hand-off. Every movie a notebook makes goes to `notebooks/output/<notebook>/`, with the Vegeta logo as a small watermark (bottom right, at most 10 % of the frame)

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
- [x] 12.1 `vegeta-ai` is the provider layer only (`ProviderConfig`, `ClaudeProvider`: structured output, images, cancel, usage/cost; `ScriptedProvider`; `vegeta ai check|models`); the copilot (`DesignSession`: proposals built in the sandbox, measured, diffed, explicitly accepted/rejected) lives in `vegeta-fidia`
- [ ] 12.2 OpenAI adapter; parameter-only proposals against `vegeta.core` criteria (see `plan.md` step 3)
- [x] 12.3 `Campaign` (now in `vegeta-fidia`): bounded agentic parameter search on `vegeta.core` revisions (criteria, objective, budget, approval policy, STOP file, resumable record); `notebooks/10_agentic_design.ipynb`
- [x] 12.4 Fidia prompt-to-3D (`vegeta-fidia`, branch `dev_fidia`): plan → CadQuery (Dedalus contract) → sandboxed build (AST screen, clean env, rlimits, process-group kill) → checks (validity, watertight, winding, budget, size, floating) → five views + sheet → vision review → revise; limits, feedback, cancel, STOP, approval, best valid revision, resumable run; GLB/glTF+bin/OBJ+MTL/STL/STEP with re-import by trimesh and VTK; `fidia run|resume|show`; `notebooks/15_fidia_prompt_to_3d.ipynb`; docs/fidia.md (incl. the Blender assessment)
- [ ] 12.5 Fidia backends: Blender (organic shapes, textures, photoreal renders) behind the same runner contract; network isolation by default where user namespaces exist

## Stage 13 — Validation (continuous)
- [x] 13.1 Unit tests in every package
- [x] 13.2 Integration tests with numerical validation; clean skips when tools are missing
- [ ] 13.3 After every stage: run tests, fix regressions, update/remove documentation (ongoing; done for stages 1–9)
