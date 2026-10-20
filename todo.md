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
- [x] 2.9 Tests (unit + CadQuery integration + CLI), `docs/dedalus.md`, `notebooks/old/01_dedalus_cad.ipynb`

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
- [x] 3.10 Tests: unit (writers/parsers), integration (STEP→Gmsh, cantilever vs beam theory, axial bar, reactions), CLI; `docs/talos.md`, `notebooks/old/02_talos_fea.ipynb`

## Stage 4 — Aeromant (OpenFOAM, template cases)
- [x] 4.1 Package skeleton (deps: numpy, matplotlib, tqdm)
- [x] 4.2 STL utilities (read/write, bbox, explicit unit scaling)
- [x] 4.3 `TemplateSpec` (Python) + template cases: `laminar_external`, `rans_ksst_external`
- [x] 4.4 `CFDCase`: prepare (copy, render placeholders, insert geometry, domain check)
- [x] 4.5 Pipeline runner: blockMesh, features, snappyHexMesh, checkMesh, solver; `OpenFOAMEnvironment` (openfoam.com and openfoam.org case files)
- [x] 4.6 Results: forceCoeffs parsing (Cd/Cl/Cm), checkMesh summary, residual/coefficient history plots
- [x] 4.7 CLI: `aeromant templates`, `prepare`, `run`, `results`
- [x] 4.8 Tests: unit (rendering, STL, parsers), integration (small known case, skip without OpenFOAM), CLI; `docs/aeromant.md`, `notebooks/old/03_aeromant_cfd.ipynb`
- [x] 4.9 `test_openfoam.sh`: per-flavour check of both templates on the installations of the machine (the `org/` case files are validated this way on an openfoam.org machine, not in CI)

## Stage 5 — Mellonia (PrusaSlicer)
- [x] 5.1 Package skeleton (deps: numpy, matplotlib, tqdm)
- [x] 5.2 `PrintSettings` (Python dicts → ini artifacts; `from_ini` for existing exports), example settings
- [x] 5.3 `Orientation` (engineer-chosen rotations, no optimisation)
- [x] 5.4 `slice_stl` via PrusaSlicer CLI, G-code preserved
- [x] 5.5 G-code parser: print time, filament (mm/cm³/g), cost, layer count, max Z
- [x] 5.6 CLI: `mellonia slice`, `mellonia parse`
- [x] 5.7 Tests: unit (parser, settings), integration (20 mm cube layer count/height, skip without PrusaSlicer), CLI; `docs/mellonia.md`, `notebooks/old/04_mellonia_print.ipynb`

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

## Stage 5l — Drongo, potato and cream delivery (branch `dev_potato`)
- [x] 5l.1 `designs/drongo.py` (`Drongo(QuadFrame)`: skids, rack-and-pinion pincer), `drongo_robot.py`, `drongo_controller.py`, `drongo_scenario.py`; notebook `08b_quadcopter_potato`: items and limits, grip window, net drop height, both variants in MuJoCo with movies, the wait, speed/altitude sweep, export; `scenarios/drongo_delivery.py`, `user_tests.sh drongo`, `tests/test_drongo.py`; `chiron.viz.frames`: `ground`, `scenery_range`, `ground_color`, a render per frame
- [ ] 5l.2 The net as a body (cloth or spring mesh) instead of a catch at its plane; the people's arms giving
- [ ] 5l.3 Wind and gusts in the garden; the rotors' downwash on the net and the people; ground effect at the pickups
- [ ] 5l.4 Frame FEA with the pincer and skid loads (notebook 08's load cases with 0.4 kg hanging below)
- [ ] 5l.5 Measured numbers: the cream cup's size and crushing force, potato bruising thresholds, notebook 08's export run

## Stage 5m — NISUS, the observation and computer-vision survey drone (branch `dev_nisus`)
- [x] 5m.1 First engineering approximation in notebook `31_nisus`: `designs/nisus.py` (`Nisus(FixedWing)`: wing, pod, main spar + rear carry-through tube, booms in two-ring root fittings, H-tail, motor mount, trays, keel skid, FEA parts), `nisus_drawings.py` (three-view, exploded view, internal placement, propeller disk), `nisus_systems.py` (variants OBS / Zero, sourced components, mass tables, CG and inertia, electrical loads and branches, propulsion requirement and map, batteries, mission energy with a derated pack, the route-based return and the 20 % reserve), `nisus_flight.py` (lattice aerodynamics, tagged derivatives, trim, CG range, envelope, flight checks, operating limits), `nisus_structure.py` (load cases, hand checks, Talos models), `nisus_cfd.py` (Aeromant plan: α and speed sweeps, fine mesh, wing alone, rotor-disk pusher), `nisus_robot.py` / `nisus_controller.py` / `nisus_scenario.py` (ChironLab six-degree-of-freedom missions with an energy-triggered return, OBS scripted pilot, Zero waypoints and Jetson failure, movies); `scenarios/nisus_mission.py`, `user_tests.sh nisus`, `tests/test_nisus.py`; component research in `designs/data/nisus_component_sources.md`
- [ ] 5m.2 Run the notebook's FEA and CFD in full and fold the results back: the CFD's Cd0 and Cm into the derivative table and the mission energy, the fitting's mesh-converged stress, the motor mount and tray across the layers
- [ ] 5m.3 Measurements: the X2216 + 9×6 on a thrust stand (static and 10–20 m/s), every power branch on the bench (the Jetson with the real pipeline), the pack's usable energy at 5 °C, printed PETG coupons across the layers, the socket and foam bonds, the carbon tubes' modulus, a sandbag wing test
- [ ] 5m.4 Confirm stock and prices of every component on the live pages (the research could read only search snippets); replace the end-of-life Jetson Nano B01 developer kit (a module on a carrier, or the Orin Nano with its mass, 9–19 V input and JetPack 6)
- [ ] 5m.5 Flight-test identification: Clp, Cnr, the pod's lateral terms and the stall; tune the autopilot gains on the identified model; first flights at 25 % MAC
- [ ] 5m.6 Aeroelastics and dynamics: the wing's torsional divergence and aileron reversal speed, the booms' modes against the propeller's 1P/2P, flutter of the tail on the booms (Talos modes)
- [ ] 5m.7 A resolved-blade CFD of the installed pusher behind the pod (`rotor_mrf_installed`), propeller noise for the observation role

## Stage 5n — NISUS-Zero bird photography: the mission software (branch `dev_nisus`)
- [x] 5n.1 Prototype in notebook `32_nisus_birds`: the general mission package `vegeta.mission` (messages, bus and executor, camera model, assumed Jetson/YOLOX compute budget, detector node with tiled search and target crop, ByteTrack-style multi-bird tracker with an IMM/EKF in camera coordinates and log range, OC-SORT re-update, target scheduler, stern photo-pass guidance with separation breakoffs, photo trigger, CLEAR-MOT/IDF1 scores) and `vegeta.mission.sim` (birds, simulated detector, kinematic aircraft, closed loop); `designs/nisus_birds.py` (the MuJoCo seam, mode 'birds' in the flight controller), `nisus_birds_movie.py`, `scenarios/nisus_birds.py`, tests in `vegeta-cli/tests/mission` and `designs/tests/test_nisus_birds.py`
- [ ] 5n.2 YOLOX for real: export to ONNX, TensorRT engines (FP16, INT8 with calibration) on the Orin Nano Super, benchmark latency and power with tiling and crops; replace the ASSUMED `effective_tflops`/`overhead_ms` in `vegeta.mission.compute`; implement `YoloxTrtDetector`
- [ ] 5n.3 Fit the simulated detector (`px50`, the score spread, false positives, box noise) on labelled flight video of birds from the IMX219 with the 6 mm lens; fine-tune YOLOX on birds seen from above and behind
- [ ] 5n.4 Hardware in the loop: the mission stack on the Jetson with replayed video and the autopilot over MAVLink (or a ROS 2 bridge of the same messages)
- [ ] 5n.5 A gimbal (standoff orbits, the bird kept in the frame through turns) or a second narrow photo camera; a quiet approach (gliding, motor at idle); per-species flight distances from the literature and from flights
- [ ] 5n.6 Tracking: appearance re-identification once real crops exist; JPDA or MHT for dense flocks; the range from the size with a species classifier
- [ ] 5n.7 Rules and ethics before flying at birds: the local drone rules near wildlife, the protected species and seasons (breeding), a minimum separation and a disturbance budget per bird

## Stage 5o — NISUS+ (formerly FALCO), the mountain survey drone (branch `dev_falco`)
- [x] 5o.1 First engineering approximation in notebook `33_nisus_plus`: `designs/nisus_plus.py` (`NisusPlus(Nisus)`: two-piece wing with spar joiner, cut flaps/ailerons posed for crow, FEA parts), `nisus_plus_drawings.py`, `nisus_plus_systems.py` (ISA + Sutherland, Li-ion pack model, signed drive map with the brake/regen region, masses, mission energy), `nisus_plus_flight.py` (crow, envelope and ceiling against altitude, climb strategy, descent and regen tables, launch), `nisus_plus_structure.py`, `nisus_plus_cfd.py`, `nisus_plus_robot.py` / `nisus_plus_controller.py` / `nisus_plus_scenario.py` (massif heightfield, mountain wind, TECS with crow and brake, six conditions, movies), `nisus_plus_birds.py` (the bird mission over the slope); `scenarios/nisus_plus_mission.py`, `scenarios/nisus_plus_birds.py`, `user_tests.sh nisus-plus`, `tests/test_nisus_plus.py`; NISUS only gained `design=` seams; Chiron `Lab.run(stop_when=)`
- [ ] 5o.2 Run the crow CFD and the brake disk and replace the ASSUMED crow ΔCD/ΔCm and the brake drag (the BEMT skips induction in the windmill state: low confidence)
- [ ] 5o.3 Bench: the AT4125 + 15×8E on a thrust stand at 10–25 m/s including active braking and the regen current into a real pack; the ESC's braking modes; the pack at −10 °C
- [ ] 5o.4 A real DEM clip instead of the analytic massif; onboard terrain (DEM + GNSS/baro, the lidar near the ground) instead of the controller's perfect terrain knowledge
- [ ] 5o.5 Flutter and divergence at TAS at 4500 m (V_NE is set in EAS); the flap hinge and horn under repeated crow; gains scheduled on identified data
- [ ] 5o.6 Weather limits: thunderstorm and icing go/no-go, lee rotor turbulence beyond the simple model; the bungee launch hardware and a hand-launch with flap test
- [ ] 5o.7 Rules and ethics before flying near eagles and vultures (protected species, nesting seasons)

## Stage 5p — FALCO, the tractor mountain aircraft (branch `dev_falco`)
- [x] 5p.1 The frame study in notebook `34_frame_study` (`designs/frame_study.py`): NISUS+'s twin booms against a single tail — hand and Talos structure (bending, torsion with the tube's own G, the asymmetric gust, the first modes), what the tail puts into the wing, the aerodynamic efficiency and the speed and acceleration of each variant with the larger propellers on an assumed motor; verdict: no wider booms on NISUS+ (the spar), a roll-wrapped single tail tube for FALCO with the 20x13
- [x] 5p.2 FALCO's design (notebook 35, `designs/falco*.py`): MERLIN's tractor body with the single roll-wrapped tail tube of the study, NISUS+'s wing, the 24 cells re-connected 8S3P, the sourced T-Motor AT5220-A KV220 (its bench table fits the motor model) on an APC 20x15E (the propeller trade: the 20x13 cruises near its zero-thrust advance ratio), crow and brake, the tractor's installation effects, the tail tube / socket / firewall FEA, the six mountain missions and the four bird hunts with the propeller parked for the landing; the FALCO-109 marking; NISUS+ vs FALCO: climb +13-15 %, mission energy −5 %
- [ ] 5p.3 The tractor's installation effects in the CFD (the nose rotor disk case is planned, NOT RUN: the propeller's wash over the fuselage and the wing root) and the real 20x15 blade in Boreas (the APC geometry file); the AT5220's point on the 20x15 at 8S on a thrust stand (the fit carries the 18x8's calibration)
- [ ] 5p.4 The propeller parking on the bench: the ESC's brake-on-stop angle and the nudge's repeatability with the Hall sensor (AM32 bidirectional mode), the companion script's timing against the 20 m park height; the pitot's reading in the slipstream
- [ ] 5p.5 Flutter at TAS with the single tube (the FEA's isotropic G makes its torsional mode 2-3 x too stiff: measure the roll-wrapped tube's G); the fin rod 10/8 at margin ~0.9 — a carbon plate fin instead
- [ ] 5p.6 A motor of the AT4125's loss class wound for the 20-inch propeller (the study's −11 % mission energy against FALCO's −5 % with the AT5220): an AT4130 KV300 with a published table, or the AT5220 on a lighter 10S pack

## Stage 5q — Atropos, laser-cut balsa (branch `dev_cut`)
- [x] 5q.1 `vegeta.atropos`: ribs from a section (skin let in, cap-strip slots, tube and lightening holes, LE/TE stock cuts, the hinge split with a bevel), kerf compensation, a deterministic shelf nesting on stock sheets, SVG per operation layer and GRBL G-code (xTool D1 Pro class), time and material; explicit `CutSettings` (sheet, machine, operations), `examples` as an illustration; CLI `rib/nest/info/settings`; tests, `docs/atropos.md`
- [x] 5q.2 FALCO's wing and tail built up in balsa on the carbon tubes (`designs/nisus_plus_balsa.py`, `falco_balsa.py`, notebook 36): rib stations, the D-box and trailing-edge sheeting, stock, the sheets, the cut time, the mass against the foam wing, the D-box torsion without the foam
- [ ] 5q.3 Cut a test strip on the D1 Pro: the kerf and the passes for 1.5 / 2 / 3 mm balsa and 2 mm ply, into a measured settings module; a rib-to-tube fit check (clearance 0.3 mm assumed)
- [ ] 5q.4 Sheeting blanks as cut parts (today: flat trapezoids from the arc lengths, a table), an engraved part name on every rib, a better nesting (first-fit by area, holes) when the sheets count matters

## Stage 5r — Kestrel-4Q, the zero-stagger biplane quadrotor tailsitter (branch `dev_kestrel`)
- [x] 5r.1 Aeromant `aircraft_quad_rotor_disks`: four rotor disks on one thrust axis, each at its own rpm and sense; overlapping disk zones refused; both dialects, prepare-only tests, `docs/aeromant.md`
- [x] 5r.2 The design (notebook 40, `designs/kestrel*.py`): the CAD with zero stagger built in and the clearance rules; masses, CG, inertia; Boreas drive tables; the hold point (the allocation's rank per sense of rotation — only the diagonal pairing — the attainable moments, a motor out, the hover in wind); the biplane in the vortex lattice against Prandtl, the bracing's drag (round vs faired), the neutral point with the propellers, the full-range polar, the cruise and its trim by differential thrust; the truss as a plane frame; the ground stance
- [x] 5r.3 The transitions both ways through the stall (`kestrel_transition`): blown and free-stream spans, rotor limits and lag, a pitch PID scheduled on airspeed, allocation to differential thrust and elevons, the height loop's vertical-speed limits (no tail slide), the roll check for a one-sided stall; the matrix (three vehicles, three schedules, headwind, gusts), the back transition's strategies, the verdict
- [x] 5r.4 The CFD at the transition points (`kestrel_cfd`): the airframe turned to each α with the four rotor disks at each pair's rpm, the bare 0–90° sweep; the solved coefficients fed back into the transition model as a correction (`CFDAirframe`) and the matrix flown again
- [ ] 5r.5 Run the CFD where OpenFOAM is installed (screening, then the worst stalled point fine); an unsteady (URANS / DES) case at the worst point — steady RANS beyond the stall is a time average at best; read the rotor disks' in-plane force from the solver log against `prop_cn_alpha`. *First campaign done on another machine (screening, 400 iterations): the ten bare cases solved and are on file (`data/kestrel_cfd_results.json`, plugged into notebook 40 § 7); all fourteen blown cases failed or diverged under SIMPLEC at U 0.9 — the disk templates now run plain SIMPLE (p 0.3, U 0.7) with a `limitVelocity` cap. Open: the blown cases again (notebook 39), a fine mesh and more iterations for the bare sweep (none reached the residual target), Kestrel-4Q-2's airframe, the section slabs.*
- [x] 5r.5a Failure is a result: `kestrel_cfd.table` lists `diverged` / `FAILED` cases with their reason and keeps them out of the correction; `CFDAirframe.from_results` returns `None` instead of raising; `kestrel_transition.simulate` ends a diverging integration as a failed transition
- [x] 5r.5b Notebook 40 as a design loop (`kestrel_loop.py`: the envelope, check tables per iteration, the iteration log) with the CFD plugged in from the results file; notebook 39 `39_kestrel_tests`: the whole CFD campaign (both aircraft, the hover, the bare sweeps, the section slabs) at a chosen quality, campaigns keyed in the results file
- [ ] 5r.5c Notebook 39 on an OpenFOAM machine: the bare sweeps first (`KESTREL_CFD_SWEEP_ONLY=1`), then the blown cases; read each failed case's log; the X-wing's CFD needs a CAD design class for it first
- [x] 5r.9 prop_lab (notebook 38, `designs/prop_lab.py`): the 9x6 alone in CFD before the airframe campaign — hover, axial inflow, the inflow at an angle on the new `rotor_mrf_oblique` template (free-stream lateral faces, normal force and pitching moment in the metrics) — beside Boreas and De Young; the calibration (blade-section scales, normal-force factor) exported to `data/prop_lab.json` and read by the drive tables, `rpm_for`, the rotor disks' polar and `prop_cn_alpha`; the campaign's points extended with the level cruise and the stall; the noise (blade-passing tones on the measured loads, broadband, dB(A)) and the sound rendered as WAV files (one rotor, four rotors beating in the hover, the transition glide)
- [x] 5r.11 The structure in Talos (`kestrel_fea.py`, notebook 40 § 8b, run in 40): the carbon frame as solid tubes (X and posts only) under the 4 g pull-up, the 75 % wing, the heading manoeuvre's torsion and the modes with the motor units; the printed spar–post joint with its across-layer tension; read against the plane frame; NOT RUN without CalculiX
- [x] 5r.13 The dossier (`kestrel_report.py`): every Kestrel notebook (38, 39, 40, 41) exports its numbers as JSON (`data/reports/<notebook>.json`, the stitched `data/kestrel_dossier.json`) and Markdown, to paste into a conversation and continue the design
- [x] 5r.14 Notebook 42 `kestrel2` (`designs/kestrel2.py`): 4Q-2 against 4Q-1 — the back transition's remedies in the simulator (crow-brake elevons, the pitch bulge, reverse thrust, the descent, the level deceleration), the performance and pursuit tables (climb, dive, acceleration, turns, endurance, range), the Kestrel-4Q-3 proposal flown through the same tables; 38 exports its transition figures to `data/figures/`
- [x] 5r.16 **One Kestrel notebook, thin workflows** (`43_kestrel_study`, `designs/kestrel_study.py`, `designs/kestrel_campaigns.py`, `side_labs/runkit.py`, `side_labs/kestrel/{airframe_cfd,structure_fea,import_legacy}.py`, `side_labs/README.md`, the rule in `docs/philosophy.md`): the unified notebook (geometry → assessment, inexpensive methods first, every number labelled analytic / empirical / solver / validated, the correlations with sources and ranges, the decision-relevant transitions only, ~2 min measured) reads CFD and FEA **runs by explicit path**, checks their state and design fingerprint, stops with the exact command when a required run is missing (`KESTREL_SCREENING=1` for an analytical pass) and launches no solver (a test guards it); the workflows write the run-directory contract (`manifest.json`, atomically updated `status.json`, a continuously flushed `run.log` with heartbeats that distinguish *alive* from *advancing*, `summary.json` only on completion, the native cases), take `--list` / `--prepare-only` / `--cases` / `--max-cases` / `--max-minutes` / `--resume`, stop their solver on SIGINT/SIGTERM and exit nonzero; the first campaign's file imported as runs; `kestrel_fea`'s regions fixed (segmented spars, the joint's socket floor: every box holds whole faces — the first FEA campaign had failed on it)
- [ ] 5r.17 **The production runs** (outstanding — none could run here: no OpenFOAM, no CalculiX, no usable Gmsh on the session machine): `airframe_cfd.py --variant 4Q-2 --campaign stall --quality fine` (CL max: the wing area hangs on it), `--campaign sweep` for 4Q-2 and 4Q-3 at fine, `--campaign cruise` and `transitions` (screening) once the sweep converged, `structure_fea.py --variant 4Q-2 --campaign all` (the first solvable FEA), 38 `prop_lab` again at fine; then `43_kestrel_study` in its default (required) mode and the 4Q-3 decision from `kestrel3.md`
- [x] 5r.19 `side_labs/cli.py` (list / info / run --detach / runs / status / tail / stop / resume / notebook, interactive without arguments; `manual_workflows.md`); `44_kestrel_fast` + `designs/kestrel_fast.py`: the one-minute comparator of candidate designs on Zephyr, Boreas points and beam rules
- [x] 5r.20 Books and chapters (`side_labs/library.py`): a book is a design (`kestrel`, code K), its chapters the numbered studies of every kind (`wf K2`, case-insensitive), shown as a tree; runs under `runs/<book>/<chapter>/`, prerequisites within a book checked before a chapter starts, solved cases cached per book (`runkit.Cache`, `--no-cache`, `wf cache` / `clean-cache`); `activate.sh` (`wf`, `wfrun`); tqdm bars over the cases and the solver inside a case (terminal only); `side_labs/gui/` — the PyQt6 window (tree, options, Run / Stop detached, live log + progress from status.json, the runs table), organised like Disk Sweeper's; `./side_labs.sh` opens it (branch `dev_workflows` from here on); **not yet run on a desktop** (the session machine has no libEGL): first launch and screenshots outstanding
- [ ] 5r.18 Workflows next: the section slabs (`--campaign sections`) read back into the notebook's § 5; a thermal workflow (motor / ESC / battery heat path in the nacelle and the pod) on the same `runkit` contract; the X-wing's CFD once it has a CAD design class
- [ ] 5r.15 4Q-3's open points: reverse thrust measured in prop_lab (a negative-rpm case), the propeller for speed (9x7.5) on the calibrated drive table, the braking autopilot's logic in 6 DOF
- [ ] 5r.12 Run the FEA where CalculiX is installed (notebook 39, or 38 § 8b); then the nacelle / motor mount, the elevon hinge and horn, the pod's spar saddles and the capsule bay under the landing load, the foot rods' touchdown — each needs its own structural CAD
- [ ] 5r.10 Run notebook 38 (prop_lab) where OpenFOAM is installed, then 39 (the campaign) and 40 on the calibrated propeller. *First campaign done (6 cells/D, 100 k cells): thrust 0.60 x BEMT, FM 0.29, the normal force zero — an under-resolved blade and a frozen rotor at one azimuth; the fit was rejected (gates added: fit rms, scale bands, both azimuths), the screening mesh raised to 10 cells/D level 4, the oblique cases run at two azimuths. Rerun;* if the thrust and torque ratios drift with J the generic section's shape is wrong — the real APC profile on Zephyr is the next step; the slipstream velocity at the wing's station (a sample line behind the disk) for the transition model's `slipstream`
- [ ] 5r.6 Roll and yaw in the transition model (6 DOF): the one-sided stall flown, not only checked; the elevons' hinge moments and servo rate; a crosswind transition
- [ ] 5r.7 The aborted transition and the fail-safes (spec § 9): a motor out in the transition, low battery, lost link — and the reduced-attitude hover for a motor out (spinning about the thrust axis)
- [x] 5r.7a The medical payload bay (notebook 40 § 11): the capsule, the pod's bays and their checks, the ellipsoidal nose, the capsule's cost empty / nominal / full
- [x] 5r.8 Kestrel-4Q-2 (notebook 40, Part II): elevons in the CAD with their hinge moments and authority; faired struts (`strut_fairing`, `fairing_figure`) with their mass and broadside area; the bracing choice (`bracing`) with the motor stations' stiffness and first modes per truss; the four changes together against 4Q in the same matrix
- [x] 5r.9 The X-wing (notebook 41, `kestrel_xwing.py`): the same motors, battery and pod on four cantilever panels at ±45°, no bracing — layout, masses with cantilever-sized spars, the hold point, the lattice, the panels' section on Zephyr, drag, cruise, stance, the transitions in the generalised panel model, against Kestrel-4Q and 4Q-2
- [ ] 5r.10 Talos FEA of the truss joints and the elevon hinges (peel at the strut ends, the motor mounts' thrust and gyroscopic loads, torsion of the wing pair with posts only); the X-wing's panel root in Talos
- [ ] 5r.11 When `dev_zephyr` is merged: run § 13 (the reflexed section), put the finalist's section into the CAD (`section` parameter from Zephyr's outline), re-run Part II's matrix on it; run the Aeromant slab cases of the finalists
- [ ] 5r.12 The ground and the touchdown in MuJoCo (Chiron): standing in gusty wind, the landing on six feet with a sink rate and a lean

## Notebook migration (separate from the Kestrel work above)
- [ ] M.1 The historical Kestrel notebooks `notebooks/old/39_kestrel_tests`, `40_kestrel`, `41_xwing`, `42_kestrel2` are kept unchanged with their outputs; they predate the workflow rule (39 runs the CFD and FEA campaign from its cells, 40 prepares FEA itself). Migrate what is still worth reading into `43_kestrel_study` sections or into workflows (the X-wing comparison as a variant table; the remedy matrix as an optional, longer transition set behind a flag), then retire them — not now
- [ ] M.2 Apply the workflow rule to the other long-running notebooks (the CFD cells of 09a, 24, 31–36, the MuJoCo benchmarks): each campaign a script under `side_labs/<project>/` on `runkit`, the notebook reading the run; `VEGETA_SKIP_OPENFOAM` becomes unnecessary where that is done

## Yanus — the door keeper (branch `dev_yanus`; `docs/yanus.md`, `docs/yanus_plan.md`)
- [x] Y.1 `vegeta-yanus`: contracts (needs, takes, gives), entry and exit gates, pins, stamps, `yn status|why|doctor|pin|check|stamp|new`
- [x] Y.2 `side_labs/flow.py` the one registry; `library` and `kestrel_campaigns` read it; the scripts' entry gate; `wf run` stops on a missing need
- [x] Y.3 Notebooks 38, 43, 44 on the gates; 43's `RUNS` dict → `notebooks/pins/43_kestrel_study.json`
- [-] Y.4 Stamp the committed exports by running 38 (with OpenFOAM) and 43 through their gates on a solver machine  *(dropped: the old Yanus is removed, S.8)*
- [x] Y.5 Rolled out to the other 38 notebooks: a contract each in `flow.py` (needs = the vegeta packages imported; gives = the files under `designs/data/`; takes = 25→25b/25c, 26→26b/27/29, 28_microjet→29, all optional), the entry gate after the imports, `GATE.leave()` last
- [-] Y.5b Tools as needs of the notebooks (ccx, OpenFOAM) once their CFD/FEA cells honour a missing solver the same way (today only DEBUG / `VEGETA_SKIP_OPENFOAM` do); pins for them after M.2  *(dropped: the old Yanus is removed, S.8)*
- [x] Y.8 Kestrel-X (`45_kestrel_x`): the X-wing as its own study, compared with Kestrel-4Q from notebook 43's report; a loud warning and Kestrel-X alone when the report is missing
- [x] Y.9 Kestrel-X workflows: the book `kestrel-x` (code X) in `flow.py` on Kestrel-X-C's CAD (`kestrel_xc.KestrelXC`) — X1 `airframe-sweep`, X2 `airframe-stall` (CFD, the bare airframe), X3 `panel` (FEA: the spar in its socket with the tip motor) — `airframe_cfd.py` / `structure_fea.py` parametrised by book (`main(book=)`, `CAMPAIGNS`, `library.kind_of`) with one-line entry points in `side_labs/kestrel_x/`; Kestrel-X-A keeps no CAD (notebook 45 draws its layout)
- [x] 5r.27 Kestrel-X-C (`kestrel_xc.py`, `kestrel_xc_campaigns.py`, notebook 48, auto): the X-wing made competitive — ±30°, 0.6 m tapered panels, the filleted roots, the payload module inside a blended fuselage (162 × 185 mm, 0.76 m, fineness 4.4, a flush camera window), the tip pods as the feet; X-C45 for the dihedral's share, X-CX (chord 0.21: longer in x) for the area / aspect-ratio trade, the section thickness as `t` in `drag_buildup` and `cantilever`; compared with X-A, X-B and Kestrel-4Q-2; the benchmark (47) reads its export
- [x] 5r.29 Mini raptors (`mini_raptor.py`, notebook 49 `49_mini_raptor`): Apus Mini, Alcotan, Gavilan on a 5-inch racing stack against the racing quad on the same parts — mass budgets, drives from Boreas, power / Wh/km / range / speed (thrust-limited and sustainable), stall, banked turns against hover reorientation, braking, transitions, targets met and unmet, sensitivity; conceptual CAD with component envelopes and propeller discs, labelled renders, dimensioned views, STEP + STL; no CFD, no FEA; `mini_raptor_print.py`: every design split into printed parts for the bed (PLA pod and fittings, LW-PLA wings), the bought stock beside them, bed fit / trailing edge / overhang checks, Mellonia slices when PrusaSlicer is present, the printed mass carried into the performance as the "as printed" rows; the biplane question answered (Apus on one wing, Gavilan keeps the cell)
- [ ] 5r.30 Mini raptors, the next questions if one goes further: the thin sections' polar at Re ~ 100 000 (Zephyr), a bare-airframe CFD book, the 5-inch drive's blade section (prop_lab at 5 inches), a structural first pass on the posts and the fin arms
- [ ] 5r.28 Kestrel-X-C's roots: the junction allowance is the build-up's largest uncertain term (it scales with chord², so X-CX carries twice X-C's) — X1 at 0 and 5° against the build-up decides it; then proper root fairings in the CAD (fillets on the union) if the CFD says they pay
- [x] Y.10 Auto mode: a notebook flagged `auto=True` (`yanus_auto`; 43, 45) runs its missing takes itself in screening quality (`Kind.screening` arguments, prerequisites first, producer notebooks headlessly) and its gate never closes; `yn auto <nb> [--dry-run]`, `YANUS_AUTO`, `YANUS_AUTO_MINUTES` (`docs/yanus.md` § 16)
- [x] W.1 `batch.yaml`: an ordered batch of chapters from one file (`side_labs/batch.py`, `wf batch`, the GUI's Batch tab writes it) for the cloud; every run and batch ends with where its output file is
- [x] 5r.21 The wing interference as a measure: `kestrel_flight.interference` (lattice gap sweep, Prandtl σ, monoplane equivalents) in notebooks 43 § 5 and 44, the CFD chapter K12 `wing-interference` (one wing / the pair / the equal-area monoplane at 0, 4, 8°, `summary["interference"]`), and `kestrel_compare.py`: FALCO, NISUS+, MERLIN, PEREGRINE scaled to Kestrel's mass and speed beside Kestrel as a monoplane
- [x] 5r.23 Kestrel-Q4T, the tandem (`kestrel_tandem.md`, `kestrel_tandem.py`, notebook 46): Q4T1N-10 designed against Q4T1P-10, Q4T1N-12, the flat control case and Kestrel-4Q-2; Munk in the lattice, the front-first stall rule, the hover lever, the horizontal start, both transitions
- [x] 5r.26 Notebook 47 `kestrel_benchmark`: the family on one table from the exports of 43, 45 and 46, categories scored 1–10 between the briefs' anchors, weighted per mission (`kestrel_benchmark.CATEGORIES`, `MISSIONS`, `FAMILY`); notebook 45 gets a 3-D view from its layout (`kestrel_drawings.xwing_3d`)
- [ ] 5r.24 Kestrel-Q4T workflows: a book `kestrel-tandem` (code T) in `flow.py` reusing `airframe_cfd.py` / `structure_fea.py` through a design key — T1 airframe-sweep per variant, T2 high-α (20–60°, where P's elevator lives: the one thing the straight-wake lattice cannot tell from N), T3 transitions with the rotor disks, T4 the spar in its pod socket (FEA); one `batch.yaml` for all three airframes
- [ ] 5r.25 Kestrel-Q4T's pull-up into the hover balloons more than the biplane's: the autopilot's entry (descent, a longer schedule) as a design parameter, and the fall out of the hover measured with the high-α CFD correction
- [ ] 5r.22 K12 with stagger and decalage as options (the lattice has them, the CAD does not), and the X-wing's panels in the same chapter once Y.9 gives it a CAD
- [x] W.3 Batch: an estimate before the start (case counts from `--list`, seconds per case from the chapter's finished runs, else a rule of thumb, `expected_minutes` override), one progress bar over every case of the batch with the remaining time, written to the batch's status.json for the GUI
- [ ] W.2 A batch step for a notebook (`notebook: 43_kestrel_study`), so a cloud batch can end with the report; and `wf batch --watch` for a batch running elsewhere (rsync + status.json)
- [-] Y.6 A header drift check (`yn check --headers`) against the scaffold; `notebooks/old/` stays out  *(dropped: the old Yanus is removed, S.8)*
- [-] Y.7 Field-level schemas for the JSON exports, if the integer schema + keys proves too weak  *(dropped: the old Yanus is removed, S.8)*

## Stage R — Races (branch `dev_workflows2`, merged into the new layout; notebooks in `notebooks/races/`)
- [x] R.1 `races/`: four drone race environments in MuJoCo for the reference 5-inch quad (hangar with three floors, industrial zone, farm, quarry), parameterised course builders → MJCF scenes with zones, a starting grid, landmarks and four cameras; the reference quad from its component budget on the existing Boreas drive table; one set of rules and a referee (zones in order, checkpoints, collisions → reset / count / stop, lap completion); a mock flyer for development; the MuJoCo runner on Drongo's geometric controller (written, not run); maps, traces, GIF replays; CLI `python -m races`; smoke tests; notebook 50 `50_four_races` (mock)
- [x] R.1b The races as Yanus chapters: book `races` (R1–R4, `side_labs/races/race.py`: the lap and three videos under the run contract, `--quality`, `--mock`), notebook 50 takes the runs (videos and statistics when on file, the mock otherwise, auto mode produces them); `races/metrics.py`: manoeuvrability statistics per lap (linear and rotational envelope, turning, tilt, thrust share, tracking error) with the aircraft's envelope and the literature
- [x] R.4 Notebook 51 `51_race_ingest` (`races/analysis.py`, `races/ingest.scan`): the laps' deep statistics (regimes, percentiles, braking, corners, straights, vertical, power duty, closest approaches, the courses' demands) and the design factors for the raptors → `designs/data/race_design_factors.json` (not connected to a design notebook yet); the first real hangar lap digested
- [x] R.2 Tune the MuJoCo autopilot per course (`./tune_autopilot.sh`: PSO then pattern search, `races/converge.py`, settings in `races/autopilot.json`, read by the race notebooks): hangar 60.1 s (1 collision; it was DNF), industrial 29.4 s, farm 31.4 s, quarry 47.4 s, each a mesh local optimum at 1/64; open refinement: a hangar lap without the collision
- [x] R.5 One notebook per course (hangar, industrial, farm, quarry) with `GENERATE_VIDEO` for the MuJoCo lap and its videos, and a final notebook combining their exports into design requirements (branch `dev_races_refactor`)
- [-] R.6 Mini raptors parked as an UNSTABLE experiment: `designs/exp/mini_raptor*.py`, `mini_races.py`, `notebooks/exp/49_mini_raptor`, tests xfail in `designs/exp/tests/`; the redesign (plank / twin / X: CAD, printed parts, notebook, tests) is unfinished and not needed now — may be re-examined later
- [-] R.3 Other aircraft on the same courses: out of scope, the races are for the quadcopter only (the mini raptors' draft `designs/exp/mini_races.py` is parked), multi-lap races, a starting-grid start with four aircraft

## Stage 5s — Zephyr, 2-D wing sections (branch `dev_zephyr`)
- [x] 5s.1 `vegeta.zephyr`: NACA 4-digit and 5-digit sections (plain and reflexed, the aft reflex adjustable), .dat files, plain flaps; thin-airfoil theory (α₀, Cm_c/4, a flap's τ) validated against Abbott & von Doenhoff; `analyze()` over NeuralFoil (optional, a confidence per point) and XFOIL (on the PATH, the command recorded); `Polar.summary()`, flap sweeps, JSON export; CLI `section/polar/flap/backends`; tests (backends skipped when missing), `docs/zephyr.md`
- [ ] 5s.2 A section notebook: the candidates for a tailless wing (symmetric, cambered, reflexed) at low Re, NeuralFoil against XFOIL where both exist, the flap data, the Reynolds and n_crit sensitivity
- [ ] 5s.3 An Aeromant 2-D-slab template (an extruded section between two slip walls) as the CFD check of a surrogate polar; a Zephyr polar as the section input of Boreas's `Airfoil` and the lattice's camber terms
- [ ] 5s.4 Multi-element sections (a slotted flap), the boundary-layer transition location in the polar, a Reynolds sweep in one call

## Stage 5t — Hyperion and Svarog, optics and the laser (moved to Alabastron, https://github.com/technolog-spectrumai/alabastron, on 2026-10-18; its open items, 5t.6 and `svarog_todo.md`, continue there)
- [x] 5t.1 `vegeta.hyperion`: materials (Sellmeier, dn/dT, κ, α), paraxial ABCD elements (spaces, thin/thick lenses, curved mirrors at incidence, interfaces, slabs, Brewster plates, thermal lenses) in a `Train`, Gaussian beams with M² (q, w(z), through a train, waists, overlap, a fibre's beam, Self's lens formula and its inverse), resonators (linear and ring: round trip, stability, eigenmode in both planes, spots on every element, Gouy phase, FSR/finesse/linewidth/Q, parameter sweeps), the four-level end-pumped laser (threshold, slope, output, Rigrod's coupler, small-signal gain, the thermal lens, the rod's temperature rise); JSON export; CLI `materials/material/lens/beam/cavity/laser`; 32 tests against Kogelnik & Li, the lensmaker, Self, Rigrod and Koechner's ranges; `docs/hyperion.md`
- [x] 5t.2 `46_svarog` + `designs/svarog.py`, `svarog_drawings.py`: the CW diode-end-pumped Nd:YAG laser — the rod and its fibre-coupled pump, threshold/slope/output and the coupler, the thermal lens vs pump, three cavity revisions on Hyperion with the stability window over pump power, the beam out and its delivery train, the report (`designs/data/svarog_design.json`, `reports/46_svarog.json`); the Yanus contract
- [x] 5t.3 `hyperion.fibre` + `47_svarog_fiber` + `designs/svarog_fiber.py`: Svarog-fiber, the Yb double-clad fibre laser — a step-index fibre's V, cutoff, Marcuse mode radius and bend loss (the LP₀₁ eigenvalue with Bessel functions by quadrature), Yb:silica cross sections with the doping set by the datasheet's absorption, the pump/signal power evolution of the FBG oscillator (shot to the gratings) and of a co-/counter-pumped amplifier, the length and coupler choices, the coil radius, SRS and the core's temperature per cooling, the beam out on Svarog's delivery train, the comparison with the rod from 46's report; 9 more tests (Bessel values, SMF-28's MFD, Rigrod's invariant, the quantum limit, energy conservation, saturation, the optimum length); CLI `fibre`, `fibre-laser`
- [x] 5t.4 `hyperion.thermal` + `hyperion.welding` + `48_svarog_weld` + `49_svarog_guard` + `designs/svarog_industrial.py`: the 1 kW welding laser (a 30/250 power stage on the fibre module, its heat budget, the 50 µm process fibre and 100/200 head, keyhole thresholds, penetration curves and process windows for steel and aluminium) and its guard (the stray beams at 20 m, eleven barriers, closed-form screening → a Jacobi steady solve → the explicit enthalpy/ablation transient only where it decides, verdicts per IEC 60825-4 class); `docs/hyperion_physics.md`, `hyperion_solvers.md`, `svarog_physics.md`, `svarog_solvers.md`
- [x] 5t.5 `51_svarog_2kw`: the 2 kW laser two ways (beam-combined modules vs one hot stage), the welds and the guard at 2 kW; `hyperion.thermal_fenicsx` on `dev_svarog_advanced` (FEniCSx backend, smoke-tested against the numpy solvers)
- [x] 5t.7 `47_svarog_iii` + `designs/svarog_commercial.py` + `svarog_commercial_drawings.py`: Svarog III — four commercial lasers' published limits (unknowns kept) through Hyperion: optics (BPP ⇄ M², fibre class, Gaussian vs imaged core), energy balance as intervals with sensitivity, thin-sheet welds at 1–3 mm on the new `hyperion.welding` thin-plate forms (`rosenthal_thin_plate`, `cooling_time_t85`, `cooling_rate_at`, `haz_width`), the 60/180-case shielding matrix at 5–100 m with PASS/FAIL/INCOMPLETE verdicts (no PASS without certified evidence, no burn-through times), fenix F4 `shield-steady`, new shielding solids in `hyperion.thermal`; tests (`test_welding.py`, `test_svarog_commercial.py`, the registry); the open validation in `svarog_todo.md`
- [ ] 5t.6 The industrial roadmap of `svarog_todo.md` (I.1 FEniCSx ablation on `dev_svarog_advanced`, I.2 the head's real optics, I.3 the weld pool, I.4 the enclosure, I.5 pulses)

## Stage 5f — Air couriers (branch `dev_velutina`)
- [x] 5f.1 `designs/velutina.py` (slim-body quadrotor: capsule nose with handle, pusher arms, fins, parachute bay; `part`, `angle_of_attack_deg`), `designs/velutina_flight.py` (terrain, ISA, wind + gusts, reduced 6-DOF flight model with set-down and hand-over, parachute estimate, movie) and `24_velutina`: mass budget, drag areas by hand + actuator-disk slipstream, Aeromant screening (fast/full presets), Boreas at the depot and the site, arm/capsule/shell FEA, modes and Campbell, the mission with precision statistics and energy, Chronos fatigue, printing, movie, JSON; `scenarios/velutina_mission.py`, `user_tests.sh velutina`; `core.Revision.run_cfd(processors=)`
- [ ] 5f.2 Parachute in CFD (opening, canopy drag, the body in the canopy's wake); the bridle anchored to the arm frame and a reefed canopy (the shell fails the opening shock in 24 §5); a drop test case
- [ ] 5f.3 Propeller guard rings for the hand-over mode (a hand near the rotors): CAD, mass, their drag in the CFD
- [ ] 5f.4 Four rotor disks in the whole-aircraft CFD (`aircraft_quad_rotor_disks` now exists, stage 5r.1): switch Velutina's cases to it — the slipstream over the arms and fins instead of the analytic increment
- [x] 5f.5 **Velutina v2 — infrastructure inspection** (`designs/velutina_inspection.py`, notebook 24 §11–12): a half-real parked 2 MW turbine, wind shear + tower shadow, the camera model (pixel size, blur and overlap → scan speed; sharp frames inside a tolerance → dwell time from the simulated station keeping), four passes per blade with suspect-point dwells, the flight with its track error, energy and battery charges, a movie; `scenarios/velutina_mission.py --mode turbine`
- [ ] 5f.7 Velutina v2 next: the camera nose as a Dedalus part (mass, CG, its drag in the CFD), a blade-relative sensor model (the blade's edge in the camera frame instead of GNSS + a fixed error), the turbine's wake with the rotor idling, the nacelle and the hub in the inspection path
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

## Stage 5h — MERLIN, wildfire sampling (branch `dev_merlin`)
- [x] 5h.1 `designs/merlin.py` (`FixedWing` subclass; EDF / tractor / pusher noses; drag build-up) and `designs/merlin_flight.py`
  (propulsor tables from `boreas.ducted` and BEMT with the installation models, the race, Gaussian smoke plume and the source
  estimate, the mission, the smoke movie) and `26_merlin`: which propulsor reaches a fire 5 / 8 / 10 / 20 / 30 km away first;
  `scenarios/merlin_mission.py`, `user_tests.sh merlin`, `tests/test_merlin.py`
- [x] 5h.5 Reach / return / landing times (`race_table`); a fire up a mountain (`fire_elevation_m`: a constant gentle climb on the dash, the descent home)
- [x] 5h.6 Multi-stage fans (`boreas.ducted` `stages`, notebook 25 §14); the propulsor library (notebook 25 §16,
  `designs/propulsor_maps.py`, `data/propulsor_maps.json`) read by notebooks 26 and 27; `27_merlin_race` (branch `dev_rave`):
  5 km and back and 10 km races over 324 fans (1-3 stages) and 48 propellers (2, 6, 12 blades) at three powers; `scenarios/merlin_race.py`, `propulsor_maps.py`
- [x] 5h.7 Split by job: 25 (2/3 blades) exports results and maps, 25b the ducted fan, 25c six and twelve blades, each with
  its maps; 26 designs MERLIN and exports it, 26b flies the mission from the exports, 27 races the libraries
- [ ] 5h.2 The race in head- and tailwind; air density at the fire's altitude; battery voltage sag at 12 C; a folding pusher propeller and its blade in the tail's wakes
- [ ] 5h.3 The EDF in its duct with `rotor_mrf_installed` on MERLIN's nose; the sensor inlet in CFD (where the sensor breathes)
- [ ] 5h.4 Belly-landing load case on the fuselage; the wing at the dash gust load with a spar (the printed wing alone is weak there)

## Stage 5i — Microjet and AGUYA (branch `dev_jet`)
- [x] 5i.1 `vegeta.boreas.microjet`: the turbojet cycle on its operating line, calibrated to a datasheet (thrust, fuel, EGT),
  maps, export; three catalogue classes
- [x] 5i.2 Aeromant `jet_external` (an aircraft with a running jet engine: intake and nozzle faces, exhaust tracer, plume
  samples) and `compressor_mrf` (an impeller MRF speed-line point), both compressible (`rhoSimpleFoam`, openfoam.com);
  named extra surfaces in `CFDCase`
- [x] 5i.3 Talos `RadialTemperature`, thermal expansion, stiffness and yield at temperature (CalculiX `*TEMPERATURE`)
- [x] 5i.4 `designs/turbojet.py` (impeller, turbine wheel, engine, compressor CFD passage) and `28_microjet`
- [x] 5i.5 `designs/aguya.py`, `designs/aguya_flight.py` (fuel burn, tank sizing) and `29_aguya` (race against MERLIN, hot jet,
  own exhaust, gust FEA, modes)
- [ ] 5i.6 Run the compressor speed line and refit the cycle's compressor from it; the jet CFD at the dash
- [ ] 5i.7 Turbine creep life (Larson-Miller) from the hot-wheel FEA; an annular combustor model; spool-up dynamics
- [ ] 5i.8 AGUYA's mission movie with the smoke (reuse `merlin_flight.render_movie`); the race in wind and at altitude

## Stage 5j — Tracked rover "Pekari" (branch `dev_track`)
A caterpillar-drive rover in the class of the wheeled rover of notebook 11 (~19 kg empty, a 5 kg payload in its
basket; the Onager is a 380 kg machine), planned in the same way: base
rover → two arms → basket, then the gears and track, a drive-type comparison with performance maps, and Chiron missions
of wheels (passive and active suspension, like the Onager) against tracks and half-tracks (the triangular-track
Catagon is its own product, Stage 5k). Nothing to
reuse for tracks, gearboxes or soil: no sprocket, gear or Bekker/terramechanics model exists yet, Chiron contacts are
rigid and it has only `Weld` equalities. Sources: Wikipedia "Continuous track" and "Half-track"; The Western Producer,
"Triangular tracks for STX and Magnum" (both blocked by the network proxy here, so fetch them before 5j.3/5j.6).
- [x] 5j.1 Stage 1, base tracked rover: `designs/pekari_rover.py` `PekariRover(Design)` + `components/pekari_rover.py` (`part` =
  rover/hull/track_module/sprocket/idler/road_wheel/track_link; rear drive sprocket, front idler with tensioner, road
  wheels on bogies, return rollers; static `contact_length`, `track_gauge`, `overall`), `designs/pekari_rover_robot.py` in the
  `onager_robot.py` pattern (`DESIGN`, `CAD`, `MATERIALS`, `PARTS_KG`, `ASSUMPTIONS`, `geometry`, `mass_budget`, battery
  placed for CG over the contact patch) and `30_pekari_rover` §1–§3: mass budget and CG, nominal ground pressure W/(2bL) and
  mean maximum pressure (Rowland MMP), steering ratio L/B ≤ 1.8, resistance (rolling, internal track loss, grade, drag),
  tractive effort against μW and soil shear (Bekker–Wong), top speed per grade, acceleration, range and endurance, static
  stability (longitudinal/lateral tip-over, side slope, step climb, trench crossing)
- [x] 5j.2 Gears and drive train: motor → gearbox (planetary, or spur + planetary) → sprocket; ratio from the hill-climb
  torque against top speed; gear sizing (module, tooth counts, Lewis bending, AGMA contact stress by hand, Talos FEA on a
  tooth); efficiency chain; gearbox life from Chronos rainflow on the mission torque; sprocket–link mesh (pitch, tooth
  count, chordal speed variation); motor and gearbox entries in `actuators.py`
- [x] 5j.3 Track: rubber band vs. linked pads, live vs. dead track; pre-tension, sag and the tension against throwing a
  track; road-wheel load distribution and pressure peaks; suspension options (rigid, bogie, torsion bar, Christie,
  Horstmann); skid-turn steering moment and lateral resistance (minimum turn radius against motor torque), clutch-brake vs.
  regenerative (differential) steering; wear and internal losses; Talos FEA on the link/pin and the sprocket, fatigue, life
- [x] 5j.1–5j.3 done in `30_pekari_rover` (real FEA; designs `pekari_rover.py`, `pekari_rover_robot.py`, `terramechanics.py`,
  `gears.py`, `tracks.py`; tests `designs/tests/test_pekari_rover.py`, `components/tests` CASES); the Chiron
  model came after (5j.1b). Follow-ups from the numbers:
  - [ ] 5j.3a Measure the track's internal loss (f_in = 0.035 + 0.002 v assumed: most of the resistance on soft soil)
  - [ ] 5j.3b Rubber pads or sharper grousers: the 30° climb fails on gravel (26°) and grass (20°) — traction, not power
  - [ ] 5j.3c Thinner links (FEA: 6.3 MPa, SF 16 at the tight side) or a rubber band: the belts are 4.8 kg of 19 kg
  - [ ] 5j.3d Hinge seals or sacrificial bushings: sandy hinges use the idler take-up in ~500 km (clean: ~50,000 km)
  - [ ] 5j.3e A 20-tooth sprocket (polygon effect 3.4 % at 48 Hz → 1.2 %); the gears nitrided (contact SF 1.27 at stall)
- [x] 5j.1b The Pekari Rover in MuJoCo (`30_pekari_rover` §7): `pekari_rover_robot.pekari()` (each track a row of 11
  rollers, the road wheels on bogies with passive hinges, velocity servos at the belt speed, a roller may carry 1/3 of
  its side's drive force) and `pekari_controller.py` (`TrackDrive` legs, `uneven_ground`, `leg_table`): 3 m, a 90° turn
  on 1 m, 2 m over rough soil with a 60 mm log and stones; the path, side forces against `tracks`, and a movie
- [x] 5j.1c Three trials of the unchanged rover in MuJoCo (`30_pekari_rover` §7.2, `pekari_controller.TRIALS`), each
  under `FailureRules` so the movie ends at the failing frame with the outcome on it (`end_card`): micro-hills the
  rover's own radius (diameter 1.44 m, slopes sized to 15°) — crossed; a 30° hill up and down — stalls at the foot
  of the ramp (traction: μ 0.6 gives 122 N, the slope asks 118 N + the internal loss; 26° was the hand limit); mud
  (`MudHook`: μ 0.25, Bekker compaction + viscous drag, the soft layer drifting sideways at 15 % W) — crossed, pushed
  0.24 m sideways. MuJoCo's ground stays rigid: no rut; a deformable height field is 5j.7's `TerramechanicsHook`
- [ ] 5j.4 Stage 2, two arms: `PekariManus(PekariRover)` in the Manus pattern (parent parameters via `replace`, Manus arm
  geometry), `pekari_manus_robot.py` reusing `onager_manus_robot` (`_arm`, `arm_ik`, `arm_fk`, `ARM_ACTUATORS`,
  `ARM_GAINS`, `STOW`) as `extra_children`; CG shift and tip-over margin over the arm workspace with payload, track
  pressure redistribution, actuator checks; `30b_pekari_manus`
- [ ] 5j.5 Stage 3, basket: the basket is Pekari's (and Catagon's) default configuration — built with the base rover in
  5j.1 (`PekariRover` `part="basket"`, `basket_*` parameters, the Sweeper's sheet-metal basket; the 5 kg payload sits in
  it; loaded vs. empty pressure, CG and the drive re-checked on the loaded grade are in `30_pekari_rover`). Left for
  stage 3 with the arms: the payload range against stability (a load chart as for the Atlas, payload mass × position in
  the basket) and the pick → place-in-basket reach check; in `30b_pekari_manus`
- [ ] 5j.6 Drive-type comparison and performance maps: `designs/terramechanics.py` (Bekker–Wong soil table: dry sand,
  loose sand, clay, snow, mud, grass, gravel, asphalt with k_c, k_φ, n, c, φ, K; wheel sinkage, compaction resistance,
  drawbar pull–slip; track with uniform and MMP pressure; rigid-surface μ and C_rr; step, trench and slope criteria).
  Drive types at equal mass and payload:
  - 4-wheel skid steer; 6-wheel rocker-bogie; 4 wheels on active legs (Onager)
  - full tracks (Pekari Rover)
  - [-] (not for now) half-track: steered front wheels + rear track units (`PekariHalftrack(Pekari)`: Ackermann front axle, short rear
    track); front/rear load split, wheel steering vs. track braking, road speed and efficiency against soft-soil
    traction, the front-wheel sinkage penalty
  - triangular (delta) track units: a separate product, **Catagon** (Stage 5k); it joins this comparison once built
  - legged (robot dog) for reference, optional
  `designs/mobility_maps.py` in the `propulsor_maps.py` library pattern (`build`, `save`, `load`, JSON in `data/`):
  drawbar pull/weight vs. slip per soil, max grade × soil, speed × grade → power and efficiency, cost of transport,
  obstacle height and trench width, go/no-go heat maps (terrain × drive type), radar summary (matplotlib `contourf` /
  `imshow`); `31_drive_comparison`
- [ ] 5j.7 Terrain race, its own notebook `32_terrain_race` (MuJoCo/Chiron), not part of `30_pekari_rover`/`31_drive_comparison`;
  shared code in `designs/terrain_race.py` (course, hooks, runner, tables): the track as a multi-roller approximation (6–8 road-wheel cylinders per side,
  `role="foot"`, plus sprocket and idler, one velocity command per side, capsule pads between rollers to keep contact over
  steps, internal loss as `frictionloss`; a closed chain of pad links needs a `Connect` equality in `chiron/robot.py`,
  deferred); soft soil as a `TerramechanicsHook` via `ChironLab.add_hook` (Bekker sinkage resistance and a slip-limited
  thrust cap per contact foot, by terrain zone); one course from `ch.Custom` zones (rocks and steps, 20–30° slope, loose
  sand, mud, side slope). Robots on the same course:
  - (a) wheels, passive suspension (`rover` geometry, a new `rover_robot.py`)
  - (b) wheels, active suspension (Onager Sentinel: `onager_robot` + `Drive`)
  - (c) full tracks: `pekari_rover_robot.pekari()` and `pekari_controller.TrackDrive` exist (5j.1b); add a slip-aware
    torque limit
  - [-] (not for now) (d) half-track (`pekari_halftrack_robot.py`: steering hinge servos on the front wheels + a rear multi-roller track;
    Ackermann blended with the track-speed differential)
  - (e) Catagon (Stage 5k), once built: four delta units, each on a pitch hinge at the hub with spring/stop limits;
    inside each a walking-beam bogie on its own hinge with 4 mid-rollers and 2 end idlers, all `role="foot"`
  Mission via `PhasedMission`: cross the course → pick an object with the arms → put it in the basket → return;
  `scenarios/terrain_race.py` → `scenarios/output/terrain_race_*.mp4` + JSON; per-segment success, time, energy (Wh/m), slip, max
  tilt, sinkage, stall; `benchmark/terrain_race/` with `run_trials` over terrain × drive × seed, report and success-rate heat
  maps from `stats.cell_rates`
- [ ] 5j.8 Tests + docs: `components/pekari*.py` in `test_ground.py` `CASES`; `designs/tests/test_pekari*.py`
  (defaults == `DESIGN`, stored CAD numbers, mass budget, standing, flat drive) and
  `test_terrain_race.py` (course zones, hook forces, slow race); `terramechanics` against
  published Wong examples; `designs/README.md`, `scenarios/scenarios.md`

## Stage 5k — Catagon, the two-segment quad-track rover (later; a separate product from Pekari)
Like Pekari, it carries a payload basket by default. Two body segments joined by an articulation joint (steering by articulation, plus pitch/roll oscillation between the
segments), each segment on a pair of triangular (delta) track units — four units in all, as the Quadtrac tractors and
the STX/Magnum track conversions.
- [ ] 5k.1 The delta unit, from the user's reference image: a large elevated drive wheel on the hub at the apex, driving
  the rubber belt's inner lugs by friction/positive drive; two large end idlers (front and rear); 4 small mid-rollers on
  a walking-beam bogie frame that pivots under the hub; a tensioner on one idler; a rubber belt with chevron tread. CAD
  `designs/catagon.py` `Catagon(Design)` + `components/catagon.py` (`part` = catagon/front_segment/rear_segment/basket/
  articulation/delta_unit/drive_wheel/end_idler/mid_roller/bogie_frame), `designs/catagon_robot.py` (mass budget, CG
  per segment), notebook `33_catagon`
- [ ] 5k.2 Unit checks: the drive wheel clear of mud and rocks; unit pitch oscillation (stops ± deg, the anti-rotation
  link) on uneven ground; contact length and footprint against a wheel; how the hub load spreads over the idlers and
  mid-rollers; belt wrap angle and slip on the drive wheel (capstan, T1/T2 = e^(μθ)); added mass, height and gear ratio
  (the drive wheel is smaller than a wheel it would replace); reuse `gears`, `tracks`, `terramechanics`
- [ ] 5k.3 Articulation: steering torque and the articulation actuator, turning radius against articulation angle, the
  load split between the segments, tip-over with the segments yawed and rolled, hitch loads (Talos FEA)
- [ ] 5k.4 Catagon in the drive comparison (5j.6) and the terrain race (5j.7); tests + docs

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
- [x] 12.3 `Campaign` (now in `vegeta-fidia`): bounded agentic parameter search on `vegeta.core` revisions (criteria, objective, budget, approval policy, STOP file, resumable record); `notebooks/demo/10_agentic_design.ipynb`
- [x] 12.4 Fidia prompt-to-3D (`vegeta-fidia`, branch `dev_fidia`): plan → CadQuery (Dedalus contract) → sandboxed build (AST screen, clean env, rlimits, process-group kill) → checks (validity, watertight, winding, budget, size, floating) → five views + sheet → vision review → revise; limits, feedback, cancel, STOP, approval, best valid revision, resumable run; GLB/glTF+bin/OBJ+MTL/STL/STEP with re-import by trimesh and VTK; `fidia run|resume|show`; `notebooks/demo/15_fidia_prompt_to_3d.ipynb`; docs/fidia.md (incl. the Blender assessment)
- [ ] 12.5 Fidia backends: Blender (organic shapes, textures, photoreal renders) behind the same runner contract; network isolation by default where user namespaces exist

## Stage 13 — Validation (continuous)
- [x] 13.1 Unit tests in every package
- [x] 13.2 Integration tests with numerical validation; clean skips when tools are missing
- [ ] 13.3 After every stage: run tests, fix regressions, update/remove documentation (ongoing; done for stages 1–9)

## Stage S — Authoritative notebooks, side labs, Ozaru
Notebooks own the analysis and its story. Side labs (`side_labs/`, formerly `workflows/`) run optional, expensive
calculations and fill the Porunga cache; neither imports, launches or names the other. The only exchange is a
Porunga key built from a plain-JSON identity (`docs/cache_identities.md`) and its payload. Ozaru (`vendor/ozaru`,
git subtree) supplies Porunga, Mimesis and Yanus (requirement checks).
- [x] S.1 Mock baseline: every notebook executed with the mocks on (no CFD, FEA or FEniCSx during the refactor); mocks on by default (DEBUG = True); notebooks split into devices/ parts/ scenarios/
- [x] S.2 Vendor Ozaru (subtree 33702e4, signed), install it offline from `vendor/ozaru` in `install_local.sh`; `scripts/update_ozaru.sh`; duplicate `mimesis/` removed
- [x] S.3 Move the engineering design modules and data from `notebooks/designs/` to `designs/`
- [x] S.4 Rename `workflows/` → `side_labs/` (`workflows.sh` → `side_labs.sh`); drop every notebook reference from it (test: `side_labs/tests/test_separation.py`); notebook declarations in `notebooks/flow.py` until S.8
- [x] S.5 Cache identities for Kestrel (`designs/kestrel_keys.py`, `docs/cache_identities.md`; key-agreement test)
- [x] S.6 The Kestrel side labs publish with `porunga.save` (`side_labs/kestrel/_kestrel.publish`); runkit's case cache unchanged
- [x] S.7 Notebook 43 reads with `porunga.check` / `load` (`kestrel_study.inputs_from_cache`); pins and `find_runs` removed; Kestrel moved to `notebooks/exp/`
- [x] S.8 Old Yanus removed: `vegeta-yanus/`, `yn`, `notebooks/flow.py`, `docs/yanus*.md`, `yanus2.md`; the side labs' books and chapters in `side_labs/registry.py` (needs from Ozaru's Yanus, prerequisites by `library.check_prerequisites`); each notebook prints its own needs table (Ozaru's Yanus); 45/46/48 read notebook 43's report directly
- [-] S.9 Legacy results into Porunga: parked
- [x] S.10 Mock-only tests: `user_tests.sh` (vegeta, design, workflow, notebook parts; `--no-*`, `-v`, `--include-experiments`); `VEGETA_NO_SOLVERS=1` blocks CalculiX, OpenFOAM, PrusaSlicer and XFOIL; MuJoCo, the aero models (BEMT, lattice, transitions, NeuralFoil) and Gmsh mocked
- [x] S.10b Notebook runner: one installed kernel for all (a dead `python3` kernelspec failed 26b, 27 and 28 in 2 s), at most 4 side by side, one retry of a start-up failure, the log's tail on a failure
- [x] S.10c Experiments: Kestrel notebooks, `28_microjet` and the Kestrel-X design modules (`designs/exp/`) out of the default runs
- [x] S.11 Docs: README (layout, notebooks / side labs / Ozaru, mock tests), installation (offline Ozaru, `user_tests.sh`), side labs README and manual (Porunga publishing, pins gone); offline check of a clean clone done (Ozaru 2.4)
- [ ] S.12 Check by hand later (left out of the fast mock checks): the long Chiron/MuJoCo notebooks and the demos that need an API key
