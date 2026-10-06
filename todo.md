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

## Stage 5l — Drongo, potato and cream delivery (branch `dev_potato`)
- [x] 5l.1 `designs/drongo.py` (`Drongo(QuadFrame)`: skids, rack-and-pinion pincer), `drongo_robot.py`, `drongo_controller.py`, `drongo_scenario.py`; notebook `08b_quadcopter_potato`: items and limits, grip window, net drop height, both variants in MuJoCo with movies, the wait, speed/altitude sweep, export; `scenarios/drongo_delivery.py`, `user_tests.sh drongo`, `tests/test_drongo.py`; `chiron.viz.frames`: `ground`, `scenery_range`, `ground_color`, a render per frame
- [ ] 5l.2 The net as a body (cloth or spring mesh) instead of a catch at its plane; the people's arms giving
- [ ] 5l.3 Wind and gusts in the garden; the rotors' downwash on the net and the people; ground effect at the pickups
- [ ] 5l.4 Frame FEA with the pincer and skid loads (notebook 08's load cases with 0.4 kg hanging below)
- [ ] 5l.5 Measured numbers: the cream cup's size and crushing force, potato bruising thresholds, notebook 08's export run

## Stage 5f — Air couriers (branch `dev_velutina`)
- [x] 5f.1 `designs/velutina.py` (slim-body quadrotor: capsule nose with handle, pusher arms, fins, parachute bay; `part`, `angle_of_attack_deg`), `designs/velutina_flight.py` (terrain, ISA, wind + gusts, reduced 6-DOF flight model with set-down and hand-over, parachute estimate, movie) and `24_velutina`: mass budget, drag areas by hand + actuator-disk slipstream, Aeromant screening (fast/full presets), Boreas at the depot and the site, arm/capsule/shell FEA, modes and Campbell, the mission with precision statistics and energy, Chronos fatigue, printing, movie, JSON; `scenarios/velutina_mission.py`, `user_tests.sh velutina`; `core.Revision.run_cfd(processors=)`
- [ ] 5f.2 Parachute in CFD (opening, canopy drag, the body in the canopy's wake); the bridle anchored to the arm frame and a reefed canopy (the shell fails the opening shock in 24 §5); a drop test case
- [ ] 5f.3 Propeller guard rings for the hand-over mode (a hand near the rotors): CAD, mass, their drag in the CFD
- [ ] 5f.4 Four rotor disks in the whole-aircraft CFD (`aircraft_rotor_disks` has two): the slipstream over the arms and fins instead of the analytic increment
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

## Stage 5m — PEREGRINE, the folding-wing farm drone (branch `dev_peregrine`)
- [x] 5m.1 `designs/peregrine.py` (`Peregrine(Merlin)`: wings A large fixed / B small fixed / C hinged, `fold_deg`,
  `hinge_y_frac`, hinge lug, `planform`, `fold_limit`, folded drag build-up), `designs/peregrine_flight.py` (vortex lattice in
  streamwise strips, stability and trim vs fold, BOM and lab hours, drive with fixed losses, envelopes, birds, stoop, gaps,
  prop-hang / hand throw / roof drop / perched wind, hinge loads, boids herding, sortie and movie, decision table) and
  `28_peregrine` (Parts 1–3: does folding pay? — no, B wins most weightings; folding pays only for roof rest in wind);
  `tests/test_peregrine.py`, `user_tests.sh peregrine`
- [x] 5m.2 Merge `dev_potato` (Drongo) into `dev_peregrine` so Part 4 can reuse Drongo's pincer, robot and controller
- [ ] 5m.3 **Part 4 — the mid-air catch** (planned, not implemented). Drongo flies holding a long scientific probe; PEREGRINE
  swoops down, takes it with talons, flies on and delivers the sensor undamaged to the base. Calm air, then unsteady wind.
  MuJoCo through Chiron, like Drongo; all three wings A/B/C. Design worked out so far:
  - *Handover geometry*: a falcon strikes from above, so Drongo holds the probe **pointing up** as a mast in its pincer, the top
    ~0.40 m above its rotor plane; both fly the same course with Drongo at ~PEREGRINE's 1.25 × stall (10–12 m/s; Drongo
    tilts ~8° for that, easy), so PEREGRINE overtakes at 0.5–1 m/s (air-to-air refuelling style), not 10 m/s.
  - *Probe* (`PROBE`, assumptions): 0.55 m × 32 mm cylinder, 0.25 kg, μ 0.6; undamaged = squeeze ≤ 60 N per pad, shock
    ≤ 30 g, no propeller/ground contact, net arrest within its load. Carried vertical it has ~0.018 m² of drag — about four
    times PEREGRINE's whole Cd·A (budget it; it may rotate in the grip).
  - *Talons* (`peregrine.py`: `talons`, `talon_*`, `part="talons"`): two pads closing along y on a rail under the belly at
    the CG, on **legs ~0.20 m long**: the probe passes under the nose first, so the grip zone must sit below the propeller
    disc (r 0.114 m) with margin — short legs leave a vertical window of only ~8 cm, 0.20 m legs ~18 cm. Open gap ~0.14 m
    (±5 cm lateral window), pads 0.12 m long × 0.08 m tall. Extended legs add ~0.0036 m² Cd·A (≈ the airframe's): retract them.
  - *Closing speed is the crux*: Drongo's rack-and-pinion servo closes at ~0.11 m/s per jaw (≈0.5 s), but the pads' 0.12 m
    dwell at 0.5–1 m/s overtaking is 0.1–0.25 s — so a **spring-loaded snap closure** (latch released by a trigger, the
    falcon's tendon lock) with the servo only to re-open/release; compare both in a capture-window analysis.
  - *Grip*: Drongo's arithmetic as is (`drongo_robot.grip_needed` / `grip_holds` with the probe as an item): 20 N per pad
    holds ~9.8 g with μ 0.6, under the 60 N limit.
  - *Simulation* (`designs/peregrine_catch.py`): PEREGRINE as the Chiron `Robot` (origin at the CG, Chiron frame x forward /
    y left = notebook frame rotated 180° about z; fuselage capsule, wing quads from `peregrine.planform` as boxes, tail, fin,
    talons as two servo slides); Drongo as a free `Prop` built from `drongo_robot.drongo().root` with the jaws stripped and
    every geom `role="visual"` (no contacts; hits judged geometrically), flown by `drongo_controller.Flight` set up by hand
    for a prop (mass, inertia from `model.body_inertia`, mixer) and a `drongo_robot.Rotors` subclass for a prop body with
    air-relative drag; the probe a free `Prop` held by a `Weld` to Drongo, released with `lab.set_weld` after the grip
    signal + a radio delay. Start velocities set in the controller's `reset` (Chiron's reset zeroes qvel); `settle=0`.
    One controller object computes the OU gust (as `velutina_flight.simulate`), PEREGRINE's 6-DOF aero (CL0, CLα, Cm0, Cmα,
    Cmδe from `peregrine_flight.aero` moved to the CG, stall blend to a flat plate, Cmq ~ −8 from the tail volume, Clp
    −0.45, Clδa from p b/2V = 0.09, CYβ −0.4, Cnβ 0.06, Cnr −0.12, Clβ 0.05; thrust from `peregrine_flight.drive`) and
    its autopilot (course → bank, γ → load factor → α → elevator with trim feed-forward, throttle on airspeed); guidance:
    speed-matched pursuit of the probe top, 3 m above until 20 m behind, then the swoop down to the grip window; trigger when
    the pads straddle the probe. Body forces are per body (PEREGRINE, Drongo, probe), so no hook overwrites another.
  - *Judge* (Drongo's `Watch` idea): pad squeeze from `lab.contact_force`, the probe's shock (specific force), probe vs
    PEREGRINE's propeller disc and PEREGRINE vs Drongo's rotors checked geometrically, ground contact, the net catch.
  - *Home*: climb, cruise to the base, release over Drongo's people's `NET` from ≤ 4 m above it with the fall's lead
    (≈ 9 m at 10 m/s); arrest load from the total speed, not only the vertical.
  - Notebook 28 sections 18–25 (job and probe, talons, handover geometry and clearance, calm catch A/B/C, gust Monte Carlo,
    home and net, movie with `chiron.viz.frames` + `drongo_scenario._hud`, does folding help the catch, export
    `peregrine_catch.json`); `tests/test_peregrine_catch.py` (talon CAD, grip, level-flight trim, weld holds until
    released, one calm catch, judge flags over-squeeze). Quick runs only (`PEREGRINE_QUICK`); no CFD.
- [ ] 5m.4 Fold-in-flight dynamics (unsteady aero while folding), measured LW-PLA and PETG coupons, the hinge in CFD

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
  `gears.py`, `tracks.py`; tests `notebooks/designs/tests/test_pekari_rover.py`, `components/tests` CASES); the Chiron
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
- [ ] 5j.8 Tests + docs: `components/pekari*.py` in `test_ground.py` `CASES`; `notebooks/designs/tests/test_pekari*.py`
  (defaults == `DESIGN`, stored CAD numbers, mass budget, standing, flat drive) and
  `test_terrain_race.py` (course zones, hook forces, slow race); `terramechanics` against
  published Wong examples; `notebooks/designs/README.md`, `scenarios/scenarios.md`

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
- [x] 12.3 `Campaign` (now in `vegeta-fidia`): bounded agentic parameter search on `vegeta.core` revisions (criteria, objective, budget, approval policy, STOP file, resumable record); `notebooks/10_agentic_design.ipynb`
- [x] 12.4 Fidia prompt-to-3D (`vegeta-fidia`, branch `dev_fidia`): plan → CadQuery (Dedalus contract) → sandboxed build (AST screen, clean env, rlimits, process-group kill) → checks (validity, watertight, winding, budget, size, floating) → five views + sheet → vision review → revise; limits, feedback, cancel, STOP, approval, best valid revision, resumable run; GLB/glTF+bin/OBJ+MTL/STL/STEP with re-import by trimesh and VTK; `fidia run|resume|show`; `notebooks/15_fidia_prompt_to_3d.ipynb`; docs/fidia.md (incl. the Blender assessment)
- [ ] 12.5 Fidia backends: Blender (organic shapes, textures, photoreal renders) behind the same runner contract; network isolation by default where user namespaces exist

## Stage 13 — Validation (continuous)
- [x] 13.1 Unit tests in every package
- [x] 13.2 Integration tests with numerical validation; clean skips when tools are missing
- [ ] 13.3 After every stage: run tests, fix regressions, update/remove documentation (ongoing; done for stages 1–9)
