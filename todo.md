# Vegeta — TODO

Vegeta is the R&D notebook set: fast and dirty, and it stays that way (the blessed imperfection rule, README).
Rules: the tools stay independent packages; the stable physics of standard parts comes from Ogma (`libs/ogma/`,
`import ogma as og`); Piccolo never imports Vegeta, and a machine graduates there once its design is frozen.
Check quickly with `./test.sh --smoke` and the mocked tests (`VEGETA_DEBUG=1`, `-p vegeta.mock.pytest_plugin`);
the full suites are in `user_tests.sh`. Anything the refactor finds missing in Ogma is added to Ogma on the way. Earlier stages (1–13, 5b–5t, R, S) and their open items: [`old_todo.md`](docs/old_todo.md).

Legend: `[ ]` open · `[x]` done · `[-]` deferred

## Stage V1 — The split with Piccolo (2026-10-20)
- [x] V1.1 Smoke suite `./test.sh --smoke` (under 30 s) and the pre-move baseline (`docs/migration/baseline.md`, robotics goldens recorded before the move)
- [x] V1.2 Yax design interchange (STL + `yax.json`) and `vegeta.dedalus.export_yax` (exact OCC mass properties, checked against the mesh integration)
- [x] V1.3 Chiron to its own repo (`technolog-spectrumai/chiron`, history kept, `chiron.mock` split out of `vegeta.mock`); `vegeta.chiron` stays here until the last machine hatches
- [x] V1.4 (reverted 2026-10-21: the races came home — Piccolo is ROS-centric full simulation; `races/` flies `ogma.rotors` and `designs/data/`, its tests run in the smoke beside the rest) The racing quad hatched → Piccolo v2: `races/`, the autopilot scripts, the race notebooks, `side_labs/races`, the races book (history kept; path map in `docs/MOVED.md`); `labs/aero/designs/drongo_controller.py` stays (Drongo has not hatched)
- [x] V1.5 Boundary test: nothing in Vegeta imports Piccolo; engineering code imports no robotics (races included)
- [x] V1.6 README: the blessed imperfection rule — go fast and break things; Vegeta stays the R&D notebook set

## Stage V2 — Ogma vendored (Ogma O0–O2)
- [x] V2.1 `vendor/ogma` (squashed subtree, `scripts/update_ogma.sh`, `vendor/OGMA_REVISION`); Yax moved from Ozaru to Ogma
- [x] V2.2 `kestrel_systems.cg_inertia` and `nisus_systems.cg_inertia` as thin adapters over `og.mass.Budget` (goldens identical)
- [x] V2.3 `designs/gears.py`, `terramechanics.py` (→ `og.soil`), `tracks.py`, `cutting.py`, `actuators.py` moved to Ogma with history; designs import `og.*`, the old modules removed
- [x] V2.4 Ozaru's `mimesis` renamed `mimezia` (vendored Ozaru, imports, scripts, docs, `MIMEZIA_*` variables)

## Stage V3 — Designs on Ogma's physics (Ogma O3)
- [x] V3.1 `og.atmosphere`: `nisus_plus_systems` atmosphere and density altitude, `velutina_flight.isa_density`
- [x] V3.2 `og.wind`: `nisus_robot.Wind` (Nisus+'s `MountainWind` builds on it)
- [x] V3.3 `og.batteries`: `nisus_plus_systems` cells and packs, `nisus_systems` bought packs
- [x] V3.4 `og.rotors`: Drongo's `rotor_layout`, `Rotors` wrench and `Flight.mix`; `kestrel_flight.induced_velocity`
- [x] V3.5 `og.propulsion`: `nisus_systems.PropulsionMap`, `nisus_plus_systems.NisusPlusDrive` (`ElectricDrive`), `merlin_flight.Unit`, `kestrel_systems.DriveTable`
- [x] V3.6 `og.aero`: the Nisus and Nisus+ aero hooks (one stability model, crow optional; air angles and wrench), `kestrel_flight` `Polar`/`viterna_cd90`/`_cf`, `merlin_flight.Airframe`, `peregrine_flight.vlm`, the turbulent Cf copies (aguya, falco, merlin, nisus, peregrine, frame_study, air_propeller, merlin_flight)
- [x] V3.7 `og.motors`: `falco_systems._static_point` and the frame study's static loop (`operating_point`), the Nisus+/Falco resistance fits, the Onager Atlas tilt, Pekari roller and Perseus arm servos (`lever`)
- [x] V3.8 The `quat_to_R` / `euler_from_R` copies (`nisus_robot`, `drongo_controller`) → `og.frames` (the parked `exp/mini_races.py` keeps its own)
- [x] V3.9 The battery-for-CG solves of `kestrel_systems`, `kestrel_tandem`, `nisus_systems`, `nisus_plus_systems` (and so Falco), `peregrine_flight` → `og.mass.place_mass_for_cg` / `cg_with` (added to Ogma for tables that keep their own sums; tables identical before and after); `onager_robot`, `robot_dog_robot` (robotics, they graduate later) and the parked `kestrel_xwing.py` (then in `exp/`) keep theirs
- [x] V3.10 Velutina's wind → `og.wind`: `Wind.steady` (`from_direction`, `linear_shear`), `SiteWind`'s profile (`power_law`), the gust and navigation-error steps (`ou_euler_step`) in `simulate` and `follow_path` — all added to Ogma for it; the tower shadow and wake stay here (site-specific); episodes identical before and after
- [x] V3.11 `G` / `RHO` unified on `og.units`: the design models (`merlin_flight`, `nisus_systems`, `nisus_structure`, `nisus_plus_systems`, `falco_flight`, `falco_structure`, `onager_sweeper_cfd`) 9.81 → 9.80665 (weights −0.034 %, its own commit); every local `RHO`/`RHO0` = 1.225 and `G` = 9.80665 an import (no change); `og.aero.Airframe.g` = standard gravity. The Chiron robots and controllers keep 9.81 (Chiron's MuJoCo gravity) until they graduate
- [x] V3.12 Falco's `fuselage:` shell row gets its ellipsoidal-shell inertia (`nisus_systems.cg_inertia` matches `pod:` and `fuselage:`): Falco Iyy +4.4 %, Izz +1.8 %, Ixx +0.1 % (mass and CG unchanged; Nisus and Nisus+ unchanged)
- [x] V3.13 Boreas's `Motor` / `Battery` are Ogma's (`og.motors.DCMotor`, `og.batteries.NominalPack` — added for it) under Boreas's names, and `boreas.microjet.isa` takes the ISA from `og.atmosphere` (Ogma is a vendored library like numpy, not a Vegeta tool, so tool independence holds); Boreas tests 56/56 before and after

## Stage V4 — Vegeta application (after Ogma O6)
- [x] V4.1 Designs take their mass budgets, parts, propulsion and aero from `og.*` throughout: a sweep after V3 moved the last duplicates — the torque constant (`og.motors.kt`: `falco_systems`, `frame_study`), hover momentum theory (`og.rotors.hover_induced_velocity` / `ideal_hover_power`: `kestrel_loop`, `kestrel_study`), Velutina's ISA constants (`og.atmosphere`). Left on purpose: the Chiron robots and controllers (their servos, `cg`, `Drive`, 9.81 gravity) until they graduate (V5), and the parked `designs/exp/`
- [x] V4.2 Ogma's CAD strategy (`og.cad`, Ogma O6) carries the CAD ↔ physics link: `vegeta.dedalus.export_yax` takes its exact mass properties and the Yax mass section from `og.cad` (moved there); the designs' own geometry stays CadQuery in Dedalus (machine geometry is not Ogma's)

## Stage V6 — Standard files between platforms (2026-10-21)
- [x] V6.1 Ogma comes home: the parts library moved from its own repo into `libs/ogma/` with its history (`import ogma as og` unchanged); `vendor/ogma` removed; smoke runs `libs/ogma/tests`
- [x] V6.2 Yax moved to its own repo (`technolog-spectrumai/yax`) and becomes the format converter (USD hub); vendored at `vendor/yax` (`scripts/update_yax.sh`)
- [x] V6.5 Chiron stays home: `vegeta.chiron` is the one Chiron (the robots are designed and tested here); the chiron repository is retired and Piccolo no longer vendors it — a graduated robot goes to Piccolo as MJCF written through Yax
- [x] V6.3 Vegeta exchanges designs with Piccolo only as standard files (URDF/SDF/USD/MJCF/STEP/JSBSim) written through Yax
  - [x] Yax 2 converts every format (vendor/yax); Piccolo reads its fixtures and Gazebo models as standard files
  - [x] the export path: `python designs/export.py <machine> --to out/<machine>.usda --also sdf [--lump]` — Chiron robot → `to_mjcf()` → Yax; 14 machines (`--list`); `designs/tests/test_export.py` in smoke
  - [x] part data in the USD as `data["ogma"]`: Drongo's rotors (position, spin, mass, disk inertia, kF = T_max/ω_max², kM, ω_max, τ up/down) for Gazebo's motor models; `--lump` = one rigid body minus the rotors
  - [x] Piccolo takes it: `vehicles/received/drongo.usda` → `drako_drongo` through `yax sync` (Piccolo P2.8b)
  - [x] tags, not flags: every property (a path: `link:<name>/mass`, `data:propulsion/propellers/apc`, …) carries codes of ≤ 8 alphanumerics; Yax assigns the `00…` system tags (`00core` never withheld), the designs theirs (`boreas`, `nb08`, `fitted`, `assumed` in the parts functions' `tags`), the user more (`--tag PATTERN=CODE`); `--keep`/`--drop`/`--tags FILE` choose by tag; consumers validate by tag code (`tools/consumers/piccolo.json`); vendored Yax a1d758a
  - [x] YX2 stage 0: the export's main file is `.yx2` (Yax's validated USD subset: units, frame, stable ids, feature metadata, canonical text; vendored Yax 7b0310a); Vegeta → YX2 → Yax → YX2 is byte-identical for every registered machine, full and selected; `.usda` and the other formats as before
  - [x] selective export (`docs/export.md`): `tools/export.py --include/--exclude/--full/--features FILE` picks Yax's features independently (withheld, never invented; the USD records it); `--consumer tools/consumers/piccolo.json` reports served / not served per feature and format without changing the export (`--strict` exits 1); the robot's `sources` and `notes` leave as provenance, Drongo's densities as a materials catalogue; vendored Yax 5bee917 (features)
  - [x] propeller maps as part data (`vegeta.labkit.propulsion`, `labs/aero/designs/propulsion_parts.py`): Drongo's 5x4.3 from notebook 08's Boreas system, NISUS's 9x6 from its propulsion map, NISUS+'s 15x8 and FALCO's 20x15 from their electric drives — geometry, CT/CP against J, the grid, the motor, the reduced constants with their residual; Yax (f62820d) writes them as JSBSim propellers (`--also jsbsim`; NISUS flies on its own map in `tools/tests/test_export.py`) and reports the reduction for SDF; the single-propeller spins are assumed (`SPIN`)
  - [ ] part data for the rovers and walkers (wheel radii, servo limits) when Piccolo maps one — a wheeled diff-drive rover for Iguana first

## Stage V7 — Labs on a common library layer (2026-10-21)
- [x] V7.1 The lab rules hold in place: `libs/labkit` (bootstrap, machine registry, control, movie, kinematics, parts); `vegeta.mock.on_install` (designs coarsen themselves); the export reads each lab's `designs/machines.py`; `tools/labs.py check` in smoke
- [x] V7.2 The libraries live in `libs/` (vegeta-cli, -core, -ai, -fidia, ogma, labkit)
- [x] V7.3 `labs/{aero,land,legged,marine,studio}` in lifecycle folders (`notebooks, designs, sims, campaigns, exp, common, tests`); `components/` folded into `designs/`; all lab tests pass on mocks
- [x] V7.4 `tools/lab.py` (test, run, new-exp, promote, demote, export, readme), generated lab READMEs (machines' stages); `user_tests.sh` mocks only, per lab
- [x] V7.5 The graph effects measured (`docs/anatomy`): 0 cross-lab imports, 0 libs → lab imports, modularity 0.419 → 0.455, each lab reached by itself only (entropy 0), the libraries by several (0.57–0.99)
- [x] V7.6 Notebooks start with `bootstrap.lab("<lab>", DEBUG)` (61 notebooks: the `sys.path.insert` cells and `mock.setup` replaced; every notebook's bootstrap and imports run on mocks)
- [x] V7.7 Each lab's `common/` holds what its studies repeated (measured: functions defined identically in 2+ notebooks): `aero_kit` (plain, lower_skins), `land_kit` (terrain rocks/drops, pin stresses), `legged_kit` (pin_stress) — 25 copies in 19 notebooks replaced by imports; helpers that read notebook globals stay. The tools' copied helpers (`result.py` ×9, `_process.py` ×3, `cache.py`, `_progress.py`, `_watermark.py`) stay copies — each tool is standalone by rule (`test_independence`) — and are kept identical: `tools/sync_copies.py`, `tests/smoke/test_tool_copies.py`
- [x] V7.8 Marine has tests (the boat, the submarine and their assemblies build), studio has tests (every notebook bootstraps and its imports run)

## Stage K4 — K4TQ tilt-wing medical transport (plan: `labs/aero/notebooks/k4_plan.md`, `k4_work.md`)
- [ ] K4.1 **Check the drag of the wing-turning mechanism.** Each wing root is a rotating joint (±90° tilt) next to a fixed fuselage, so it cannot be faired like a fixed wing root. Sources to quantify:
  - the gap between the rotating root rib and the fuselage side (leakage from the pressure to the suction side: lost root lift, more induced drag);
  - the 42 mm tilt cartridge and the fold-hinge lugs standing out of the fuselage line;
  - any servo or gear bulge;
  - the fairing step at intermediate tilt (transition) and at the cruise trim angles (±5°).

  Stage 1 assumes 4 × 4 cm² of drag area for all of it (`k4tq_sizing.ROOT_GAP_CDA`, ESTIMATE). Check it:
  - with a root-junction CFD case in Stage 3 (cruise at 0° and ±5° tilt, gap 1 / 2 / 4 mm);
  - against fixes: a seal (brush or elastomer lip), a root disc fairing that turns with the wing, the cartridge sunk inside the fuselage line.

  Feed the result back into the drag build-up and the range.

  Stage 2 (2026-10-21) moved the tilt drive (stub, bearings, servo, gears) inside the fuselage: only the 12 mm fold knuckle, the hinge and locking pins and
  the stub's passage through the skin remain in the flow. The CFD question narrows to that knuckle, its gap and the root plate at 0 / ±5° tilt.
- [x] K4.2 Stage 1: the transition corridor solved as one least-power balance per speed (tilt, rear differential tilt, thrust, split; SLSQP with continuation): trimmed everywhere; wing-borne from 21 m/s, the free span stalled 1–20 m/s (the CFD question, R10); `phase1.md` regenerated
- [x] K4.3 Fuselage length: config B at 1.2 m with 15x10 (disc gap 159 mm, front motor at 63 % in the hover, the best R3 margin) — approved 2026-10-21
- [x] K4.5 Stage 2: the CAD masses (+12 %) broke the Stage 1 design (the mission −18 Wh, the sagged T/W 1.43): revised to MN501-S KV300, 16x10, 6S3P (`designs/data/k4tq/stage2_motor_sweep.csv`), Stage 1 rerun on the CAD masses (revision B)
- [ ] K4.6 Motor data to verify with T-Motor's datasheets before the BOM (Stage 4): the MN501-S KV300's no-load current (estimated 1.0 A) and its 40 A / 180 s rating at the hover's 6S current; the MN5006 / MN505-S rows likewise
- [x] K4.7 Wing sections by flight phase (NeuralFoil, `designs/k4tq_sections.py`, `exp/K4TQ_airfoils.ipynb`): front NACA 4415, rear NACA 2415 → K4TQ-F1 "Freddy" / F2; the four designs compared in `exp/K4TQ_compare.ipynb`
- [ ] K4.8 K4TQ-A1 / A2 (NACA 2412 front) leave 14.4 mm of section depth at the 120 mm front tip chord for a Ø16 spar that needs 18 mm: adopt K4TQ-F1's (Freddy's) sections, or taper the spar to Ø12 at the tip (Stage 4 structure decides)
- [ ] K4.9 Run the section CFD check (`exp/K4TQ_airfoils.ipynb`, DEBUG = False, ~30–45 min) — confirm NeuralFoil's ranking before freezing Freddy
- [ ] K4.12 K4TQ-F3 "Freddy 3" / K4TQ-G3 "Ginna 3" (the fast autonomous designs, `dev_kestrel5`): G3's tilt servo reserve is ≈1.07 at 44 m/s — a 6.5 N·m class servo first; the all-moving wing's divergence and flutter at 40–45 m/s (Stage 4); the Aeronaut CAM blades' rpm limit against the hover rpm and their stock (16x13, 17x13); Ginna's / G2's drive (U8 II KV170 + 20x10) is an assumed class, not a shelf item (the real twin: MN601-S KV170); the drive model at 40+ m/s (beyond the pitch speed) checked against a bench run
- [ ] K4.11 K4TQ-G "Ginna" (K4TQ-B "Bertha" parked 2026-10-21: too heavy) (`dev_kestrel5`): the CAD of the scaled roots and packing (the render's checks flag the frame/gear and the cartridge envelopes sized for Freddy), the transition corridor at the new scale, datasheets for the 12S motors (U8 II, U12 II) and the Mejzlik propellers, Bertha's thin R3 margin (≈6 Wh), the operation category (Bertha: specific category)
- [-] K4.10 K4TQ-D1 "Dorothy" (the powered in-flight fold for smoke zones): studied on branch `dev_kestrel4`, rejected 2026-10-21 — +0.66 kg (D2 ~+0.95 kg) for a 20–30° tuck; see `rejected.md`
- [ ] K4.4 Smoke: the three race tests (`test_autopilot_telemetry_and_side_panel`, `test_neutral_scenario_is_the_nominal_lap`, `test_mujoco_second_with_tuned_autopilot`) fail in the combined `./test.sh --smoke` run with MuJoCo 3.16 installed but pass alone — an order/state dependence in the race suite, not K4TQ
## Stage B — vegeta-bionic: glove teleoperation, the puppet arm, assistive exoskeletons (2026-11-03)
The plan is `docs/bionic.md`.
- [x] B0 the plan and the design decisions (`docs/bionic.md`)
- [x] B1 `libs/vegeta-bionic` skeleton: hand, sensors, SimGlove, estimate, link (in-process); the risk file (`libs/vegeta-bionic/RISK.md`); 12 tests
- [x] B2 retarget + vibration feedback + teleop: a scripted hand drives Onager Manus's two pincers in MuJoCo; demonstrations recorded and replayed; latency sweep: ≤ 50 ms round trip for the catch, 500 ms still fine for still grasps
- [ ] B3 the glove's CAD (printed mounts, three hand sizes); choose the electronics; link over serial/UDP
- [ ] B4 force feedback in simulation: a brake, then a motor, on the glove's linkages; haptic-loop stability with delay
- [ ] B5 the puppet arm: a scaled printed copy of the Manus arm with magnetic encoders; teach, record, replay
- [ ] B6 the whole arm from the glove (wrist IMU + tracker → arm IK)
- [ ] TODO exoskeleton: grip assist (fingers) — risk file first
- [ ] TODO exoskeleton: wrist and forearm support (wrist extension)
- [ ] TODO exoskeleton: elbow and shoulder support (against gravity)
## Stage V8 — ÉMERILLON, the speed quadcopter (2026-11-10)
- [x] V8.1 The market: `labs/aero/designs/emerillon_market.py` — motors, propellers, ESCs, 6S packs, electronics and filament with price, shop, spread and `kind` (web search of 2026-11-10); `bom`, `price` (low / used / high)
- [x] V8.2 The flight model: `emerillon_flight.py` — airframe from the parts, mass budget (`og.mass`), drag build-up with a margin, Boreas with a Mach-aware section and propeller maps, the drive under its current limits and the pack's sag, trim on body lift, top speed, the run, `evaluate`
- [x] V8.3 The trade: 336 builds, the front of speed against price, budget / knee / fastest, also inside the propeller makers' rpm limits; `labs/aero/sims/emerillon_trade.py` → `designs/data/emerillon_trade.json`
- [x] V8.4 CAD: `emerillon.py` (`Emerillon`: body, arms, pods, fins; `aircraft`, `arm`, `nose`), sized by `cad_parameters(config)`; defaults = the knee build
- [x] V8.5 `37_emerillon` (unexecuted; runs on the mocks), `labs/aero/tests/test_emerillon.py`, README
- [x] V8.5b ER1 (261 km/h) and ER2 (383 km/h) as named variants; the model at altitude (`air_at`: density, speed of sound, viscosity through the drive, the drag and the trim)
- [x] V8.5c Altitude: `emerillon_mission.ceiling` / `ceiling_options` — the pack, not the thrust, sets it (ER1 ~3.5 km, ER2 ~4.5 km as built); `ef.HIGH` the packs for 4 km and 7 km
- [x] V8.5d The flights: `emerillon_mission.py` (point mass with a body axis, the drive tabulated over altitude, speed and throttle) and five scenarios — wildfire, mountain, sea, climb, palace; `labs/aero/sims/emerillon_mission.py`
- [x] V8.5d2 Long range: `cruise_power`, `range_curve`, `range_options` (the best-range speed ~140 km/h; ER1 ~8 km, ER2 ~11 km as built, ~18 km on two packs: `ef.LONG`) and the `range` scenario (out against the wind to 90 % of the radius and back)
- [x] V8.5e Book `emerillon` (E) in the side labs: E1 `airframe-drag`, E2 `airframe-alpha` (CFD), E3 `arm` (FEA) — `emerillon_campaigns.py` behind the shared scripts; prepared, **not run** (CFD cases written with `--prepare-only`; the FEA mesh needs Gmsh, not on the machine that wrote this)
- [ ] V8.5f Run E1–E3 for ER1 and ER2 on the workstation; put E1's margin in place of `DRAG_MARGIN` and E2's normal force in place of `normal_force_area`; read E3's modes against the rotor's 1P (the beam rule puts the arm's first mode at 100–150 Hz, below 1P at top speed: the rotors run through it)
- [ ] V8.6 A bench run of the knee build's motor and propeller (static thrust, current, rpm) to calibrate the propeller map — nothing here is measured
- [ ] V8.7 CFD: the solid aircraft at the design's top speed (the notebook's cell, NOT RUN so far) for the drag margin; the fast build at Mach 0.5 on the compressible template; the propellers as rotor disks
- [ ] V8.8 The lab's own high-pitch 7-inch blade (`prop_lab`), its FEA spinning (`talos.Centrifugal`) and a proof spin; real blade tables in place of the generic planforms
- [ ] V8.9 The pitch-over, the stability of the bullet with its fins (neutral point against the CG) and the braking with attitude dynamics (Chiron / MuJoCo: the scenarios are a point mass so far); the propellers' own normal force in the trim; the descents (the flat fall, the tail-first catch: vortex-ring state is not modelled); a cold pack at altitude
- [ ] V8.10 Quotes for the rows marked `assumed` (the 3120 at 900 KV, the own propellers, the 7x7E), a pack larger than 2 Ah with a real C rating, PETG and PA6-CF slicer profiles
- [ ] V8.11 MERLIN and ÉMERILLON are the same bird in two languages: decide whether MERLIN is renamed (26, 26b, 27, `merlin*.py`, the sims and tests; PEREGRINE and FALCO import it)
## Stage V9 — The air match (branch `dev_air_match`, 2026-11-10)
- [x] V9.1 The arena in MuJoCo (`labs/aero/sims/air_match/arena.py`): the field, a hoop on a 2 m stick at each end, drifting balloons (free bodies, weight taken off), the Émerillons as rigid bodies from the mass budget
- [x] V9.2 The match (`match.py`): the races' geometric controller on a nose-up body, the rotor hook (lagged thrusts, the drive's limit, reaction torques, drag and normal force), the referee (goals through the hoop, fouls on aircraft and balloons, crashes, restarts), rule-based pilots
- [x] V9.3 The chase: a token only whose carrier scores, taken by a tag (within 2 m for 0.35 s, no contact); `--plain` plays without it
- [x] V9.4 `emerillon_air_match.py` (a match, `--series`), the summary sheet and the replay video (`replay.py`), `labs/aero/tests/test_air_match.py`
- [x] V9.4b Step 2, the formation chase: `air_match/swarm.py` (Nomios's wedge and column, the path-following column, the Hungarian assignment, the morph, the slot command and the separation reflex, re-written here) and `formation.py` (seven behind a hare round a racetrack: wedge on the straights, column through the turns and the hoops); `--mode formation`
- [x] V9.4c Steps 3 and 4, the group token chase (`match.chase`, `Rules.mode = "keepaway"`): every aircraft for itself, a point per second with the token, a bonus per hoop; `--mode chase3`, `--mode chase7`; the match generalised to any number of players and teams (colours, starts on a ring, the steering with priorities for the ground and the field's edge)
- [ ] V9.4d The chase of seven is rough (8-25 faults a minute: the carrier cornered into the ground or out of the field): a carrier that escapes along the wall and upward, chasers that keep Nomios's flock separation among themselves, a larger field
- [ ] V9.4e The formation's launch (two contacts in the first three seconds: seven take off in a line behind the hare) and its wedge on the far straight (the slot error rises to 2 m among the balloons)
- [x] V9.4f The circus race (`air_match/circus.py`, `--mode circus`): seven agents, seven strategies (a line's radius and height, an airspeed cap, how hard the turns and the brakes, the room left), three laps anticlockwise round a spina with two posts on the Circus Maximus's plan at 0.4 scale; balloons that drift with the wind and come back in upwind; gates, restarts from the last gate; `tournament` (the grid turned each race) and `solo`
- [x] V9.4g The circus, second round: the slipstream (`WAKE`, `Race.wake_at`), the adaptive strategies (`drafter`, `adaptive`, `tempered`; `Race.adapt`), the racing tilt limit (72°), the changing wind (`CHANGING`, `Race.blow`, `--gusts`); four tournaments rerun (README)
- [ ] V9.4h The circus: a wake computed rather than assumed (and one that disturbs the follower), a drafter that keeps its own pace until the last lap, strategies that defend, more seeds per tournament (confidence in the order), the gusts' moments on the aircraft
- [ ] V9.5 A MuJoCo camera video (`replay.mujoco_video`, written, not run: no OpenGL on the machine that wrote it)
- [ ] V9.6 2 v 2 and 3 v 3 (the arena's start slots and the token rules are per team already; the pilots need passing and marking), three hoops a side
- [ ] V9.7 Pilots that learn or search (the rule-based ones never feint); a keeper; a foul-drawing rule that cannot be gamed
- [ ] V9.8 Aerodynamic moments and propeller wash on the balloons; the pack and the windings over a match (the energy is counted, nothing is limited)

## Stage V9 — The Pekari family: Pekari-2M, Pumba and the fleet (2026-11-10, branch `dev_pumba`)
Design note: [`docs/pekari_pumba.md`](docs/pekari_pumba.md). The Pekari Rover itself is unchanged; both machines build on `pekari_rover_robot` as Perseus does.
- [x] V9.1 Architecture assessment: what exists (Pekari Rover, the variant pattern, Manus/Perseus arms, Ogma, Chiron, Dedalus, Yax export) and what does not (no fleet, shared-map, comms or supervision layer; no WOTAN/MIMIR/Wagenburg; Chiron is one robot per world)
- [x] V9.2 WILK (Stohid Technology) as a reference: manufacturer-reported 180 kg / 200 kg / 8 km/h / 1700 x 1170 x 700 mm in `pumba_robot.WILK_REFERENCE`; the principles taken, not its geometry. The trade press disagrees on whether WILK is tracked or wheeled (open: a datasheet or photographs)
- [x] V9.3 `pekari_arms.py`: one arm description (5-7 DOF, Manus conventions): IK/FK, statics, rated load against reach (peak / SF 1.5 and continuous), workspace, power, the Chiron link tree; three arms (Pekari-2M light, Pumba main 6-DOF, Pumba aux)
- [x] V9.4 `pekari_family.py`: `Platform` (chassis + arm mounts + stabilizers + parts), mass and CG in 3-D, support polygon and tip-over, lift capacity (min(arm, 0.75 x tipping)), drive check, power and endurance, capabilities (the fleet contract), BOM and cost, the Chiron robot, a MuJoCo stand test
- [x] V9.5 Pekari-2M (`pekari_2m_robot.py`, CAD `pekari_2m.Pekari2M`): Pekari one size up, two 5-DOF arms (5 kg at 0.5 m), 55 kg empty, 30 kg (20-40) on the deck, 0.58 m wide, 0.42-0.45 m high folded
- [x] V9.6 Pumba (`pumba_robot.py`, CAD `pumba.Pumba`): WILK-class envelope, 6-DOF main arm (60 kg at 1.2 m, 1.92 m reach) + 5-DOF aux arm, four stabilizers, 444 kg empty, 150 kg (100-200) on the deck, 8 km/h, 4.8 kWh
- [x] V9.7 Ogma actuator catalogue: `harmonic 12 Nm, brake`, `harmonic 30 Nm, brake`, `cycloidal 1600 Nm, brake`, `pumba track drive 4 kW, 48 V` (class values, assumed)
- [x] V9.8 `fleet.py`: deterministic heterogeneous fleet — capability auction, width/headroom-aware A*, shared map (time-stamped merge), frontier exploration, debris tasks, reassignment on failure / blocked path, comms-loss recovery (return to last link, offline completion reported, duplicates cancelled), supervisor (e-stop on a separate safety radio, hold, approvals, pre-approval), drone and third-party profiles
- [x] V9.9 `fleet_scenario.py` + `sims/fleet_response.py`: two Pekari-2M, Pumba and a drone at a collapsed industrial hall — every operator task done in 392 s with a drive fault, comms loss and an e-stop; deterministic
- [x] V9.10 `sims/pekari_family_checks.py`: numbers, lift / workspace / to-scale figures (`docs/figures/`), MuJoCo stand tests; registered `pekari_2m` and `pumba` for export (YX2 written for both)
- [x] V9.11 Tests: `labs/land/tests/test_pekari_family.py`, `test_fleet.py` (51 tests; the whole land lab passes on mocks with MuJoCo and CadQuery installed)
- [x] V9.18 Pumba redesigned light, the first design kept as **Calydon** (`calydon_robot.py`, `calydon.Calydon`, `CALYDON_MAIN_ARM` / `CALYDON_AUX_ARM`, `calydon track drive 4 kW, 48 V`: numbers unchanged, 444 kg). Pumba: chassis 167 kg (WILK's reported ~180 kg), 236 kg with its 6-DOF main arm (`PUMBA_MAIN_ARM`, 30 kg at 1.0 m, 1.58 m, 46.5 kg), Pekari-2M's arm as the second arm and two front stabilizers; a 180 mm rubber band track (`Platform.band_track_kg_m`, 5 kg/m assumed) in place of PA6 links and pins, 6 road wheels a side at 25 % fill, 2 mm shell, welded Al frame, `pumba track drive 2.5 kW, 48 V`, 2.9 kWh; still 150 kg (100-200) on the deck, 8 km/h, 20 deg loaded; `prr.pekari(budget=...)` (additive keyword) so the Chiron model carries the band track; the fleet's debris pieces are 30 kg; 62 tests
- [ ] V9.19 A rubber-track supplier sheet for Pumba's 180 mm band (mass per metre, pitch, sprocket engagement, minimum wheel diameter) in place of `BAND_TRACK_KG_M`; FEA of the welded Al frame under the main arm's ~470 N m
- [x] V9.20 Pumba in the forest (`designs/pumba_forest.py`, `sims/pumba_forest_movie.py` → `sims/output/pumba_forest.mp4`): MuJoCo trail run between trees and shrubs over a rough floor and two branches, the main arm picks the fallen log off the trail and Pumba drives on; pure-pursuit track steering, IK moves retaken from the hull's pose, an alignment step on the measured joints, the hold modelled as a weld once both jaws press (> 20 N); fixes on the way: the family's jaw hinge axes (+ now closes, as documented: it opened before), Pumba's grab opens 55 deg and squeezes at ~35 N m; `tests/test_pumba_forest.py` (the mission is `slow`)
- [ ] V9.21 Holding by contact alone (no weld): stiffer jaw/prop contacts or a serrated jaw model so the log stays between the jaws in MuJoCo; a real belt-track contact model (the roller row wedges on thin round sticks)
- [x] V9.22 Interchangeable tools: `pekari_arms.Tool` / `TOOLS` / `with_tool` (coupler-checked: `qc` heavy arms, `iso` light arms), `cut_check` (Perseus's stall-SF measure), `pekari_family.with_tool`, `pumba_robot.PUMBA_SAW` / `TOOL_KIT`, `pumba(tool=...)`, the CAD's `tool` parameter and saw drawing, the `cut` pose; Pumba with Perseus's 2 kW saw: wrist SF 1.28 cutting outstretched at loads x1.5 (Perseus 1.56); the pruning saw is beyond the light arm
- [x] V9.23 `designs/land_comparison.py` + `sims/land_comparison.py` → `docs/land_comparison.md` and `docs/figures/land_comparison.png`: Pekari Rover, Pekari-2M, Pumba (grab / saw), Perseus, Calydon on one set of rules (CAD envelope, drive and endurance on gravel, lift, saw margin, tip angle, cost)
- [ ] V9.24 Pumba's 400 N m wrist option for sawing at Perseus's margin; a forest cutting scenario with the saw on Pumba (as Perseus's `SawCutter`)
- [x] V9.25 Perseus options (`designs/perseus_options.py`, `sims/perseus_options.py` → `docs/perseus_options.md`): Perseus's chassis kept; seven configurations judged on the cut zone, saw margin, grab lift (min of drives and 75 % tipping), second-hand reach / hold / tip margin, drive and endurance. Findings: change the head (quick coupler at the boom's wrist: saw, grab 22 kg to 1.8 m), not the boom; the 6-DOF joint arm is lighter but reaches less (58 % vs 70 %); no second hand on this chassis (light: reaches 0 %; Pumba-size: 12-16 % of cuts, 7-8 cm tip margin; long: tips)
- [ ] V9.26 Perseus: a quick coupler at the boom's wrist with saw and grab heads (the boom's own Variant, as `pekari_arms.TOOLS` for the joint arms); the 400 N m wrist as an option
- [x] V9.27 Timon, the servicing rover (`designs/timon_robot.py`, CAD `timon.Timon`, `docs/timon.md`): sized from the service job by `check` / `sweep` (25 hulls, 9 pass, the lightest 1200 x 620 mm: the deck decides, stability does not bind); Pumba's 6-DOF arm + Pekari-2M's arm as the second hand, two-module cradle and tool rack, tow hitch, 4.5 kN winch, rear earth anchor (braked tracks alone would need ~440 kg); 236 kg, 1.52 x 1.02 x 0.74 m; registered for export; in the comparison
- [x] V9.28 Fleet service tasks (`fleet.py`): `swap_battery` (the low robot waits, the task follows it) and `recover` (tow home at half speed) raised when a `service` / `tow` robot is in the fleet; unreachable service tasks cancelled; `fleet_scenario.build(with_timon=True)`: Timon swaps P1's module and tows P2 home; `tests/test_timon.py`
- [ ] V9.29 Timon: the spade's hold and the docking accuracy on real ground; a MuJoCo module swap (a module as a free body in a cradle); a tool-change task in the fleet; charging spare modules on the deck; confined-space recovery (a Pekari-2M with a tow line)
- [ ] V9.12 Arm controller with gravity feed-forward and a MuJoCo pick-and-place with real contacts (a box from the deck; Pumba's grab on a beam); wrist-camera models
- [ ] V9.13 Calydon structure: main-arm tube deflection and stresses, the steel subframe (Talos FEA); track links and pins at 600 kg — decide steel-rubber or band track (Pumba took the band track: V9.18)
- [ ] V9.14 MuJoCo tip test on a side slope with the arm at reach, with and without stabilizers, against `pf.tipping_load`; brakes holding on power loss
- [ ] V9.15 Fleet: robots as obstacles (deconfliction), multi-robot tasks (two robots on one load), battery-aware scheduling with charging, scenario sweeps (fault times, comm ranges); message schemas with a ROS 2 mapping for Piccolo; a learned policy only as a challenger to the deterministic baseline
- [ ] V9.16 A notebook for Pekari-2M and Pumba (as 30 / 37); quotes for the strain-wave and cycloidal modules, the track drive and the battery (the BOM is class prices: EUR 20-47 k / 68-143 k per unit, parts only); a bench test of one arm joint at its rated torque
- [ ] V9.17 Pre-existing failures seen while checking (not from this work, present on the base commit): `libs/ogma/tests/test_parts.py::test_module_reproduces_vegeta[actuators]` (the golden predates the `gear_ratio` / `gear_efficiency` fields) and four export tests in `tools/tests/test_export.py` (`test_selective_export_withholds_and_says_so`, `test_design_and_user_tags`, `test_consumer_requirements_report_and_never_override_the_flags`, `test_yx2_is_the_main_file_and_keeps_every_selected_property`)

## Stage V5 — Later: graduations to Piccolo
- [-] V5.1 Drongo
- [-] V5.2 Nisus / Nisus+ / Falco (+ `vegeta.mission` → Piccolo's `onboard`)
- [-] V5.3 The Onager series, Pekari, Perseus
- [-] V5.4 The robot dog and the Myropods (Chiron and MuJoCo leave Vegeta with them)

## Stage V8 — Strato: a solar-electric stratospheric airship (HAPS), branch `dev_strato` (2026-10-26)
Approved 2026-10-26: physics in Ogma, SciPy optimiser, the 15 m demonstrator reported as it is, built-in wind
climatology; GaAs array baseline, CIGS alternative. Assumptions and the verified/estimate table: `labs/aero/designs/STRATO.md`.
Question: can a solar-electric, helium-filled airship hold station at 18–22 km over Europe for weeks or months with a
communications or observation payload, which engineering factor limits it, and which configurations are feasible?
Vegeta does the engineering only (geometry, physics, energy, sizing, optimisation); station-keeping *control* and
autonomy belong to Piccolo — here station-keeping is a power/airspeed budget against a wind profile, not a controller.

Inspection (what is reused, what is missing):
- Reused: `og.mass.Budget` (mass, CG, inertia), `og.batteries` (cells/packs), `og.propulsion` / `vegeta.boreas` (BEMT
  props at low Re), `og.aero.cf_turbulent` (skin friction), `og.wind.Wind` (turbulence and gusts on top of a profile),
  `og.units`, Dedalus `Design`/`Parameter` (CadQuery → STEP/STL), `dedalus.viz.plot3d` (pyvista), `dedalus.export_yax`
  (Yax `.yx2` for Piccolo later), Vegeta Core revisions for recorded studies; Talos (CalculiX) for the solid fin /
  gondola-frame FEA and Aeromant (OpenFOAM) for one hull-drag CFD check — both only where the solvers are installed
  (neither is in this container: those steps are written and tested on mocks, results NOT RUN until run on a workstation).
- Missing: `og.atmosphere` is the troposphere only (linear lapse; wrong above 11 km); no gravity vs altitude, no
  lifting-gas, solar, envelope or stratospheric-wind model; no generic optimiser (the races' PSO/GPS/CMA strategies in
  `labs/aero/sims/races/` are bound to autopilot settings and lap scores, so the study gets a small generic wrapper).
- Fidia (AI modelling) is not needed; Mellonia/Atropos do not apply at airship scale.

Layout (stratosphere physics in its own files; verified physics separated from estimates):
- Ogma, verified physics with tests pinned to published reference values:
  `ogma/stratosphere.py` (US Standard Atmosphere 1976, 0–86 km, geopotential altitude, T/p/ρ/μ/a, ΔT offsets for hot
  and cold days, g(h) = g0·(Rₑ/(Rₑ+h))²), `ogma/lifting_gas.py` (He/H₂ ideal gas, purity, gross/net lift, superheat,
  sealed superpressure vs vented zero-pressure, pressure height, ballonet fraction), `ogma/solar.py` (solar position,
  day length, extraterrestrial irradiance with orbit eccentricity, clear-sky beam/diffuse transmittance above the
  tropopause, incidence on a curved hull patch), `ogma/envelope.py` (thin-shell membrane stresses of a pressurised body
  of revolution — hoop/meridional, bending from lift/weight distribution — laminate catalogue with sources).
  `og.atmosphere` stays as it is (its goldens unchanged); `og.stratosphere` agrees with it below 11 km (test).
- Aero lab, design models (estimates marked as such in each docstring and in `STRATO.md`):
  `designs/strato.py` (Dedalus CAD: hull, fins, gondola/payload bay, solar array, propellers, battery bay — `part=`
  selects), `strato_geometry.py` (hull profiles: ellipsoid, NPL/Gertler-58-like 4-parameter profile, Zeppelin-like
  ogive; volume, area, CB, solar patch area — analytic, checked against the CAD), `strato_mass.py` (`og.mass.Budget`
  from areal densities and specific powers), `strato_aero.py` (hull drag vs fineness and Re — Hoerner/Young volumetric
  Cd; fins, gondola, interference; Munk moment and fin stability, added mass), `strato_thermal.py` (gas superheat: solar
  absorption, IR exchange with Earth and sky, convection; day/night gas temperature), `strato_energy.py` (PV with
  temperature derating, MPPT, battery charge/discharge, hotel + payload loads, propulsive power from Boreas props),
  `strato_wind.py` (stratospheric wind climatology by month and latitude for 40–60°N — the summer ~20 km wind minimum;
  values from published ERA5-based studies, cited, marked estimate), `strato_flight.py` (ascent ODE, float
  equilibrium, station-keeping), `strato_feasibility.py` (constraints → structured verdict with every rejection reason),
  `strato_study.py` (sweeps, optimisation, Pareto); `sims/strato_mission.py` (multi-day endurance runner);
  `campaigns/strato/` (scripted study runs writing JSON/CSV/HTML/PNG); `notebooks/37_strato.ipynb`; `designs/STRATO.md`
  (assumptions, sources, verified vs estimate table).

Tasks:
- [x] V8.0 `dev_strato` branch; this plan; the answers folded in
- [x] V8.1 `og.stratosphere`: USSA-1976 layers to 86 km, geopotential ↔ geometric altitude, gravity vs altitude;
  tests vs the USSA-1976 tables (e.g. 20 km: 216.65 K, 5474.9 Pa, 0.08803 kg/m³) and vs `og.atmosphere` below 11 km
- [x] V8.2 `og.lifting_gas`: buoyancy, sealed vs vented, superheat, ballonet / pressure height; tests by hand
  calculation and conservation (mass of gas constant when sealed; vented loses gas above pressure height)
- [x] V8.3 `og.solar`: solar position (NOAA algorithm), irradiance on tilted/curved surfaces; tests vs NOAA calculator
  values for given dates and latitudes, day length at solstices/equinoxes
- [x] V8.4 `og.envelope`: membrane stresses, required thickness for a safety factor, laminate catalogue; tests vs
  closed-form cylinder/sphere results
- [x] V8.5 (`designs/strato.py`, `strato_geometry.py`, `campaigns/strato/cad.py`, `viewer.py` — three.js page, `tests/test_strato_cad.py`) Parametric geometry + CAD: 15 m × 3.5 m demonstrator (fineness 4.3), scaling to 30–80 m (scale by length
  with fineness free); STEP/STL/glTF; interactive 3D (pyvista in the notebook, a self-contained HTML viewer for the
  report); analytic volume/area within 0.5 % of the CAD measurement (test)
- [x] V8.6 (`strato_design.py`) Mass budget and CG/CB, trim ballast; stability (CB above CG, metacentric height, fin volume vs Munk moment)
- [x] V8.7 (`strato_aero.py`; the Aeromant case NOT RUN: no OpenFOAM here) Aerodynamics: hull + appendage drag vs Re at altitude; validation vs published wind-tunnel drag of airship
  hulls (Hoerner, Akron/ZRS-4 model) within the stated band; one Aeromant RANS case template (NOT RUN here)
- [x] V8.8 (`strato_thermal.py`) Thermal: gas superheat day/night and its effect on lift and superpressure
- [x] V8.9 (`strato_sim.day_cycle`, `availability`) Energy: PV + battery + loads over 24 h by date and latitude; energy-closure verdict; seasonal maps 40–60°N
- [x] V8.10 (`strato_wind.py`, `strato_sim`) Wind and station-keeping: monthly profile 15–25 km, required airspeed and propulsive power, % time on
  station, best altitude per month
- [x] V8.11 (`strato_sim`: `day_cycle`, `endurance`, `ascent`; campaign stages detail/sweeps/constraints) Simulations: ascent 0 → 20 km (sealed and vented), float equilibrium, day/night cycles, wind disturbance
  (profile + `og.wind` turbulence/gusts), payload and altitude sensitivity, structural and thermal limits, multi-day
  endurance (power, SOC, altitude, superpressure, position drift)
- [x] V8.12 (`strato_feasibility.py`) Feasibility gate: reject designs that do not float (lift < mass), overstress the envelope, cannot close the
  24 h energy balance, cannot exceed the wind, are statically unstable, or need negative masses/areas — every reason
  reported, nothing silently clipped
- [x] V8.13 (`strato_study.py`: SciPy differential evolution + payload polish, Sobol sampling, non-dominated filter) Optimisation: design vector (length, fineness, solar coverage, battery energy, propulsion power, float
  altitude, payload) → objectives (endurance/station-keeping %, payload ↑; mass, drag, energy ↓) under the feasibility
  constraints; single-objective by the chosen optimiser, Pareto fronts by non-dominated sorting of a sampled/evolved
  population; seeds fixed, runs reproducible
- [x] V8.14 (`campaigns/strato/run.py`, `report.py` → `results/report.md`; notebook `37_strato.ipynb`) Outputs: 15/30/50/80 m comparison table, mass and energy budgets, sweeps, Pareto fronts, feasibility report
  (Markdown + HTML), the notebook; every number written by a run, none typed in by hand
- [x] V8.15 (`libs/ogma/tests/test_{stratosphere,lifting_gas,solar,envelope}.py`; `labs/aero/tests/test_strato_{cad,models,golden,campaign}.py`) Tests: Ogma unit tests in `libs/ogma/tests`, lab tests in `labs/aero/tests/test_strato*.py` (mocks, fast,
  added to smoke), regression goldens for the four reference designs; `./test.sh --smoke` green; existing goldens
  unchanged
- [x] V8.16 `STRATO.md`: assumptions, sources, the verified/estimate split, the conclusion and the limiting factors
- [ ] V8.18 **Strato-Sigma** (2026-10-26): the smallest Strato for imaging only — a daylight EO camera + a downlink to a
  ground station (no relay to aircraft; ~4.5 kg, 50 W by day, 0 W at night), summer (Jun–Aug, 50°N), on station ≥ 90 %
  of days, ≥ 60 days of gas: `Design.payload_power_night`, `strato_study.smallest_length`, campaign stage `sigma`
  (`results/sigma.json`, `sigma.png`, `cad/strato_sigma_<L>m.step` and parts, report section)
- [x] V8.19 **Zeppelin-Micro** (2026-10-26): the cheapest maritime imaging airship at a few hundred metres — daylight
  camera + video link, hours per deployment; tethered aerostat vs free-flying micro blimp, sized against a design sea
  wind, cost vs operability; with and without a thermal camera (`designs/zeppelin_micro.py`, `campaigns/strato/zeppelin_micro_study.py`, `tests/test_zeppelin_micro.py`;
  Dedalus `with_*` switches and a `tether` part for the aerostat CAD)
- [x] V8.20 Campaign complete; conclusions in `labs/aero/designs/STRATO.md` (feasible in the stratosphere from ~35–50 m in
  summer and ~80 m year-round at 50°N; the 15 m demonstrator cannot reach it; limits: envelope mass, the payload's night
  power, battery, helium leakage, winter winds). Fixed on the way: the endurance sim's ballast drop now keeps a 1 %
  margin (it declared a sealed ship lost on a knife-edge equilibrium); sensitivity is deterministic (the re-optimised
  one is dominated by optimiser noise, ±50 kg)
- [ ] V8.21 Next: ERA5 winds for a real site; Aeromant CFD of the hull and Talos FEM of the fins on a workstation; a
  laminate datasheet; a longer optimiser budget (or a gradient polish) for the payload numbers
- [-] V8.17 Later: graduation to Piccolo (export through Yax; control and autonomy there), FEM of the envelope as a
  membrane (Talos is solid-element only), unsteady CFD
