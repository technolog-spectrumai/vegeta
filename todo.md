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

## Stage V5 — Later: graduations to Piccolo
- [-] V5.1 Drongo
- [-] V5.2 Nisus / Nisus+ / Falco (+ `vegeta.mission` → Piccolo's `onboard`)
- [-] V5.3 The Onager series, Pekari, Perseus
- [-] V5.4 The robot dog and the Myropods (Chiron and MuJoCo leave Vegeta with them)

## Stage V8 — Strato: a solar-electric stratospheric airship (HAPS), branch `dev_strato` (2026-10-26) — PLAN, awaiting approval
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
- [ ] V8.0 `dev_strato` branch; this plan; answers to the open questions folded in
- [ ] V8.1 `og.stratosphere`: USSA-1976 layers to 86 km, geopotential ↔ geometric altitude, gravity vs altitude;
  tests vs the USSA-1976 tables (e.g. 20 km: 216.65 K, 5474.9 Pa, 0.08803 kg/m³) and vs `og.atmosphere` below 11 km
- [ ] V8.2 `og.lifting_gas`: buoyancy, sealed vs vented, superheat, ballonet / pressure height; tests by hand
  calculation and conservation (mass of gas constant when sealed; vented loses gas above pressure height)
- [ ] V8.3 `og.solar`: solar position (NOAA algorithm), irradiance on tilted/curved surfaces; tests vs NOAA calculator
  values for given dates and latitudes, day length at solstices/equinoxes
- [ ] V8.4 `og.envelope`: membrane stresses, required thickness for a safety factor, laminate catalogue; tests vs
  closed-form cylinder/sphere results
- [ ] V8.5 Parametric geometry + CAD: 15 m × 3.5 m demonstrator (fineness 4.3), scaling to 30–80 m (scale by length
  with fineness free); STEP/STL/glTF; interactive 3D (pyvista in the notebook, a self-contained HTML viewer for the
  report); analytic volume/area within 0.5 % of the CAD measurement (test)
- [ ] V8.6 Mass budget and CG/CB, trim ballast; stability (CB above CG, metacentric height, fin volume vs Munk moment)
- [ ] V8.7 Aerodynamics: hull + appendage drag vs Re at altitude; validation vs published wind-tunnel drag of airship
  hulls (Hoerner, Akron/ZRS-4 model) within the stated band; one Aeromant RANS case template (NOT RUN here)
- [ ] V8.8 Thermal: gas superheat day/night and its effect on lift and superpressure
- [ ] V8.9 Energy: PV + battery + loads over 24 h by date and latitude; energy-closure verdict; seasonal maps 40–60°N
- [ ] V8.10 Wind and station-keeping: monthly profile 15–25 km, required airspeed and propulsive power, % time on
  station, best altitude per month
- [ ] V8.11 Simulations: ascent 0 → 20 km (sealed and vented), float equilibrium, day/night cycles, wind disturbance
  (profile + `og.wind` turbulence/gusts), payload and altitude sensitivity, structural and thermal limits, multi-day
  endurance (power, SOC, altitude, superpressure, position drift)
- [ ] V8.12 Feasibility gate: reject designs that do not float (lift < mass), overstress the envelope, cannot close the
  24 h energy balance, cannot exceed the wind, are statically unstable, or need negative masses/areas — every reason
  reported, nothing silently clipped
- [ ] V8.13 Optimisation: design vector (length, fineness, solar coverage, battery energy, propulsion power, float
  altitude, payload) → objectives (endurance/station-keeping %, payload ↑; mass, drag, energy ↓) under the feasibility
  constraints; single-objective by the chosen optimiser, Pareto fronts by non-dominated sorting of a sampled/evolved
  population; seeds fixed, runs reproducible
- [ ] V8.14 Outputs: 15/30/50/80 m comparison table, mass and energy budgets, sweeps, Pareto fronts, feasibility report
  (Markdown + HTML), the notebook; every number written by a run, none typed in by hand
- [ ] V8.15 Tests: Ogma unit tests in `libs/ogma/tests`, lab tests in `labs/aero/tests/test_strato*.py` (mocks, fast,
  added to smoke), regression goldens for the four reference designs; `./test.sh --smoke` green; existing goldens
  unchanged
- [ ] V8.16 `STRATO.md`: assumptions, sources, the verified/estimate split, the conclusion and the limiting factors
- [-] V8.17 Later: graduation to Piccolo (export through Yax; control and autonomy there), FEM of the envelope as a
  membrane (Talos is solid-element only), unsteady CFD
