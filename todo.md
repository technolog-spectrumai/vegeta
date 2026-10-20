# Vegeta — TODO

Vegeta is the R&D notebook set: fast and dirty, and it stays that way (the blessed imperfection rule, README).
Rules: the tools stay independent packages; the stable physics of standard parts comes from Ogma (`vendor/ogma`,
`import ogma as og`); Piccolo never imports Vegeta, and a machine graduates there once its design is frozen.
Check quickly with `./test.sh --smoke` and the mocked tests (`VEGETA_DEBUG=1`, `-p vegeta.mock.pytest_plugin`);
the full suites are in `user_tests.sh`. Earlier stages (1–13, 5b–5t, R, S) and their open items: [`old_todo.md`](old_todo.md).

Legend: `[ ]` open · `[x]` done · `[-]` deferred

## Stage V1 — The split with Piccolo (2026-10-20)
- [x] V1.1 Smoke suite `./test.sh --smoke` (under 30 s) and the pre-move baseline (`docs/migration/baseline.md`, robotics goldens recorded before the move)
- [x] V1.2 Yax design interchange (STL + `yax.json`) and `vegeta.dedalus.export_yax` (exact OCC mass properties, checked against the mesh integration)
- [x] V1.3 Chiron to its own repo (`technolog-spectrumai/chiron`, history kept, `chiron.mock` split out of `vegeta.mock`); `vegeta.chiron` stays here until the last machine hatches
- [x] V1.4 The racing quad hatched → Piccolo v2: `races/`, the autopilot scripts, the race notebooks, `side_labs/races`, the races book (history kept; path map in `MOVED.md`); `designs/drongo_controller.py` stays (Drongo has not hatched)
- [x] V1.5 Boundary test: nothing in Vegeta imports `races` or Piccolo
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
- [ ] V3.8 The `quat_to_R` / `euler_from_R` copies (`nisus_robot`, `drongo_controller`) → `og.frames`
- [ ] V3.9 The 7 battery-for-CG solves → `og.mass.place_for_cg`
- [ ] V3.10 Velutina's wind → `og.wind`
- [ ] V3.11 `G` / `RHO` unified on `og.units` (9.81 → 9.80665, 1.2 → 1.225): each golden that moves in its own commit, the reason written down
- [ ] V3.12 Falco's `fuselage:` rows get their shape inertia (a reported number change)
- [-] V3.13 Boreas's `Motor` / `Battery` vs `og.motors` / `og.batteries`: kept while the tools stay independent packages

## Stage V4 — Vegeta application (after Ogma O6)
- [ ] V4.1 Designs take their mass budgets, parts, propulsion and aero from `og.*` throughout
- [ ] V4.2 CAD through Ogma's CAD strategy (`og.cad`); `export_yax`'s mass properties move to Ogma

## Stage V5 — Later: graduations to Piccolo
- [-] V5.1 Drongo
- [-] V5.2 Nisus / Nisus+ / Falco (+ `vegeta.mission` → Piccolo's `onboard`)
- [-] V5.3 The Onager series, Pekari, Perseus
- [-] V5.4 The robot dog and the Myropods (Chiron and MuJoCo leave Vegeta with them)
