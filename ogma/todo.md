# Ogma — todo

Progress of the Ogma refactoring (plan: O0–O7). Notes, decisions and known issues: [works.md](works.md).

## O0 — repo + Yax
- [x] Yax moved from Ozaru with history
- [x] `scripts/test.sh [--smoke]`, `user_tests.sh`, no-host-imports test
- [x] Ozaru drops `yax/`
- [x] vendored in Vegeta and Piccolo (`vendor/ogma`, `update_ogma.sh`)

## O1 — foundations
- [x] `og.units`, `og.frames`
- [x] `og.mass` (shapes, Budget, `place_for_cg`)
- [x] Vegeta `kestrel_systems.cg_inertia`, `nisus_systems.cg_inertia` over `og.mass`
- [ ] the 7 battery-for-CG solves → `place_for_cg`
- [ ] Falco `fuselage:` rows get their shape inertia (reported number change)

## O2 — generic modules moved with history
- [x] `og.gears`, `og.soil`, `og.tracks`, `og.cutting`, `og.actuators`
- [x] Vegeta designs import `og.*`
- [ ] `og.motors` (DC motor, servo law from Chiron, transmissions)

## O3 — consolidate duplicates
- [x] `og.atmosphere`
- [x] `og.wind`
- [x] `og.batteries`
- [x] `og.rotors` (wrench about the COM; races fixed in Piccolo)
- [x] `og.propulsion`
- [x] Vegeta adopts `og.propulsion` (Nisus, Nisus+, Merlin, Kestrel)
- [x] `og.aero` (Cf, Airframe, Polar, stability coefficients, aero wrench, vlm)
- [ ] Vegeta adopts `og.aero`
- [ ] Velutina wind → `og.wind`
- [ ] unify `G` / `RHO` (each golden change its own commit)

## O4 — graduate machines to Piccolo
- [x] racing quad (Piccolo v2) — the only graduation for now

## O5 — MuJoCo strategy
- [ ] `og.mujoco`: Ogma parts → MJCF fragments (lazy `mujoco` extra)
- [ ] Chiron test: Ogma shape inertia == MuJoCo `inertiafromgeom`

## O6 — CAD strategy
- [ ] `og.cad` (CadQuery, lazy extra)

## Vegeta application
- [ ] Vegeta vendors Ogma and builds its designs on it (`vendor/ogma`)

## Later
- [ ] Drongo
- [ ] Nisus / Nisus+ / Falco (+ `vegeta.mission` → `onboard`)
- [ ] Onager series, Pekari, Perseus
- [ ] robot dog, Myropods (Chiron leaves Vegeta)
- [ ] O7 — Gazebo strategy: `og.gazebo` (SDF, lazy extra)
