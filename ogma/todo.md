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
- [x] the battery-for-CG solves → `place_mass_for_cg` (5 design tables; robots later)
- [x] Falco `fuselage:` rows get their shape inertia (Iyy +4.4 %, Izz +1.8 %)

## O2 — generic modules moved with history
- [x] `og.gears`, `og.soil`, `og.tracks`, `og.cutting`, `og.actuators`
- [x] Vegeta designs import `og.*`
- [x] `og.motors` (DC motor, servo law from Chiron, transmissions)
- [x] Vegeta adopts `og.motors`
- [x] Chiron takes its servo law from Ogma (Chiron vendors Ogma; chiron 2368d21, its test_ogma checks every shape)

## O3 — consolidate duplicates
- [x] `og.atmosphere`
- [x] `og.wind`
- [x] `og.batteries`
- [x] `og.rotors` (wrench about the COM; races fixed in Piccolo)
- [x] `og.propulsion`
- [x] Vegeta adopts `og.propulsion` (Nisus, Nisus+, Merlin, Kestrel)
- [x] `og.aero` (Cf, Airframe, Polar, stability coefficients, aero wrench, vlm)
- [x] Vegeta adopts `og.aero` (vegeta 8ae1f7a)
- [x] Velutina wind → `og.wind`
- [x] unify `G` / `RHO` on the design models (Chiron robots keep 9.81 until they graduate)

## O4 — graduate machines to Piccolo
- [x] racing quad (Piccolo v2) — the only graduation for now

## O5 — MuJoCo strategy
- [x] `og.mujoco`: Ogma parts → MJCF fragments (lazy `mujoco` extra); `og.mass.Capsule` added
- [x] Ogma shape inertia == MuJoCo `inertiafromgeom` (Ogma's test_mujoco and Chiron's test_ogma)

## O6 — CAD strategy
- [x] `og.cad` (CadQuery, lazy extra)
- [x] Vegeta's `export_yax` takes its mass properties from `og.cad`

## Vegeta application
- [x] Vegeta vendors Ogma and builds its designs on it (`vendor/ogma`; Vegeta todo V3–V4 done 2026-10-21)

## Later
- [ ] Drongo
- [ ] Nisus / Nisus+ / Falco (+ `vegeta.mission` → `onboard`)
- [ ] Onager series, Pekari, Perseus
- [ ] robot dog, Myropods (Chiron leaves Vegeta)
- [ ] O7 — Gazebo strategy: `og.gazebo` (SDF, lazy extra)
