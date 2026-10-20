# Ogma (`import ogma as og`)

Standard machine parts as **data + behaviour**: dataclasses with their sources, pure functions, JSON catalogues.
numpy only, SI units, organised by physics.

| Module | Parts | Status |
|---|---|---|
| `og.units` | constants (g, ISA sea level, air), unit conversions | done |
| `og.frames` | quaternions, rotations, Euler angles, frame conventions | done |
| `og.mass` | mass items and shapes (rod, plate, shell; box, sphere, cylinder, ellipsoid as MuJoCo), budgets, CG and inertia, placing an item for a target CG | done (O1) |
| `og.gears`, `og.soil`, `og.tracks`, `og.cutting`, `og.motors` | moved from Vegeta's designs/ and Chiron | O2 |
| `og.batteries`, `og.atmosphere`, `og.wind`, `og.propulsion`, `og.rotors`, `og.aero` | consolidated from Vegeta's machines | O3 |

Tests: each moved piece reproduces the numbers Vegeta's code gave on fixed inputs (`tests/golden/`, recorded from
Vegeta at the commit named in the file); published references on top where they exist.
