# Ogma (`import ogma as og`)

Standard machine parts as **data + behaviour**: dataclasses with their sources, pure functions, JSON catalogues.
numpy only, SI units, organised by physics.

| Module | Parts | Status |
|---|---|---|
| `og.units` | constants (g, ISA sea level, air), unit conversions | done |
| `og.frames` | quaternions, rotations, Euler angles, frame conventions | done |
| `og.mass` | mass items and shapes (rod, plate, shell; box, sphere, cylinder, capsule, ellipsoid as MuJoCo), budgets, CG and inertia, placing an item for a target CG | done (O1) |
| `og.gears`, `og.soil`, `og.tracks`, `og.cutting`, `og.actuators` | moved from Vegeta's designs/ with history; reproduce it exactly | done (O2) |
| `og.motors` | DC motor (Boreas's), resistance fit and operating point, the servo law (Chiron's), levers and gearboxes | done (O3) |
| `og.atmosphere` | the ISA (+ΔT), Sutherland viscosity, density altitude; reproduces Vegeta's three copies | done (O3) |
| `og.wind` | steady + OU turbulence + 1−cos gust (Nisus's model; terrain layers subclass it) | done (O3) |
| `og.batteries` | Li-ion cells and packs (sag, cold, charge limits), bought packs; catalogues | done (O3) |
| `og.rotors` | X layout, allocation, desaturating mixer, spin-up lag, wrench about the COM, frame drag, Glauert induced velocity | done (O3) |
| `og.propulsion` | propulsion maps (signed T, Q on V × rpm), the electric drive (map + motor + ρ scaling), flight units, drive tables | done (O3) |
| `og.mujoco` | MuJoCo strategy: geoms (box, sphere, cylinder, capsule, ellipsoid) and explicit inertials from Ogma parts, budget bodies, `compile_check` (lazy `mujoco` extra) | done (O5) |
| `og.cad` | CAD strategy (CadQuery, lazy `cad` extra): exact mass properties of a solid (from Dedalus), Yax mass section, `MassProperties` in SI, solids from every budget shape, assemblies from budgets (checked against the budget), STEP/STL export | done (O6) |
| `og.aero` | skin friction (turbulent, mixed), Merlin's parabolic airframe, Kestrel's full-range polar, Nisus's stability coefficients (+ crow), air angles and aero wrench, Peregrine's vortex lattice | done (O3) |

Tests: each moved piece reproduces the numbers Vegeta's code gave on fixed inputs (`tests/golden/`, recorded from
Vegeta at the commit named in the file); published references on top where they exist.
