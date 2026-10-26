# Strato — a solar-electric stratospheric airship (HAPS), Stage V8

**Question.** Can a helium-filled, solar-electric airship hold station at 18–22 km over Europe for weeks or months
with a communications or observation payload; which engineering factor limits it; which configurations are feasible?

**Scope.** Vegeta does the engineering: geometry, physics, energy, sizing, optimisation. Station-keeping here is a
power and airspeed budget against a wind climatology, not a controller; guidance, control and autonomy belong to
Piccolo, which receives a frozen design through Yax when Strato graduates (V8.17, later).

**Results.** `labs/aero/campaigns/strato/results/report.md` is written by the campaign from its result files (every
number in it comes from a run). The conclusions below quote it.

## Files

| layer | file | what |
|---|---|---|
| physics (Ogma, verified) | `libs/ogma/src/ogma/stratosphere.py` | U.S. Standard Atmosphere 1976 to 86 km, geopotential altitude, gravity vs altitude |
| | `libs/ogma/src/ogma/lifting_gas.py` | buoyancy, sealed (superpressure) and vented (zero-pressure) gas cells |
| | `libs/ogma/src/ogma/solar.py` | NOAA sun position, irradiance above the clouds, incidence on facets, horizon dip |
| | `libs/ogma/src/ogma/envelope.py` | membrane forces of pressurised bodies of revolution, beam loads, wrinkling |
| geometry, CAD | `strato_geometry.py`, `strato.py` | the hull profile and part layout (fractions of the hull); the Dedalus CAD |
| estimates | `strato_catalogue.py` | component data: PV, batteries, laminates, structure, propulsion, gas |
| | `strato_wind.py` | parametric wind climatology, 35–65°N, by month and altitude |
| models | `strato_aero.py` | drag build-up, Lamb's coefficients, Munk moment vs fins, propeller power |
| | `strato_thermal.py` | two-node gas superheat and cell temperature |
| | `strato_design.py` | design point: gas fill, superpressure, envelope plies, mass budget, trim, lift |
| | `strato_sim.py` | day/night cycle, station-keeping availability, endurance, ascent |
| | `strato_feasibility.py` | the gate: problems (rejected) and warnings |
| | `strato_study.py` | seasons, optimisation (SciPy differential evolution), Sobol sampling, Pareto, sweeps |
| campaign | `campaigns/strato/cad.py`, `viewer.py`, `run.py`, `report.py` | CAD files and pictures, the 3D page, every study, the report |
| tests | `libs/ogma/tests/test_{stratosphere,lifting_gas,solar,envelope}.py`, `labs/aero/tests/test_strato_*.py` | reference values, limits, rejections, regression goldens |
| notebook | `notebooks/37_strato.ipynb` | one design walked through the models |

Units SI throughout (kg, m, s, W, Pa, K); the CAD is built in millimetres (Dedalus's unit) from parameters in metres.
Body frame: x aft from the nose tip, y starboard, z up. Times UTC.

## Verified physics and estimates

| model | status | basis |
|---|---|---|
| atmosphere 0–86 km, gravity | verified | USSA-1976 tables at 0/11/20/25/30/50 km (rel. 2e-4) |
| buoyancy, sealed/vented cells | verified | ideal gas, Archimedes; conservation tests |
| sun position, day length | verified | NOAA algorithm: solstice declination, equation-of-time extremes, Earth–Sun distance |
| membrane forces, beam loads | verified | membrane theory: cylinder and sphere limits, hand-solved beam |
| hull volume, area, centre of buoyancy | verified | analytic profile; CAD solid within 0.01 %; half-ellipsoid + paraboloid closed form |
| added mass, Munk moment | verified | Lamb's coefficients (k1, k2 at fineness 4 and 10) |
| beam irradiance at altitude | **estimate** | Meinel clear-sky relation with pressure-scaled air mass (~95 % of TSI at 20 km, noon) |
| drag | **estimate** | Hoerner form factors; hull C_DV in the published 0.015–0.035 band (plausibility only) |
| fin effectiveness, stability | **estimate** | low-aspect-ratio lift slope, hull-wake factor 0.8 |
| propeller | physics × **estimate** | actuator-disc efficiency × profile factor 0.85 |
| superheat, cell temperature | **estimate** | two-node lumped model (one envelope temperature), coefficients below |
| winds | **estimate** | parametric climatology tuned to the published features (jet, summer minimum near 20 km, winter vortex); not ERA5 |
| component data | **estimate** | technology-class values (below), not datasheets |
| CFD of the hull, FEM of fins/gondola | NOT RUN | OpenFOAM and CalculiX are not installed in the build container; Aeromant/Talos cases are the next step on a workstation |

## Assumptions (the estimates, with their values)

- **Solar array** (installed, encapsulated): GaAs (baseline) η 26 % at 25 °C, −0.20 %/K, 0.25 kg/m²; CIGS
  (alternative) η 15 %, −0.36 %/K, 0.45 kg/m²; absorptance 0.90, emissivity 0.85; MPPT 97 %, harness 98 %.
- **Batteries** (pack level): Li-ion 220 Wh/kg, 97 %/97 % charge/discharge, 90 % usable; Li–S 350 Wh/kg, 93 %/95 %,
  85 % usable; a 10 % reserve is never planned to be used.
- **Envelope laminates** (per ply): Vectran 0.20 kg/m², 80 kN/m; Zylon 0.13 kg/m², 90 kN/m; UHMWPE 0.15 kg/m², 60 kN/m;
  safety factor 4 (creep, seams, cold, UV); seams and patches +10 %; solar absorptance 0.20, emissivity 0.80.
- **Structure**: fins 0.35 kg/m² of planform; nose battens and keel 6 % of the envelope; gondola 20 % of what it
  carries; ballonet film 0.08 kg/m² over 45 % of the envelope area (vented designs); avionics 8 kg and 60 W; night
  heaters 10 % of payload + avionics power; 10 % growth margin on everything but the payload.
- **Propulsion**: 1.6 kg/kW rated (motor, gearbox, propeller, mount, ESC); motor 92 %, ESC 97 %; 60 % at the stern,
  40 % on two vectoring units.
- **Gas**: helium, 98 % purity, 0.1 %/day leakage.
- **Payload**: 10 W per kg (communications or observation; 5–20 W/kg typical).
- **Thermal**: Earth's outgoing long-wave radiation 240 W/m² below, 15 W/m² of sky IR above, envelope–gas convection
  0.5 W/(m² K), laminate heat capacity 1300 J/(kg K); forced convection from the turbulent flat plate at the airspeed.
- **Wind**: jet 27 ± 10 m/s at 11 km (winter +), stratospheric westerlies to +30 m/s per 10 km above 16 km in winter
  (stronger poleward), easterlies to −12 m/s per 10 km in summer; day-to-day σ 3 m/s (summer) to 9 m/s (winter) at
  20 km; Gaussian daily components.
- **Design rules**: the hull is filled so it is just full at night at the minimum superpressure (shape: 1.2 × the
  dynamic pressure at top speed, the bending wrinkling pressure, at least 50 Pa) plus an optional night reserve; the
  envelope carries the noon superpressure plus bending with the safety factor (plies added as needed, at most 6);
  the gondola is placed to put the CG over the CB.

## What the gate rejects (problems) and flags (warnings)

Rejected: invalid inputs (negative energies, unknown parts, impossible hulls: fineness outside 2–12, the maximum
diameter outside 20–60 % of the length); **does not float** at the design altitude; **cannot trim** (the gondola
would leave 25–65 % of the length); CG not below the CB; the envelope needs more than 6 plies; the energy does not
close even in calm air. Flagged: static instability in pitch/yaw (needs active control: Piccolo), a vented design's
shape needing more than its valve setting. Scenario requirements (availability, gas life, payload) are not physics:
the optimiser treats them as constraints and the report says which design met them.

## Conclusions

See the section of the same name at the end of `campaigns/strato/results/report.md` and the summary in `todo.md`
(Stage V8); filled in from the campaign results.
