# Strato — a solar-electric stratospheric airship (HAPS), Stage V8

**Question.** Can a helium-filled, solar-electric airship hold station at 18–22 km over Europe for weeks or months
with a communications or observation payload; which engineering factor limits it; which configurations are feasible?

**Scope.** Vegeta does the engineering: geometry, physics, energy, sizing, optimisation. Station-keeping here is a
power and airspeed budget against a wind climatology, not a controller; guidance, control and autonomy belong to
Piccolo, which receives a frozen design through Yax when Strato graduates (V8.17, later).

**Results.** `labs/aero/campaigns/strato/results/report.md` is written by the campaign from its result files (every
number in it comes from a run). The conclusions below quote it.

## Strato-Sigma

The smallest member of the family, for imaging only: a daylight EO camera on a small gimbal (~3 kg, 25 W) and an
S/X-band downlink to a ground station (~1.5 kg, 25 W), both off at night, no relay to aircraft (estimates). Sized for
the summer scenario (June–August at 50°N, on station ≥ 90 % of days, ≥ 60 days of gas) by `strato_study.smallest_length`:
bisection on the length with the rest of the design optimised at each step. Campaign stage `sigma`; results in the
report's "Strato-Sigma" section.

## Zeppelin-Micro

The cheapest maritime imaging airship (it stays in the troposphere, so it is a Zeppelin, not a Strato; the
Strato name is kept for ships that float in the stratosphere): a daylight camera with a video link at a few hundred metres over the sea, for
hours per deployment. A tethered aerostat (no propulsion; the line holds it in the wind) is compared with a
free-flying micro blimp, both sized against a design sea wind with the same physics; parts, prices and the sea wind
are estimates (`zeppelin_micro.py`), and the costs compare options rather than price a product. Study script
`campaigns/strato/zeppelin_micro_study.py`; results in the report's "Zeppelin-Micro" section.

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

## Conclusions (campaign of 2026-10-26; numbers from `campaigns/strato/results/report.md`)

All numbers below come from models built on the estimates listed above (component data, wind climatology); read
them as engineering estimates, not measured performance.

1. **Feasible, in the stratosphere, from about 35–50 m.** A sealed, solar-electric, helium superpressure airship
   holds station at about 18 km over 50°N:
   - **in summer** (June–August, on station ≥ 90 % of days, ≥ 60 days of gas): the smallest optimised general ship is
     50 m (about 140 kg of payload); the imaging-only **Strato-Sigma** is 34.4 m (35.6 m with a day-and-night thermal
     camera);
   - **year-round** (≥ 50 % of days on station in the worst month): the smallest is 80 m (about 390 kg of payload).
   In the 30-day July simulation the 80 m ship held 18.2 km and its station all month with ballast to spare.
2. **The 15 m × 3.5 m demonstrator cannot reach the stratosphere.** At 20 km its hull lifts about 24 kg in all and
   its envelope alone weighs about 30 kg. Its best design floats at about 6 km and holds station on only about 20 % of
   July days.
3. **What limits it** (deterministic sensitivity of the 80 m ship, summer, base 538 kg of payload):
   - **lift versus envelope mass**: square–cube, ±20 % of payload for ±30 % of laminate mass. It decides the minimum
     size; the envelope is 30–46 % of the mass;
   - **the payload's night power**: the largest lever, 354–618 kg for 20–5 W per kg of payload;
   - **battery specific energy**: −100 kg for −20 %;
   - **helium leakage**: −108 kg for twice the leak rate;
   - **winter winds and short days**: a year-round ship carries 2.5 times the summer ship's battery;
   - **wind variability**: −70 kg for 30 % more.
   **Not limiting in summer**: PV efficiency (energy is in surplus) and the thermal model's least certain coefficient.
   GaAs beats CIGS by about 16 % of payload in summer and 25 % year-round, through the array's mass.
4. **The best configurations** (all optimiser picks): fat hulls (fineness 2.5–3.2), floating at the bottom of the band
   (18.0–18.6 km, the densest air above the summer wind minimum), one ply of Zylon-class laminate, a sealed
   superpressure hull with a night reserve. Vented designs are not viable here: the 80 m ship's vented variant does not
   even float, because of its ballonet.
5. **Below the stratosphere it is a Zeppelin, not a Strato.** **Zeppelin-Micro**, a maritime imaging airship at
   200 m, is cheapest as a tethered aerostat: about 6.2 m × 2.1 m, 6.5 kg, roughly €7.4k in its first year (estimate);
   about 6.6 m and €11.7k with a thermal camera.
6. **What is not settled.**
   - The optimiser's payloads scatter by about ±50 kg between seeds (which lengths meet a scenario is robust).
   - The winds are a parametric climatology, not ERA5 data.
   - The hull CFD and fin FEM are not run (no OpenFOAM or CalculiX in the build container).
   - Station-keeping is an energy and airspeed budget, not a controller (that is Piccolo's job).
   - Static pitch/yaw stability is marginal; the ships need active control.

   Next: real wind data for a site, CFD of the hull, a laminate datasheet, and a longer optimiser budget for the
   payload numbers.
