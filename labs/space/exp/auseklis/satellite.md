# Auseklis — design choices and their reasons

This document explains **why** the satellite model is the way it is: what each choice is, what it replaced or ruled out,
where the number comes from, how much to trust it, and what to change first. It describes the code in this folder
(`physics.py`, `variants.py`, `mission.py`, `rendezvous.py`, `capture.py`, `cad.py`, `trade.py`, `external/`); the
numerical results live in `REPORT.md` (generated) and the summary in `README.md`.

**Status of everything below: preliminary concept-level engineering. Nothing is validated.** No flight hardware
datasheet was consulted for any value; no real solver (CalculiX, Gmsh, MuJoCo) has been run.

## 0. How to read the tags

| tag | meaning | in code |
|---|---|---|
| **VERIFIED** | a physical constant or definition with a public source | module constants in `physics.py` (`MU`, `RE`, `J2`, `G0`, `SIGMA`, `S0`, `EARTH_IR`, `ALBEDO`), with the source in a comment. They are plain floats, *not* `Tagged` objects, so they are not in the assumption register |
| **CALCULATED** | produced by a script here from other values | everything `REPORT.md` prints |
| **ASSUMED** | a design choice or class-typical value, with its reason | `provenance.assumed(...)`, 47 call sites; listed in `REPORT.md` ("Assumption register") |

Be aware of a **gap in the register**: several inputs are still inline literals and therefore *not* tagged or listed
(§15 lists them). The register is a lower bound on the assumptions, not the full set.

## 1. How the brief was interpreted

The brief gave preliminary requirements "not validated performance" and asked that infeasible requirements be
*identified, not forced*. That shaped three rules used throughout:

1. **Report the closure honestly.** When the bottom-up mass does not land in the 350–500 kg window, the code says so
   (`within_mass_window`) rather than inflating margins until it does.
2. **Give every requirement a number, even when it is a judgement.** E.g. "conservative velocity limits" became a table
   (§10) that the independent safety monitor enforces, so it can be tested and challenged.
3. **Prefer a transparent closed-form model to an opaque high-fidelity one**, and say where a high-fidelity run is
   needed (`external/`). Closed forms are testable against hand calculations (`tests/test_auseklis.py`).

| brief item | what was done | what was not |
|---|---|---|
| wet mass 350–500, target 400 kg | two budget modes: *sized to 500 m/s* and *filled to 400 kg* (§5) | no flight-heritage mass check |
| 550 km, 97.6° | sun-synchronous inclination computed from J2 (97.59°) as a consistency check (§3) | no LTAN/β-angle study beyond β=0 worst case |
| 5-year life | array degradation over 5 y; mission campaign limited to 5 y (§8) | no wear-out reliability model |
| Δv 500 m/s incl. reserves | variant sizing target; 10 % propellant reserve in campaigns | reserves are not itemised per manoeuvre |
| propulsion: green mono, bi-prop, electric; hybrid | variants B, A, C; C is the hybrid (Hall + chemical RCS) (§4) | no thrust-profile optimisation of a hybrid transfer |
| 1.5 kW EOL, Li-ion | array sized from 1.5 kW EOL; battery from worst eclipse (§6) | no battery cycle/thermal model |
| ADCS, nav, comms, computing, thermal | mass allocations and sensor-class assumptions | no ADCS control sim, no link budget, no thermal network |
| capture: grippers vs arms vs docking | one variant each (§4, §11) | no gripper contact model, no net/harpoon |
| disposal: controlled reentry or device | both *hand-over to 5-year decay* and *controlled reentry* (§8) | no disposal-device mass |
| rendezvous at 10 km, 1 km, 100 m, 10 m | four gates with nav error, speed limit, keep-out, passive safety (§10) | no EKF; σ values are inputs |
| uncooperative debris | tumble rates and capture momentum (§11) | no pose-estimation or tumble-rate estimation |
| CadQuery CAD, STEP | done for all variants (§12) | no detailed mechanisms/harness |
| astrodynamics sim with J2, drag, relative motion | closed-form J2 and drag, Clohessy–Wiltshire (§3, §10) | **no numerical J2+drag propagator yet** |
| multibody capture, FEA | numpy 1-D contact + angular momentum (§11); FEA and MuJoCo *prepared* (§13) | not run |
| component selection with real specs, BOM, costs | **not done** | needs datasheet verification (§16) |

## 2. Frame and units conventions

- SI units in all physics modules; **millimetres** in CAD and in the Talos scripts (`mm-N-MPa`).
- Body frame (CAD): **+X toward the target** (capture end), −X aft with the main thrusters, ±Y solar wings, +Z zenith,
  −Z nadir. Origin at the bus centre. Capture and proximity sensors share the +X face so everything that looks at the
  target points the same way; the thrust axis is along −X so a burn pushes the satellite toward +X.
- Relative-motion (Hill) frame for rendezvous: x radial up, y along-track, z cross-track (Clohessy–Wiltshire).

## 3. Orbit and environment (`physics.py`)

**Reference orbit.** 550 km circular, i = 97.6°: v = 7585 m/s, period 95.65 min. The inclination is *computed* from the
J2 nodal-precession condition (`raan_rate_j2 = Ω_sun`) as a cross-check: 97.59°. Choice rationale: the brief's number is
consistent with a sun-synchronous orbit, which also gives near-constant illumination geometry; so the cheap
consistency test is worth having. Only the mean Sun rate `2π/year` is used (not a refined tropical-year value).

**Hohmann, spiral, phasing, plane change.** Standard formulas (Vallado). Checked against 200 km → GEO (3.93 km/s vs the
textbook 3.935). Two choices worth noting:

- `phasing_dv` is an **impulsive two-burn phasing orbit** (equal burns at the common apse); it refuses a phasing orbit
  whose perigee falls below RE + 120 km. The cost falls roughly as 1/N with the number of phasing revolutions, so the
  mission model spends *time* (10 days) to save Δv.
- The **electric variant** uses the same impulsive numbers for phasing and a low-thrust Edelbaum spiral (`|v1−v2|`) for the
  tow. For the small manoeuvres involved this is a reasonable first estimate; it neglects the extra Δv of
  imperfect thrust steering and gravity losses (small at LEO for these Δv).

**Plane change and RAAN.** `plane_change_dv = 2 v sin(Δi/2)` (≈132 m/s/deg at 550 km). A RAAN gap can instead be closed by
differential nodal drift (`differential_raan_rate_deg_per_day`: ~0.010 deg/day for ±20 km, ≈100 days per degree). These
two numbers are why the throughput model **assumes all targets are already in the servicer's plane** (§8). This is the
single largest optimism in the throughput results and is stated in every summary.

**Eclipse.** Cylindrical-shadow formula (Wertz): 37.2 % of the orbit (35.6 min) at β = 0. The budget uses β = 0 because
it is the worst case and the nodal local time of the final orbit is not fixed by the brief. A 10:30 or dawn–dusk orbit
would have much less eclipse (dawn–dusk none), so the battery is **conservative** unless the mission can choose its LTAN.

**Atmosphere (the most sensitive assumption).** A small log-linear table of density vs altitude for three solar-activity
cases (`_RHO`), "consistent with NRLMSISE-00/Jacchia class models" in order of magnitude only. Why a table instead of a
real model: no dependency, reproducible, trivially testable (monotonic in altitude and activity). Why three cases: because
the lifetime and the hand-over altitude vary by *orders of magnitude* with solar activity, so a single number would
hide the main uncertainty (REPORT: 250 kg removals/life 4 → 14 → 22 for solar min → moderate → max). **Replace with
NRLMSISE-00 or JB2008 and a real F10.7/Ap forecast before quoting any lifetime.**

`lifetime_days` integrates the circular decay rate `da/dt = −ρ (C_d A/m) √(μ a)` in 1 km steps down to 150 km. It is
applied to a *circular orbit of equal semi-major axis*, which **over-estimates** the life of the elliptical hand-over orbit
(perigee dominates drag), so the resulting hand-over altitude is conservative on that point; the tumbling-averaged
`C_d A/m` is not.

## 4. Why these three variants (`variants.py`)

The brief asked to evaluate green mono, bipropellant and electric propulsion and three capture approaches, and to
compare at least three configurations. Mapping each propulsion option to a *different* capture approach makes the
trade explore a wider space with three points, instead of nine near-duplicates:

| | A | B | C |
|---|---|---|---|
| propulsion | storable bipropellant | green monopropellant (ADN-class) | Hall electric + green RCS ("hybrid") |
| capture | 2 m robotic arm + gripper | active docking adapter | short boom + compliant 3-finger gripper |
| intent | agility, tumbling targets | simplicity, cooperative targets | propellant economy, slow tows |

- **Why A has the arm:** a tumbling uncooperative stage needs standoff, reach and compliance, which only an arm gives.
  It is the mass and mechanism-risk worst case, and the matrix says so.
- **Why B has docking:** it is the lightest and most mature capture option; it is deliberately *restricted to cooperative
  targets* (`cooperative_only=True`) so the model can show what that limit costs.
- **Why C is the hybrid:** electric propulsion makes the largest tows cheap in propellant but cannot do the delicate
  proximity operations (thrust ~54 mN, response in seconds–minutes) or an impulsive reentry burn; so the Hall thruster
  handles phasing and tows and a **chemical RCS handles rendezvous, capture and aborts** (60 m/s reserved).
  The 60 m/s is an assumption sized to ≈12 m/s per removal × a handful of removals plus abort margin.
- **Not modelled:** green-mono *and* electric, nets/harpoons/tethers, ion-beam shepherds, or a drag-sail disposal kit.
  Nets/harpoons are out because the brief says "gripper, arm, docking"; a drag sail is the natural next *disposal*
  variant (mass-cheap) and is a recommended follow-up.

## 5. Mass budget and closure (`variants.budget`)

### Bottom-up subsystem masses (kg, ASSUMED, identical across variants unless noted)

| item | kg | reason |
|---|---:|---|
| structure | 52 | Al honeycomb panels + machined frames; ~13 % of 400 kg, typical for a small servicer |
| thermal | 13 | MLI, radiators, heaters, thermistors |
| harness | 22 | ~5 % of mass, two redundant strings |
| avionics | 9 | two rad-tolerant OBCs, I/O, and the *independent* safety-monitor unit (§10) |
| GNC | 24 | 2 star trackers, 2 IMUs, 2 GNSS, sun sensors, 4 wheels, 3 magnetorquers |
| proximity sensors | 14 | scanning LiDAR, wide + narrow cameras, thermal IR, second independent range sensor |
| comms | 9 | 2 S-band transponders + patches, X-band transmitter |
| battery + PCDU | 10 | Li-ion pack electronics, PCDU, array drive |
| solar array | 22.2 | **calculated**: 6.16 m² × 3.6 kg/m² (rigid panels incl. hinges, yoke) |
| battery cells | 6.6 | **calculated**: 791 Wh / 120 Wh/kg pack-level |
| capture | 38 / 20 / 26 | A arm / B docking / C gripper + boom |
| propulsion hardware | 14 / 10 / 16+9 | thrusters, valves, lines; C = Hall + spare + PPU + RCS |

Nominal dry mass comes to 234 / 212 / 233 kg. A **25 % growth margin** (AIAA S-120-style concept-level 25–30 %) gives
292 / 265 / 291 kg dry. The margin is applied to dry mass only; propellant is computed exactly.

### Two ways to close

1. **Sized to 500 m/s:** the propellant is whatever gives the *satellite alone* 500 m/s. This is the requirement read
   literally and gives wet masses of 355 / 339 / 311 kg — **B and C fall below the 350 kg floor**. That is reported as a
   finding, not fixed by adding mass.
2. **Filled to 400 kg:** tanks filled to the target wet mass; Δv capability is the *result* (~800 / ~830 / ~3640 m/s).
   Mission campaigns use this mode, since an operator would fly the spare capacity.

### The sizing algorithm

Rocket equation `m_p = m_empty (e^{Δv/(Isp g0)} − 1)` with tank mass `f × m_p`, which makes the system implicit; for the
single-system variants it has the closed form `m_p = m_dry (e^x − 1) / (1 − f (e^x − 1))`, and variant C (two systems) is
solved by a short fixed-point loop. Two modelling choices in C:

- the **RCS burns first at full wet mass** (a conservative ordering: the largest mass gives the most RCS propellant);
- the xenon system takes the rest (440 m/s in the 500 m/s case).

Tank fractions (A 0.10, B 0.12, C xenon 0.15, RCS 0.12) are ASSUMED: xenon is stored supercritical at high pressure in
heavier vessels per kg. The RCS fraction 0.12 is an inline constant (not in the register).

### How to read "Δv 500 m/s"

The requirement says *total Δv budget*; the code reads it as the capability of the satellite **at its own mass**. But a
tow burns propellant against the **combined** mass (satellite + debris, up to ~900 kg), so the *same* Δv costs more
propellant, and the ledger (`mission.removal`) counts that. This is why a 500 m/s budget does not translate into
500 m/s of tow Δv. This reading is the most consequential interpretation in the document; if the requirement is intended
for the *pair* the numbers change materially.

## 6. Power, battery and eclipse (`variants.array_sizing`, `power_modes`)

- **Array:** `A = P_eol / (S0 η pf (1−d)^t (1−L_inc)(1−L_T)(1−L_other))` with η = 0.29 (triple-junction BOL, class-typical),
  packing 0.85, degradation 2 %/yr over 5 y (radiation, thermal cycling, UV), and inline losses of 8 % (incidence with a
  single-axis drive), 10 % (operating temperature) and 3 % (other). Result: **6.16 m²**.
  Why size to EOL: the requirement is stated as end-of-life 1.5 kW.
- **Battery:** energy = `P_ecl t_ecl / (DoD η_b)` with 380 W, 35.6 min, DoD 30 % (a depth consistent with ~28 000 cycles in
  5 years for Li-ion), discharge efficiency 0.95 → **791 Wh**.
  *Inconsistency to know about:* the battery is sized at 380 W while the `power_modes` table puts rendezvous at 420 W and
  capture at 480 W. If rendezvous/capture happen in eclipse, the battery is ~11–26 % short. Fix by sizing to the
  worst-case eclipse operating mode, or by constraining operations to sunlight.
- **Power modes:** cruise 190, rendezvous 420, capture 480, safe 120, chemical tow 300 W, electric tow 300 + 800 W; a mode
  is "ok" if it uses ≤ 90 % of 1.5 kW. The loads are ASSUMED round numbers, not a bottom-up load list. The electric tow
  fits (1100 W, 400 W margin) *only in sunlight*, which is why the Hall duty cycle is 0.65 (§8).
- **Thermal:** `equilibrium_temperature` (a single-node isothermal balance with solar, albedo, Earth-IR and internal
  heat) is implemented and unit-conventions documented, but **no thermal report or radiator sizing was produced**; the
  13 kg thermal mass is an allocation, not a result.

## 7. Propulsion choices

| | Isp | thrust | why this value |
|---|---:|---:|---|
| A bi-prop | 290 s | 44 N | small storable bipropellant engines (2 × ~22 N class), **system-level** not engine-level Isp |
| B green mono | 235 s | 22 N | ADN-blend monoprop, system-level; 4 × ~5 N + 1 N thrusters |
| C Hall | 1500 s | 54.4 mN | 800 W input, η = 0.5: `F = 2ηP/(Isp g0)`; 600–1000 W class Hall thrusters |
| C RCS | 220 s | 10 N | green monoprop RCS |

The 10 N used for C's RCS in the burn-time estimate is an inline constant. Isp values are *system-level class-typical*
numbers, deliberately a little under quoted engine values to leave room for transients and pulse-mode operation; all are
tagged ASSUMED and should be replaced by datasheet values (§16).

Hybrid in the brief's sense ("efficient transfers and precise maneuvering") is variant C: the electric stage does the
efficient transfers and the chemical RCS does the precise manoeuvres. A genuinely *combined* hybrid trajectory
(chemical for the time-critical phase, electric for the remainder) was not optimised.

## 8. Mission model (`mission.py`)

### Debris classes (ASSUMED)

| class | mass | C_d A/m (m²/kg) | cooperative | tumble | shape |
|---|---:|---:|---|---|---|
| 100 kg | 100 | 0.022 | yes (fixture) | 0.5 °/s | R 0.4 m × L 1.2 m |
| 250 kg | 250 | 0.030 | no | 2 °/s | R 0.6 m × L 2.8 m |
| 500 kg | 500 | 0.030 | no | 3 °/s | R 0.8 m × L 3.6 m |

`C_d` = 2.2 and a tumbling-averaged area (≈ 1.0, 3.4, 6.8 m²). Tumble rates are within the range reported for old rocket
bodies (a few °/s) but are *chosen*, not measured. Real catalogued targets (TLE-based) are not used because the brief
did not name any; this is a clarifying question that remains open.

### Per-removal ledger

Legs: **phasing** (90°, 10 days) → **altitude change** (20 km Hohmann) → **rendezvous + inspection** (12 m/s, on the RCS in
C) → **tow** at the pair mass to the hand-over orbit → **return** to 550 km alone. Propellant per leg uses the actual
mass at that point; the satellite gets lighter through the sequence; the tow leg adds the debris mass.

Disposal modes:

- **`decay5` (default):** hand the debris over on an orbit that decays within **5 years** (the ESA Zero Debris direction;
  the older 25-year rule would need far less Δv). The altitude is found by bisection on `lifetime_days` for the
  debris's `C_d A/m`. Implemented as one tangential burn at apogee that lowers perigee to the same semi-major axis as
  the circular hand-over orbit; for C, a spiral to that circular altitude.
- **`reentry`:** perigee lowered to 60 km (~140 m/s on the pair, then ~140 m/s back alone). Not available for C:
  a low-thrust spiral cannot make an impulsive reentry burn.

Why 5 years: it is the stricter current guidance and gives the more *conservative* Δv; the alternative (25-year) would
inflate the throughput. Why "hand over then return" instead of towing all the way: the debris only needs a decaying
orbit, and the satellite must come back to the operating altitude for the next target; the two burns cost about equal Δv.
The satellite's own end-of-life disposal is covered only by a 10 % propellant reserve, not by a calculation.

### Campaign

`campaign` repeats removals of one debris class until the main propellant (less 10 %) or RCS propellant runs out, or the
5-year clock runs out. Durations: max(10 days phasing, burn time) + altitude change + 8 days of operations + tow + return,
× 1.25 contingency; the Hall variant divides burn time by a 0.65 duty (sunlight, power, pointing).

**What it does not include** (and why the result is an upper bound): target availability and in-plane geometry, ground
contact/operations scheduling, failures and retries, learning from one removal to the next, and RAAN/inclination
differences between targets. Every ledger assumes a target 20 km away in altitude and 90° in phase *in the same plane*.

## 9. Sensitivities that drive the conclusions

| change | effect |
|---|---|
| solar activity moderate → minimum | 250 kg hand-over 538 → 455 km, Δv 45 → 137 m/s, removals/life 14 → 4 (A) |
| disposal 5-year decay → controlled reentry | Δv ~45 → ~312 m/s per removal, 14 → 1 removal/life (A) |
| capture type | B cannot take uncooperative debris at all |
| thrust level (C) | ~27–30 days per removal vs ~23 days for chemical, and ≈4 removals limited by RCS propellant |
| targets not in plane | ~132 m/s per degree of inclination; ~100 days per degree of RAAN by drift |

The first two dominate; they are why a real atmosphere model and an explicit disposal-policy decision come first.

## 10. Rendezvous and approach safety (`rendezvous.py`)

| gate | sensor class | σ_pos (assumed) | v_max | keep-out | horizon |
|---|---|---:|---:|---:|---:|
| 10 km | relative GNSS/ground-aided + wide camera, angles-only | 60 m | 5 m/s | 5 km | 90 min |
| 1 km | narrow camera + angles-only range / LiDAR acquisition | 8 m | 0.5 m/s | 500 m | 90 min |
| 100 m | LiDAR + camera pose estimation | 0.6 m | 0.1 m/s | 30 m | 30 min |
| 10 m | LiDAR + camera + proximity sensors | 0.03 m | 0.02 m/s | 3 m | 5 min |

- **Speed limits are deliberately conservative**: roughly 0.05–0.2 % of range per second (5 m/s at 10 km, 2 cm/s at 10 m). At
  the speeds shown, closing from 10 km to 10 m takes ~1.5 hours of pure approach before inspection holds. They are
  assumptions intended to be tightened or relaxed by a proper navigation and abort-time analysis.
- **Passive safety** = with thrusters off from a hold at the gate range, the first-percentile minimum separation over the
  horizon exceeds the keep-out. Method: Clohessy–Wiltshire propagation of a Monte-Carlo sample (60 samples, seeded for
  reproducibility) of velocity errors `σ_v` (0.05, 0.02, 0.005, 0.001 m/s). The horizons are chosen as "time to
  recognise a failure and abort" (long far away, a few minutes at 10 m).
- **Result:** the 1 km gate **fails** (356 m vs a 500 m keep-out) at σ_v = 0.02 m/s; at 0.01 m/s it passes (636 m). The
  result is kept as a design finding (needs a better velocity estimate or a safety-ellipse hold), not tuned away.
- **Collision probability** (`pc_2d`): the integral of an isotropic 2-D Gaussian position error over the combined
  hard-body disc, evaluated on a grid; verified against the closed form `1 − exp(−R²/2σ²)` for a centred encounter.
  The table's value is "if on a nominal collision course", which is a conservative bound, *not* a mission Pc. A
  covariance-based Pc along the real trajectory requires the EKF covariance (not built).
- **Independent safety monitor** (`SafetyMonitor`): a separate class with *no access to the guidance state*; it sees only
  range, closing speed and lateral offset (the second range sensor in §5) and can only return `abort`. It enforces the
  speed table (with 20 % tolerance) and a ±10° approach corridor inside 100 m. It is independent by construction in
  the model; independence in flight (separate processor, power, sensor) is a hardware requirement not yet analysed.
- **Cooperative vs uncooperative:** modelled through the capture system and tumble (§11), not through different
  navigation: a cooperative target would allow retro-reflectors and a lower σ, which is not credited here (conservative).

## 11. Capture and coupled dynamics (`capture.py`)

Light numpy estimates that the MuJoCo scenario (§13) is meant to supersede or cross-check:

- **Angular momentum at capture:** rigid, instantaneous capture, servicer initially non-rotating; `H = r × m v + I_d ω_d`
  about the combined centre of mass; the combined rate is `I⁻¹ H`. Verified by tests (zero debris spin and zero relative
  velocity gives zero momentum; `I ω = H`). For the 500 kg, 3 °/s case: H ≈ 32 N·m·s, combined rate ≈ 0.9 °/s.
- **Detumble propellant:** `H / (arm · Isp · g0)` with a 0.8 m RCS lever arm and 220 s → about 19 g. Reaction wheels
  could absorb *part* of it, but the wheels' storage capacity was not modelled; using RCS is conservative for the wheels
  and trivial for propellant.
- **Contact:** a 1-D two-mass spring–damper (arm stiffness 2 kN/m, damping 300 N·s/m, 5 cm/s closing) integrated with
  semi-implicit Euler; peak force ≈ 27 N against the undamped bound `v √(kμ)` = 33 N, with the reduced mass μ = 222 kg.
  The stiffness and damping are illustrative; they are the design handles for an arm compliance requirement.
- **What is not captured:** off-centre contact torques, three-dimensional approach, grasp-fixture geometry, flexible arm
  modes, slosh, wheel gyroscopics, and the real controller. The rigid-capture assumption is the main idealisation.

## 12. CAD (`cad.py`, `render.py`)

**Why CadQuery via Dedalus/Assembly:** it is the repo's CAD toolchain, produces STEP, and `labkit.assembly.Assembly`
gives named parts for per-part colour, mass and STEP export.

**Parameters** (`Auseklis.parameters`): `variant`, `wet_kg`, bus X/Y/Z (900 × 800 × 850 mm), panel thickness (25 mm), wing
panels (3 per wing), panel span/chord (1000 × 1050 mm), arm length, dish diameter. The bus is a **hollow shell**
(wall = panel thickness) plus one equipment deck above the tanks.

**Why that bus size:** 0.9 × 0.8 × 0.85 m is ~0.6 m³, a density of ~660 kg/m³ at 400 kg; this fits a rideshare
envelope and leaves room for a 100–110 litre tank. Not optimised.

**Tank sizing from the budget:** propellant volume = `m_p / bulk_density / 0.9` (10 % ullage). Bulk densities (ASSUMED):
1200 kg/m³ storable bipropellant, 1250 ADN monoprop, 1500 dense xenon. Tanks are cylinders with hemispherical domes along X,
split 45/55 into fuel/oxidiser for A, and `tank_radius` finds the smallest radius whose tank fits inside the bus;
`ValueError` if it cannot (the oversized-tank test). A single-tank approximation is used for B and C.

**Array geometry:** 2 wings × 3 panels × 1.05 m² = 6.3 m² ≥ the 6.16 m² needed; the test asserts it. Panels are drawn
fully deployed in one plane; the sun-tracking joint and stowed configuration are not modelled.

**Mass properties (`mass_properties`):** each budget category's mass is spread over its CAD parts **by volume**, each part is
a point mass at its centre of volume plus a bounding-box inertia, then summed (parallel-axis). Tank parts carry the
propellant. This gives a self-consistent CoG/inertia that closes to the budget wet mass exactly (tested), but it is an
*estimate*: a real value needs per-part masses from the BOM. Observed CoG offsets (e.g. +0.15 m along X on A) come from the arm.

**Interference check:** `test_no_unintended_part_interference` intersects every pair of parts whose bounding boxes touch and requires < 1 cm³ of overlap, apart from two intended contacts (wheels on the deck, the arm elbow) and the corner thrusters embedded 3 mm in the wall. Writing it found two real mistakes that were then fixed: the X-band dish overlapped a star tracker (dish moved to x = +0.15 X), and in variant B the LiDAR and narrow camera sat inside the docking ring (ring and cone reduced to r = 180 mm, a 205 mm clearance to the nearest sensor).

**Simplifications:** thrusters, sensors and antennas are cylinders/cones; the dish is a revolved arc; fingers are boxes;
no fasteners, harness, hinges, MLI. The capture hardware geometry is schematic and **must not be used for load
calculations**.

**Rendering:** `render.py` tessellates each OpenCascade part and draws shaded triangles with matplotlib's 3-D toolkit.
No GPU or display needed. The zoomed isometric shows the bus transparent so tanks, equipment deck and wheels are visible.
(Matplotlib is not a CAD viewer: painter's-algorithm artefacts between overlapping parts are expected.)

## 13. Prepared heavy-tool runs (`external/`)

Policy: **nothing here runs a solver by default.** Each script writes its inputs and prints the plan; `--run` is required
for a real run; in `VEGETA_DEBUG=1` mode the Talos calls are mocked and the MuJoCo run is refused outright (mock
numbers from a physics engine would be misleading).

**`bus_fea.py` (Talos → Gmsh + CalculiX).**
- *Equivalent-density shell:* the 25 mm honeycomb shell is modelled as solid 6061-T6 with density `M_wet / V_shell` so a
  body load carries the whole wet mass (3.39e-9 t/mm³ for variant A, ~26 % above aluminium). It is a *load-transfer*
  trick, not a stiffness model: real sandwich panels are less stiff, so the modal result is optimistic and the stresses
  are not a panel check.
- *Interface:* the aft face fixed (a stand-in for the separation ring).
- *Loads:* 7 g axial and 3 g lateral quasi-static (**placeholders** until a launcher user guide is chosen); required yield
  factor 1.25 and first-mode requirement 45 Hz are assumptions typical for small-satellite rideshare.
- *Material:* E 68.9 GPa, ν 0.33, yield 240 MPa as typical handbook values to be confirmed against MMPDS.
- *Mesh:* 30 mm second-order tetrahedra (coarse, adequate for a first run).

**`mujoco_capture.py`.** Writes an MJCF with two free bodies (servicer; debris as a uniform cylinder with initial spin and
closing velocity), a zero-gravity world, and a **soft weld** equality constraint switched on at contact. The inertias come
from `cad.mass_properties` and the cylinder formula. The `run` function logs the peak constraint force and the coupled
rate. It is a rigid-body stand-in for gripper contact; the real contact geometry needs a mesh-based model.

Both scripts' results require external verification (also listed in `README.md`).

## 14. Weighted trade matrix (`trade.py`)

Criteria and weights (**ASSUMED**): uncooperative capture 0.25, removals per life (250 kg) 0.20, maturity 0.15,
propellant per removal 0.10, mission time 0.10, mass margin 0.10, mechanism risk 0.10.
Quantities are min–max scaled across the three variants; "maturity" (A 4, B 4, C 3) and "mechanism risk" (A 2, B 4, C 3)
are my 1–5 judgements. Totals: A 0.69, C 0.63, B 0.22.

Because the weights are assumed, `weight_sensitivity` draws 2000 Dirichlet perturbations (concentration 40) around them
and reports the share each variant wins (A 71 %, C 29 %, B 0 %). Read it as "A is favoured under these preferences and B
is disqualified by cooperative-only capture". It is **not** a recommendation; real weights need stakeholders, and the
matrix omits cost, which is not yet estimated.

## 15. Known limitations, inconsistencies and untagged inputs

Be skeptical of these first:

1. **Untagged inline constants** (not in the register): array incidence/temperature/other losses (8 %/10 %/3 %), RCS tank
   fraction 0.12, C's RCS thrust 10 N, ullage 10 %, bulk densities, the `power_modes` loads, navigation σ_v, speed limits,
   keep-outs, horizons and corridor angle, the trade weights and judgement scores, the capture arm stiffness/damping.
2. **Battery vs rendezvous load mismatch** (§6).
3. **VERIFIED constants are not `Tagged` objects**, so they do not appear in the register (sources are code comments).
4. **Solar-activity table** is order-of-magnitude (§3) and dominates the throughput result.
5. **Throughput is an upper bound** (in-plane targets, no failures, no scheduling).
6. **Tow Δv for the electric variant** uses the circular-spiral approximation and a fixed 0.65 duty cycle.
7. **Mass properties** are volume-weighted estimates, not BOM-based.
8. **No cost, reliability/FMEA, link budget, ADCS control, EKF, or thermal network** exists yet.
9. **The coupled capture** is rigid-body; flexible arm modes and contact are missing.
10. **Phasing for the electric variant** uses an impulsive formula with a duration floor; it is acceptable only because
    the manoeuvre is small relative to the thrust level.

## 16. What to do next (in order of leverage)

1. Replace the atmosphere table with NRLMSISE-00/JB2008 and run the J2+drag numerical propagator; decide the disposal policy
   (5-year decay vs reentry vs a drag-sail kit).
2. Fix the battery/operations inconsistency and tag the inline constants.
3. Choose a launcher; replace the load placeholders; run the bus FEA for real and refine the sandwich model.
4. Build the EKF and a covariance-based Pc; resolve the 1 km passive-safety shortfall.
5. Select real components from datasheets (thrusters, wheels, star trackers, battery, TT&C, LiDAR), then BOM and cost.
6. FMEA and reliability; AIT roadmap; launch and regulatory review.
7. Replace the capture stand-in with a gripper contact model and run the MuJoCo scenario.

## 17. Decision log

| # | decision | alternatives considered | reason | reversibility |
|---|---|---|---|---|
| D1 | Closed-form physics first | high-fidelity propagator, GMAT/Orekit | testable, fast, no dependency; the brief asked for equations and sensitivities | add a propagator beside, keep closed forms as cross-check |
| D2 | Three variants crossing propulsion × capture | 3 propulsion × 3 capture = 9 | a wider space with three points; matches "compare at least three configurations" | easy: variants are data |
| D3 | Two budget modes (500 m/s vs 400 kg) | one mode | the requirement and the target conflict for B and C; both are informative | trivial |
| D4 | Δv read as satellite-alone capability, tows counted at pair mass | pair-mass Δv requirement | literal reading of the brief; the ledger exposes the consequence | one parameter |
| D5 | 5-year decay hand-over as default disposal | 25-year, controlled reentry only | stricter current practice and conservative Δv; reentry is modelled as an alternative | switch `mode` |
| D6 | β = 0 eclipse for the battery | LTAN-specific eclipse | LTAN not fixed by the brief | easy |
| D7 | Log-linear 3-case density table | NRLMSISE-00 now | dependency-free; exposes solar-activity sensitivity | replace `physics.density` |
| D8 | Rigid capture + 1-D contact | multibody contact model | cheap cross-check; heavy model prepared in MuJoCo | supersede |
| D9 | Independent safety monitor as a separate class | guidance-internal check | the brief demands independently validated safety constraints | keep; harden |
| D10 | Equivalent-density shell for FEA | orthotropic sandwich | needs only a STEP and gives a load-transfer first look; flagged optimistic | replace model |
| D11 | Hollow-shell bus with one deck | detailed frame, internal webs | enough to size tanks and place equipment; webs conflicted with tanks | refine in CAD |
| D12 | No commercial part numbers yet | pick from memory | values from memory risk being wrong; datasheet verification is a deliberate later step | add `components.py` |
| D13 | Mocks for FEA, refusal for MuJoCo | mock both | Talos has maintained mocks; a mock rigid-body physics run would mislead | n/a |
| D14 | Keep the unflattering results (1 km passive safety, mass floor, solar-min throughput) | tune parameters until green | the brief says identify infeasible requirements | n/a |

---

# Part II — Detailed design description

Part I explained the modelling choices. Part II describes **the satellite itself**, subsystem by subsystem: what it is,
where it sits, how it is meant to work, why it was designed this way, and what is still undesigned. The geometry
numbers are read from the CadQuery model (`cad.py`, bounding boxes in mm, body frame of §2). Everything that is a
*concept* rather than something the code computes is marked **[concept]**; everything that is only a placeholder in
the model is marked **[placeholder]**.

## II.1 System overview

Auseklis is a ~400 kg, three-axis-stabilised servicer that (1) phases to a target in the same sun-synchronous plane,
(2) approaches it through four gates (10 km → 1 km → 100 m → 10 m), (3) inspects it, (4) captures it, (5) tows the pair
to a decaying orbit, (6) releases it and returns to ~550 km for the next target. Its design driver is **repeatability**:
propellant for ~4–14 removals, a robust capture mechanism, and an approach that is passively safe at every gate.

### Operating modes **[concept]**

| mode | purpose | what is active | power (model, W) |
|---|---|---|---:|
| launch | stowed, off | none (safing relays only) | 0 |
| LEOP / commissioning | detumble, deploy arrays, checkout | OBC, ADCS, S-band, heaters | ~190 (cruise) |
| cruise / phasing | drift or thrust to the target plane/phase | ADCS, GNSS, S-band, propulsion | 190 (or 300 + 800 electric) |
| far rendezvous (10 km → 1 km) | angles-only / relative GNSS navigation | cameras, GNSS, S-band, thrusters | 420 |
| near rendezvous + inspection (1 km → 10 m) | LiDAR/camera pose, fly-around, tumble estimation | all proximity sensors, independent safety monitor | 420 |
| capture | final approach, grasp | capture mechanism, force sensing, thrusters off or minimal | 480 |
| stabilisation / checkout | detumble the pair, verify the grasp | RCS, wheels, capture sensing | 480 |
| tow | lower perigee with the pair attached | main propulsion (Hall: +800 W) | 300 or 1100 |
| release + return | hand over, climb back | capture release, main propulsion | 300 |
| safe | loss of control or anomaly | OBC-B, sun-pointing, S-band, minimal heaters | 120 |

Mode power values are the `power_modes` table in `variants.py`; the transitions and entry/exit rules are not yet
specified.

## II.2 Configuration and layout

**Stowed vs deployed.** Only the *deployed* configuration is modelled. The envelope of each variant:

| | A arm | B docking | C gripper |
|---|---|---|---|
| bus (X × Y × Z) | 900 × 800 × 850 mm | same | same |
| aft extent (nozzle tips) | −680 mm | −680 mm | −540 mm |
| forward extent (capture tip) | +2526 mm | +640 mm | +1590 mm |
| array span (Y) | ±3680 mm | same | same |
| overall length (X) | 3.2 m | 1.3 m | 2.1 m |

**Equipment placement (identical in all variants unless noted).** Coordinates are bounding boxes in mm.

| item | X | Y | Z | placement reason |
|---|---|---|---|---|
| bus shell (25 mm walls) | −450…450 | −400…400 | −425…425 | ~0.6 m³; 118 L of wall material in the model |
| equipment deck | inside the shell | | z ≈ +170…+195 | carries avionics, battery and wheels above the tanks; keeps tanks low so the CoG sits near the geometric centre |
| avionics box | −150…150 | 50…270 | 182…292 | on the deck, +Y half |
| battery box | −130…130 | −260…−60 | 182…322 | on the deck, −Y half, separated from avionics (thermal, propagation) |
| reaction wheels (×4) | ±300 ±70 | ±280 ±70 | 182…227 | deck corners, spread for inertia authority; **drawn as flat cylinders, not a pyramid [placeholder]** |
| tanks | A: −400…364 | A: ±(15…325) | −390…−80 | low in the bus; A two tanks side by side, B/C one on the axis |
| X-band dish on a 230 mm mast | −75…225 (centre +0.15 X) | −350…−50 | 425…716 | on the zenith face away from the star trackers |
| S-band patches (×2) | −55…55 | 105…215 | ±(425…437) | zenith + nadir for near-omni coverage |
| star trackers (×2) | −315…−225 | ±(155…245) | 425…555 | zenith face, aft end, 90 mm baffles, spread in Y for redundancy against Sun/Earth/Moon blinding |
| LiDAR | 450…615 | 140…260 | 152…272 | +X face, looks at the target |
| wide camera | 450…530 | −230…−170 | 182…242 | +X face |
| narrow camera | 450…630 | −240…−160 | −252…−172 | +X face, long tube for the inspection range |
| thermal-IR camera | 450…520 | 172…228 | −240…−184 | +X face |
| capture mechanism | +X from 450 | on axis | −112…28 (A) | on the +X face so it points where all the sensors point |
| main engines / Hall thrusters | −680…−450 (A/B), −540…−450 (C) | ±(90…230) | ∓70 (A/B), −150…−20 (C) | on the aft face so thrust pushes toward +X |
| RCS thrusters (×8) | ±(392…428) | ±(398…452) | ±(367…403) | at the eight bus corners |

**Mass properties** (from `cad.mass_properties`, 400 kg wet, about the CoG; values are volume-weighted estimates):

| | CoG (x, y, z) m | Ixx | Iyy | Izz (kg m²) |
|---|---|---:|---:|---:|
| A | (+0.146, −0.005, −0.001) | 192 | 142 | 284 |
| B | (+0.028, 0.000, +0.007) | 189 | 57 | 200 |
| C | (+0.068, 0.000, 0.000) | 189 | 90 | 231 |

The arm moves A's CoG 15 cm forward, which matters for the thrust line (§II.4). The large Ixx and Izz come from the two
3.3 m solar wings; Iyy is small because the wings lie along Y. A 3.3 m wing at one end with a 400 kg body means the
array flexible modes will probably be low-frequency (well below the bus modes; to be computed, not assumed), which constrains the ADCS bandwidth during capture.
**[concept]** (not analysed).

## II.3 Structure

- **Concept:** an aluminium-honeycomb box with 25 mm sandwich panels, machined corner fittings, and a single equipment
  deck. The 52 kg allocation is ~13 % of the wet mass.
- **Why a closed box:** a box is the simplest manufacturable structure that offers panel-mounted equipment, a rigid
  interface, and straightforward modular access; the brief asked for "aluminium or composite panels with modular
  equipment mounting". Aluminium honeycomb was chosen over composite because it is lower risk and cheaper at low
  quantities; composite is a mass-saving option for later.
- **Equipment mounting:** the deck takes the avionics, battery and wheels; sensors on external faces; the tanks sit on
  the floor panel. Panels are the modularity boundary (an equipment panel can be integrated and tested separately).
- **Loads:** only a quasi-static FEA has been *prepared* (7 g axial, 3 g lateral, aft-face fixed, first mode ≥ 45 Hz, factor
  1.25 on yield). The equivalent-density model is optimistic for modal frequency (§13).
- **Open conflict [concept]:** the FEA fixes the **aft** face as the launch interface, but the aft face carries the main
  engines (A/B) or Hall thrusters (C). A real design needs the separation ring elsewhere (e.g. a side adapter or the −Z
  face) or a ring with the nozzles inside its aperture; the nozzle tips protruding 230 mm (A/B) rule out a simple
  ring-on-the-aft-face. This must be resolved with the launcher user guide.
- **Not designed:** the stowed array envelope, hold-down and release mechanisms, deployment hinges, thermal expansion
  interfaces, grounding, and fastener schedule.

## II.4 Propulsion subsystem

| | A | B | C |
|---|---|---|---|
| propellant | storable bipropellant (fuel + oxidiser) | ADN-based green monoprop | xenon (Hall) + green monoprop (RCS) |
| propellant mass at 400 kg wet | 98 kg | 121 kg | 84 kg Xe + 11 kg monoprop |
| tank volume | 40.8 L fuel + 49.9 L oxidiser | 107 L | 62 L Xe + 3 L RCS |
| main thrust | 44 N (2 × 22 N) | 22 N (4 × 5 N + 1 N class) | 54 mN (Hall, 800 W) |
| tank + hardware mass | 9.8 + 14 kg | 14.5 + 10 kg | 13.9 + 25 kg |
| delta-v at 400 kg wet | ~800 m/s | ~830 m/s | ~3.6 km/s |

**Layout choices [concept].** Tanks sit low in the bus and near the centre so the CoG moves little as propellant is
used (the CAD tanks are along X, 40–60 cm long, with hemispherical domes). Main engines on the aft face thrust along
+X, the direction of the capture axis, so a controlled burn is also the "push the pair" direction used by the tow. Variant A has
two engines 18 cm either side of the axis; the 15 cm forward CoG shift from the arm means the thrust line is offset from
the CoG and must be corrected by wheels and RCS during burns. Variant C carries a second Hall thruster as a **cold
spare**, not as extra thrust (the power budget supports one).

**Feed system [concept].** A: pressurant-fed, regulated, two tanks with propellant-management devices. B: blow-down or
regulated monoprop with a catalytic bed preheater. C: xenon flow controller and PPU for the Hall thruster, a separate
low-pressure green-monoprop RCS. The `tank_fraction` values cover tanks, pressurant and fittings; the 8–25 kg
hardware figures are allocations.

**Known deficiency in the CAD [placeholder].** The eight RCS thrusters are drawn as short cylinders along ±Y at the
corners. That gives lateral force and some torques, but **no ±X force authority and no clean ±Z force**, which
proximity operations require. A real RCS needs ≥ 12–16 thrusters in canted clusters (the usual six-DoF layout), or the
main engines must double as the forward/aft authority. The Δv ledger does not depend on the layout, but the layout
affects plume impingement on the target and the arrays; that is an unanalysed risk.

**Plume safety [concept].** Thrusters never fire toward the target inside 100 m; the capture axis (+X) has no thrusters;
main engines are aft; during tow the plume points away from the pair. Hall-thruster plume erosion of the arrays (the
thruster is 79–209 mm off axis, behind the arrays in X) needs a plume model.

## II.5 Electrical power

- **Generation:** two wings, three rigid panels each, 1000 × 1050 mm panels with 15 mm gaps and a 250 mm yoke, in the
  bus mid-plane (z = 0), 6.3 m² in total against the 6.16 m² required. One rotary drive per wing (a single-axis sun
  tracker) keeps the incidence loss near the 8 % assumed.
- **Why rigid panels:** at 6 m² the stowed folded stack is small enough for a rigid-panel design, which is more
  reliable and cheaper than flexible rolled arrays; mass is allocated at 3.6 kg/m².
- **Storage:** 791 Wh Li-ion (6.6 kg cells), sized for the β = 0 eclipse (35.6 min) at 30 % DoD. ~28 000 cycles in 5 years is the
  driver for the shallow DoD; it is conservative for cell chemistry such as NMC/NCA used in space-qualified packs.
- **Regulation [concept]:** a regulated 28 V bus from a PCDU with MPPT, battery charge/discharge regulators, latching
  current limiters, and prioritised load shedding (safe mode keeps the OBC, receiver, heaters). The 10 kg
  battery + PCDU allocation covers it.
- **Capture-time power:** the mode table shows 480 W at capture. The battery was sized at 380 W in eclipse (§6): the
  capture operation should be planned in sunlight or the battery enlarged by ~25 %.
- **Hall thruster** takes 800 W from a 1500 W EOL supply; with ~300 W bus load the margin is 400 W. Thrusting is
  restricted to sunlight and good pointing, which is the 0.65 duty used for the duration estimates.

## II.6 Avionics, software and fault management

- **Hardware [concept]:** two cross-strapped radiation-tolerant OBCs (one cold, one hot), a data-handling unit with
  the CAN/SpaceWire links to subsystems, a mass memory, and **a separate safety-monitor unit** (the 9 kg avionics
  allocation includes it). The safety monitor is a small independent processor with its own range sensor and power
  switch; it is the "independently validated safety" the brief asks for.
- **Safety monitor logic** (modelled in `SafetyMonitor`): speed vs range table (§10), approach corridor ±10° inside 100 m,
  and an abort command. An abort is a thrusters-off, retreat burn along a pre-planned direction that is passively
  safe; the retreat design is not done.
- **FDIR [concept]:** three layers — unit-level (watchdogs, limits), subsystem-level (redundancy switchover), and
  mission-level (abort to a safe relative orbit or to safe mode). Loss of navigation inside 100 m triggers an abort,
  not a re-acquisition attempt. No FMEA exists yet; it is the next reliability deliverable.
- **Autonomy:** autonomous collision avoidance is a requirement; the model implements only the speed/corridor monitor.
  Autonomous inspection, tumble-rate estimation and capture timing are not developed.
- **Software [concept]:** a flight-software architecture with a hard real-time layer (control, monitor) and a mode manager;
  not designed.

## II.7 Guidance, navigation and control

| phase | navigation sensors | actuators | note |
|---|---|---|---|
| cruise / phasing | GNSS + IMU + star trackers | wheels, magnetorquers, main propulsion | ground-supplied target ephemeris (TLE-class, km-level) |
| far (10 km → 1 km) | wide camera (angles-only), relative GNSS if cooperative, thermal-IR at night | RCS | observability of range from angles-only needs manoeuvres (e.g. a deliberate radial offset or co-elliptic hold) |
| near (1 km → 100 m) | narrow camera, LiDAR acquisition | RCS | passive-safety holds required at each gate |
| inspection (100 m → 10 m) | LiDAR point cloud, camera pose, thermal-IR | RCS, wheels | target shape and tumble estimation; fly-around |
| capture | LiDAR + camera + arm/force sensors | arm + RCS-off | closing speed 2–10 cm/s |

- **Star trackers** (2) are on the zenith face to see deep space and avoid Earth limb; two for redundancy against Sun/Moon
  blinding. **GNSS** antennas (two) are on the zenith face with the S-band patches **[concept]** (not drawn).
- **Wheels:** four in a layout that gives three-axis control with one failed wheel; momentum storage and torque are not
  sized. During capture the combined system can have ~32 N m s of angular momentum (500 kg, 3 °/s), which exceeds a
  typical small wheel set's storage, so the RCS detumble (§11) is the baseline and wheels trim.
- **Magnetorquers** (3) desaturate the wheels in cruise; at 550 km the field is adequate.
- **Control during tow:** the pair is a 900 kg body with a different CoG and inertia; the controller needs the new inertia
  (from the capture-time estimation) and a flexible-mode notch for the arrays.
- **Why the camera and LiDAR are redundant by design:** the brief requires LiDAR and cameras; cameras give the long-range
  angles-only estimate and are the fallback if the LiDAR is blinded or the target is glinting.

## II.8 Communications

- **S-band TT&C:** two transponders on separate patches (zenith and nadir) give near-omnidirectional coverage so the
  command link survives loss of attitude. Rate of order 10–100 kbit/s [concept]; no link budget exists.
- **X-band data downlink (optional):** a 300 mm dish on a 230 mm mast on the zenith face, for downlinking inspection imagery
  that is too large for S-band. A fixed dish needs attitude pointing to the ground station; a gimbal is not modelled.
- **Why the dish moved:** it originally overlapped a star tracker's baffle; the mast is now at x = +0.15 X.
- **Not designed:** ground segment, contact schedule, autonomy during communication gaps. The ~35 min per orbit with no
  ground contact makes autonomy the practical requirement.

## II.9 Thermal design

- **Concept [concept]:** MLI blanket on the whole bus; radiator area on the bus faces that see cold space most of the orbit (which faces
  depends on the LTAN and array orientation, to be decided by the thermal analysis); heaters on tanks, battery and the capture
  mechanism; thermistors and thermostats as independent protection. The 13 kg allocation matches that.
- **What exists:** `equilibrium_temperature`, a single-node steady-state balance. It is not yet applied to this design.
- **Hot and cold cases to analyse:** a hot case with the Hall thruster firing in sunlight and the battery charging; a cold case
  in safe mode in eclipse with minimum heater power; both over the 5-year degradation of coatings.
- **Specific concerns:** the propellant tanks need temperature control (green monopropellants have minimum storage
  temperatures that must be taken from the chosen propellant's datasheet); the arm joint lubricants; LiDAR and camera
  focal planes.

## II.10 Capture mechanisms in detail

### A — robotic arm (`capture_arm_*`, `capture_gripper`)

- **Geometry as modelled:** a 120 mm base on the +X face, two 900 mm links (half of `arm_length` 1800 mm each), spherical
  elbow and wrist joints, and a two-finger end effector (220 mm long fingers, 140 mm apart). The second link is raised 25°
  above the first so that the tip rises 386 mm. The tip is 2.08 m ahead of the +X face.
- **Allocation vs drawing:** the 38 kg mass allocation is for a "7-DoF-class" arm with controller; **the CAD shows only a
  2-link arm with two fingers, a schematic of one posture**, not the full kinematic chain.
- **Why an arm:** a tumbling uncooperative target needs standoff (the satellite stays ≥ 2 m away at the first touch),
  reach, and compliance to absorb residual relative motion; capture impulse and angular momentum are then shared
  through an arm of known stiffness.
- **Design handles:** arm compliance (model: 2 kN/m, 300 N·s/m) and closing speed (5 cm/s) set the peak contact force
  (27 N). Fingers must tolerate the diameter ranges and fixture geometries of the targets.
- **Risks:** mechanism complexity, joint failure modes, control around flexible arrays, and thermal gradients on the
  arm. It is the highest mass and the lowest margin variant.

### B — docking adapter (`docking_ring`, `docking_cone`)

- **Geometry:** a 180 mm-radius, 40 mm-thick ring with 140 mm bore, and a 170 mm-deep cone narrowing from r 175 to 60 mm,
  cut hollow with a 150 → 40 mm inner profile. The radius is limited by the sensors, which sit ≥ 205 mm off axis.
- **Why this size:** the interface has to stay inside the sensor ring and still be large enough to guide a cooperative
  fixture. It is not a standard interface (no IDSS/iSSI geometry is claimed); selecting a standard is an open item.
- **Constraint:** it works only with targets that carry a compatible fixture, which is why B is cooperative-only in
  every calculation.
- **Strength:** lightest (20 kg) and most mature; the capture can be done with a soft-dock and latch sequence at
  very low closing speed.

### C — compliant gripper (`capture_boom`, `capture_hub`, `capture_gripper`)

- **Geometry:** an 810 mm boom (0.45 × `arm_length`), a 140 mm hub and three 260 mm fingers spaced at 120° around the axis, with the
  tip 1.59 m ahead of the +X face.
- **Why a boom + three-finger gripper:** a shorter reach than the arm, no joints (lower risk, 26 kg), and the three fingers
  grasp a cylindrical or conical feature (a launch-adapter ring or nozzle) — the most common uncooperative feature on
  upper stages. The "compliant" part (finger flexure and a damped boom) is an allocation; no mechanism is modelled.
- **Limit:** it needs a target feature within reach and cannot adapt to an arbitrary tumbling attitude as an arm can.

### Capture sequence [concept]

Inspect → match the target's rotation about its main axis (or approach along the spin axis) → final approach at ≤ 2–10 cm/s with the
monitor armed → contact → grasp (force-limited) → stabilise (RCS detumble, wheels trim) → checkout (grasp verified by force and
pose) → mode change to tow. A failed grasp triggers a retreat to the 10 m gate rather than a retry in contact.

## II.11 Mass roll-up (400 kg wet, kg)

| item | A | B | C |
|---|---:|---:|---:|
| structure | 52.0 | 52.0 | 52.0 |
| thermal | 13.0 | 13.0 | 13.0 |
| harness | 22.0 | 22.0 | 22.0 |
| avionics | 9.0 | 9.0 | 9.0 |
| GNC | 24.0 | 24.0 | 24.0 |
| proximity sensors | 14.0 | 14.0 | 14.0 |
| comms | 9.0 | 9.0 | 9.0 |
| battery + PCDU | 10.0 | 10.0 | 10.0 |
| solar array | 22.2 | 22.2 | 22.2 |
| battery cells | 6.6 | 6.6 | 6.6 |
| capture | 38.0 | 20.0 | 26.0 |
| propulsion hardware | 14.0 | 10.0 | 25.0 |
| **dry nominal** | **233.8** | **211.8** | **232.8** |
| growth margin 25 % | +58.4 | +52.9 | +58.2 |
| tanks | 9.8 | 14.5 | 13.9 |
| propellant | 98.0 | 120.8 | 84.1 + 11.0 |
| **wet** | **400.0** | **400.0** | **400.0** |

The margin applies to the dry masses before adding tanks and propellant. Note how much of the wet mass is *allowance*
(growth margin ≈ 58 kg): it is the main buffer against datasheet masses coming in higher.

## II.12 Requirement verification matrix

| requirement | verified by | state |
|---|---|---|
| wet mass 350–500 kg | budget, CAD mass properties | calculated; B/C below 350 kg if sized to 500 m/s |
| 550 km, 97.6° | J2 sun-synchronous check | calculated (97.59°) |
| 5-year life | array degradation; campaign duration | calculated; no reliability model |
| Δv 500 m/s incl. reserves | variant sizing, campaign reserve | calculated; interpretation in §5 |
| 1.5 kW EOL | array sizing | calculated (6.16 m² vs 6.3 m² drawn) |
| eclipse-sized Li-ion | battery sizing | calculated; mismatch with rendezvous loads (§6) |
| ADCS: wheels, MTQ, thrusters | allocation | **not analysed** |
| navigation: GNSS, IMU, ST, cameras, LiDAR | sensor-class σ table | assumed; **EKF needed** |
| comms: S-band, X-band | allocation | **no link budget** |
| redundant computing | architecture concept | not analysed |
| thermal control | allocation, equilibrium function | **not analysed** |
| structure/modular mounting | CAD, prepared FEA | FEA not run; interface conflict (§II.3) |
| capture comparison | three variants + momentum/contact | partial |
| disposal | hand-over and reentry ledgers | calculated; satellite's own disposal not computed |
| FDIR, collision avoidance | safety monitor model | partial |
| approach at 10 km, 1 km, 100 m, 10 m | gate table | calculated; 1 km fails passive safety |
| mission profiles 100/250/500 kg | mission ledgers | calculated (upper bound) |

## II.13 Top design risks and the order to attack them

1. **Atmosphere and disposal policy** drive removals/life by 5× (§9). Mitigation: real model and an explicit policy.
2. **Capture of an uncooperative tumbling stage** is physically hard and only schematic here. Mitigation: gripper
   contact model, test on air-bearing or robotic hardware-in-the-loop, target feature survey.
3. **Launch interface conflicts with the aft engines** (§II.3). Mitigation: choose the launcher and interface early.
4. **RCS layout cannot translate along ±X/±Z** (§II.4). Mitigation: a full six-DoF thruster layout with plume analysis.
5. **1 km passive safety** shortfall. Mitigation: better velocity estimate, safety-ellipse hold, tighter gate spacing.
6. **Battery/operations mismatch** (§6). Mitigation: size to the worst operating mode or constrain operations to sunlight.
7. **Array flexible modes** coupled with capture control. Mitigation: modal analysis, notch filter, or a stiffer array for the capture phase.
8. **Dry-mass growth**: ~58 kg of margin protects it, but nothing in the numbers is datasheet-based yet.
9. **Propellant management for green monoprop** (freezing, catalyst bed life) and Hall thruster plume on the arrays.
10. **No FMEA or reliability model**, so the 5-year, multi-removal claim has no failure data behind it.
