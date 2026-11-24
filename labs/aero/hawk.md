# HAWK 2 — the ASTOR-TWIN interceptor, as it stands

State of the design on 2026-11-24, branch `dev_astor_hawk`.

HAWK is a compact, autonomous, fixed-wing aircraft that catches small quadcopters by
firing a net at them. In the code it is **ASTOR-TWIN** (`labs/aero/designs/astor_twin.py`),
the default design of `labs/aero/sims/astor.sh`.

**HAWK 2** is the aircraft with the wing doubled for margin (2026-11-24). **HAWK1** is the
aircraft before that (span 1.38 m); it stays in the code as a named variant because all
the CFD so far was run on it.

How to read the numbers:

- **CFD** — screening-level RANS on the user's machine. Coarse meshes, all failing the
  quality check, about half the angles converged. Treat as ±15 %. **Every CFD number
  here is HAWK1's.**
- **Carried over** — HAWK 2's aerodynamics are HAWK1's CFD scaled by area. The wing is
  HAWK1's enlarged 1.414 times in every length, with the same section, flaps and aspect
  ratio. This holds until HAWK 2's own CFD is run.
- **Model** — the flight model (`astor_v_flight`) or the MuJoCo simulation built on it.
- **HAND** — an estimate with no model or catalogue behind it. The weakest numbers.

**Nothing below has been confirmed by a HAWK 2 CFD, FEA, stability run or campaign.**
Those are prepared for the user to run (section 11).

---

## 1. What it is

| | HAWK 2 | HAWK1, for comparison |
|---|---|---|
| Mass, ready to fly | 8.04 kg (HAND budget) | 6.92 kg |
| Span / length / height | 1.95 m / 1.11 m / 0.36 m | 1.38 m / 0.88 m |
| Stowed span | 1.32 m | — |
| Wing area | 0.513 m² | 0.257 m² |
| Wing loading | 15.7 kg/m² | 26.9 kg/m² |
| Power | 2 × 4 kW, 13-inch variable-pitch propellers (HAND) | 2 × 4 kW, 11-inch fixed |
| Static thrust / weight | 2.56 (HAND) | 1.84 |
| Stall, flaps up / down | 16.5 / 14.8 m/s | 21.6 / 19.4 m/s |
| g limit | 8 | 8 |
| Nets | 3 | 3 |

Pack: 12S 5000 mAh LiPo, 222 Wh, 1.52 kg. Launch by catapult, recovery by parachute.

## 2. Why the wing was doubled

HAWK1 was sized at the minimum. A sizing study (`astor_twin_margin.py`) set seven margin
targets and HAWK1 missed six. Doubling the wing was the change that met the most.

| | HAWK1 | HAWK 2 |
|---|---|---|
| Closing speed on an 8 / 14 m/s quadcopter | 17 / 11 m/s | 11 / 5 m/s |
| Turn radius at 25 m/s | 52 m | 25 m |
| Turn radius at 30 m/s | 45 m | 24 m |
| Load factor at 40 m/s | 4.1 g | 7.0 g |
| 8 g reached from | 60 m/s | 50 m/s |
| Top speed | about 90 m/s | about 81 m/s |

The top speed was **not** kept. See section 9.

## 3. Airframe

**Wing.** One tapered trapezoid at 12° of leading-edge sweep: 368 mm chord at the centre
to 165 mm at the tip, half span 977 mm. Section 12 % thick with 4 % camber.

- **Inner piece** through the fuselage, carrying both motors, with plain flaps (30 % chord)
  behind the propellers.
- **Outer pieces** from 420 mm out, each with a flaperon along 5–96 % of its length.
- **Swing-back pivots** at the joint, near the trailing edge. Swung back 90° the outer
  pieces lie along the body, outboard of the inner wing: 1.32 m across instead of 1.95 m.
  This is for the transport box. In flight the wing stays spread.

**Motors.** Two nacelles on the inner wing, 225 mm either side of the centre. Disc
clearances: 25 mm to the fuselage, 120 mm between the discs, 30 mm inside the pivot.

**Fuselage.** 940 mm long (150 mm longer than HAWK1, added at the tail), 70 mm diameter.

**Tail.** Two all-moving halves (365 mm each, 182 → 95 mm chord), mounted **low** on the
body's sides, plus one upright fin (236 mm high) with a rudder. The halves move together
for pitch and against each other for roll. Low, they sit in the propeller wash, below the
wing's wake at high angles, and under the swung-back wing.

## 4. Net launcher

GAVILAN's recoilless launcher (`gavilan_flight.NET`): one tube, three cartridges.

| | |
|---|---|
| Net | 2.4 m square, 150 g, leaves at 40 m/s, 2.95 m across open |
| Effective range | 4–10 m (its own test: 22 of 24 at a hovering quadcopter) |
| Tube travel | level to **45° down** (was 15°); the mount also pans ±35° (HAND) |
| Air vessel | 1.0 L at 30 bar inside the tail cone, stations 535–915 mm |
| Mass | 2.04 kg with three nets, plus 90 g for the pan-tilt mount (both HAND) |

## 5. Flight controller

- Angle-of-attack limiter that follows the flap position.
- Yaw damper and pitch damper. Without the yaw damper HAWK1's Dutch roll was barely damped.
- 8 g limit.
- **Flap schedule:** reflexed 6° up below 2° angle of attack for fast flight, neutral by
  4°, 20° down on command for slow tight turning.
- **Crow brake** (flaperons up, inboard flaps down) when ASTOR is faster than commanded
  in the attack. Its drag is a textbook estimate until CFD.
- **Differential thrust** for yaw, mixed with the rudder.

## 6. What the flight model now includes

| | Source |
|---|---|
| Lift, drag, moment, flaps up and down | HAWK1's CFD, carried over by area |
| Lift slope and moments | HAWK 2's own lattice plus HAWK1's CFD-to-lattice difference |
| Static margin | 11 % |
| Thrust: 202 N static, 142 N at 30 m/s, 84 N at 60 m/s | HAND for the low-speed end |
| Propeller-wash lift on the inner wing | HAND (momentum theory × 0.8) |
| Crow brake drag, differential thrust | HAND |

Aerodynamics beyond 24° angle of attack are a flat-plate blend, not data.

## 7. Structure

The 8 g limit had nothing sized behind it. The wing spar now has a closed-form sizing
(`astor_twin_fea.reference`) and an FEA prepared but **not run**.

| | Root | Pivot |
|---|---|---|
| Tube | 25 × 22 mm carbon | 16 × 13 mm carbon |
| Bending moment at 8 g × 1.5 | 166 N·m | 51 N·m |
| Stress against 400 MPa allowable | 270 MPa | 223 MPa |
| Margin | +0.48 | +0.79 |

Tip deflection at ultimate load is 87 mm (9 % of the half span). Spars weigh 262 g for
both wings. Not covered: the pivot's lugs and lock, skin and ribs, the tail spigot, the
fin, the launcher trunnion, flutter.

## 8. The attack

`astor_netshot.py`: MuJoCo aircraft, a tracker with 0.1 s latency and 0.3 m noise, the
net reduced to the launcher's measured numbers.

- **Straight targets:** pursuit onto a point trailing the quadcopter.
- **Weaving:** the tracker fits a mean line plus the weave; ASTOR flies to where the
  quadcopter will be.
- **Jinking:** ASTOR chases directly, flaps down, up to 68° of bank, rolling level for
  the last 28 m.
- **Turn back after a miss:** flaps down at the steepest bank the lift allows.
- **Abort:** the attack ends when 70 % of the pack's usable energy is spent.

Two optional tactics exist, **off by default and never flown**: starting high and diving
(`pounce_m`), and a quadcopter that only reacts when ASTOR is outside a cone behind it
(`blind_cone_deg`).

**Results so far**

| | |
|---|---|
| Last campaign (HAWK1, 60 sorties, before the jinking chase) | 46 of 60 = 77 %; straight 18/18, weaving 17/17, jinking 11/25 |
| HAWK 2, single debug sorties | straight, weaving and jinking: 3 of 3 netted on the first pass |

Three debug sorties are a smoke test, not a rate. The target is 80 %.

## 9. What was traded, and the contradictions settled

- **Top speed was not kept.** Doubling the wing adds drag that thrust at speed cannot
  cover: about 81 m/s against 90. The variable-pitch propellers protect the fast end of
  the thrust but cannot add to it. More power would recover it.
- **Folding is for the box, not for speed.** Earlier work showed sweeping the wing in
  flight bought nothing (HAWK1's swept states had the same drag and less lift). That
  still stands. Because the pivots now exist, one CFD case with the outer pieces swept
  to 60° is included to measure it on the big wing rather than assume.
- **The pack now has room.** HAWK1 left 215 mm of body ahead of the air vessel, which a
  12S 5000 mAh pack did not fit. The longer tail moves the vessel aft: 365 mm.
- **The launcher mount is in the mass budget.** It was not.
- **The tube can now follow a target in a turn or flare** (45° of travel, was 15°).
- **Flight model drag was overstated** for the cambered wing by about 20 % (corrected).
- **The CFD phase used to change the design list** for the phases after it. Fixed.

## 10. What is not known

- **No HAWK 2 CFD, FEA, stability run or campaign exists.**
- **Variable-pitch propellers** are not in the propulsion library. Static thrust could
  be 2.3 or 2.8 times weight rather than 2.56.
- **The braking flare** (flying nose-high at the target's speed) is not implemented.
  The model has no measured aerodynamics above 24°. The high-angle CFD cases and the
  propeller-wash model are the groundwork for it.
- **Mass** is 8.04 kg on HAND estimates, 0.6 kg more than the sizing study assumed.
- **Tail spigot** crosses the body low, where the air vessel sits. The vessel must be
  notched or shortened there. Not drawn.
- **Full power lasts 1.4 minutes.**
- **The net** has only been tested from a fixed launcher at a hovering quadcopter.
  HAWK 2 closes at 5–11 m/s instead of 11–17 m/s, which is nearer those conditions but
  still not them.
- **Motor controllers** are assumed in the nacelles; the refill bottle has no place.

## 11. To run

Discard your local copy of the CFD data file once, then pull:

```bash
git checkout -- labs/aero/designs/data/astor_v_cfd_screening.yaml
```

```bash
git pull origin dev_astor_hawk
```

CFD of HAWK 2, 38 cases: clean and flaps-down to 60°, reflexed, crow brake, swept 60°.

```bash
bash labs/aero/sims/astor.sh -j 14 --cfd-only
```

FEA of the wing spar at 8 g, 3 cases (needs CalculiX and Gmsh):

```bash
bash labs/aero/sims/astor_fea.sh -j 8
```

Stability, performance card and the 60 sorties, after the CFD:

```bash
bash labs/aero/sims/astor.sh -j 14 --skip-cfd
```

Every run is restartable, shows a progress bar, has `--help` and `--dry-run`, and writes
one timestamped YAML.

## 12. Files

| | |
|---|---|
| Design (CAD, mass, checks) | `labs/aero/designs/astor_twin.py` |
| Flight model | `labs/aero/designs/astor_v_flight.py` |
| CFD cases | `labs/aero/designs/astor_v_cfd.py` |
| Spar sizing and FEA cases | `labs/aero/designs/astor_twin_fea.py` |
| Net-shot sortie | `labs/aero/designs/astor_netshot.py` |
| Sizing study | `labs/aero/designs/astor_twin_margin.py` |
| Drawings and STEP files | `labs/aero/designs/data/astor_twin/hawk2_*` |
