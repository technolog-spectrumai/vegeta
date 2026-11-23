# HAWK — the ASTOR-TWIN interceptor, as it stands

State of the design on 2026-11-23, branch `dev_astor_hawk`, commit `a509fcd`.

HAWK is a compact, autonomous, fixed-wing aircraft that catches small quadcopters by
firing a net at them. In the code it is called **ASTOR-TWIN** (`labs/aero/designs/astor_twin.py`);
it is the default design of `labs/aero/sims/astor.sh`.

How to read the numbers:

- **CFD** — screening-level RANS run on the user's machine. Coarse meshes (about 130,000
  cells), every mesh failed its quality check, about half the angles converged. Treat as ±15 %.
- **Model** — the flight model (`astor_v_flight`) or the 6-DOF MuJoCo simulation built on it.
- **HAND** — an estimate with no model or catalogue behind it. These are the weakest numbers.

---

## 1. What it is

| | |
|---|---|
| Role | Chase a quadcopter of 1–2 kg and net it; three nets per sortie |
| Layout | Mid-wing monoplane, two tractor propellers on the wing, all-moving tail, one fin |
| Mass, ready to fly | 6.92 kg with launcher, three nets and pack (HAND budget) |
| Span / length / height | 1.38 m / 0.88 m / 0.40 m |
| Wing area | 0.257 m² (wing loading 26.9 kg/m²) |
| Power | 2 × 4 kW electric motors, 11-inch folding propellers |
| Pack | 12S 5000 mAh LiPo, 222 Wh, 1.52 kg (HAND: two 12S 2500 mAh in parallel) |
| Launch | Catapult or rail at about 28 m/s; a hand launch fails (model) |
| Recovery | Parachute |

## 2. Airframe

**Fuselage.** 790 mm long, 70 mm diameter, nose camera at the tip. No motor on the nose
or tail.

**Wing.** One plain tapered trapezoid, fixed at 12° of leading-edge sweep.

- Chord 260 mm at the centre plane to 117 mm at the tip; half span 691 mm.
- Section 12 % thick with 4 % camber. **This section is under review**, see section 8.
- Built in three pieces: an inner piece through the fuselage carrying both motors, and two
  outer pieces that plug onto joiner tubes 235 mm from the centre (for transport).
- **Flaperons** along 5–96 % of each outer piece, 30 % of the chord: roll control, and
  drooped together as flaps.
- **Inboard plain flaps** on the inner pieces, 30 % chord, directly behind the propellers.

**Motors.** Two nacelles on the inner wing, 190 mm either side of the centre. The propeller
discs clear the fuselage side by 15 mm and each other by 101 mm.

**Tail.** Two all-moving flat halves (270 mm each, 135 → 70 mm chord) that move together
for pitch and against each other for roll, plus one upright fin on the body's back
(175 mm high) with a rudder.

**What it no longer has.** The wing does not swing or fold, there are no fins under the
tail, no legs or talons, no tube launch and no dropped nets. Those belonged to the earlier
ASTOR-V and ASTOR-T, which are still in the code (`--design v | boost | t`).

## 3. Net launcher

GAVILAN's recoilless net launcher (`gavilan_flight.NET`), one tube with three cartridges.

| | |
|---|---|
| Net | 2.4 m square, 150 g cartridge, leaves at 40 m/s, 2.95 m across when open |
| Effective range | 4–10 m (the launcher's own test: 22 captures in 24 at a hovering quadcopter) |
| Tube | 300 mm, under the forward belly, on a trunnion 447 mm from the nose |
| Tube travel | Level to 15° down; the mount also pans ±35° (HAND servo mount, 150 °/s) |
| Air vessel | 1.0 L at 30 bar, **inside the fuselage**, shaped to the tail cone (stations 385–765 mm, 64 mm at its widest) |
| Refill | 0.2 L, 300 bar bottle for the second and third shots (HAND; no place assigned yet) |
| Launcher mass | 2.04 kg with three nets, reload feed and gas (HAND) |

In cruise the tube lies level: CFD gives 10 cm² less drag area than with it 15° down.

## 4. Aerodynamics

All from CFD, on the aircraft's own wing area.

| | Flaps up | All flaps down 20° |
|---|---|---|
| Maximum lift coefficient | 0.92 | 1.14 |
| Stall speed at 6.92 kg | 21.6 m/s | 19.4 m/s |
| Angle of maximum lift | 14° | about 4°, then flat to 16° |

- Flaps add 0.65 to the lift at a fixed angle.
- Static margin 10 % with the centre of gravity at 385 mm from the nose.
- Level-flight drag at 6.92 kg: 7.9 N at 30 m/s, 30 N at 60 m/s, 55 N at 80 m/s.
- Thrust (propulsion library, both propellers): 125 N static, 108 N at 30 m/s, 84 N at
  60 m/s, 68 N at 80 m/s.

## 5. Flight controller

- **Angle-of-attack limiter** under every elevator command; it follows the flap position.
- **Yaw damper** on the rudder, with the steady rate of a turn washed out.
- **Pitch-rate damper** on the elevator.
- **g limit: 8 g.** The user's choice. 100 g of structure is budgeted for it (HAND);
  nothing is structurally sized.

## 6. Stability and handling (model, on CFD)

From the user's stability run at commit `4576ec5` (5 g limit at that time), dampers on:

| | 30 m/s | 45 m/s | 60 m/s | 24 m/s, flaps down |
|---|---|---|---|---|
| Dutch-roll damping | 0.25 | 0.35 | 0.46 | 0.17 |
| Short-period damping | 0.71 | 0.62 | 0.62 | 0.77 |
| Roll rate, first turn | 446 °/s | 690 °/s | 933 °/s | 304 °/s |
| Spiral | stable | stable | stable | stable |

Without the dampers the Dutch roll is barely damped (0.04–0.06); the yaw damper is
required, not optional. No test stalled the aircraft under the limiter.

## 7. Performance

**Turning** (point-mass on the CFD tables, 8 g limit, flaps used where they help):

| Speed | 25 m/s | 30 m/s | 40 m/s | 50 m/s | 60 m/s | 80 m/s |
|---|---|---|---|---|---|---|
| Load factor | 1.6 g | 2.3 g | 4.1 g | 6.4 g | 8.0 g | 6.3 g |
| Turn radius | 52 m | 45 m | 41 m | 41 m | 46 m | 104 m |
| What limits it | lift | lift | lift | lift | structure | thrust |

**Performance card** (model, user's run at commit `4576ec5`, flaps not used by the card):

| | |
|---|---|
| Top speed | 80.7 m/s |
| Best climb | 54 m/s at 45 m/s |
| Endurance | 29 min at 25.6 m/s; range 45 km |
| Time at full power | 1.2 min |
| Catapult launch | safe in 1.6 s from 27.5 m/s |

The card's mission radius (0.4 km economy) and the 10 km reserve (−27 %) are poor. They
come from the cambered wing's drag in fast flight, see section 8.

## 8. Open question: the wing section

CFD shows the 4 % cambered wing costs about 25 % more drag than the earlier thin wing
(8 %, 2 % camber) from 45 m/s upward, for only +0.06 of maximum lift. The flaps are what
bring the low-speed lift (+0.20).

| Speed | Thin wing | Current wing | Difference |
|---|---|---|---|
| 30 m/s | 8.4 N | 7.9 N | −6 % |
| 45 m/s | 13.0 N | 16.3 N | +25 % |
| 60 m/s | 23.6 N | 30.3 N | +28 % |
| 80 m/s | 44.7 N | 55.3 N | +24 % |

Two fixes are prepared as CFD cases and **not yet run**: the same 12 % wing with 2 %
camber (flaps up and down), and the current wing with every flap reflexed 6° up in fast
flight. The wing will be chosen on those results.

```bash
bash labs/aero/sims/astor.sh -j 14 --section-cfd
```

## 9. The attack

Simulated in `astor_netshot.py`: MuJoCo 6-DOF aircraft, a tracker with 0.1 s latency and
0.3 m noise, and the net reduced to the launcher's measured numbers.

1. ASTOR comes up behind the quadcopter along its track, slightly above it, at 1.3 times
   its stall speed (about 28 m/s).
2. The mount pans and tilts the tube onto the predicted meeting point.
3. It fires when the net is predicted to pass within 60 % of its half span, between 4.5
   and 9.5 m of flight.
4. After a miss it turns back (about 24 s) and fires the next net.

Three behaviours are handled differently:

- **Straight:** proportional navigation onto a point trailing the quadcopter.
- **Weaving:** the tracker fits a mean line plus the weave and ASTOR flies to where the
  quadcopter will be on arrival.
- **Jinking:** once the quadcopter's heading leaves its mean track by more than 25°, ASTOR
  chases it directly with flaps down, up to 62° of bank, and rolls level for the last 28 m.

**Result of the last 60-sortie campaign** (user's run, commit `4576ec5`, before the
jinking chase existed):

| Quadcopter | Netted | Rate |
|---|---|---|
| Straight | 18 of 18 | 100 % |
| Weaving | 17 of 17 | 100 % |
| Jinking | 11 of 25 | 44 % |
| **All** | **46 of 60** | **77 %** |

ASTOR was never lost. The target is 80 %. With the jinking chase, 9 of 13 jinking debug
sorties were netted (7 of 10 that had failed in the campaign); the campaign has not been
re-run with it.

## 10. What is not yet known or not done

**Design**

- The wing section (section 8).
- **Internal layout.** No item-by-item balance exists. The pack has about 215 mm of body
  ahead of the air vessel and its fit in a 70 mm fuselage is unchecked. The motor
  controllers are assumed to sit in the nacelles. The refill bottle has no place.
- **Tail spigot.** It crosses the body where the air vessel sits and must pass over or
  under it; not drawn.
- **Structure.** Nothing is sized for 8 g.
- **Mass.** The whole budget is HAND estimates on top of the earlier ASTOR-T budget.
- **Pan-tilt mount.** Its mass is not in the budget.
- **Mount travel.** The tube cannot point above the wing plane, so ASTOR cannot shoot while
  turning hard toward a target.

**Aerodynamics and control**

- All CFD is screening level.
- The lift the propellers' slipstream adds to the inner wing is not in the flight model.
  A HAND estimate says it could halve the low-speed turn radius at full throttle.
- The approach guidance uses the flaps only against jinking targets; the performance card
  does not use them at all.
- At the 25.6 m/s best-endurance speed a 2.7 m/s gust reaches the stall; it should loiter
  at 30 m/s or more.
- The stability output wrongly flags "stalled" with flaps down above 3°.

**Net**

- The net has only been tested from a fixed launcher at a hovering quadcopter. Fired from
  an aircraft at 28 m/s it leaves at about 68 m/s through the air. This is the largest
  untested step in the whole concept.
- A netted quadcopter is counted as a win; what it does after is a simple fall.

## 11. Files and commands

| | |
|---|---|
| Design (CAD, mass, checks) | `labs/aero/designs/astor_twin.py` |
| Flight model | `labs/aero/designs/astor_v_flight.py` |
| CFD cases | `labs/aero/designs/astor_v_cfd.py` |
| Net-shot sortie | `labs/aero/designs/astor_netshot.py` |
| Turning study | `labs/aero/designs/astor_twin_agility.py` |
| Launcher for every simulation | `labs/aero/sims/astor.sh` |
| Drawings, STEP, STL, result files | `labs/aero/designs/data/astor_twin/` |

```bash
bash labs/aero/sims/astor.sh -j 14                  # CFD, stability, 60 net-shot sorties
```

```bash
bash labs/aero/sims/astor.sh -j 14 --netshot-only   # the sorties alone
```

```bash
bash labs/aero/sims/astor.sh -j 14 --section-cfd    # the wing-section question, 24 cases
```

Every run is restartable, shows a progress bar and writes one timestamped YAML.
