# Kestrel-4Q-3 — design brief, first principles to technicals

Status: proposal, 2026-10-17. Bought parts: four motors, four propellers, four ESCs, one battery, the flight
stack, four servos, two carbon spar tubes. Everything else is printed. Numbers below are first-order sizing; the
notebooks (38 → 42) are where they get checked, and the fine CFD sweep in 39 decides the wing area (§ 3).

## 1. Mission → requirements

| Requirement | Value | Why |
|---|---|---|
| Launch / land | vertical, from a 1 m² pad, no runway | tailsitter, four rotors |
| Loiter | ≥ 30 min at ≤ 16 m/s | battery energy ÷ cruise power |
| Dash | ≥ 30 m/s for ≥ 2 min | pursuit; thrust-limited speed = prop pitch speed |
| Turn | corner radius ≤ 12 m at 16 m/s | 3 g structural, CL max limited |
| Payload | 150 g in a removable 90 g capsule, plus camera | as 4Q-1/2 |
| Transitions | forward and back with ≤ 3 m height change, no stall, no ballooning | notebook 42's criteria |
| Take-off mass | ≤ 2.0 kg | keeps hover under 250 W electrical on 2216-class motors |

## 2. Why this configuration

- Four rotors give hover and attitude control with no tilt mechanism and no servo in the hover loop.
- Two wings in a biplane put 0.38 m² of lift in a 1.2 m span and give the four posts a structural box.
- Tailsitting means the same motors serve hover and cruise; the price is the back transition, which 4Q-2 failed
  by ballooning. 4Q-3 pays that price with reverse thrust on the lower pair and crow-brake elevons.
- Faired posts and no X-bracing (4Q-2) because Cd0 drops from 0.067 to 0.023 and endurance nearly doubles.

## 3. Sizing

**Mass budget (kg).** motors 4×0.066, props 4×0.018, ESCs 4×0.014, battery 0.36, flight stack 0.085, camera 0.045,
servos 4×0.012, wiring 0.05, pod 0.22, wings 0.34, posts/nacelles/fins 0.14, capsule 0.09, payload 0.15 →
**1.92 kg (18.8 N)**.

**Wing area from stall speed.** W/S = ½ρV_s²·CL max. The campaign measured CL max 0.52–0.53 (screening mesh);
the model assumed 0.81. Design to the measurement:

| CL max | S = 0.28 m² (span 1.0) | S = 0.384 m² (span 1.2, chord 0.16) |
|---|---|---|
| 0.55 (measured) | V_s 14.1 m/s, cruise 16 is 1.13 V_s | **V_s 12.1 m/s, cruise 16 is 1.3 V_s** |
| 0.80 (model) | V_s 11.7 | V_s 10.0 |

So 4Q-3 is span 1.2 m, chord 0.16 m, gap 0.30 m, two wings → S 0.384 m², AR 7.5 per wing. If the fine sweep
recovers CL max ≥ 0.75, drop back to span 1.1 / chord 0.15.

**Hover.** Four 10-inch discs, A = 0.203 m², disc loading 93 N/m², induced velocity 6.2 m/s, ideal power 116 W;
at FM 0.6 and drive efficiency 0.8 → **~240 W electrical, 60 W per motor**, hover throttle ≈ 0.55, T/W ≈ 2.5.

**Cruise.** With Cd0 0.025 on S_ref 0.384 and e 0.8:

| V [m/s] | CL | L/D | drag [N] | P elec [W] | endurance [min] | range [km] |
|---|---|---|---|---|---|---|
| 14 | 0.41 | 10.9 | 1.7 | 57 | 41 | 34 |
| 16 | 0.31 | 9.7 | 1.9 | 72 | 33 | 31 |
| 20 | 0.20 | 7.2 | 2.6 | 117 | 20 | 24 |
| 30 | 0.09 | 3.5 | 5.4 | 346 | 7 | 12 |

(39 Wh usable from 4S 3300 mAh at 80 %; avionics 7 W.) A 4S 5000 mAh adds 140 g and takes loiter to ~48 min.

**Dash.** The 9×6 tops out at its pitch speed, 33 m/s at 13 000 rpm. A 10×7 moves that to 38 m/s; 4Q-3 flies
a **10×7** if prop_lab (38) confirms the 2216 stays under 25 A at full throttle static, else a 10×6.

## 4. Technical definition

**Airframe parameters** (`kestrel2.params("4Q-3")`): span 1200, chord 160, gap 300, motor_y 450, prop_diameter 254,
prop_x 135, bracing "posts", strut_fairing true, post_diameter 10, elevons true, elevon_chord 0.32,
elevon_inner_y 300, pod as 4Q-2, fins scaled with the gap, section: 12 % thick, camber 1.5 % at 40 %
with 1° of reflex via elevon trim (replaces the fixed-reflex 23112; a slab comparison in 39 decides).

**Bought parts.**

| Item | Choice | Note |
|---|---|---|
| Motors ×4 | 2216-class outrunner, 880–900 KV, 4S | 25 A max; hover at 7 A |
| Props ×4 | APC 10×7E (or 10×6E) | two CW, two CCW; reversed they give ~35 % thrust |
| ESCs ×4 | 30 A BLHeli_32, 3D (bidirectional) mode, in the nacelles | reverse thrust for the back transition |
| Battery | 4S 3300 mAh LiPo (option 5000) | 140 mm bay, CG on the spar |
| Flight controller | ArduPilot-capable H7 board, ArduPlane QuadPlane tailsitter mode | GNSS, compass, telemetry, receiver |
| Servos ×4 | 12 g metal-gear digital, one per elevon | crow brake = upper up, lower down |
| Spars ×2 | pultruded carbon tube Ø10×8 mm, 1.2 m | the only non-printed structure |

**Printed parts.** Pod in two halves (PETG/ASA, 1.2 mm walls, 6 mm bulkheads at the spar and battery), wing
skins in LW-PLA on the carbon tube (ribs every 80 mm, hinge line printed in), four faired posts with the
nacelles and ESC bays, two fins and the ventral fin with the feet, the capsule shell, the elevon horns and
the servo frames. Print orientation: wings spanwise in 300 mm segments joined on the tube; posts vertical.

**Structure to check in FEA (40 § 8b).** Case A 3 g pull-up, case B hover with one motor at T_max, case C
landing 2 m/s on two feet; ultimate 1.5×, PETG knock-down 0.5. First-order: root bending at 3 g is ~5 N·m per
wing against ~25 N·m for the Ø10×8 carbon tube.

**Control laws.** Hover and transitions as in 42: allocation to differential thrust + elevons, back transition
= crow brake to 1.15 V_s in level flight, then pitch-up with −3 N on the lower pair, the pitch bulge past the
vertical as fallback. The "decelerate to 13 m/s" rule is dropped; it is replaced by the stall-based one.

## 5. What decides each number

| Number | Settled by |
|---|---|
| wing area (span 1.2 vs 1.1) | fine bare sweep in 39 (CL max 8–16°) |
| prop (10×7 vs 10×6) | 38 prop_lab fine campaign (static current, reverse thrust at azimuths 0/90) |
| section (reflex vs elevon trim) | 23112 vs cambered slab at 14–16° in 39 |
| structure | 40 § 8b once the region bug is fixed |
| transitions | 42 remedy matrix on the 4Q-3 vehicle |
