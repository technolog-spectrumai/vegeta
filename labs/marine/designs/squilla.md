# Squilla — design brief (v0)

Named after *Squilla mantis*, the mantis shrimp: raptorial claws that strike fast, and the most elaborate eyes in
the sea. Squilla is **Nephro's intervention variant** (`nephro.md`). It keeps the body of revolution, the housings,
the thrusters and the autonomy, but it is built to work at a site, not to survey 100 km of cable.

## Decisions

| question | decision |
|---|---|
| role | better than Nephro at intervention; no 100 km survey range (a 20 km transit out and back, 4 h of work per dive) |
| jobs | cut and clear (ropes, nets, fishing gear, debris on cables, moorings, piles); grab, tag, retrieve (≤ 8 kg in water); close inspection by touch (UT, CP stab, coating) |
| vision | every sensible configuration compared (`Squilla.compare_sensors`); default **C: stereo + polarisation + imaging sonar** |

## What changes from Nephro

| | Nephro | Squilla |
|---|---|---|
| length | 3.0 m | **2.2 m** |
| battery | 8 kWh: 97 km at 1.5 m/s | **4.5 kWh**: 20 km transit + 4 h of work |
| arms | one 5-DOF folding arm | **6-DOF work arm** (1.0 m, raptorial cutter-gripper: 3 kN in 150 ms, spring-assisted) + **4-DOF hold arm** (0.7 m clamp jaw, contact probes) |
| hover | 4 tunnel thrusters | **8 tunnel thrusters** (4 vertical, 4 lateral): vertical margin +1.0 with 8 kg lifted |
| skids | two low skids | wider, longer skids for landed work |
| sensors | coil array, side-scan, multibeam, sub-bottom | a vision set chosen for manipulation (below) |
| mass | 165 kg | **139 kg** (vision C) |

## Vision configurations

Each sensor is scored 0–2 for five aspects (clear water, turbid water, ropes and fibres, 3-D grasping, material
ID); a set scores the best of its sensors per aspect. Prices and powers are class values.

| set | sensors | price | power | battery margin | score |
|---|---|---|---|---|---|
| A | stereo + polarisation | €10.5k | 96 W | +14 % | 7 |
| B | stereo + imaging sonar | €15.0k | 110 W | +11 % | 6 |
| **C** | **stereo + polarisation + imaging sonar** | **€19.5k** | **116 W** | **+10 %** | **8** |
| D | stereo + imaging sonar + laser line scanner | €45.0k | 145 W | +6 % | 7 |
| E | stereo + hyperspectral + imaging sonar | €55.0k | 135 W | +7 % | 8 |
| F | all five | €89.5k | 176 W | +1 % | 10 |

**C is the default**: it sees in turbid water (sonar) and finds ropes and fibres against the seabed (the
polarisation camera cuts backscatter and shows fibres), which is most of cut-and-clear, at a fifth of F's price.
D (the laser scanner) is the upgrade when grasping precision matters (retrieving small objects); E (hyperspectral)
only when telling materials apart is the job. Every set fits the Jetson AGX Orin (165–235 of 275 TOPS).

## Open questions

* the raptorial cutter: spring-assisted strike mechanism, reset time, and the reaction the hold arm must take
* the hold arm's clamp: one jaw for cables Ø80–250 mm, mooring chain and pile braces, or swappable jaws
* landed work in current: skids and thrusters against 1 m/s broadside when the work forces it off the nose-in heading
* the polarisation camera's real gain in North Sea turbidity: measure before committing to C over B
