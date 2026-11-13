# Scampi — design brief (v0)

Small work-class electric AUV for **offshore wind**. It swims first and lands on the seabed to work. It succeeds the
Sikarian Lobster (`origin/dev_lobster`, Nefri / Ornatus: a 6–15 kg six-legged walker with two pincers and a tail
thruster). Scampi drops the walking legs for four fixed landing skids and becomes a hover-capable vehicle with a hold
arm and a work arm.

## Decisions (design questions, round 1–2)

| question | decision |
|---|---|
| market | offshore wind first (fixed-bottom; floating-wind moorings later) |
| size | small work class, 60–120 kg in air, ~1.5 m, workboat davit launch |
| approach | universal vehicle + swappable tool skids (belly bay, wet-mate 48 V + Ethernet); each skid trimmed neutral with its own foam |
| comms | untethered; acoustic modem; optional fibre micro-tether for supervised arm work |
| arms | one 6-DOF work arm + one 4-DOF hold (grabber) arm |
| depth | 300 m rating (housings at 1.5 × 3.1 MPa) |
| jobs | monopile and cable inspection; marine-growth cleaning; light intervention; sensor / CP anode work |
| endurance | 6–8 h per dive |
| autonomy | NVIDIA Jetson AGX Orin 64GB (15–60 W) |

## Decisions (round 3)

| question | decision |
|---|---|
| launch and recovery | davit or A-frame from a workboat (CTV / survey vessel); Scampi swims to the turbine and back |
| first tool skid | cleaning brush + CP / UT probe: clean a weld or anode, then measure cathodic protection and wall thickness |
| sourcing | buy the thrusters (T500 class), design the arms in-house (oil-filled, electric, the hold-and-work pair) |
| next step | MuJoCo scenario at a monopile: boat → pile in a tidal current, clamp with the hold arm, clean and probe a weld, land on the seabed (the Lobster's water model) |

## v0 numbers (`python labs/marine/designs/scampi.py`)

* 1.5 × 0.8 × 0.56 m without arms; **118 kg** in air; displaces 115 L; +0.5 kg buoyant; 43 L syntactic foam.
* Al 6061-T6 housings Ø200 × 7 mm: buckling 6.6 MPa against 4.5 MPa (margin +0.47); hoop stress 65 MPa.
* Eight T500-class thrusters (four vectored 45°, four vertical): 444 N surge/sway, 628 N heave; holds station in
  1.24 m/s sideways (1.69 m/s ahead) against the 1.0 m/s requirement.
* A 7 h dive (transit, 3 h inspection, 1 h cleaning, 1 h intervention, 1 h landed) takes 3.4 kWh of a 4.5 kWh pack
  (4.05 kWh usable): +19 % margin. Hotel load (Jetson, INS, DVL, sonar, cameras, lights) 237 W.

All the bought-part numbers are class values (`PARTS`, `THRUSTER`), not quotes.

## Comparable vehicles

| vehicle | class | notes |
|---|---|---|
| Nauticus Aquanaut | transforming AUV/ROV, ~1,050 kg, 3.5 m, 300 m | two 7-DOF arms; ~30 s to transform ([IEEE Robots](https://robots.ieee.org/robots/aquanaut)) |
| Saab Seaeye Sabertooth | hybrid AUV/ROV, 3.6–4 m, 1,200 / 3,000 m | resident in a seabed dock for months (Equinor); manipulator options ([Saab](https://saabseaeye.com/solutions/underwater-vehicles/sabertooth-single-hull)) |
| Sikarian Lobster (ours) | 6–15 kg walker | the arms, pincers and water model Scampi starts from |

Scampi sits well below both: a tenth of Aquanaut's mass, one workboat and a davit. It trades their depth and
endurance for cost per unit, which offshore wind's repetitive shallow work rewards.

## Open questions (next rounds)

* the work and hold arms: joint torques for brushing and probing, link sizing, sealing (oil-filled, compensated)
* the davit's launch cage: does Scampi ride in a cage through the splash zone, or hang from a latch
* navigation near monopiles: DVL loses lock close to the pile; sonar SLAM or a pile-relative camera mode
