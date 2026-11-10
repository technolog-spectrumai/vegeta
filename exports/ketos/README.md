# Ketos export

The fjord salmon farm's machines for [Ketos](https://github.com/technolog-spectrumai/ketos): meshes (one JSON per
coloured part, metres, x forward, z up) and a physics card per machine (`*.card.json`, `ketos.card` 1.0; kinds rov, crawler, boat, hull, pen, barge, structure, tms).
Made by `labs/marine/designs/ketos_export.py` from the CadQuery models and the designs' sizing:

| machine | design | in Ketos |
|---|---|---|
| cyclopterus | `cyclopterus.py` | the resident pen robot: inspect, brush, clip repair, morts |
| shrimp | `shrimp.py` | the tethered backpack craft: O2 profiles, inspection, tags |
| workboat | `survey_boat.py` at 12 m | the service boat that carries Shrimp |
| pen | `salmon_pen.py` | the pen collar (the net is simulated by Ketos) |
| barge | `feed_barge.py` | the feed barge with the robots' docks |
| scampi | `scampi.py` (arms folded) | the work-class AUV: mooring-line and seabed inspection, tool skids |
| echensis | `echensis.py` | the tethered hull crawler: cleaning with capture, UT/CP inspection (z = 0 the hull) |
| ship_hull | `merchant_hull.py` | a 120 m merchant hull alongside, Echensis's work (z = 0 the waterline) |
| wellboat | `wellboat.py` | the 75 m wellboat: two wells, freshwater/thermal delice treatment |
| homarus | `homarus.py` | Ketos Oil's work-class ROV: 4.2 t, 7 hydraulic thrusters, two manipulators, umbilical power, 250 m tether from the TMS |
| jacket | `oil_platform.py` `Jacket` | the 4-leg steel jacket in 110 m: the mesh by part, every member, node and anode on the card |
| dp_vessel | `oil_platform.py` `SupportVessel` | the 90 m DP2 ROV support vessel with the A-frame LARS (Hs limit, DP limits on the card) |
| tms | `oil_platform.py` `Tms` | the top-hat tether management system |
| manifold | `oil_platform.py` `Manifold` | a subsea manifold with the ROV valve panel (class-4 buckets, hot stabs) |

    python labs/marine/designs/ketos_export.py exports/ketos
| squilla | `squilla.py` (arms stowed) | Ketos Oil's resident drone: the 2.2 m intervention AUV, 4.5 kWh, docked on the jacket |
| resident_dock | `oil_platform.py` `ResidentDock` | Squilla's garage on a leg at -80 m: inductive charger, fibre to the topside |
| riser_clamp | `oil_platform.py` `RiserClamp` | a riser guide clamp (two bolted halves, an arm to the leg) |
| debris | `oil_platform.py` `Debris` | dropped objects on the seabed: drum, scaffold tube, wire coil, grating |
| nephro | `nephro.py` | Ketos Cable's cable AUV: 3 m torpedo, 8 kWh, the coil-array cable tracker (depth of burial), MBES, SBP; ~100 km a dive |
| trencher | `subsea_cable.py` `JetTrencher` | Ketos Cable's jetting ROV trencher: 16 t, 2 x 300 kW jet pumps, 3 m swords, tracks; speed by soil on the card |
| cable_vessel | `subsea_cable.py` `CableVessel` | the 130 m cable-lay and repair vessel: 7000 t carousel, stern chute, jointing container, the trencher's LARS |
| cable | `subsea_cable.py` `CableSegment` | a 10 m segment of a 3-core 220 kV export cable (0.24 m, 90 kg/m, MBR 4 m) |
| cable_joint | `subsea_cable.py` `CableJoint` | a rigid universal repair joint with its bend restrictors (96 h to make one) |
| pinna | `pinna.py` `Pinna` | the autonomous drone-carrier catamaran (no crew): 24 x 10 m, 108 t, Walrus davits, basket robot, crate packer, 4 VTOL pads, catapult, recovery net, autonomy mast |
| gannet | aero `gannet.py` `Gannet` | the cargo quad (air card, `air: vtol`): 16.4 kg with a 5 kg crate, 28" rotors, 1150 Wh, radius 13.5 km, deck Hs 2 m |
| fulmar | aero `fulmar.py` `Fulmar` (on `fixed_wing.FixedWing`) | the long-haul fixed-wing (air card, `air: fixed_wing`): 3.6 m, 20 kg with 5 kg, 8.5 Wh/km, radius 65 km, catapult + net |
