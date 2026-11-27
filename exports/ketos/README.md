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
| resident_dock | `oil_platform.py` `ResidentDock`, `squilla_dock.py` | Squilla's garage clamped by two split collars to leg A1 at -80 m (fits A1/B1/B2, not the riser leg A2): 2 kW inductive charger (2.2 h), fibre to the topside, docking 157 s, cross current 0.42 m/s |
| riser_clamp | `oil_platform.py` `RiserClamp` | a riser guide clamp (two bolted halves, an arm to the leg) |
| debris | `oil_platform.py` `Debris` | dropped objects on the seabed: drum, scaffold tube, wire coil, grating |
| nephro | `nephro.py` | Ketos Cable's cable AUV: 3 m torpedo, 8 kWh, the coil-array cable tracker (depth of burial), MBES, SBP; ~100 km a dive |
| trencher | `subsea_cable.py` `JetTrencher` | Ketos Cable's jetting ROV trencher: 16 t, 2 x 300 kW jet pumps, 3 m swords, tracks; speed by soil on the card |
| cable_vessel | `subsea_cable.py` `CableVessel` | the 130 m cable-lay and repair vessel: 7000 t carousel, stern chute, jointing container, the trencher's LARS |
| cable | `subsea_cable.py` `CableSegment` | a 10 m segment of a 3-core 220 kV export cable (0.24 m, 90 kg/m, MBR 4 m) |
| cable_joint | `subsea_cable.py` `CableJoint` | a rigid universal repair joint with its bend restrictors (96 h to make one) |
| zarco | `zarco.py` `Zarco` | the unmanned, robot-run drone-carrier catamaran (no crew at all): 24 x 10 m, 108 t; robots (card `robots`: reach, payload, cycle, power, faults, price) for the davits, baskets, packing, crate loading, battery swaps, deck handling, cleaning; 4 VTOL pads, catapult, net, autonomy mast |
| esmeralda | `esmeralda.py` `Esmeralda` (on `Zarco`) | Zarco's guard-duty variant: the clam line and Walrus davits out, a boat bay with a 13.5 m stern ramp for Barracuda, a 160 dB hailer, 2 searchlights, a 3000 L/min bow cannon, a long-range EO/IR turret; 4 Gannets + 2 Fulmars; `changed_vs_zarco` on the card |
| barracuda | `barracuda.py` `Barracuda` | the guard carrier's unmanned interceptor boat: 9 m deep-V with a foam collar, 2 x 220 kW on waterjets, 47 kn, speed by Hs on the card, EO/IR + radar ranges, hailer, light, strobe, water cannon; stern-ramp launch and recovery to Hs 1.95 m (class values) |
| fishing_boat | `small_craft.py` | a contact (kind `craft`): 20 m trawler/longliner, 157 t, 9.7 kn, RCS ~300 m2, AIS class A, gear in the water |
| pilot_boat | `small_craft.py` | a contact: 16 m pilot boat, 27 t, 22 kn, AIS class A, white over red |
| yacht | `small_craft.py` | a contact: 12 m sloop, 12 t, 8 kn under engine (4-8 sailing), often no AIS |
| skiff | `small_craft.py` | a contact: 7 m open skiff, 1.65 t, 35 kn in calm (15 kn in Hs 1 m: its crew's impacts), RCS ~1 m2, no AIS, often no lights |
| saucer | aero `ufo.py` | the alien pirates' flying saucer (kind `craft`, airborne, **fictional**, `ais: false`): 12 m disc, 6.4 t (a medium twin helicopter's class), moves like a helicopter - hovers, 78 m/s, 18 m/s sideways, 40 deg/s turn on the spot, 45 deg bank; collision primitives; RCS, IR, flak figures; the alien Jolly Roger on a mast (`saucer_flag`, textured: `alien_jolly_roger.png`) |
| cigar | aero `ufo.py` | the alien pirates' winged cigar (kind `craft`, airborne, **fictional**, `ais: false`): 18 m spindle, 10 m swept wings, 8.5 t (a business jet's class), moves like a jet - 70 to 215 m/s low (245 high), 2.5 g turns (1.8 km radius at 200 m/s); the flag on both sides of its fin (`cigar_flag`, textured) |
| gannet | aero `gannet.py` `Gannet` | the cargo quad (air card, `air: vtol`): 16.4 kg with a 5 kg crate, 28" rotors, 1150 Wh, radius 13.5 km, deck Hs 2 m; `payloads`: cargo, or the EO/IR surveillance pod (`eoir_payload.py`: 2.2 kg, 42 min loiter, DRI by altitude and weather) |
| fulmar | aero `fulmar.py` `Fulmar` (on `fixed_wing.FixedWing`) | the long-haul fixed-wing (air card, `air: fixed_wing`): 3.6 m, 20 kg with 5 kg, 8.5 Wh/km, radius 65 km, catapult + net; `payloads`: cargo, or the cooled EO/IR pod (3.8 kg, 150 min loiter; a radar weighed and not fitted) |
