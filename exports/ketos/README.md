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
