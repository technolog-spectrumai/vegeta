# Ketos export

The fjord salmon farm's machines for [Ketos](https://github.com/technolog-spectrumai/ketos): meshes (one JSON per
coloured part, metres, x forward, z up) and a physics card per machine (`*.card.json`, `ketos.card` 1.0; kinds rov, crawler, boat, hull, pen, barge).
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

    python labs/marine/designs/ketos_export.py exports/ketos
