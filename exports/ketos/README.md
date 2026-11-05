# Ketos export

The fjord salmon farm's machines for [Ketos](https://github.com/technolog-spectrumai/ketos): meshes (one JSON per
coloured part, metres, x forward, z up) and a physics card per machine (`*.card.json`, `ketos.card` 1.0).
Made by `labs/marine/designs/ketos_export.py` from the CadQuery models and the designs' sizing:

| machine | design | in Ketos |
|---|---|---|
| cyclopterus | `cyclopterus.py` | the resident pen robot: inspect, brush, clip repair, morts |
| shrimp | `shrimp.py` | the tethered backpack craft: O2 profiles, inspection, tags |
| workboat | `survey_boat.py` at 12 m | the service boat that carries Shrimp |
| pen | `salmon_pen.py` | the pen collar (the net is simulated by Ketos) |
| barge | `feed_barge.py` | the feed barge with the robots' docks |

    python labs/marine/designs/ketos_export.py exports/ketos
