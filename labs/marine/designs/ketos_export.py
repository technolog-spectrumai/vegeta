"""Export the marine lab's farm machines to Ketos (the fjord salmon-farm simulator): for each machine a JSON
triangle mesh per coloured part, in metres, x forward, z up (Ketos's ``models/`` format, the nomios model of schema
2.3), and a physics card (mass, displacement, size, thrusters, battery, power, tools) that Ketos's vehicle model
reads. The meshes come from the CadQuery models; the numbers from each design's sizing.

    python ketos_export.py OUT_DIR            # e.g. ../../../exports/ketos, or ketos/models/vegeta
    python ketos_export.py OUT_DIR --only cyclopterus,shrimp

The card is the contract between Vegeta and Ketos (``ketos.card`` 1.0):

    {"schema":"ketos.card","version":"1.0","name":"cyclopterus","kind":"rov"|"boat"|"pen"|"barge",
     "parts":[{"model":"cyclopterus_hull","color":[r,g,b]}], "size":[L,W,H], "mass_kg":..., "displacement_L":...,
     "thrusters":[{"at":[x,y,z],"axis":[ax,ay,az],"max_N":...}], "battery_Wh":..., "hotel_W":..., "tools":{...}, ...}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CARD = {"schema": "ketos.card", "version": "1.0"}


def _mesh(shape, name, color, tol_mm, scale=0.001, ang=0.5):
    """A CadQuery workplane -> the nomios model dict (metres)."""
    verts, tris = shape.val().tessellate(tol_mm, ang)
    v = []
    for q in verts:
        v += [round(q.x * scale, 4), round(q.y * scale, 4), round(q.z * scale, 4)]
    t = [int(i) for tri in tris for i in tri]
    xs, ys, zs = v[0::3], v[1::3], v[2::3]
    size = [round(max(xs) - min(xs), 3), round(max(ys) - min(ys), 3), round(max(zs) - min(zs), 3)]
    return {"name": name, "source": "vegeta labs/marine/designs (CadQuery)", "closed": True,
            "color": [round(c, 3) for c in color], "size": size, "vertices": v, "triangles": t}


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _centre_z(models, below=0.0):
    """Move every part so the machine's middle height is at z = 0 (an ROV's pose is its centre)."""
    zs = [z for m in models for z in m["vertices"][2::3]]
    mid = (max(zs) + min(zs)) / 2 - below
    for m in models:
        m["vertices"][2::3] = [round(z - mid, 4) for z in m["vertices"][2::3]]
    return mid


def _x_centre(models):
    xs = [x for m in models for x in m["vertices"][0::3]]
    mid = (max(xs) + min(xs)) / 2
    for m in models:
        m["vertices"][0::3] = [round(x - mid, 4) for x in m["vertices"][0::3]]
    return mid


def cyclopterus():
    from cyclopterus import Cyclopterus, TOOLS, PARTS
    d = Cyclopterus()
    p = Cyclopterus.p({"tool": "brush"})
    s = Cyclopterus.sizing({"tool": "brush"})
    th = None
    for at, a in d.thruster_layout(p):
        t = d._thruster(p, at, a)
        th = t if th is None else th.union(t)
    parts = [("cyclopterus_hull", d._hull(p).union(d._disc(p)), "#5d8a3a"),
             ("cyclopterus_thrusters", th, "#222222"),
             ("cyclopterus_fingers", d._fingers(p).union(d._tool(p)), "#e67e22")]
    models = [_mesh(sh, n, _hex(c), 6.0) for n, sh, c in parts]
    dz = _centre_z(models)
    thrusters = [{"at": [round(x / 1000, 3), round(y / 1000, 3), round(z / 1000 - dz, 3)], "axis": list(a), "max_N": 50.0}
                 for (x, y, z), a in d.thruster_layout(p)]
    card = CARD | {"name": "cyclopterus", "kind": "rov", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"] / 1000, p["width"] / 1000, p["height"] / 1000],
                   "mass_kg": round(s["mass_in_air_kg"], 2), "displacement_L": round(s["displacement_L"], 2),
                   "net_buoyancy_kg": 0.3, "cd": [0.8, 0.9, 1.0], "thrusters": thrusters,
                   "battery_Wh": p["battery_kwh"] * 1000, "hotel_W": round(s["hotel_W"], 1),
                   "max_depth_m": 50.0, "inspect_speed_m_s": p["inspect_speed"], "camera_swath_m": p["camera_swath_m"],
                   "tools": {k: {"mass_kg": v[0], "power_W": v[2], "price_eur": v[3], "what": v[4]} for k, v in TOOLS.items()},
                   "grip_hold_N": round(s["hold_N"], 1), "tether": None, "resident": True,
                   "price_eur": round(s["unit_price_eur"])}
    return card, models


def shrimp():
    from shrimp import Shrimp, TOOLS, TETHER_D, TETHER_CD
    d = Shrimp()
    p = Shrimp.p()
    s = Shrimp.sizing()
    parts = [("shrimp_shell", d._shell(p).union(d._handle(p)).union(d._skids(p)), "#e8734a"),
             ("shrimp_thrusters", d._thrusters(p), "#222222"),
             ("shrimp_pod", d._tool(p).union(d._lights(p)), "#2980b9")]
    models = [_mesh(sh, n, _hex(c), 3.0) for n, sh, c in parts]
    dz = _centre_z(models)
    thrusters = [{"at": [round(x / 1000, 3), round(y / 1000, 3), round(z / 1000 - dz, 3)], "axis": [round(c, 4) for c in a],
                  "max_N": p["thrust_each_N"]} for (x, y, z), a in d.thruster_layout(p)]
    card = CARD | {"name": "shrimp", "kind": "rov", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"] / 1000, p["width"] / 1000, p["height"] / 1000],
                   "mass_kg": round(s["mass_in_air_kg"], 2), "displacement_L": round(s["displacement_L"], 2),
                   "net_buoyancy_kg": 0.05, "cd": [0.7, 0.9, 1.0], "thrusters": thrusters,
                   "battery_Wh": p["battery_wh"], "hotel_W": round(s["hotel_W"], 1), "max_depth_m": 50.0,
                   "inspect_speed_m_s": p["inspect_speed"], "camera_swath_m": p["camera_swath_m"],
                   "tools": {k: {"mass_kg": v[0], "power_W": v[2], "price_eur": v[3], "what": v[4]} for k, v in TOOLS.items()},
                   "tether": {"length_m": 50.0, "diameter_m": TETHER_D, "cd": TETHER_CD},
                   "resident": False, "price_eur": round(s["kit_price_eur"])}
    return card, models


def workboat():
    """The farm's service boat: the survey boat's V hull at 12 m (a farm workboat)."""
    from survey_boat import SurveyBoat
    import cadquery as cq
    d = SurveyBoat()
    p = {q.name: q.default for q in SurveyBoat.parameters} | {"length": 12000.0, "beam": 4600.0, "depth": 1900.0, "shell": 60.0,
                                                               "bow_rise": 700.0, "bracket_height": 300.0, "pod_diameter": 300.0}
    hull = d._hull_solid(p)
    cabin = cq.Workplane("XY").box(3800, 3000, 2300, centered=(False, True, False)).translate((6200, 0, p["depth"]))
    crane = cq.Workplane("XY").add(cq.Solid.makeCylinder(250, 3200, cq.Vector(3200, 1400, p["depth"]))) \
        .union(cq.Workplane("XY").add(cq.Solid.makeCylinder(150, 5000, cq.Vector(3200, 1400, p["depth"] + 3100), cq.Vector(-0.6, -0.5, 0.3))))
    models = [_mesh(hull, "workboat_hull", _hex("#d9dde0"), 40.0), _mesh(cabin, "workboat_cabin", _hex("#2c3e50"), 40.0),
              _mesh(crane, "workboat_crane", _hex("#f39c12"), 40.0)]
    _x_centre(models)
    _centre_z(models, below=-(p["depth"] / 1000) / 2 + 0.7)        # the waterline 0.7 m above the keel
    card = CARD | {"name": "workboat", "kind": "boat", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [12.0, 4.6, 1.9], "mass_kg": 14000.0, "max_speed_m_s": 6.0, "thrust_N": 18000.0,
                   "cd": [0.35, 1.2, 1.0], "carries": ["shrimp"], "price_eur": 900000}
    return card, models


def salmon_pen():
    from salmon_pen import SalmonPen
    d = SalmonPen()
    p = SalmonPen.p({"stanchions": 24})
    s = SalmonPen.sizing()
    models = [_mesh(d.build(p), "pen_collar", _hex("#1b1b1b"), 450.0, ang=0.6)]
    card = CARD | {"name": "pen", "kind": "pen", "parts": [{"model": "pen_collar", "color": models[0]["color"]}],
                   "circumference_m": p["circumference"], "radius_m": round(s["radius_m"], 3), "net_depth_m": p["net_depth"],
                   "cone_depth_m": p["cone_depth"], "net_area_m2": round(s["net_area_m2"]), "volume_m3": round(s["volume_m3"])}
    return card, models


def feed_barge():
    from feed_barge import FeedBarge
    d = FeedBarge()
    p = FeedBarge.p()
    parts = [("barge_hull", "hull", "#7f8c8d"), ("barge_silos", "silos", "#ecf0f1"), ("barge_house", "house", "#c0392b"),
             ("barge_dock", "dock", "#f1c40f")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 150.0) for n, k, c in parts]
    card = CARD | {"name": "barge", "kind": "barge", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"], p["beam"], p["depth"]], "draft_m": p["draft"],
                   "docks": [list(map(lambda c: round(c, 3), q)) for q in d.dock_positions(p)],
                   "feed_t": round(FeedBarge.sizing()["feed_t"], 1)}
    return card, models


MACHINES = {"cyclopterus": cyclopterus, "shrimp": shrimp, "workboat": workboat, "pen": salmon_pen, "barge": feed_barge}


def export(out, only=None):
    os.makedirs(out, exist_ok=True)
    written = []
    for name, fn in MACHINES.items():
        if only and name not in only:
            continue
        card, models = fn()
        for m in models:
            with open(os.path.join(out, m["name"] + ".json"), "w") as fh:
                json.dump(m, fh, separators=(",", ":"))
        with open(os.path.join(out, name + ".card.json"), "w") as fh:
            json.dump(card, fh, indent=1)
        tris = sum(len(m["triangles"]) // 3 for m in models)
        written.append((name, len(models), tris))
        print(f"{name}: {len(models)} parts, {tris} triangles")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--only", help="comma-separated machines: " + ",".join(MACHINES))
    a = ap.parse_args()
    export(a.out, set(a.only.split(",")) if a.only else None)
