"""Export the marine lab's farm machines to Ketos (the fjord salmon-farm simulator): for each machine a JSON
triangle mesh per coloured part, in metres, x forward, z up (Ketos's ``models/`` format, the nomios model of schema
2.3), and a physics card (mass, displacement, size, thrusters, battery, power, tools) that Ketos's vehicle model
reads. The meshes come from the CadQuery models; the numbers from each design's sizing.

    python ketos_export.py OUT_DIR            # e.g. ../../../exports/ketos, or ketos/models/vegeta
    python ketos_export.py OUT_DIR --only cyclopterus,shrimp

The card is the contract between Vegeta and Ketos (``ketos.card`` 1.0):

    {"schema":"ketos.card","version":"1.0","name":"cyclopterus","kind":"rov"|"crawler"|"boat"|"hull"|"pen"|"barge",
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


#: Scampi's tool skids on the belly bay (mass kg, power W while working, price EUR, what): class estimates (assumed),
#: the cleaning skid's 250 W from scampi.energy_profile.
SCAMPI_TOOLS = {
    "cleaning": (12.0, 250.0, 18000.0, "rotary brush skid: marine growth off mooring lines, chains and anchors"),
    "cp": (2.0, 2.0, 6000.0, "CP + contact probe on the hold arm: anode and chain potential readings"),
    "survey": (6.0, 60.0, 25000.0, "seabed survey skid: profiler + sidescan for the moorings' footprints and the mort pile"),
}


def scampi():
    """The work-class AUV for the farm's moorings (lines, chains, anchors) and the seabed under the pens."""
    from scampi import Scampi, THRUSTER, PARTS
    d = Scampi()
    p = {q.name: q.default for q in Scampi.parameters} | {"arm_pose": "folded"}
    s = Scampi.sizing()
    th = None
    for at, a in d.thruster_layout(p):
        t = d._thruster(p, at, a)
        th = t if th is None else th.union(t)
    th = th.union(d._housing(p, "electronics")).union(d._housing(p, "battery"))
    body = d._frame(p).union(d._fairing(p)).union(d._foam(p))
    for sx in (1, -1):
        for sy in (1, -1):
            body = body.union(d._skid(p, sx, sy))
    arms = d.build(p | {"part": "work_arm"}).union(d.build(p | {"part": "hold_arm"})).union(d._tool_skid(p))
    parts = [("scampi_body", body, "#f1c40f"), ("scampi_thrusters", th, "#222222"), ("scampi_arms", arms, "#c0392b")]
    models = [_mesh(sh, n, _hex(c), 5.0) for n, sh, c in parts]
    dz = _centre_z(models)
    thrusters = [{"at": [round(x / 1000, 3), round(y / 1000, 3), round(z / 1000 - dz, 3)], "axis": [round(c, 4) for c in a],
                  "max_N": THRUSTER["forward_N"]} for (x, y, z), a in d.thruster_layout(p)]
    # price: the bought parts and housings at class prices (assumed), x 1.6 build and margin -> ~EUR 190 k
    card = CARD | {"name": "scampi", "kind": "rov", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"] / 1000, p["width"] / 1000, p["height"] / 1000],
                   "mass_kg": round(s["mass_in_air_kg"], 2), "displacement_L": round(s["displacement_L"], 2),
                   "net_buoyancy_kg": s["net_buoyancy_kg"], "cd": [0.9, 1.0, 1.1], "thrusters": thrusters,
                   "battery_Wh": p["battery_kwh"] * 1000, "hotel_W": round(s["hotel_W"], 1),
                   "max_depth_m": 300.0, "inspect_speed_m_s": 0.3,            # scampi.energy_profile's inspection pass
                   "camera_swath_m": 2.5,                                       # 4 cameras at ~1.5 m stand-off (assumed)
                   "tools": {k: {"mass_kg": v[0], "power_W": v[1], "price_eur": v[2], "what": v[3]} for k, v in SCAMPI_TOOLS.items()},
                   "tether": None, "resident": True, "price_eur": 190000}
    return card, models


def echensis():
    """The tethered hull crawler: cleans (with capture) and inspects a ship's hull alongside. z = 0 the hull (the
    tracks' contact plane), z out of the hull; x the crawl direction, the crawler (chassis + head) centred in x, y."""
    from echensis import Echensis, HEADS
    d = Echensis()
    p = Echensis.p()
    s = Echensis.sizing()
    hd = HEADS[p["head_type"]]
    body = d._chassis(p).union(d._suction(p)).union(d._housing(p))
    parts = [("echensis_body", body, "#16a085"), ("echensis_tracks", d._track(p, 1).union(d._track(p, -1)), "#222222"),
             ("echensis_head", d._head(p).union(d._swim_thrusters(p)), "#e67e22")]
    models = [_mesh(sh, n, _hex(c), 4.0) for n, sh, c in parts]
    _x_centre(models)
    pw = s["power_W"]
    card = CARD | {"name": "echensis", "kind": "crawler", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [round((p["length"] + p["head_length"]) / 1000, 3), round(2 * p["brush_diameter"] * 0.95 / 1000 + 0.06, 3),
                            round(p["height"] / 1000 + p["housing_od"] / 1000, 3)],
                   "mass_kg": round(s["mass_in_air_kg"], 2),
                   "displacement_L": round((s["mass_in_air_kg"] - s["in_water_kg"]) / 1.025, 2),   # 2 kg heavy in water
                   "crawl_speed_m_s": hd["speed_m_s"], "clean_width_m": hd["swath_m"],
                   "capture_efficiency": 0.9,          # class value: in-water cleaning with capture is held to >= 90 %
                   "adhesion_N": round(s["magnet_N"] + p["suction_N"], 1),
                   "clean_power_W": round(pw["total"] - pw["hotel"], 1), "hotel_W": round(pw["hotel"], 1),
                   "tether": {"length_m": p["hose_length"], "diameter_m": 0.12, "cd": 1.2},   # the 100 mm hose + power/fibre
                   "probes": ["ut", "cp"],
                   "swim_speed_m_s": 0.5,              # four T200-class swim thrusters, ~100 kg: class value
                   "price_eur": 240000}                # crawler + quay filter unit, class estimate (assumed)
    return card, models


def ship_hull():
    """A 120 m merchant ship's hull alongside (Echensis's work): z = 0 the waterline, x = 0 midship, y = 0 centreline."""
    from merchant_hull import MerchantHull
    d = MerchantHull()
    p = MerchantHull.p()
    models = [_mesh(d.build(p), "ship_hull", _hex("#8e2b2b"), 150.0, ang=0.6)]
    card = CARD | {"name": "ship_hull", "kind": "hull", "parts": [{"model": "ship_hull", "color": models[0]["color"]}],
                   "length_m": p["length"], "beam_m": p["beam"], "draft_m": p["draft"], "depth_m": p["depth"],
                   "wetted_area_m2": round(1.7 * p["length"] * p["draft"] + p["cb"] * p["length"] * p["beam"], 1)}   # Denny-Mumford
    return card, models


def wellboat():
    """The 75 m wellboat that comes alongside the pens for delice treatment: z = 0 the waterline, centred."""
    from wellboat import Wellboat
    d = Wellboat()
    p = Wellboat.p()
    s = Wellboat.sizing()
    parts = [("wellboat_hull", "hull", "#1f3a5f"), ("wellboat_house", "house", "#ecf0f1"), ("wellboat_wells", "wells", "#f1c40f"),
             ("wellboat_crane", "crane", "#e67e22")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 120.0, ang=0.6) for n, k, c in parts]
    _x_centre(models)
    card = CARD | {"name": "wellboat", "kind": "boat", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"], p["beam"], p["depth"]], "mass_kg": round(s["mass_kg"]), "max_speed_m_s": p["max_speed"],
                   "wells_m3": round(s["wells_m3"]), "treatment": ["freshwater", "thermal"], "price_eur": round(p["price_eur"])}
    return card, models


def homarus():
    """The work-class ROV for the oil field (Ketos Oil): umbilical-powered, from a TMS on the vessel's LARS."""
    from homarus import Homarus, TOOLS
    d = Homarus()
    p = Homarus.p()
    s = Homarus.sizing()
    parts = [("homarus_frame", "frame", "#7f8c8d"), ("homarus_foam", "foam", "#f1c40f"), ("homarus_thrusters", "thrusters", "#222222"),
             ("homarus_arms", "arms", "#c0392b"), ("homarus_skid", "skid", "#2c3e50")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 12.0) for n, k, c in parts]
    dz = _centre_z(models)
    th = Homarus.thruster_layout(p)
    thrusters = [{"at": [round(x / 1000, 3), round(y / 1000, 3), round(z / 1000 - dz, 3)], "axis": [round(c, 4) for c in a],
                  "max_N": (p["h_thrust_kn"] if a[2] == 0 else p["v_thrust_kn"]) * 1000} for (x, y, z), a in th]
    card = CARD | {"name": "homarus", "kind": "rov", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"] / 1000, p["width"] / 1000, p["height"] / 1000 + 0.3],
                   "mass_kg": p["mass_kg"], "displacement_L": round(s["displacement_L"], 1),
                   "net_buoyancy_kg": p["net_buoyancy_kg"], "cd": [0.9, 1.0, 1.1], "thrusters": thrusters,
                   "thruster_W": round(s["thruster_W"]), "battery_Wh": None, "power": "umbilical",
                   "hpu_W": p["hpu_kw"] * 1000, "hotel_W": s["hotel_W"], "max_depth_m": p["rating_m"],
                   "transit_speed_m_s": 0.8, "inspect_speed_m_s": 0.25, "camera_swath_m": 2.0,
                   "tools": {k: {"mass_kg": v[0], "power_W": v[1], "price_eur": v[2], "what": v[3]} for k, v in TOOLS.items()},
                   "manipulators": [{"side": "right", "functions": 7, "reach_m": 1.5, "lift_kg": 270},
                                    {"side": "left", "functions": 5, "reach_m": 1.2, "lift_kg": 450}],
                   "tether": {"length_m": p["tether_m"], "diameter_m": 0.032, "cd": 1.2, "from": "tms"},
                   "resident": False, "price_eur": round(s["price_eur"])}
    return card, models


def jacket():
    """The oil platform's jacket in 110 m (Ketos Oil): the mesh by part and the structure as data on the card."""
    from oil_platform import Jacket
    d = Jacket()
    p = Jacket.p()
    g = Jacket.members(p)
    parts = [("jacket_structure", "structure", "#c9a227"), ("jacket_anodes", "anodes", "#b0b7bf"),
             ("jacket_conductors", "conductors", "#5d6d7e"), ("jacket_risers", "risers", "#1c2833"),
             ("jacket_topside", "topside", "#95a5a6")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 90.0, ang=0.6) for n, k, c in parts]
    card = CARD | {"name": "jacket", "kind": "structure", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "water_depth_m": p["water_depth"], "deck_m": p["deck"], "batter": p["batter"], "half_top_m": p["half_top"],
                   "top_m": p["top"], "levels": g["levels"], "members": g["members"], "nodes": g["nodes"],
                   "anodes": g["anodes"], "anode_kg": p["anode_kg"], "sizing": Jacket.sizing(p)}
    return card, models


def dp_vessel():
    """The DP2 IMR/ROV support vessel (Ketos Oil): z = 0 the waterline, centred; the LARS on the starboard side."""
    from oil_platform import SupportVessel
    d = SupportVessel()
    p = SupportVessel.p()
    parts = [("dp_vessel_hull", "hull", "#1f618d"), ("dp_vessel_house", "house", "#ecf0f1"), ("dp_vessel_lars", "lars", "#e67e22")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 120.0, ang=0.6) for n, k, c in parts]
    dx = _x_centre(models)
    ob = d.overboard(p)
    card = CARD | {"name": "dp_vessel", "kind": "boat", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"], p["beam"], p["depth"]], "draft_m": p["draft"], "mass_kg": 6.5e6,
                   "max_speed_m_s": 6.5, "dp": {"class": 2, "max_wind_m_s": p["dp_wind_m_s"], "max_current_m_s": p["dp_current_m_s"]},
                   "lars": {"overboard": [round(ob[0] - dx, 2), round(ob[1], 2), 0.0], "max_hs_m": p["lars_hs_m"],
                            "winch_m_min": 30.0},
                   "carries": ["homarus", "tms"], "day_rate_eur": p["day_rate_eur"]}
    return card, models


def tms():
    """The top-hat tether management system: the ROV's garage on the lift umbilical, its 250 m excursion tether."""
    from oil_platform import Tms
    d = Tms()
    p = Tms.p()
    models = [_mesh(d.build(p), "tms_body", _hex("#d35400"), 15.0)]
    _centre_z(models)
    card = CARD | {"name": "tms", "kind": "tms", "parts": [{"model": "tms_body", "color": models[0]["color"]}],
                   "size": [p["diameter"] + 0.8, p["diameter"] + 0.8, p["height"]], "mass_kg": p["mass_kg"],
                   "tether": {"length_m": p["tether_m"], "diameter_m": p["tether_d"], "cd": 1.2},
                   "umbilical": {"diameter_m": p["umbilical_d"], "cd": 1.2, "length_m": 600.0}}
    return card, models


def manifold():
    """A subsea production manifold with an ROV valve panel (stage 4 of Ketos Oil): z = 0 the mudmat's sole."""
    from oil_platform import Manifold
    d = Manifold()
    p = Manifold.p()
    models = [_mesh(d.build(p | {"part": "frame"}), "manifold_frame", _hex("#f4d03f"), 40.0),
              _mesh(d.build(p | {"part": "panel"}), "manifold_panel", _hex("#2e86c1"), 20.0)]
    card = CARD | {"name": "manifold", "kind": "structure", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"], p["width"], p["height"]]} | Manifold.panel(p)
    return card, models


MACHINES = {"cyclopterus": cyclopterus, "shrimp": shrimp, "workboat": workboat, "pen": salmon_pen, "barge": feed_barge,
            "scampi": scampi, "echensis": echensis, "ship_hull": ship_hull, "wellboat": wellboat,
            "homarus": homarus, "jacket": jacket, "dp_vessel": dp_vessel, "tms": tms, "manifold": manifold}


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
