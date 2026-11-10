"""Export the marine lab's farm machines to Ketos (the fjord salmon-farm simulator): for each machine a JSON
triangle mesh per coloured part, in metres, x forward, z up (Ketos's ``models/`` format, the nomios model of schema
2.3), and a physics card (mass, displacement, size, thrusters, battery, power, tools) that Ketos's vehicle model
reads. The meshes come from the CadQuery models; the numbers from each design's sizing.

    python ketos_export.py OUT_DIR            # e.g. ../../../exports/ketos, or ketos/models/vegeta
    python ketos_export.py OUT_DIR --only cyclopterus,shrimp

The card is the contract between Vegeta and Ketos (``ketos.card`` 1.0):

    {"schema":"ketos.card","version":"1.0","name":"cyclopterus","kind":"rov"|"crawler"|"boat"|"hull"|"pen"|"barge"|"air",
     "parts":[{"model":"cyclopterus_hull","color":[r,g,b]}], "size":[L,W,H], "mass_kg":..., "displacement_L":...,
     "thrusters":[{"at":[x,y,z],"axis":[ax,ay,az],"max_N":...}], "battery_Wh":..., "hotel_W":..., "tools":{...}, ...}

An ``air`` card (a drone that flies: Gannet, Fulmar) has ``air`` ("vtol" | "fixed_wing"), ``mass_kg`` (empty, with
the pack), ``mtow_kg``, ``payload_kg``, ``battery_Wh`` / ``usable_Wh``, ``cruise_m_s``, ``power_cruise_W`` (+ ``_empty``),
``power_hover_W`` (+ ``_empty``; None for a fixed-wing), ``wh_per_km`` (+ ``_empty``), ``range_km``, ``radius_km``,
``endurance_min``, ``wind_limit_m_s``, ``deck_hs_limit_m`` (the landing on a deck's sea-state limit), ``launch`` and
``recovery`` ("pad" | "catapult" | "net"), ``swap_s``.
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


def squilla():
    """Squilla, the intervention AUV, as Ketos Oil's resident drone (stage 3): docked on the jacket, battery-powered."""
    from squilla import Squilla
    d = Squilla()
    p = Squilla.p() | {"arm_pose": "stowed"}
    s = Squilla.sizing()
    parts = [("squilla_body", "body", "#e74c3c"), ("squilla_arms", "arms", "#2c3e50"), ("squilla_skids", "skids", "#7f8c8d")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 4.0) for n, k, c in parts]
    dz = _centre_z(models)
    L, R = p["length"] / 1000, p["diameter"] / 2000
    th = [{"at": [round(-L / 2, 3), round(sy * R * 0.6, 3), 0.0], "axis": [1.0, 0.0, 0.0], "max_N": 157.0} for sy in (1, -1)]
    th += [{"at": [round(x * L, 3), 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "max_N": 50.0} for x in (0.3, 0.15, -0.15, -0.3)]
    th += [{"at": [round(x * L, 3), 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "max_N": 50.0} for x in (0.3, 0.15, -0.15, -0.3)]
    card = CARD | {"name": "squilla", "kind": "rov", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [L, 2 * R + 0.2, 2 * R + 0.25], "mass_kg": round(s["mass_in_air_kg"], 1),
                   "displacement_L": round(s["mass_in_air_kg"] / 1.025 + 1.0, 1), "net_buoyancy_kg": 1.0, "cd": [0.3, 1.0, 1.1],
                   "thrusters": th, "battery_Wh": p["battery_kwh"] * 1000, "hotel_W": round(s["work_W"] * 0.4, 1),
                   "work_W": round(s["work_W"], 1), "transit_W": round(s["transit_W"], 1), "max_depth_m": 300.0,
                   "transit_speed_m_s": 1.0, "inspect_speed_m_s": 0.3, "camera_swath_m": 1.5,
                   "tools": {"camera": {"what": "stereo + polarisation + imaging sonar"}, "cp": {"what": "CP stab on the hold arm"},
                             "ut": {"what": "UT thickness probe"}, "cutter": {"what": "raptorial cutter-gripper, 3 kN"}},
                   "lift_kg": 8.0, "tether": None, "resident": True, "price_eur": 450000}
    return card, models


def resident_dock():
    """Squilla's dock on the jacket: the garage, the charger, the fibre to the topside."""
    from oil_platform import ResidentDock
    d = ResidentDock()
    p = ResidentDock.p()
    models = [_mesh(d.build(p), "resident_dock_frame", _hex("#16a085"), 20.0)]
    card = CARD | {"name": "resident_dock", "kind": "dock", "parts": [{"model": "resident_dock_frame", "color": models[0]["color"]}],
                   "size": [p["length"], p["width"], p["height"]], "depth_m": p["depth"], "charge_W": p["charge_kw"] * 1000,
                   "mass_kg": p["mass_kg"], "comms": "fibre", "price_eur": p["price_eur"]}
    return card, models


def riser_clamp():
    from oil_platform import RiserClamp
    d = RiserClamp()
    p = RiserClamp.p()
    models = [_mesh(d.build(p), "riser_clamp_body", _hex("#d4ac0d"), 10.0)]
    card = CARD | {"name": "riser_clamp", "kind": "structure", "parts": [{"model": "riser_clamp_body", "color": models[0]["color"]}],
                   "size": [p["arm"] + p["riser_d"] + 0.16, p["riser_d"] + 0.16, p["height"]], "riser_d_m": p["riser_d"]}
    return card, models


def debris():
    """Dropped objects for the seabed survey: one card, a mesh per kind (z = 0 on the seabed)."""
    from oil_platform import Debris
    d = Debris()
    cols = {"drum": "#2471a3", "tube": "#909497", "coil": "#6e2c00", "grating": "#b7950b"}
    models = [_mesh(d.build({"kind": k}), f"debris_{k}", _hex(c), 10.0) for k, c in cols.items()]
    card = CARD | {"name": "debris", "kind": "debris", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "kinds": {k: {"model": f"debris_{k}", "mass_kg": w} for k, w in
                             {"drum": 30.0, "tube": 25.0, "coil": 120.0, "grating": 60.0}.items()}}
    return card, models


def nephro():
    """Nephro, the cable AUV (Ketos Cable): follows a buried cable on its coil-array tracker, surveys the route."""
    from nephro import Nephro, PARTS as NPARTS, REQUIREMENTS
    d = Nephro()
    p = Nephro.p()
    s = Nephro.sizing()
    parts = [("nephro_body", "body", "#f39c12"), ("nephro_antennae", "antennae", "#2c3e50"), ("nephro_arm", "arm", "#c0392b"),
             ("nephro_skids", "skids", "#7f8c8d")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 4.0) for n, k, c in parts]
    dz = _centre_z(models)
    L, R = p["length"] / 1000, p["diameter"] / 2000
    th = [{"at": [round(-L / 2, 3), round(sy * (R * 0.25 + 0.09), 3), round(-dz, 3)], "axis": [1.0, 0.0, 0.0], "max_N": 120.0}
          for sy in (1, -1)]
    th += [{"at": [round(x, 3), 0.0, round(-dz, 3)], "axis": [0.0, 0.0, 1.0], "max_N": 50.0} for x in (0.18 * L, -0.22 * L)]
    th += [{"at": [round(x + 0.16, 3), 0.0, round(-dz, 3)], "axis": [0.0, 1.0, 0.0], "max_N": 50.0} for x in (0.18 * L, -0.22 * L)]
    card = CARD | {"name": "nephro", "kind": "rov", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [L + 0.5, p["antenna_span"] / 1000, 2 * R + 0.2], "mass_kg": round(s["mass_in_air_kg"], 1),
                   "displacement_L": round((s["mass_in_air_kg"] + 1.0) / 1.025, 1), "net_buoyancy_kg": 1.0, "cd": [0.25, 1.0, 1.1],
                   "thrusters": th, "battery_Wh": p["battery_kwh"] * 1000, "hotel_W": round(s["hotel_W"], 1),
                   "survey_W": round(s["survey_W"], 1), "hover_W": round(s["hover_W"], 1), "max_depth_m": REQUIREMENTS["depth_m"],
                   "transit_speed_m_s": 1.5, "survey_speed_m_s": REQUIREMENTS["survey_speed_m_s"],
                   "survey_altitude_m": REQUIREMENTS["survey_altitude_m"], "range_km": round(s["range_km"], 1),
                   "inspect_speed_m_s": 0.5, "camera_swath_m": 3.0,
                   "tracker": {"kind": "3-axis coil array (active tone / passive field)", "span_m": p["antenna_span"] / 1000,
                               "dob_max_m": 3.0, "dob_sigma_m": 0.08, "lateral_sigma_m": 0.15},
                   "sensors": [k for k in NPARTS if any(w in k for w in ("sonar", "multibeam", "profiler", "tracker", "cameras", "DVL", "INS"))],
                   "tools": {"camera": {"what": "2 cameras + lights"}, "tracker": {"what": "coil array: cable position and depth of burial"},
                             "mbes": {"what": "multibeam: bathymetry, exposures, free spans"}, "sbp": {"what": "sub-bottom profiler: burial in soft sediment"},
                             "gripper": {"what": "5-DOF arm: lift and tag the cable, place markers"}},
                   "tether": None, "resident": False, "price_eur": 420000}
    return card, models


def trencher():
    """The jetting ROV trencher (Ketos Cable stage 3): umbilical-powered from the cable vessel, tracks on the seabed."""
    from subsea_cable import JetTrencher, SOILS
    d = JetTrencher()
    p = JetTrencher.p()
    s = JetTrencher.sizing()
    parts = [("trencher_frame", "frame", "#34495e"), ("trencher_foam", "foam", "#f1c40f"), ("trencher_thrusters", "thrusters", "#222222"),
             ("trencher_swords", "swords", "#e67e22"), ("trencher_tracks", "tracks", "#1b2631")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 30.0) for n, k, c in parts]
    dz = _centre_z(models)
    thrusters = [{"at": [round(x, 3), round(y, 3), round(z - dz, 3)], "axis": [round(c, 4) for c in a], "max_N": p["thrust_kn"] * 1000}
                 for (x, y, z), a in JetTrencher.thruster_layout(p)]
    card = CARD | {"name": "trencher", "kind": "trencher", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"], p["width"], p["height"]], "mass_kg": p["mass_kg"], "displacement_L": round(s["displacement_L"], 1),
                   "net_buoyancy_kg": round(-(p["mass_kg"] - s["displacement_L"] * 1.025), 1), "cd": [1.0, 1.1, 1.2],
                   "thrusters": thrusters, "thruster_W": round(s["thruster_W"]), "battery_Wh": None, "power": "umbilical",
                   "power_W": s["power_W"], "jet_W": p["jet_kw"] * 1000, "hotel_W": p["hotel_kw"] * 1000, "max_depth_m": p["rating_m"],
                   "transit_speed_m_s": 0.6, "on_tracks_N": round(s["on_tracks_N"]), "trench_width_m": p["trench_width"],
                   "sword_depth_m": p["sword_depth"], "max_trench_speed_m_h": p["max_speed_m_h"], "jet_eff": p["jet_eff"],
                   "soils": {k: {"specific_kJ_m3": v["specific_kJ_m3"], "max_depth_m": v["max_depth_m"]} for k, v in SOILS.items()},
                   "trench_m_h": s["trench_m_h"],
                   "tether": {"length_m": p["umbilical_m"], "diameter_m": 0.06, "cd": 1.2, "from": "vessel"},
                   "resident": False, "price_eur": p["price_eur"], "day_rate_eur": p["day_rate_eur"]}
    return card, models


def cable_vessel():
    """The cable-lay and repair vessel (Ketos Cable): carousel, stern chute, jointing container, the trencher's LARS."""
    from subsea_cable import CableVessel
    d = CableVessel()
    p = CableVessel.p()
    s = CableVessel.sizing()
    parts = [("cable_vessel_hull", "hull", "#922b21"), ("cable_vessel_house", "house", "#ecf0f1"), ("cable_vessel_deck", "deck", "#e67e22")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 150.0, ang=0.6) for n, k, c in parts]
    dx = _x_centre(models)
    ob = CableVessel.overboard(p)
    card = CARD | {"name": "cable_vessel", "kind": "boat", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"], p["beam"], p["depth"]], "draft_m": p["draft"], "mass_kg": 1.4e7, "max_speed_m_s": 6.0,
                   "dp": {"class": 2, "max_wind_m_s": p["dp_wind_m_s"], "max_current_m_s": p["dp_current_m_s"]},
                   "chute": [round(ob["chute"][0] - dx, 2), 0.0, 0.0],
                   "lars": {"overboard": [round(ob["lars"][0] - dx, 2), round(ob["lars"][1], 2), 0.0], "max_hs_m": p["lars_hs_m"]},
                   "carousel_t": p["carousel_t"], "carousel_km": round(s["carousel_km"], 1), "lay_speed_m_h": p["lay_speed_m_h"],
                   "crane_t": p["crane_t"], "joint_hs_m": p["joint_hs_m"], "carries": ["trencher", "homarus", "nephro"],
                   "day_rate_eur": p["day_rate_eur"]}
    return card, models


def cable():
    """A 10 m cable segment (the viewer strings it along the route) and the cable's properties."""
    from subsea_cable import CableSegment, CABLE
    p = CableSegment.p()
    models = [_mesh(CableSegment().build(p), "cable_segment", _hex("#17202a"), 5.0)]
    card = CARD | {"name": "cable", "kind": "cable", "parts": [{"model": "cable_segment", "color": models[0]["color"]}],
                   "segment_m": p["length"]} | CABLE
    return card, models


def cable_joint():
    from subsea_cable import CableJoint
    p = CableJoint.p()
    models = [_mesh(CableJoint().build(p), "cable_joint_body", _hex("#d4ac0d"), 10.0)]
    card = CARD | {"name": "cable_joint", "kind": "joint", "parts": [{"model": "cable_joint_body", "color": models[0]["color"]}],
                   "size": [p["length"] + 2 * p["restrictor"], p["diameter"], p["diameter"]], "mass_kg": p["mass_kg"],
                   "joint_h": p["joint_h"], "price_eur": p["price_eur"]}
    return card, models


def walrus():
    """Walrus, the clam-picking seabed crawler (Ketos Clams): tracks, a belly gantry with soft grippers and a pulsed
    jet, two digging arms, baskets on float lines, tethered to the harvest boat. z = 0 the track bottom (on the bed)."""
    from walrus import Walrus, PARTS, CLAM, REQUIREMENTS, LAYOUT, prop, thruster_layout
    d = Walrus()
    p = Walrus.p()
    s = Walrus.sizing()
    parts = [("walrus_tracks", "tracks", "#1b2631"), ("walrus_frame", "frame", "#7f8c8d"), ("walrus_housing", "housing", "#2e86c1"),
             ("walrus_gantry", "gantry", "#e67e22"), ("walrus_basket", "basket", "#f4d03f"), ("walrus_thrusters", "thrusters", "#222222"),
             ("walrus_lights", "lights", "#ecf0f1"), ("walrus_arms", "arms", "#c0392b")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 4.0) for n, k, c in parts]
    _x_centre(models)
    card = CARD | {"name": "walrus", "kind": "crawler", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [p["length"] / 1000, p["width"] / 1000, p["height"] / 1000],
                   "mass_kg": round(s["mass_wet_vehicle_kg"], 2), "mass_with_tether_kg": round(s["mass_in_air_kg"], 2),
                   "sink_kg": round(s["sink_kg"], 2), "lift_margin": round(s["lift_margin"], 2),
                   "crawl_speed_m_s": p["crawl_speed"], "swath_m": p["swath"], "pickers": p["pickers"], "pick_s": p["pick_s"],
                   "jet": {"kind": "pulsed water jet + nozzle", "power_W": PARTS["jet pump + nozzle (pulsed)"][2]},
                   "arms": {"count": 2, "dig_depth_m": p["dig_depth"] / 1000, "dig_fraction": p["dig_fraction"],
                            "dig_m3_h": round(s["dig_m3_h"], 3), "duty": round(s["dig_duty"], 3)},
                   "basket": {"kg": p["basket_kg"], "litres": 30, "count": 2, "swap_min": s["trip_min"], "on": "float line"},
                   "sizing_chute": True, "clam": CLAM,
                   "swim_speed_m_s": round(s["v_swim_m_s"], 3), "transit_m": p["transit_m"], "nav_error_m": round(s["nav_error_m"], 2),
                   "hold_current_m_s": 0.5 * math.sqrt(1 + s["hold_current_margin"]),
                   "thruster_layout": LAYOUT, "net_buoyancy_kg": round(-s["sink_kg"], 2),
                   "thrusters": [{"at": [round(c, 3) for c in t["at"]], "axis": [round(c, 4) for c in t["axis"]], "max_N": round(t["max_N"], 1),
                                  "diameter_m": t["diameter_m"], "ducted": True, "rpm": round(prop(t)["rpm"]), "power_W": round(prop(t)["power_W"], 1)}
                                 for t in thruster_layout(p)],
                   "battery_Wh": None, "power": "tether", "power_W": round(s["power_W"], 1), "boat_power_W": p["boat_power_W"],
                   "hotel_W": round(sum(w for _, _, w, _ in PARTS.values()), 1),
                   "tether": {"length_m": 150.0, "diameter_m": 0.012, "cd": 1.2, "from": "clam_boat"},
                   "max_depth_m": REQUIREMENTS["depth_m"], "shift_h": 8.0, "shifts": p["shifts"],
                   "rate_per_h": round(s["rate_per_h"]), "rate_kg_h": round(s["rate_kg_h"], 2), "day_kg": round(s["day_kg"], 1),
                   "price_eur": round(s["price_eur"])}
    return card, models


def clam_basket():
    """The 30 L basket on its float line: released full from the crawler's back, hauled by the crew."""
    from walrus import Walrus
    p = Walrus.p()
    models = [_mesh(Walrus()._basket(p), "clam_basket_body", _hex("#f4d03f"), 4.0)]
    _x_centre(models)
    _centre_z(models)
    card = CARD | {"name": "clam_basket", "kind": "basket", "parts": [{"model": "clam_basket_body", "color": models[0]["color"]}],
                   "capacity_kg": p["basket_kg"], "litres": 30, "mass_kg": 1.5, "float_line_m": 35.0, "haul_min": 3.0}
    return card, models


def clam_boat():
    """The harvest boat Walrus works from: the farm workboat's hull (its meshes), a davit per crawler, the tether
    reels and topside, the USBL head, a sorting table and chilled clam bins."""
    card, _ = workboat()
    card = card | {"name": "clam_boat", "carries": ["walrus"], "davits": 4, "tether_reels": 4, "usbl": True,
                   "hold_kg": 3000.0, "crew": 2, "haul_min_per_basket": 3.0, "power_W": 6000.0, "day_rate_eur": 1200.0,
                   "price_eur": 450000}
    return card, []


def pinna():
    """Pinna, the autonomous drone-carrier catamaran (Ketos Clams stages 6-8): Walrus davits and reels, the basket
    robot and the packer, four VTOL pads, the catapult and the recovery net, the autonomy mast. z = 0 the waterline."""
    from pinna import AUTONOMY, LOADS, Pinna
    d, p, s = Pinna(), Pinna.p(), Pinna.sizing()
    parts = [("pinna_hulls", "hulls", "#34495e"), ("pinna_deck", "deck", "#95a5a6"), ("pinna_house", "house", "#ecf0f1"),
             ("pinna_flight_deck", "flight_deck", "#7f8c8d"), ("pinna_pads", "pads", "#f1c40f"), ("pinna_catapult", "catapult", "#c0392b"),
             ("pinna_net", "net", "#2c3e50"), ("pinna_mast", "mast", "#d0d3d4"), ("pinna_davits", "davits", "#e67e22")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 40.0) for n, k, c in parts]
    z_fd = (d._z_deck(p) + 3600 + 150) / 1000
    card = CARD | {"name": "pinna", "kind": "boat", "autonomous": True, "crew": 0, "operators_ashore": round(s["operators_ashore"], 3),
                   "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [s["length_m"], s["beam_m"], round(p["hull_depth"] / 1000 + 7.8, 1)], "draught_m": s["draught_m"],
                   "displacement_t": s["displacement_t"], "mass_kg": round(1000 * (s["lightship_t"] + s["deadweight_t"])),
                   "service_kn": s["service_kn"], "max_kn": s["max_kn"], "service_kW": s["service_kW"], "bed_load_kW": s["bed_load_kW"],
                   "power_W": round(1000 * s["bed_load_kW"]), "battery_kWh": p["battery_kwh"], "endurance_days": s["endurance_days"],
                   "fuel_l_h_on_bed": s["fuel_l_h_on_bed"], "loads_kW": {k: round(v, 2) for k, v in LOADS.items()}, "autonomy": AUTONOMY,
                   "carries": ["walrus", "gannet", "fulmar"], "davits": p["davits"], "tether_reels": p["davits"], "usbl": True,
                   "haul_min_per_basket": p["basket_haul_s"] / 60.0, "basket_robot": True,
                   "packing": {"crate_kg": p["crate_kg"], "pack_s": p["pack_s"], "store_kg": p["crate_store_kg"], "ice": True},
                   "hold_kg": p["crate_store_kg"],
                   "flight_deck": {"z_m": round(z_fd, 2), "pads": [[x / 1000, y / 1000] for x, y in d._pad_at(p)], "pad_cycle_s": p["pad_cycle_s"],
                                   "land_lock_s": 40.0, "crate_load_s": 40.0, "swap_s": 90.0, "launch_s": 30.0,
                                   "catapult": {"at": [-0.2, 0.0], "cycle_s": p["catapult_cycle_s"], "stroke_m": 7.0},
                                   "net": {"at": [-9.1, 0.0], "cycle_s": p["net_cycle_s"], "size_m": [6.0, 4.0]},
                                   "deck_hs_limit_m": p["deck_hs_limit"], "hangar": {"gannet": p["hangar_gannets"], "fulmar": p["hangar_fulmars"]},
                                   "packs": {"gannet": p["gannet_packs"], "fulmar": p["fulmar_packs"]}, "charger_kW": round(s["charging_kW"] / 12, 2), "chargers": 12},
                   "rates": {k: s[k] for k in ("vtol_launches_per_h", "vtol_recoveries_per_h", "fixed_wing_launches_per_h", "fixed_wing_recoveries_per_h",
                                               "crates_per_h", "pack_kg_h", "baskets_per_h")},
                   "price_eur": s["price_eur"], "day_rate_eur": s["day_rate_eur"]}
    return card, models


def _aero():
    aero = os.path.join(HERE, "..", "..", "aero", "designs")
    if aero not in sys.path:
        sys.path.insert(0, aero)


def gannet():
    """Gannet, the cargo quad (labs/aero/designs/gannet.py): an air card - mass, payload, cruise, range, the pack,
    the power at cruise and in the hover, the wind and deck limits. z = 0 the skids' bottom."""
    _aero()
    from gannet import Gannet
    d, p, s = Gannet(), Gannet.p(), Gannet.sizing()
    parts = [("gannet_body", "body", "#1f618d"), ("gannet_arms", "arms", "#17202a"), ("gannet_rotors", "rotors", "#aab7b8"),
             ("gannet_skids", "skids", "#566573"), ("gannet_crate", "crate", "#f5f5f5")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 6.0) for n, k, c in parts]
    _x_centre(models)
    card = CARD | {"name": "gannet", "kind": "air", "air": "vtol", "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [round(2 * p["arm"] / math.sqrt(2) / 1000 + p["prop_in"] * 0.0254, 2)] * 2 + [0.55],
                   "mass_kg": s["empty_kg"], "mtow_kg": s["mtow_kg"], "payload_kg": p["payload_kg"], "battery_Wh": p["battery_wh"],
                   "usable_Wh": s["usable_Wh"], "cruise_m_s": p["cruise"], "climb_m_s": 4.0, "cruise_alt_m": 60.0,
                   "power_cruise_W": s["cruise_W"], "power_cruise_empty_W": s["cruise_empty_W"],
                   "power_hover_W": s["hover_W"], "power_hover_empty_W": s["hover_empty_W"],
                   "wh_per_km": s["wh_per_km"], "wh_per_km_empty": s["wh_per_km_empty"], "range_km": s["range_km"], "radius_km": s["radius_km"],
                   "endurance_min": s["endurance_hover_min"], "wind_limit_m_s": s["wind_limit_m_s"], "deck_hs_limit_m": s["deck_hs_limit_m"],
                   "launch": "pad", "recovery": "pad", "swap_s": s["swap_s"], "reserve": 0.15, "price_eur": s["price_eur"]}
    return card, models


def fulmar():
    """Fulmar, the long-haul fixed-wing (labs/aero/designs/fulmar.py, on fixed_wing.FixedWing): an air card;
    catapult launch and net recovery. Nose toward +x (the CAD's x runs aft: mirrored)."""
    _aero()
    from fulmar import Fulmar
    d, p, s = Fulmar(), Fulmar.p(), Fulmar.sizing()
    models = [_mesh(d.build(p | {"part": "aircraft"}), "fulmar_body", _hex("#f39c12"), 6.0)]
    for m in models:
        m["vertices"][0::3] = [-x for x in m["vertices"][0::3]]
        m["vertices"][1::3] = [-y for y in m["vertices"][1::3]]
    _x_centre(models)
    card = CARD | {"name": "fulmar", "kind": "air", "air": "fixed_wing", "parts": [{"model": "fulmar_body", "color": models[0]["color"]}],
                   "size": [round(p["fuselage_length"] / 1000, 2), p["span"] / 1000, 0.6],
                   "mass_kg": s["empty_kg"], "mtow_kg": s["mtow_kg"], "payload_kg": p["payload_kg"], "battery_Wh": p["battery_wh"],
                   "usable_Wh": s["usable_Wh"], "cruise_m_s": p["cruise"], "climb_m_s": 3.0, "cruise_alt_m": 150.0, "stall_m_s": s["stall_m_s"],
                   "power_cruise_W": s["cruise_W"], "power_cruise_empty_W": s["cruise_empty_W"], "power_hover_W": None, "power_hover_empty_W": None,
                   "wh_per_km": s["wh_per_km"], "wh_per_km_empty": s["wh_per_km_empty"], "range_km": s["range_km"], "radius_km": s["radius_km"],
                   "endurance_min": s["endurance_min"], "wind_limit_m_s": s["wind_limit_m_s"], "deck_hs_limit_m": s["deck_hs_limit_m"],
                   "launch": "catapult", "launch_m_s": s["launch_m_s"], "recovery": "net", "swap_s": 120.0, "reserve": 0.15,
                   "price_eur": s["price_eur"]}
    return card, models


MACHINES = {"cyclopterus": cyclopterus, "shrimp": shrimp, "workboat": workboat, "pen": salmon_pen, "barge": feed_barge,
            "scampi": scampi, "echensis": echensis, "ship_hull": ship_hull, "wellboat": wellboat,
            "homarus": homarus, "jacket": jacket, "dp_vessel": dp_vessel, "tms": tms, "manifold": manifold,
            "squilla": squilla, "resident_dock": resident_dock, "riser_clamp": riser_clamp, "debris": debris,
            "nephro": nephro, "trencher": trencher, "cable_vessel": cable_vessel, "cable": cable, "cable_joint": cable_joint,
            "walrus": walrus, "clam_basket": clam_basket, "clam_boat": clam_boat,
            "pinna": pinna, "gannet": gannet, "fulmar": fulmar}


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
