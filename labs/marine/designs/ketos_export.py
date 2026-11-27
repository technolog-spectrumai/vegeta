"""Export the marine lab's farm machines to Ketos (the fjord salmon-farm simulator): for each machine a JSON
triangle mesh per coloured part, in metres, x forward, z up (Ketos's ``models/`` format, the nomios model of schema
2.3), and a physics card (mass, displacement, size, thrusters, battery, power, tools) that Ketos's vehicle model
reads. The meshes come from the CadQuery models; the numbers from each design's sizing.

    python ketos_export.py OUT_DIR            # e.g. ../../../exports/ketos, or ketos/models/vegeta
    python ketos_export.py OUT_DIR --only cyclopterus,shrimp

The card is the contract between Vegeta and Ketos (``ketos.card`` 1.0):

    {"schema":"ketos.card","version":"1.0","name":"cyclopterus","kind":"rov"|"crawler"|"boat"|"hull"|"pen"|"barge"|"air"|"craft",
     "parts":[{"model":"cyclopterus_hull","color":[r,g,b]}], "size":[L,W,H], "mass_kg":..., "displacement_L":...,
     "thrusters":[{"at":[x,y,z],"axis":[ax,ay,az],"max_N":...}], "battery_Wh":..., "hotel_W":..., "tools":{...}, ...}

An ``air`` card (a drone that flies: Gannet, Fulmar) has ``air`` ("vtol" | "fixed_wing"), ``mass_kg`` (empty, with
the pack), ``mtow_kg``, ``payload_kg``, ``battery_Wh`` / ``usable_Wh``, ``cruise_m_s``, ``power_cruise_W`` (+ ``_empty``),
``power_hover_W`` (+ ``_empty``; None for a fixed-wing), ``wh_per_km`` (+ ``_empty``), ``range_km``, ``radius_km``,
``endurance_min``, ``wind_limit_m_s``, ``deck_hs_limit_m`` (the landing on a deck's sea-state limit), ``launch`` and
``recovery`` ("pad" | "catapult" | "net"), ``swap_s``.

The UFOs (the alien pirates' fictional saucer and cigar; ``labs/aero/designs/ufo.py``) are ``craft`` cards with
``airborne`` and ``fictional`` true: the craft fields as the boats' (``speed_vs_hs`` flat, ``ais`` False), and beside
them ``model`` ("helicopter" | "airplane": how it moves), ``kinematics`` (the limits ``ufo.step`` applies),
``collision`` (primitives in the card's frame) and ``bounding_radius_m``, ``flak`` (time in each gun's range on typical
passes, tracking rate, lead), ``flag``. A part may carry ``texture`` (an image file written beside the meshes); its
model then has ``uv`` (two per vertex, (0, 0) the image's bottom left) and ``texture``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
    import cadquery as cq
    from scampi import ARMS, EFFECTORS, Scampi, THRUSTER, PARTS
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
    # the drawn vehicle: body, thrusters, the tool skid with the holder. The arms and the tools are not in it: a
    # viewer animates them (the folded arms are exported apart, ``scampi_arms``, for a static picture)
    skid = d._tool_skid(p).union(d._tool_holder(p))
    arms = d.build(p | {"part": "work_arm"}).union(d.build(p | {"part": "hold_arm"}))
    arms = arms.union(d._gripper(p, "hold", 0.0)).union(d._gripper(p, "work", 0.0))
    parts = [("scampi_body", body, "#f1c40f"), ("scampi_thrusters", th, "#222222"), ("scampi_skid", skid, "#27ae60")]
    models = [_mesh(sh, n, _hex(c), 5.0) for n, sh, c in parts]
    extra = [_mesh(arms, "scampi_arms", _hex("#c0392b"), 5.0), _mesh(d._cutter(p), "scampi_cutter_parked", _hex("#8e44ad"), 5.0)]
    dz = _centre_z(models + extra)
    # the end effectors as tools in their own frame (metres; origin the wrist flange, +x out of the wrist, jaws
    # opening along y, z up): the viewer puts them on its arms and moves the jaws and the blade
    tools = [("scampi_gripper_l", d._gripper_local(0.0, jaws=False), "#1b4f72"),
             ("scampi_gripper_r", d._gripper_local(0.0, jaws=False), "#7b241c"),
             ("scampi_gripper_jaw", cq.Workplane("XY").box(72, 14, 36).union(cq.Workplane("XY").box(40, 22, 30).translate((16, 0, 0))), "#5d6d7e"),
             ("scampi_cutter", d._cutter_local(blade=False), "#8e44ad"),
             ("scampi_cutter_blade", d._cutter_blade_local().translate((-215, 0, -19)), "#d0d3d4")]
    tool_models = [_mesh(sh, n, _hex(c), 1.0) for n, sh, c in tools]
    ef = Scampi.effectors(p)
    g, c = EFFECTORS["gripper"], EFFECTORS["cutter"]
    slots = {s: d.holder_pose(p, s)[0] for s in d.HOLDER_SLOTS}
    grip = lambda arm, model: {"arm": arm, "model": model, "jaw_model": "scampi_gripper_jaw", "jaw_at_x_m": 0.134,
                               "jaw_offset_m": 0.007, "tip_m": 0.17, "jaw_opening_m": [0.0, g["jaw_opening_mm"][1] / 1000],
                               "grip_force_N": g["grip_force_N"], "close_s": g["close_s"],
                               "wrist_torque_Nm": ARMS[arm]["wrist_torque_Nm"],
                               "payload_full_reach_kg": ARMS[arm]["payload_full_reach_kg"],
                               "mass_kg": g["mass_kg"], "displacement_L": g["volume_L"], "power_W": g["power_W"],
                               "price_eur": g["price_eur"]}
    effectors = {
        "gripper_l": grip("hold", "scampi_gripper_l"), "gripper_r": grip("work", "scampi_gripper_r"),
        "cutter": {"arm": "work", "quick_change": True, "model": "scampi_cutter", "blade_model": "scampi_cutter_blade",
                   "blade_at_m": [0.215, 0.0, 0.019], "blade_stroke_m": 0.034, "tip_m": 0.24, "throat_mm": c["throat_mm"],
                   "cut_force_kN": c["cut_force_kN"], "cycle_s": c["cycle_s"],
                   "max_cut_mm": {m: round(v, 1) for m, v in ef["max_cut_mm"].items()},
                   "mass_kg": c["mass_kg"], "displacement_L": c["volume_L"], "power_W": c["power_W"], "price_eur": c["price_eur"],
                   "holder": {"slots": {s: [round(v.x / 1000, 3), round(v.y / 1000, 3), round(v.z / 1000 - dz, 3)] for s, v in slots.items()},
                              "tool_dir": [-1.0, 0.0, 0.0], "starts_in": "cutter",
                              "what": "flange forward, the tool aft in the cradle; the arm's coupler latches the flange"},
                   "shoulder_m": {"work": [round((p["length"] / 2 - 40) / 1000, 3), round(-0.22 * p["width"] / 1000, 3)],
                                  "hold": [round((p["length"] / 2 - 40) / 1000, 3), round(0.22 * p["width"] / 1000, 3)]},
                   "swap_s": 60.0},
    }
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
                   "tools": {k: {"mass_kg": v[0], "power_W": v[1], "price_eur": v[2], "what": v[3]} for k, v in SCAMPI_TOOLS.items()}
                   | {k: {"count": EFFECTORS[k]["count"], "mass_kg": EFFECTORS[k]["mass_kg"], "power_W": EFFECTORS[k]["power_W"],
                          "price_eur": EFFECTORS[k]["price_eur"], "what": EFFECTORS[k]["what"]} for k in ("gripper", "cutter")},
                   "effectors": effectors,
                   # + the effectors (grippers, cutter, holder: EUR 28 k) x the same 1.6 build and margin
                   "tether": None, "resident": True, "price_eur": 190000 + round(1.6 * ef["price_eur"])}
    return card, models + extra + tool_models


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
    import squilla_dock as sd
    r = sd.report()
    card |= {"leg": "A1", "legs_that_fit_at_depth": r["fits_at_80_on"], "fitting_depths_m": r["fitting_depths"],
             "clearance_m": r["best"]["clearance_m"] if r["best"] and r["best"]["leg"] == "A1" and r["best"]["depth_m"] == -p["depth"] else
             next(x["clearance_m"] for x in r["at_80"] if x["leg"] == "A1"),
             "mount": r["mounting"], "mount_mass_kg": r["mounting"]["collar_mass_kg"], "services": r["services"], "docking": r["docking"]}
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


def zarco():
    """Zarco, the autonomous drone-carrier catamaran (Ketos Clams stages 6-8): Walrus davits and reels, the basket
    robot and the packer, four VTOL pads, the catapult and the recovery net, the autonomy mast. z = 0 the waterline."""
    from zarco import AUTONOMY, LOADS, ROBOTS, Zarco, davit_card
    d, p, s = Zarco(), Zarco.p(), Zarco.sizing()
    parts = [("zarco_hulls", "hulls", "#34495e"), ("zarco_deck", "deck", "#95a5a6"), ("zarco_house", "house", "#ecf0f1"),
             ("zarco_flight_deck", "flight_deck", "#7f8c8d"), ("zarco_pads", "pads", "#f1c40f"), ("zarco_catapult", "catapult", "#c0392b"),
             ("zarco_net", "net", "#2c3e50"), ("zarco_mast", "mast", "#d0d3d4"), ("zarco_davits", "davits", "#e67e22"), ("zarco_robots", "robots", "#27ae60")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 40.0) for n, k, c in parts]
    z_fd = (d._z_deck(p) + 3600 + 150) / 1000
    card = CARD | {"name": "zarco", "kind": "boat", "autonomous": True, "crew": 0, "operators_ashore": round(s["operators_ashore"], 3),
                   "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [s["length_m"], s["beam_m"], round(p["hull_depth"] / 1000 + 7.8, 1)], "draught_m": s["draught_m"],
                   "displacement_t": s["displacement_t"], "mass_kg": round(1000 * (s["lightship_t"] + s["deadweight_t"])),
                   "service_kn": s["service_kn"], "max_kn": s["max_kn"], "service_kW": s["service_kW"], "bed_load_kW": s["bed_load_kW"],
                   "power_W": round(1000 * s["bed_load_kW"]), "battery_kWh": p["battery_kwh"], "endurance_days": s["endurance_days"],
                   "fuel_l_h_on_bed": s["fuel_l_h_on_bed"], "loads_kW": {k: round(v, 2) for k, v in LOADS.items()}, "autonomy": AUTONOMY, "robots": ROBOTS, "unmanned": True,
                   "carries": ["walrus", "gannet", "fulmar"], "davits": p["davits"], "tether_reels": p["davits"], "usbl": True,
                   "davit": davit_card(), "davit_hs_limit_m": s["davit_hs_limit_m"],
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


def barracuda():
    """Barracuda, the guard carrier's unmanned interceptor boat (``barracuda.py``): a ``boat`` card with its speeds by
    sea state, fuel by speed band, manoeuvres, sensors with their ranges, the escalation ladder's effectors, the
    collar's blocking limit and the carrier ramp's numbers. z = 0 the waterline, x = 0 amidships."""
    from barracuda import DRAUGHT_CAD_MM, Barracuda
    d, p, s = Barracuda(), Barracuda.p(), Barracuda.sizing()
    parts = [("barracuda_hull", "hull", "#7f8c8d"), ("barracuda_collar", "collar", "#e67e22"), ("barracuda_jets", "jets", "#2c3e50"),
             ("barracuda_console", "console", "#ecf0f1"), ("barracuda_mast", "mast", "#bdc3c7"), ("barracuda_cannon", "cannon", "#c0392b"),
             ("barracuda_effectors", "effectors", "#2980b9")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 15.0) for n, k, c in parts]
    for m in models:
        m["vertices"][2::3] = [round(z - DRAUGHT_CAD_MM / 1000, 4) for z in m["vertices"][2::3]]
    _x_centre(models)
    m = s["manoeuvres"]
    card = CARD | {"name": "barracuda", "kind": "boat", "role": "interceptor", "autonomous": True, "unmanned": True, "crew": 0,
                   "parts": [{"model": mm["name"], "color": mm["color"]} for mm in models],
                   "size": [s["length_overall_m"], s["beam_overall_m"], s["air_draught_mast_up_m"] + s["draught_m"]],
                   "draught_m": s["draught_m"], "freeboard_m": s["freeboard_m"], "air_draught_m": {"mast_up": s["air_draught_mast_up_m"], "mast_folded": s["air_draught_mast_folded_m"]},
                   "mass_kg": s["mass_full_kg"], "mass_half_load_kg": s["mass_half_kg"], "lightship_kg": s["lightship_kg"],
                   "fuel_L": s["fuel_L"], "propulsion": {"engines": 2, "engine_kW": s["engine_kW"], "fuel": "diesel", "propulsors": "2 waterjets with reversing buckets"},
                   "max_speed_m_s": round(s["sprint_kn_full_load"] * 0.5144, 2), "max_kn": s["sprint_kn_full_load"], "cruise_kn": s["cruise_kn"], "loiter_kn": s["loiter_kn"],
                   "thrust_N": round(m["bollard_thrust_kN"] * 1000), "reverse_thrust_N": round(m["reverse_thrust_kN"] * 1000), "cd": [0.35, 1.2, 1.0],
                   "speed_vs_hs": s["speed_vs_hs"], "fuel_bands": s["fuel_bands"], "range_nm": s["range_nm_at_cruise"], "loiter_h": s["loiter_h"],
                   "manoeuvres": m, "sensors": s["sensors"], "effectors": s["effectors"], "blocking": s["blocking"],
                   "launch_recovery": s["carrier"], "sea_state_limits": {"sustained_g": p["g_sustained"], "sprint_g": p["g_sprint"],
                                                                         "launch_recovery_hs_m": s["carrier"]["hs_limit_m"], "operating_hs_m": 2.5},
                   "autonomy": {"colregs": "rules 8-17 with radar/EO-IR/AIS fused tracks; a CPA limit of 50 m while shadowing; suspended only by the operator, "
                                           "inside a geofence round the protected platform, for a block (legal basis: flag; not settled)",
                                "links": {"radio": "MANET 4.9 GHz (video) + UHF command, line of sight to the carrier's mast: ~20 km",
                                          "satcom": "LEO broadband (global, ~50 ms) + L-band backup (~0.7 Mbit/s, ~1 s)"},
                                "lost_link": ["0-10 s: hold; the water cannon and the strobe stop at once, the hailer stops, lights stay",
                                              "10-60 s: open to >= 100 m from any contact and keep shadowing; switch to satcom",
                                              "> 5 min without any link: return to the carrier's rendezvous at 15 kn under COLREGs",
                                              "never climbs the ladder above step 1 (hail) without a live operator"]},
                   "price_eur": s["price_eur"], "class_values": True}
    return card, models


def esmeralda():
    """Esmeralda, Zarco's guard-duty variant (``esmeralda.py``): Zarco's hull, mast and flight deck; the boat bay and
    ramp for Barracuda, the guard effectors, the EO/IR turret, the guard hangar. A ``boat`` card next to Zarco's,
    with ``changed_vs_zarco``. Barracuda is drawn apart (its own card). z = 0 the waterline."""
    from esmeralda import AUTONOMY_ADDED, Esmeralda
    from zarco import AUTONOMY
    d, p, s = Esmeralda(), Esmeralda.p(), Esmeralda.sizing()
    parts = [("esmeralda_hulls", "hulls", "#34495e"), ("esmeralda_deck", "deck", "#95a5a6"), ("esmeralda_house", "house", "#ecf0f1"),
             ("esmeralda_flight_deck", "flight_deck", "#7f8c8d"), ("esmeralda_pads", "pads", "#f1c40f"), ("esmeralda_catapult", "catapult", "#c0392b"),
             ("esmeralda_net", "net", "#2c3e50"), ("esmeralda_mast", "mast", "#d0d3d4"), ("esmeralda_robots", "robots", "#27ae60"),
             ("esmeralda_ramp", "ramp", "#8e44ad"), ("esmeralda_effectors", "effectors", "#2980b9"), ("esmeralda_turret", "turret", "#16a085")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 40.0) for n, k, c in parts]
    z_fd = (d._z_deck(p) + 3600 + 150) / 1000
    card = CARD | {"name": "esmeralda", "kind": "boat", "role": "guard carrier", "variant_of": "zarco", "autonomous": True, "unmanned": True, "crew": 0,
                   "operators_ashore": round(s["operators_ashore"], 3), "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [s["length_m"], s["beam_m"], round(p["hull_depth"] / 1000 + 7.8, 1)], "draught_m": round(s["draught_m"], 2),
                   "displacement_t": s["displacement_t"], "mass_kg": round(1000 * s["mass_t"]), "trim": s["trim"],
                   "level_by_moving_fuel_m": s["level_by_moving_fuel_m"],
                   "service_kn": s["service_kn"], "max_kn": s["max_kn"], "service_kW": s["service_kW"], "station_load_kW": s["station_load_kW"],
                   "power_W": round(1000 * s["station_load_kW"]), "peak_kW": s["peak_kW"], "loads_kW": s["loads_kW"], "battery_kWh": p["battery_kwh"],
                   "endurance_days": s["endurance_days"], "fuel_l_h_on_station": s["fuel_l_h_on_station"],
                   "autonomy": AUTONOMY | AUTONOMY_ADDED, "robots": s["robots"], "carries": ["barracuda", "gannet", "fulmar"],
                   "boat": s["boat"], "bay": s["bay"], "layout": s["layout"], "sensors": s["sensors"], "effectors": s["effectors"],
                   "flight_deck": {"z_m": round(z_fd, 2), "pads": [[x / 1000, y / 1000] for x, y in d._pad_at(p)], "pad_cycle_s": p["pad_cycle_s"],
                                   "land_lock_s": 40.0, "payload_load_s": 40.0, "swap_s": 90.0, "launch_s": 30.0,
                                   "catapult": {"at": [-0.2, 0.0], "cycle_s": p["catapult_cycle_s"], "stroke_m": 7.0},
                                   "net": {"at": [-9.1, 0.0], "cycle_s": p["net_cycle_s"], "size_m": [6.0, 4.0]},
                                   "deck_hs_limit_m": p["deck_hs_limit"], "hangar": {"gannet": p["hangar_gannets"], "fulmar": p["hangar_fulmars"]},
                                   "hangar_bays": s["hangar"], "packs": {"gannet": p["gannet_packs"], "fulmar": p["fulmar_packs"]},
                                   "chargers": s["chargers"], "charging_kW": s["charging_kW"]},
                   "rates": {k: s[k] for k in ("vtol_launches_per_h", "vtol_recoveries_per_h", "fixed_wing_launches_per_h", "fixed_wing_recoveries_per_h")}
                   | {"boat_launch_s": s["boat"]["launch_s"], "boat_recovery_s": s["boat"]["recovery_s"]},
                   "sea_state_limits": {"deck_hs_m": p["deck_hs_limit"], "boat_ramp_hs_m": s["boat"]["hs_limit_m"]},
                   "changed_vs_zarco": s["changed_vs_zarco"], "price_eur": s["price_eur"], "price_with_barracuda_eur": s["price_with_barracuda_eur"],
                   "day_rate_eur": 1400.0, "class_values": True}
    return card, models


def _craft(name):
    """A small craft (``small_craft.py``): a ``craft`` card - a contact round the platform, not a robot: its size,
    displacement, speeds against the sea state, turning, acceleration, radar cross-section by aspect, lights, IR, AIS
    and crew, every number a class value with its basis. z = 0 the waterline, x = 0 amidships. No weapons."""
    import small_craft as sc
    c = sc.CRAFT[name]
    s = sc.sizing(name)
    cols = {"fishing_boat": ("#a93226", "#ecf0f1"), "pilot_boat": ("#17202a", "#e67e22"), "yacht": ("#fdfefe", "#aab7b8"),
            "skiff": ("#d6eaf8", "#566573")}[name]
    models = [_mesh(sc.build(name, "hull"), f"{name}_hull", _hex(cols[0]), 15.0),
              _mesh(sc.build(name, "superstructure"), f"{name}_superstructure", _hex(cols[1]), 15.0)]
    wl = c.get("canoe_draught_m", c["draught_m"]) if name != "skiff" else c["draught_m"]
    for m in models:
        m["vertices"][2::3] = [round(z - wl, 4) for z in m["vertices"][2::3]]
    _x_centre(models)
    zs = [z for m in models for z in m["vertices"][2::3]]
    card = CARD | {"name": name, "kind": "craft", "what": s["what"], "parts": [{"model": m["name"], "color": m["color"]} for m in models],
                   "size": [s["length_m"], s["beam_m"], round(max(zs) - min(zs), 2)], "draught_m": s["draught_m"],
                   "displacement_t": s["displacement_t"], "mass_kg": round(s["displacement_t"] * 1000), "power_kW": s["power_kW"],
                   "max_kn": s["max_kn"], "max_speed_m_s": round(s["max_kn"] * 0.5144, 2), "cruise_kn": s["cruise_kn"],
                   "speed_vs_hs": s["speed_vs_hs"], "turning_circle_m": s["turning_circle_m"], "accel_0_cruise_s": s["accel_0_cruise_s"],
                   "rcs_m2": s["rcs_m2"], "signature": {"lights": s["lights"], "ir": s["ir"], "delta_T_K": s["delta_T_K"]},
                   "ais": s["ais"], "ais_typical": s["ais_typical"], "crew": s["crew"], "behaviour": s["behaviour"],
                   "weapons": None, "class_values": True, "basis": s["basis"],
                   **{k: s[k] for k in ("max_kn_light", "fishing_kn", "sail_kn") if k in s}}
    return card, models


def fishing_boat():
    return _craft("fishing_boat")


def pilot_boat():
    return _craft("pilot_boat")


def yacht():
    return _craft("yacht")


def skiff():
    return _craft("skiff")


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
             ("gannet_skids", "skids", "#566573"), ("gannet_crate", "crate", "#f5f5f5"), ("gannet_gimbal", "gimbal", "#16a085"),
             ("gannet_ladder", "ladder", "#e74c3c")]
    models = [_mesh(d.build(p | {"part": k}), n, _hex(c), 6.0) for n, k, c in parts]
    _x_centre(models)
    e = Gannet.eoir_sizing()
    drawn = [m for m in models if m["name"] not in ("gannet_gimbal", "gannet_ladder")]
    lad = Gannet.ladder_sizing()
    card = CARD | {"name": "gannet", "kind": "air", "air": "vtol", "parts": [{"model": m["name"], "color": m["color"]} for m in drawn],
                   "size": [round(2 * p["arm"] / math.sqrt(2) / 1000 + p["prop_in"] * 0.0254, 2)] * 2 + [0.55],
                   "mass_kg": s["empty_kg"], "mtow_kg": s["mtow_kg"], "payload_kg": p["payload_kg"], "battery_Wh": p["battery_wh"],
                   "usable_Wh": s["usable_Wh"], "cruise_m_s": p["cruise"], "climb_m_s": 4.0, "cruise_alt_m": 60.0,
                   "power_cruise_W": s["cruise_W"], "power_cruise_empty_W": s["cruise_empty_W"],
                   "power_hover_W": s["hover_W"], "power_hover_empty_W": s["hover_empty_W"],
                   "wh_per_km": s["wh_per_km"], "wh_per_km_empty": s["wh_per_km_empty"], "range_km": s["range_km"], "radius_km": s["radius_km"],
                   "endurance_min": s["endurance_hover_min"], "wind_limit_m_s": s["wind_limit_m_s"], "deck_hs_limit_m": s["deck_hs_limit_m"],
                   "launch": "pad", "recovery": "pad", "swap_s": s["swap_s"], "reserve": 0.15, "price_eur": s["price_eur"],
                   "payloads": {"cargo": {"what": "a 5 kg crate on the hook", "payload_kg": p["payload_kg"], "mtow_kg": s["mtow_kg"],
                                          "power_hover_W": s["hover_W"], "power_cruise_W": s["cruise_W"], "endurance_min": s["endurance_hover_min"],
                                          "range_km": s["range_km"], "radius_km": s["radius_km"], "parts": [m["name"] for m in drawn]},
                                "eoir": {k: v for k, v in e.items() if k != "payload"} | {"what": "the surveillance pod (eoir_payload.py) on the crate hook's mount",
                                         "payload_kg": e["pod"]["mass_kg"], "parts": [m["name"] for m in models if m["name"] not in ("gannet_crate", "gannet_ladder")]},
                                "ladder": {k: v for k, v in lad.items() if k != "payload"} | {"what": "the ladder pod (eoir_payload.LADDER_POD): searchlight, strobe, loudspeaker",
                                           "payload_kg": lad["pod"]["mass_kg"], "parts": [m["name"] for m in models if m["name"] not in ("gannet_crate", "gannet_gimbal")]}}}
    return card, models


def fulmar():
    """Fulmar, the long-haul fixed-wing (labs/aero/designs/fulmar.py, on fixed_wing.FixedWing): an air card;
    catapult launch and net recovery. Nose toward +x (the CAD's x runs aft: mirrored)."""
    _aero()
    from fulmar import Fulmar
    d, p, s = Fulmar(), Fulmar.p(), Fulmar.sizing()
    models = [_mesh(d.build(p | {"part": "aircraft"}), "fulmar_body", _hex("#f39c12"), 6.0),
              _mesh(d.build(p | {"part": "aircraft", "payload": "eoir"}), "fulmar_body_eoir", _hex("#f39c12"), 6.0),
              _mesh(d.build(p | {"part": "gimbal"}), "fulmar_gimbal", _hex("#16a085"), 6.0)]
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
    e = Fulmar.eoir_sizing()
    card["payloads"] = {"cargo": {"what": "a 5 kg cargo pod", "payload_kg": p["payload_kg"], "mtow_kg": s["mtow_kg"], "power_cruise_W": s["cruise_W"],
                                  "endurance_min": s["endurance_min"], "range_km": s["range_km"], "radius_km": s["radius_km"],
                                  "launch_m_s": s["launch_m_s"], "parts": ["fulmar_body"]},
                        "eoir": {k: v for k, v in e.items() if k != "payload"} | {"what": "the surveillance pod (eoir_payload.py) in the cargo bay",
                                 "payload_kg": e["pod"]["mass_kg"], "parts": ["fulmar_body_eoir"], "gimbal_model": "fulmar_gimbal"}}
    return card, models


#: Textures the models name, and where they are made (copied beside the meshes on export).
TEXTURES = {"alien_jolly_roger.png": os.path.join(HERE, "..", "..", "aero", "designs", "data", "alien_jolly_roger.png")}


def _ufo(name):
    """A UFO (``labs/aero/designs/ufo.py``): a ``craft`` card like the small craft's, so the game takes it as it takes
    a contact - fictional game craft (``fictional``, ``airborne``) that move like a helicopter (the saucer) or a jet
    (the cigar). The craft fields (speeds in kn, ``speed_vs_hs`` - flat: the sea does not limit them - turning circles,
    acceleration, RCS by bow/beam/stern, signature, ``ais`` False) and, beside them, the kinematic limits, collision
    primitives, full signatures, flak figures and the flag (a textured panel with the owner's alien Jolly Roger).
    Saucer: z = 0 its pads' feet; cigar: z = 0 its axis; both centred on x = 0. No weapons, no AIS."""
    _aero()
    import ufo
    s = ufo.sizing(name)
    c = ufo.CRAFT[name]
    models = [_mesh(ufo.build(name, part), f"{name}_{part}", _hex(col), 10.0) for part, col in ufo.PART_COLOURS[name].items()]
    for m in models:
        m["source"] = "vegeta labs/aero/designs/ufo.py (CadQuery)"
    models.append(ufo.flag_mesh(name))
    parts = [{"model": m["name"], "color": m["color"]} | ({"texture": m["texture"]} if m.get("texture") else {}) for m in models]
    k, sig = s["kinematics"], s["signature"]
    kn = lambda v: round(v / 0.5144, 1)
    if k["model"] == "helicopter":
        v_max, v_cruise, v_start = k["v_max_m_s"], k["v_cruise_m_s"], 0.0
        turn_at = (20.0, v_cruise, v_max)
        behaviour = ("moves like a helicopter: hovers, creeps sideways and backwards, turns on the spot; patrols low and slow, "
                     "dashes at 78 m/s; appears late in a mission")
        extra = {}
    else:
        v_max, v_cruise, v_start = k["v_max_m_s"]["low"], k["v_cruise_m_s"]["low"], k["v_min_m_s"]
        turn_at = (k["v_min_m_s"], v_cruise, v_max)
        behaviour = ("moves like a jet: never below 70 m/s, turns by banking (2.5 g low); low passes at 200 m/s, "
                     "transits high out of gun range; appears late in a mission")
        extra = {"min_speed_m_s": k["v_min_m_s"], "max_speed_high_m_s": k["v_max_m_s"]["high"], "max_kn_high": kn(k["v_max_m_s"]["high"])}
    turning = {f"{kn(v):.0f}kn": round(2 * ufo.turn(name, v)["radius_m"]) for v in turn_at}       # diameters, as the boats'
    rcs = sig["rcs_m2"] | {"bow": sig["rcs_m2"]["front"], "beam": sig["rcs_m2"]["side"], "stern": sig["rcs_m2"]["rear"]}
    hot = sig["ir"]["hot_spots"][0]
    signature = {"lights": sig["lights"], "ir": f"{hot['what']}: ~{hot['T_K']:g} K over {hot['area_m2']:g} m2; skin +{sig['ir']['skin_dT_K']:g} K "
                 f"({sig['ir']['skin_dT_note']})", "delta_T_K": sig["ir"]["skin_dT_K"], "ir_hot_spots": sig["ir"]["hot_spots"],
                 "sound": sig["sound"]}
    card = CARD | {"name": name, "kind": "craft", "airborne": True, "fictional": True, "what": s["what"], "model": k["model"],
                   "parts": parts, "size": s["size"], "mass_kg": s["mass_kg"], "power_kW": None,
                   "max_kn": kn(v_max), "max_speed_m_s": v_max, "cruise_kn": kn(v_cruise), "cruise_m_s": v_cruise, **extra,
                   "speed_vs_hs": [{"hs_m": hs, "kn": kn(v_max), "limited_by": "airborne (the sea state does not limit it)"}
                                   for hs in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)],
                   "turning_circle_m": turning, "turn_rate_deg_s": ufo.turn(name, v_cruise)["rate_deg_s"],
                   "accel_0_cruise_s": round((v_cruise - v_start) / k["accel_m_s2"], 1),
                   "rcs_m2": rcs, "signature": signature, "ais": False, "ais_typical": False, "crew": s["crew"],
                   "behaviour": behaviour, "weapons": None, "class_values": True, "basis": s["basis"],
                   "kinematics": k, "collision": s["collision"], "bounding_radius_m": s["bounding_radius_m"],
                   "flak": s["flak"], "flag": s["flag"]}
    return card, models


def saucer():
    return _ufo("saucer")


def cigar():
    return _ufo("cigar")


MACHINES = {"cyclopterus": cyclopterus, "shrimp": shrimp, "workboat": workboat, "pen": salmon_pen, "barge": feed_barge,
            "scampi": scampi, "echensis": echensis, "ship_hull": ship_hull, "wellboat": wellboat,
            "homarus": homarus, "jacket": jacket, "dp_vessel": dp_vessel, "tms": tms, "manifold": manifold,
            "squilla": squilla, "resident_dock": resident_dock, "riser_clamp": riser_clamp, "debris": debris,
            "nephro": nephro, "trencher": trencher, "cable_vessel": cable_vessel, "cable": cable, "cable_joint": cable_joint,
            "walrus": walrus, "clam_basket": clam_basket, "clam_boat": clam_boat,
            "zarco": zarco, "gannet": gannet, "fulmar": fulmar, "barracuda": barracuda, "esmeralda": esmeralda,
            "fishing_boat": fishing_boat, "pilot_boat": pilot_boat, "yacht": yacht, "skiff": skiff,
            "saucer": saucer, "cigar": cigar}


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
            if m.get("texture"):
                shutil.copyfile(TEXTURES[m["texture"]], os.path.join(out, m["texture"]))
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
