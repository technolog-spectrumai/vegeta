"""Parametric CAD of Auseklis (CadQuery, via Dedalus): bus, tanks, thrusters, deployed solar arrays, antennas, sensors,
the capture system and equipment compartments, for the three variants A / B / C of ``variants.py``.

Frame (mm): +X toward the target (capture end), -X aft (main thrusters), +Y/-Y solar wings, +Z zenith, -Z nadir.
Origin at the bus centre. Tank sizes follow the propellant masses of the closed budget (filled to ``wet_kg``)."""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Parameter
from vegeta.labkit.assembly import Assembly

from . import variants as V

BULK = {"A": 1200.0, "B": 1250.0, "C": 1500.0}            # kg/m3: storable bipropellant, ADN monoprop, dense xenon (ASSUMED)


def cyl(r, h, axis="x", at=(0, 0, 0)):
    """Cylinder of radius ``r`` and length ``h`` centred at ``at``, along ``axis``."""
    plane = {"x": "YZ", "y": "XZ", "z": "XY"}[axis]
    c = cq.Workplane(plane).circle(r).extrude(h / 2, both=True)
    return c.translate(at)


def box(x, y, z, at=(0, 0, 0)):
    return cq.Workplane("XY").box(x, y, z).translate(at)


def capsule_tank(volume_l: float, radius: float):
    """Cylindrical tank with hemispherical domes along X, centred at the origin, holding ``volume_l`` litres.
    Returns (solid, overall length mm)."""
    v = volume_l * 1e6                                    # mm3
    length = max((v - 4 / 3 * math.pi * radius ** 3) / (math.pi * radius ** 2), 0.0)
    out = cq.Workplane("XY").sphere(radius).translate((length / 2, 0, 0)).union(cq.Workplane("XY").sphere(radius).translate((-length / 2, 0, 0)))
    if length > 1:
        out = out.union(cyl(radius, length, "x"))
    return out, length + 2 * radius


def tank_radius(volume_l: float, free_x: float, free_y: float, n_tanks: int) -> float:
    """Smallest radius whose tank fits the bus (length <= 0.9 free_x, side by side across 0.9 free_y)."""
    r = 80.0
    while r < 0.5 * free_y / max(n_tanks, 1) * 0.9:
        if capsule_len(volume_l, r) <= 0.9 * free_x:
            return r
        r += 5.0
    raise ValueError(f"a {volume_l:.0f} l tank does not fit the bus: enlarge bus_x/bus_y or lower wet_kg")


def capsule_len(volume_l: float, radius: float) -> float:
    return max((volume_l * 1e6 - 4 / 3 * math.pi * radius ** 3) / (math.pi * radius ** 2), 0.0) + 2 * radius


class Auseklis(Assembly):
    """Auseklis servicer. ``variant`` picks propulsion + capture hardware; the rest are free geometry parameters."""

    parameters = [
        Parameter("variant", "A", choices=("A", "B", "C"), description="A bipropellant+arm, B green monoprop+docking, C Hall+gripper"),
        Parameter("wet_kg", 400.0, "kg", min=350, max=500, description="wet mass the tanks are filled to"),
        Parameter("bus_x", 900.0, "mm", min=500), Parameter("bus_y", 800.0, "mm", min=400), Parameter("bus_z", 850.0, "mm", min=400),
        Parameter("panel_t", 25.0, "mm", min=10, description="honeycomb sandwich panel thickness"),
        Parameter("wing_panels", 3, min=1, max=6, description="panels per wing"),
        Parameter("panel_span", 1000.0, "mm", min=300, description="panel length along Y"),
        Parameter("panel_chord", 1050.0, "mm", min=300, description="panel width along X"),
        Parameter("arm_length", 1800.0, "mm", min=300, description="arm reach (variant A) / boom length scale (C)"),
        Parameter("dish_diameter", 300.0, "mm", min=0, description="X-band dish (0 = none)"),
    ]

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _bus(self, p):
        X, Y, Z, t = p["bus_x"], p["bus_y"], p["bus_z"], p["panel_t"]
        shell = box(X, Y, Z).cut(box(X - 2 * t, Y - 2 * t, Z - 2 * t))
        deck = box(X - 2 * t, Y - 2 * t, t, at=(0, 0, 0.2 * Z))            # equipment deck above the tanks
        return shell.union(deck)

    def _compartments(self, p):
        X, Y, Z = p["bus_x"], p["bus_y"], p["bus_z"]
        zd = 0.2 * Z + p["panel_t"] / 2                                  # top of the equipment deck
        out = {"avionics_box": box(300, 220, 110, at=(0.0, 0.2 * Y, zd + 55)),
               "battery_box": box(260, 200, 140, at=(0.0, -0.2 * Y, zd + 70))}
        wheels = None
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):          # four wheels (simplified as cylinders), at the deck corners
            w = cyl(70, 45, "z", at=(0.30 * X * sx, 0.28 * Y * sy, zd + 22))
            wheels = w if wheels is None else wheels.union(w)
        out["wheels"] = wheels
        return out

    def _tanks(self, p):
        b = V.budget(V.VARIANTS[p["variant"]], wet_target=p["wet_kg"])
        key, t = p["variant"], p["panel_t"]
        fx, fy = p["bus_x"] - 2 * t, p["bus_y"] - 2 * t
        vol = b["propellant"] / BULK[key] * 1e3 / 0.9                    # litres incl. 10 % ullage
        split = [("tank_fuel", 0.45 * vol), ("tank_oxidiser", 0.55 * vol)] if key == "A" else [("tank_main", vol)]
        r = max(tank_radius(l, fx, fy, len(split)) for _, l in split)
        zc = -0.5 * p["bus_z"] + t + r + 10                              # tanks rest on the floor, under the equipment deck
        out = {}
        for i, (name, litres) in enumerate(split):
            tank, _ = capsule_tank(litres, r)
            y = 0.0 if len(split) == 1 else (-1) ** i * (r + 15)
            out[name] = tank.translate((-0.02 * p["bus_x"], y, zc))
        if key == "C":                                                   # RCS tank
            out["tank_rcs"] = cq.Workplane("XY").sphere(90).translate((0.3 * p["bus_x"], 0, 0.5 * p["bus_z"] - t - 100 - 0.3 * p["bus_z"] * 0))
        return out

    def _thrusters(self, p):
        X, Y, Z = p["bus_x"], p["bus_y"], p["bus_z"]
        out = {}
        aft = -X / 2
        if p["variant"] == "C":
            for i, y in enumerate((-0.18 * Y, 0.18 * Y)):                # Hall thruster + cold spare
                out[f"hall_thruster_{i}"] = cyl(65, 90, "x", at=(aft - 45, y, -0.1 * Z))
        else:
            for i, y in enumerate((-0.2 * Y, 0.2 * Y)):                  # 22 N / 5 N class engines: chamber + nozzle cone
                ch = cyl(30, 110, "x", at=(aft - 55, y, 0))
                noz = cq.Workplane("YZ").circle(30).workplane(offset=-120).circle(70).loft().translate((aft - 110, y, 0)).rotate((aft - 110, y, 0), (aft - 110, y, 1), 0)
                out[f"main_engine_{i}"] = ch.union(noz)
        k = 0
        for sx in (1, -1):                                               # RCS clusters on the four +/-x corners, firing +/-y
            for sz in (1, -1):
                for sy in (1, -1):
                    out[f"rcs_{k}"] = cyl(18, 55, "y", at=(sx * (X / 2 - 40), sy * (Y / 2 + 25), sz * (Z / 2 - 40)))
                    k += 1
        return out

    def _arrays(self, p):
        Y, n, span, chord, t = p["bus_y"], p["wing_panels"], p["panel_span"], p["panel_chord"], 25.0
        out = {}
        for side, s in (("pos", 1), ("neg", -1)):
            yoke = cyl(25, 250, "y", at=(0, s * (Y / 2 + 125), 0))
            wing = yoke
            for i in range(n):
                y0 = s * (Y / 2 + 250 + span / 2 + i * (span + 15))
                wing = wing.union(box(chord, span, t, at=(0, y0, 0)))
            out[f"array_{side}"] = wing
        return out

    def _antennas_sensors(self, p):
        X, Y, Z = p["bus_x"], p["bus_y"], p["bus_z"]
        out = {"sband_patch_zenith": box(110, 110, 12, at=(0, 0.2 * Y, Z / 2 + 6)),
               "sband_patch_nadir": box(110, 110, 12, at=(0, 0.2 * Y, -Z / 2 - 6))}
        if p["dish_diameter"] > 0:
            r = p["dish_diameter"] / 2
            dish = cq.Workplane("XZ").moveTo(0, 0).threePointArc((r * 0.6, 0.12 * r), (r, 0.38 * r)).lineTo(r, 0.38 * r + 4).threePointArc((r * 0.6, 0.12 * r + 4), (0, 4)).close().revolve(360, (0, 0, 0), (0, 1, 0))
            out["xband_dish"] = dish.translate((0.15 * X, -0.25 * Y, Z / 2 + 230)).union(cyl(8, 230, "z", at=(0.15 * X, -0.25 * Y, Z / 2 + 115)))
        for i, y in enumerate((-0.25 * Y, 0.25 * Y)):                    # star trackers with sun baffles, on the zenith face
            out[f"star_tracker_{i}"] = cyl(45, 130, "z", at=(-0.3 * X, y, Z / 2 + 65))
        out["lidar"] = cyl(60, 90, "x", at=(X / 2 + 45, 0.25 * Y, 0.25 * Z)).union(cq.Workplane("XY").sphere(55).translate((X / 2 + 110, 0.25 * Y, 0.25 * Z)))
        out["camera_wide"] = cyl(30, 80, "x", at=(X / 2 + 40, -0.25 * Y, 0.25 * Z))
        out["camera_narrow"] = cyl(40, 180, "x", at=(X / 2 + 90, -0.25 * Y, -0.25 * Z))
        out["thermal_ir_camera"] = cyl(28, 70, "x", at=(X / 2 + 35, 0.25 * Y, -0.25 * Z))
        return out

    def _capture(self, p):
        X, Y, Z = p["bus_x"], p["bus_y"], p["bus_z"]
        x0 = X / 2
        v = p["variant"]
        out = {}
        if v == "A":                                                     # two-link arm, partly extended, with a two-finger end effector
            L = p["arm_length"]
            l1, l2 = 0.5 * L, 0.5 * L
            base = cyl(70, 120, "x", at=(x0 + 60, 0, -0.05 * Z))
            link1 = cyl(45, l1, "x", at=(x0 + 120 + l1 / 2, 0, -0.05 * Z))
            elbow = cq.Workplane("XY").sphere(60).translate((x0 + 120 + l1, 0, -0.05 * Z))
            ang = math.radians(25)
            end = (x0 + 120 + l1 + l2 * math.cos(ang), 0, -0.05 * Z + l2 * math.sin(ang))
            mid = ((x0 + 120 + l1 + end[0]) / 2, 0, (-0.05 * Z + end[2]) / 2)
            link2 = cyl(38, l2, "x").rotate((0, 0, 0), (0, 1, 0), -25).translate(mid)
            wrist = cq.Workplane("XY").sphere(48).translate(end)
            fingers = None
            for s in (1, -1):
                f = box(220, 24, 40, at=(end[0] + 130, s * 70, end[2]))
                fingers = f if fingers is None else fingers.union(f)
            out.update({"capture_arm_base": base, "capture_arm_link1": link1.union(elbow), "capture_arm_link2": link2.union(wrist),
                        "capture_gripper": fingers})
        elif v == "B":                                                   # docking adapter: ring + capture cone
            # ring and cone stay inside r = 180 mm: the LiDAR and cameras sit >= 205 mm off the axis and must keep a clear view
            ring = cyl(180, 40, "x", at=(x0 + 20, 0, 0)).cut(cyl(140, 60, "x", at=(x0 + 20, 0, 0)))
            cone = cq.Workplane("YZ").circle(175).workplane(offset=170).circle(60).loft().translate((x0 + 40, 0, 0))
            out.update({"docking_ring": ring, "docking_cone": cone.cut(cq.Workplane("YZ").circle(150).workplane(offset=170).circle(40).loft().translate((x0 + 40, 0, 0)))})
        else:                                                            # short boom + three-finger compliant gripper
            boom = cyl(40, p["arm_length"] * 0.45, "x", at=(x0 + p["arm_length"] * 0.225, 0, 0))
            hub = cyl(70, 90, "x", at=(x0 + p["arm_length"] * 0.45 + 45, 0, 0))
            fingers = None
            for k in range(3):
                a = math.radians(120 * k)
                f = box(260, 28, 40, at=(x0 + p["arm_length"] * 0.45 + 200, 0, 0)).translate((0, 0, 110)).rotate((0, 0, 0), (1, 0, 0), 120 * k)
                f = f.translate((0, 0, 0))
                fingers = f if fingers is None else fingers.union(f)
            out.update({"capture_boom": boom, "capture_hub": hub, "capture_gripper": fingers.translate((0, 0, 0)) if fingers is not None else hub})
        return out

    # ---- Assembly -----------------------------------------------------------------------------------------------
    def parts(self, p):
        parts = {"bus_shell": self._bus(p)}
        for fn in (self._compartments, self._tanks, self._thrusters, self._arrays, self._antennas_sensors, self._capture):
            parts.update(fn(p))
        return {k: (v.val() if hasattr(v, "val") else v) for k, v in parts.items()}


CATEGORY = {"bus_shell": "structure", "avionics_box": "avionics", "battery_box": "battery", "wheels": "gnc",
            "star_tracker": "gnc", "lidar": "prox_sensors", "camera": "prox_sensors", "thermal_ir": "prox_sensors",
            "sband": "comms", "xband": "comms", "array": "arrays", "tank": "tanks", "main_engine": "propulsion_hw",
            "hall_thruster": "propulsion_hw", "rcs": "propulsion_hw", "capture": "capture", "docking": "capture"}


def _category(name: str) -> str:
    return next(c for k, c in CATEGORY.items() if name.startswith(k))


def mass_properties(**overrides) -> dict:
    """Mass, centre of gravity [m] and inertia tensor [kg m^2] about the CoG, from the CAD and the budget.
    Each budget category's mass is spread over its parts by volume and each part is a point mass at its centre of volume
    plus the inertia of its bounding box (an estimate; replace with part masses from the BOM)."""
    d = Auseklis()
    p = d.resolve(**overrides)
    parts = d.parts(p)
    b = V.budget(V.VARIANTS[p["variant"]], wet_target=p["wet_kg"])
    sub = dict(b["subsystems"])
    margin = b["dry_with_margin"] / b["dry_nominal"]
    pool = {"structure": sub["structure"] + sub["thermal"] + sub["harness"] + sub["battery_pcdu"] * 0.5 + sub["comms"] * 0.0,
            "avionics": sub["avionics"], "battery": sub["battery"] + sub["battery_pcdu"] * 0.5, "gnc": sub["gnc"],
            "prox_sensors": sub["prox_sensors"], "comms": sub["comms"], "arrays": sub["arrays"], "capture": sub["capture"],
            "propulsion_hw": sub["propulsion_hw"], "tanks": b["tanks"] + b["propellant"] + b["rcs_propellant"]}
    groups = {}
    for n, s in parts.items():
        groups.setdefault(_category(n), []).append(n)
    items = []
    for cat, names in groups.items():
        vols = np.array([parts[n].Volume() for n in names])
        mass_cat = pool[cat] * (margin if cat not in ("tanks",) else 1.0)
        for n, vol in zip(names, vols):
            bb = parts[n].BoundingBox()
            m = mass_cat * vol / vols.sum()
            c = parts[n].Center()
            dims = np.array([bb.xlen, bb.ylen, bb.zlen]) / 1e3
            Ibox = m / 12 * np.array([dims[1] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[1] ** 2])
            items.append((m, np.array([c.x, c.y, c.z]) / 1e3, np.diag(Ibox)))
    M = sum(i[0] for i in items)
    cog = sum(i[0] * i[1] for i in items) / M
    I = np.zeros((3, 3))
    for m, r, Ib in items:
        d_ = r - cog
        I += Ib + m * (np.dot(d_, d_) * np.eye(3) - np.outer(d_, d_))
    return {"mass_kg": M, "cog_m": cog, "inertia_kgm2": I, "budget_wet_kg": b["wet"], "array_area_m2": 2 * p["wing_panels"] * p["panel_span"] * p["panel_chord"] / 1e6}


def export_step(path: str, **overrides) -> str:
    """Write the named assembly as STEP."""
    Auseklis().assembly(**overrides).save(path, "STEP")
    return path
