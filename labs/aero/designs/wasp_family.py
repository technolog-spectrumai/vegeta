"""The WASP family: the two official builds, one way in to each, and the builds they replaced.

``wasp`` (ÉMERILLON small, for speed) and ``wasp_micro`` (smaller still, with a payload bay) each produced two builds.
On one basis (``compare``: the same payloads, the same delivery range at 100 km/h) two cover what all four did:

- **Cotesia** (first WASP-ML, then WASP-M2) — the small carrier: 176 g with a 10 g payload, about 200 km/h, 6 km.
  Named after the small parasitoid wasps that deliver their load precisely;
- **Vespula** (first WASP-1F) — the fast one: 290 g, about 290 km/h, the most spare thrust. Named after the
  yellowjackets.

``OFFICIAL`` is the family. Everything a script needs of a build goes through this module, whichever of the two
studies it came from: ``ctx`` (the model at that build's scale), ``vehicle`` (mass, CG, inertia, speeds, thrust),
``cad`` / ``cad_parameters``, ``card`` (results and the bill of materials), ``export_yx2`` (the Yax exchange file).
``RETIRED`` says what each dropped build was and why; they stay computable through ``BUILDS``.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import emerillon_flight as ef
import emerillon_market as mk
import wasp
import wasp_micro as wm

BUILDS = {  # name -> (configuration, wasp_micro's airframe constants?, the payload it is quoted with [g])
    "Cotesia": (wm.WASP_M2, True, 10.0), "Vespula": (wasp.WASP1_FAST, False, 0.0),
    "WASP-M": (wm.WASP_M, True, 10.0), "WASP-1": (wasp.WASP1, False, 0.0),
}
OFFICIAL = {
    "Cotesia": "the small carrier: under 250 g with its payload, 20 g of load, 200 km/h, the cheapest way to move a few grams a few kilometres",
    "Vespula": "the fast one: the highest top speed and thrust margin, and the longest range with a payload",
}
KEPT = OFFICIAL
RETIRED = {
    "WASP-1": "between the two kept builds: slower and shorter-ranged than Vespula; against Cotesia it is 65 g heavier and goes two thirds "
              "as far for 20 km/h more top speed; its one reason was the 250 g line, which Cotesia is also under",
    "WASP-M": "Cotesia with a smaller pack: 40 g lighter, the same speed, less than half the range; nothing it does that Cotesia does not",
}


def slug(name: str) -> str:
    return name.lower().replace("-", "_")


def config(name: str):
    return BUILDS[name][0]


def ctx(name: str, payload_g: float | None = None):
    """The model at this build's scale, with its payload aboard (a context manager)."""
    c, micro, pay = BUILDS[name]
    return wm.micro(pay if payload_g is None else payload_g, constants=micro)


def fixed(name: str) -> list:
    return wm.FIXED if BUILDS[name][1] else wasp.FIXED


def cad_parameters(name: str) -> dict:
    c, micro, pay = BUILDS[name]
    return wm.cad_parameters(c, pay) if micro else wasp.cad_parameters(c)


def cad(name: str, **over):
    """The build as Dedalus geometry (``part``: aircraft, arm, nose; ``angle_of_attack_deg`` pitches it for the CFD)."""
    c, micro, pay = BUILDS[name]
    return wm.cad(c, pay, **over) if micro else wasp.cad(c, **over)


def vehicle(name: str):
    """What the solver scripts ask of a build: mass, CG and inertia (the budget), the geometry, the top speed and its
    rpm, the static thrust of one motor, the motor unit's mass."""
    c, micro, pay = BUILDS[name]
    with ctx(name):
        b = ef.mass_budget(c)
        pr = b.properties()
        top = ef.top_speed(c, b.mass)
        motor, prop = mk.part(c.motor), mk.part(c.propeller)
        return SimpleNamespace(name=name, config=c, label=c.label(), payload_g=pay, mass=b.mass, x_cg=float(pr.com[0]),
                               inertia=[float(pr.inertia[i][i]) for i in range(3)], inertia_matrix=np.asarray(pr.inertia), geo=ef.geometry(c),
                               V_top=float(top["V"]), rpm_top=float(top["rpm"]), thrust_top=float(top["thrust"]),
                               static_thrust=float(ef.drive(c, 0.5)["thrust"]) / 4, motor_unit_kg=(motor.mass_g + prop.mass_g) / 1e3, budget=b)


def card(name: str) -> dict:
    """One build: the delivery evaluation (speed, thrust, range with its payload), the mass by group, the printed
    grams, the bill of materials."""
    c, micro, pay = BUILDS[name]
    r = wm.delivery(c, pay, constants=micro)
    esc = mk.part(c.esc)
    with ctx(name):
        printed = ef.printed_masses(c)
    parts = [(4, mk.part(c.motor)), (4, mk.part(c.propeller)), (1, esc), (1, mk.part(c.battery))] + \
            [(1, q) for q in fixed(name) if not (q.group == "radio" and getattr(esc, "with_rx", False))]
    r.update({"name": name, "printed_g": printed,
              "bom": [{"qty": n, "part": q.name, "g_each": q.mass_g, "usd_each": q.price_usd, "kind": q.kind, "url": q.url, "note": q.note} for n, q in parts]})
    return r


def export_yx2(name: str, path, triangles: int = 20000) -> dict:
    """The build as a Yax exchange file (``.yx2``): one link with the budget's mass and inertia about its CG, the CAD as
    its visual mesh, a box hull for contact; the rotors, motor, battery, payload bay and the model's results as data.
    Frame: forward +x, up +z, the origin at the CG (the CAD's is x aft from the nose). The propellers are data, not
    links."""
    import tempfile
    import pyvista as pv
    import yax
    from yax import hub
    c, micro, pay = BUILDS[name]
    veh, r = vehicle(name), card(name)
    geo = veh.geo
    cg = np.array([veh.x_cg, 0.0, 0.0])
    flip = np.diag([-1.0, -1.0, 1.0])
    with tempfile.TemporaryDirectory() as tmp:
        stl = Path(tmp) / "body.stl"
        cad(name).export_stl(str(stl), tolerance=0.05)
        mesh = pv.read(str(stl)).triangulate().clean()
    if mesh.n_cells > triangles:
        mesh = mesh.decimate(1 - triangles / mesh.n_cells)
    pts = (np.asarray(mesh.points) / 1000.0 - cg) @ flip.T
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    rr, L = geo["r_arm"] / math.sqrt(2), geo["length"]
    link = yax.Link(name="body", mass=float(veh.mass), com=(0.0, 0.0, 0.0), inertia=flip @ veh.inertia_matrix @ flip.T, shapes=[
        hub.Shape(name="cad", kind="mesh", role="visual", points=pts, faces=faces, uri=f"vegeta designs:wasp_family.{name}", rgba=(0.91, 0.35, 0.05, 1.0)),
        hub.Shape(name="hull", kind="box", role="collision", size=(L / 2, rr, rr), pos=(float(veh.x_cg - L / 2), 0.0, 0.0))])
    prop, motor, bat, esc = (mk.part(k) for k in (c.propeller, c.motor, c.battery, c.esc))
    rotors = [{"name": f"rotor_{i + 1}", "pos": [float(veh.x_cg - geo["x_prop"]), -sy * rr, sz * rr], "axis": [1.0, 0.0, 0.0], "spin": int(sy * sz),
               "diameter_m": geo["prop_d"], "pitch_m": prop.p_in * 0.0254, "blades": prop.blades, "kg": prop.mass_g / 1e3}
              for i, (sy, sz) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1)))]
    cr = r["cruise"]
    data = {"config": c.label(), "payload_kg": pay / 1e3, "rotors": rotors,
            "motor": {"name": motor.name, "kv_rpm_per_V": motor.kv, "resistance_ohm": motor.r_ohm, "max_current_A": motor.max_a, "kg": motor.mass_g / 1e3},
            "battery": {"name": bat.name, "cells": bat.cells, "capacity_Ah": bat.ah, "energy_Wh": bat.cells * 3.7 * bat.ah, "kg": bat.mass_g / 1e3},
            "board": esc.name, "mass_groups_kg": {k: v / 1e3 for k, v in r["mass_groups_g"].items()},
            "model": {"top_speed_m_s": float(veh.V_top), "static_thrust_N": float(4 * veh.static_thrust), "cruise_m_s": wm.CRUISE_KMH / 3.6,
                      "cruise_power_W": None if cr is None else float(cr["electric_w"]), "range_m": None if r["range_km"] is None else float(r["range_km"]) * 1e3,
                      "hover_power_W": float(r["hover_w"])},
            "dimensions_m": {"length": L, "body_diameter": geo["d"], "across_propeller_tips": float(r["across_tips"]), "motor_axis_radius": geo["r_arm"]}}
    if pay:
        data["payload_bay"] = {"bore_m": wm.BAY_D, "pos": [float(veh.x_cg - geo["nose"]), 0.0, 0.0]}
    m = yax.Machine(name=slug(name), links=[link], joints=[],
                    data={"wasp": data, "provenance": {"design": "labs/aero/designs/wasp_family.py", "notes": "mass and inertia from the mass budget (point masses at "
                          "the parts' stations); performance from emerillon_flight's model on shop-listed parts; nothing built or tested"}},
                    producer={"tool": "vegeta labs/aero wasp_family", "machine": slug(name), "lab": "aero"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = yax.save(m, path)
    back, _ = yax.read(path)
    return {"file": str(path), "bytes": path.stat().st_size, "report": str(report), "mass_kg": float(back.links[0].mass),
            "triangles": int(len(back.links[0].shapes[0].faces)), "links": len(back.links)}


def compare(payloads=(0.0, 10.0), names=None) -> list:
    rows = []
    for name in names or BUILDS:
        c, micro, _ = BUILDS[name]
        for pay in payloads:
            r = wm.delivery(c, pay, constants=micro)
            cr = r["cruise"]
            rows.append({"build": name, "official": name in OFFICIAL, "payload_g": pay, "mass_g": round(r["mass_g"], 1), "top_kmh": round(r["kmh"], 0),
                         "static_tw": round(float(r["static_tw"]), 2), "length_mm": round(r["length"] * 1e3), "across_tips_mm": round(r["across_tips"] * 1e3),
                         "battery_wh": round(r["wh"], 2), "cruise_w": None if cr is None else round(cr["electric_w"], 1), "hover_w": round(float(r["hover_w"]), 1),
                         "range_km": None if r["range_km"] is None else round(r["range_km"], 1), "minutes": None if r["minutes"] is None else round(r["minutes"], 1),
                         "price_usd": round(r["price"], 0), "motor_dT": round(float(r["motor_dT"])), "fails": r["fails"], "label": r["label"]})
    return rows
