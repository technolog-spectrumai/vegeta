"""ALBATROS for Ketos (notebook 50): the aircraft, its cargo hexacopter and the Émerillon carrier as Ketos machines —
meshes (one JSON per coloured part: metres, x forward, y left, z up, the origin at the centre of gravity) and a card
per machine (``ketos.card`` 1.0; kinds ``aircraft`` and ``multirotor``) with the numbers Ketos Albatros flies them by.

    python labs/aero/designs/albatros_ketos.py exports/ketos [../ketos/models/vegeta]

The aircraft's card carries the sized baseline (``albatros_flight.size("catapult")``): masses, the speeds and rates of
each phase of the sortie and its fuel flow (the phase's fuel over its time), the catapult and the pass. The
hexacopter's card carries the release (``albatros_drop``: the height it uses, the time to its hover, the pass height)
and its own flight; the Émerillon's the carrier of ``albatros_flood``. The marine lab's exporter
(``labs/marine/designs/ketos_export.py``) writes the same two formats for the ships and the underwater machines.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def mesh(shape, name: str, color, transform, tolerance: float = 2.0, angular: float = 0.35) -> dict:
    """A CadQuery shape as a Ketos mesh: tessellated, moved by ``transform`` (an (n, 3) array in mm -> metres)."""
    verts, tris = shape.tessellate(tolerance, angular)
    v = transform(np.array([[p.x, p.y, p.z] for p in verts]))
    return {"name": name, "source": "vegeta labs/aero/designs (CadQuery)", "closed": True, "color": [round(c, 3) for c in color],
            "size": [round(float(s), 3) for s in (v.max(axis=0) - v.min(axis=0))],
            "vertices": [round(float(x), 4) for x in v.ravel()], "triangles": [int(i) for t in tris for i in t]}


def albatros_parts(d) -> list:
    """The sized aircraft in three colours: fuselage, wing and tail, engine pods."""
    import albatros as al

    p = d.params
    a = al.Albatros()
    L = al.Albatros.layout(p)
    x0, zc = L["x_ac"], -p["fuselage_height"] / 2

    def tr(v):                                    # the design's frame (x aft, z up, mm) -> Ketos (x forward, m, at the centre)
        return np.column_stack([-(v[:, 0] - x0), -v[:, 1], v[:, 2] - zc]) / 1000.0

    return [mesh(a._fuselage(p).val(), "albatros_fuselage", (0.93, 0.94, 0.95), tr),
            mesh(a._wing(p).union(a._vtail(p)).val(), "albatros_wing", (0.86, 0.33, 0.12), tr),
            mesh(a._nacelles(p).val(), "albatros_engines", (0.22, 0.23, 0.25), tr)]


def albatros_card(d, req) -> dict:
    import albatros_flight as F

    f, la = d.flight, F.launch(d)
    ph = f.phases

    def rate(*names):
        t = sum(ph[n]["time"] for n in names if n in ph)
        return round(sum(ph[n]["fuel"] for n in names if n in ph) / t, 5) if t else 0.0

    climb_t = ph["climb out"]["time"]
    return {"schema": "ketos.card", "version": "1.0", "name": "albatros", "kind": "aircraft",
            "parts": [{"model": "albatros_fuselage", "color": [0.93, 0.94, 0.95]}, {"model": "albatros_wing", "color": [0.86, 0.33, 0.12]},
                      {"model": "albatros_engines", "color": [0.22, 0.23, 0.25]}],
            "span_m": round(d.params["span"] / 1000, 2), "length_m": round(d.params["fuselage_length"] / 1000, 2),
            "mass_kg": round(d.mtow_kg, 1), "empty_kg": round(d.mtow_kg - d.fuel_kg - sum(v for k, v in d.budget.items() if k.startswith("bay load")), 1),
            "fuel_capacity_kg": round(d.fuel_kg, 1), "reserve": req.reserve, "bay_load_kg": round(sum(v for k, v in d.budget.items() if k.startswith("bay load")), 1),
            "engines": d.engines, "thrust_n": round(F._static(d)), "design_radius_km": req.radius_km,
            "cruise_altitude_m": req.cruise_altitude_m, "cruise_speed": round(f.cruise_speed_out, 1), "cruise_speed_light": round(f.cruise_speed_back, 1),
            "climb_rate": round(req.cruise_altitude_m / climb_t, 2), "climb_speed": round(ph["climb out"]["dist"] / climb_t, 1),
            "descent_rate": round(req.cruise_altitude_m / ph["descent to the vessel"]["time"], 2),
            "descent_speed": round(ph["descent to the vessel"]["dist"] / ph["descent to the vessel"]["time"], 1),
            "fuel_kg_s": {"start": rate("start"), "climb": rate("climb out", "climb home"), "cruise": rate("cruise out"), "cruise_light": rate("cruise home"),
                          "descent": rate("descent to the vessel", "descent home"), "pass": rate("slow pass and release"), "approach": rate("approach and landing")},
            "start_s": round(ph["start"]["time"]), "stall_speed": round(la["stall_speed"], 1), "launch_speed": round(la["launch_speed"], 1),
            "catapult_stroke_m": round(la["catapult_stroke_m"], 1), "catapult_g": la["catapult_g"], "runway_roll_m": round(la["runway_roll_m"]),
            "pass_speed": round(la["release_speed"], 1), "approach_speed": round(la["approach_speed"], 1), "landing_roll_m": round(la["landing_roll_m"]),
            "turn_rate_deg_s": 6.0, "wind_limit_m_s": 20.0}


def hexa_parts_and_card(d, option) -> tuple:
    import albatros as al
    import albatros_drop as dr
    import albatros_flight as F

    hexa = option.vehicles[0]
    shape = al.Albatros().generate(**dict(d.params, part="multirotor")).shape
    part = mesh(shape, "cargo_hexa_frame", (0.95, 0.55, 0.10), lambda v: v / 1000.0, tolerance=1.5)
    pkg = dr.package(hexa)
    v_rel = F.launch(d)["release_speed"]
    rel = dr.simulate(pkg, v_rel)
    mc = dr.monte_carlo(pkg, v_rel, 200)
    card = {"schema": "ketos.card", "version": "1.0", "name": "cargo_hexa", "kind": "multirotor",
            "parts": [{"model": "cargo_hexa_frame", "color": [0.95, 0.55, 0.10]}], "rotors": hexa.rotors, "mass_kg": round(hexa.mass_kg, 2),
            "cargo_kg": hexa.cargo_kg, "span_m": round(hexa.span_m, 2), "battery_wh": round(hexa.battery_kg * 160.0), "hover_w": round(hexa.hover_power_w),
            "cruise_speed": 15.0, "descent_rate": 4.0, "landing_speed": 0.8, "deck_hs_limit_m": 2.5, "wind_limit_m_s": 14.0,
            "release": {"pass_speed": round(v_rel, 1), "pass_height_m": round(mc["pass_height_m"]), "height_lost_m": round(rel["height_lost_m"]),
                        "time_to_hover_s": round(rel["time_s"], 1), "downrange_m": round(rel["downrange_m"]), "drogue_s": round(rel["events"]["drogue out"], 2),
                        "arms_locked_s": round(rel["events"]["arms locked"], 2), "success": round(mc["success"], 3)}}
    return part, card


def emerillon_parts_and_card() -> tuple:
    import albatros_flood as fl
    import emerillon as em
    import emerillon_flight as ef

    car = fl.carrier()
    c = ef.VARIANTS["ER2"]
    shape = em.Emerillon().generate(**ef.cad_parameters(c)).shape
    L = car.length_m * 1000

    def tr(v):                                    # x aft from the nose -> x forward, at the middle
        return np.column_stack([-(v[:, 0] - L / 2), -v[:, 1], v[:, 2]]) / 1000.0

    part = mesh(shape, "emerillon_body", (0.95, 0.42, 0.08), tr, tolerance=0.6)
    card = {"schema": "ketos.card", "version": "1.0", "name": "emerillon", "kind": "multirotor", "parts": [{"model": "emerillon_body", "color": [0.95, 0.42, 0.08]}],
            "rotors": 4, "mass_kg": round(car.gross_kg, 2), "cargo_kg": car.payload_kg, "span_m": round(2 * car.arm_r_m + car.prop_d_m, 2),
            "length_m": round(car.length_m, 2), "battery_wh": round(car.pack_wh * car.usable, 1), "hover_w": round(car.hover_power()),
            "cruise_speed": 32.0, "descent_rate": 6.0, "landing_speed": 0.7, "deck_hs_limit_m": 1.5, "wind_limit_m_s": 16.0, "tail_sitter": True,
            "release": {"pass_height_m": 100, "height_lost_m": 25, "time_to_hover_s": 2.0, "downrange_m": 40, "success": 1.0}}
    return part, card


def export(out, *copies) -> list:
    import albatros_bay as B
    import albatros_flight as F

    req = F.Requirements()
    option = B.options(req.cargo_kg)["hexacopter"]
    d = F.size("catapult", req, option)
    docs = {f"{m['name']}.json": m for m in albatros_parts(d)}
    docs["albatros.card.json"] = albatros_card(d, req)
    part, card = hexa_parts_and_card(d, option)
    docs[f"{part['name']}.json"], docs["cargo_hexa.card.json"] = part, card
    part, card = emerillon_parts_and_card()
    docs[f"{part['name']}.json"], docs["emerillon.card.json"] = part, card
    written = []
    for folder in (out, *copies):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, doc in docs.items():
            (folder / name).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
            written.append(str(folder / name))
    return written


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for w in export(*sys.argv[1:]):
        print(w)
