"""Export the pearl farm's machines to Ketos (``models/vegeta``): Dolphin (``labs/marine/designs/dolphin.py``) and
the Emerillion (``labs/aero/designs/emerillion.py``) as ``ketos.card`` cards with their part meshes, in the format
Ketos's ``ketos_core.models`` reads (metres, x forward, z up, the mesh's lowest point at the keel / the pads).

    python tools/ketos_export_pearl.py /path/to/ketos/models/vegeta
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs" / "marine" / "designs"))
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))


def mesh_json(name, shape, color, scale=1e-3, tol=2.0):
    """A Ketos mesh (``name``, ``closed``, ``color``, ``size``, ``vertices``, ``triangles``) from a CadQuery shape."""
    verts, tris, off = [], [], 0
    for s in shape.solids().vals():
        v, t = s.tessellate(tol, 0.5)
        verts += [[round(q.x * scale, 4), round(q.y * scale, 4), round(q.z * scale, 4)] for q in v]
        tris += [[a + off, b + off, c + off] for a, b, c in t]
        off += len(v)
    xs, ys, zs = (sorted(c[i] for c in verts) for i in range(3))
    return {"name": name, "source": "vegeta (CadQuery): tools/ketos_export_pearl.py", "closed": True, "color": list(color),
            "size": [round(xs[-1] - xs[0], 3), round(ys[-1] - ys[0], 3), round(zs[-1] - zs[0], 3)],
            "vertices": verts, "triangles": tris}


def export(out_dir: Path):
    import cadquery as cq
    from dolphin import Dolphin, EMERILLION
    from emerillion import Emerillion, hover, pearl_run, mass_kg
    out_dir.mkdir(parents=True, exist_ok=True)
    # --- Dolphin: x forward along the hull, z up from the keel (the design's frame already)
    p = Dolphin.p()
    s = Dolphin.sizing(p)
    parts = [("dolphin_hull", "hull", (0.17, 0.37, 0.54)), ("dolphin_sail", "sail", (0.20, 0.27, 0.33)), ("dolphin_hangar", "hangar", (0.50, 0.55, 0.60)),
             ("dolphin_pad", "pad", (0.90, 0.49, 0.13)), ("dolphin_wetwell", "wetwell", (0.75, 0.22, 0.17)), ("dolphin_propulsor", "propulsor", (0.13, 0.13, 0.13))]
    d = Dolphin()
    card = {"schema": "ketos.card", "version": "1.0", "name": "dolphin", "kind": "submarine",
            "parts": [{"model": n, "color": list(c)} for n, _, c in parts],
            "size": [p["length"] / 1000, p["hull_d"] / 1000, (p["hull_d"] + 500) / 1000], "mass_kg": round(s["mass_kg"], 1),
            "cruise_m_s": p["cruise"], "max_speed_m_s": 2.0, "prop_W": round(s["prop_W"], 1), "hotel_W": round(s["hotel_W"], 1),
            "battery_kwh": 12.0, "depth_m": 60.0, "freeboard_m": p["freeboard"] / 1000, "surface_rate_m_s": 0.5,
            "hangar": {"drones": 2, "hatch_s": 60.0, "lift_s": 60.0, "charge_h": EMERILLION["charge_h"]},
            "pearl_cell": {"oysters_per_h": 400.0, "capsule_pearls": 200, "capsule_kg": 0.2},
            "wet_well": {"baskets": 4, "hook_s": 90.0}, "price_eur": round(s["price_eur"])}
    for n, part, c in parts:
        (out_dir / f"{n}.json").write_text(json.dumps(mesh_json(n, d.build({**p, "part": part}), c)))
    (out_dir / "dolphin.card.json").write_text(json.dumps(card, indent=1))
    # --- Emerillion: the design's frame has the nose at the origin along +X; Ketos wants x forward, z up, the
    # lowest point at z = 0; a tail-sitter stands on its fins, so the card carries both poses as the sim's business
    pe = Emerillion.p()
    e = Emerillion()
    eparts = [("emerillion_fuselage", "fuselage", (0.95, 0.77, 0.06)), ("emerillion_wing", "wing", (0.95, 0.77, 0.06)),
              ("emerillion_tail", "tail", (0.30, 0.30, 0.30)), ("emerillion_canopy", "canopy", (0.55, 0.75, 0.90))]
    h, r = hover(), pearl_run(10.0)
    ecard = {"schema": "ketos.card", "version": "1.0", "name": "emerillion", "kind": "drone",
             "parts": [{"model": n, "color": list(c)} for n, _, c in eparts],
             "size": [pe["length"] / 1000, pe["span"] / 1000, (pe["fin_span"] * 2 + pe["fuselage_d"]) / 1000], "mass_kg": round(mass_kg(), 2),
             "payload_kg": 0.2, "thrust_to_weight": round(h["thrust_to_weight"], 2), "hover_W": round(h["hover_W"]), "cruise_m_s": 28.0, "cruise_W": 420.0,
             "climb_m_s": 2.0, "descent_m_s": 1.5, "battery_wh": 222.0, "reserve": 0.2, "range_km_reserve": round(r["range_km_reserve"], 1),
             "stall_m_s": round(r["stall_m_s"], 1), "charge_h": 1.0, "tail_sitter": True, "price_eur": 4800}
    for n, part, c in eparts:
        shape = e.build({**pe, "part": part}).translate((0, 0, pe["fin_span"] + pe["fuselage_d"] * 0.2))   # the pads at z = 0 when it lies level
        (out_dir / f"{n}.json").write_text(json.dumps(mesh_json(n, shape, c, tol=1.0)))
    (out_dir / "emerillion.card.json").write_text(json.dumps(ecard, indent=1))
    return [str(out_dir / "dolphin.card.json"), str(out_dir / "emerillion.card.json")]


if __name__ == "__main__":
    print(export(Path(sys.argv[1])))
