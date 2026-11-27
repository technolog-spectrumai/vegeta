"""The sea game's vessels as Chiron robots, for the Yax export (``machines.py``, ``tools/publish.py``): each one rigid
link (a free-floating body; no joints), its mass and centre of gravity those of the sizing.

* **Zarco** and **Esmeralda**: the two demihulls and the cross deck as box geoms carrying the hull structure's mass;
  every other weight of the sizing as a point mass at its centre (``esmeralda.ZARCO_CENTRES`` for Zarco's,
  ``Esmeralda.weights`` for Esmeralda's: the boat aboard, stowed on its cradle). Root frame: the waterline amidships.
* **Barracuda**: the hull as a box carrying the hull's and the collar's mass, the other items of the sizing's mass
  budget as point masses at the centres in ``BARRACUDA_CENTRES`` (class estimates from the CAD), the fuel and the
  entrained water at half load; the fuel tank's position chosen so the centre of gravity is the one the planing
  model assumes (``lcg_frac`` of the waterline length from the transom). Root frame: the waterline amidships.

Units SI; x forward, y left, z up. The hydrodynamics (resistance, seakeeping, the speeds) are on the Ketos cards, not
here: a rigid body with its mass properties and its looks is what a simulator takes from these files.
"""
from __future__ import annotations

from vegeta.chiron import Geom, Link, PointMass, Robot

HULL_RGBA = (0.20, 0.29, 0.37, 1.0)

#: Barracuda's items' centres: (x from the transom, z above the keel) in m (class estimates from ``barracuda.py``'s
#: CAD, assumed). Keys are prefixes of the sizing's ``items``.
BARRACUDA_CENTRES = {
    "hull": (3.7, 0.55), "foam collar": (4.3, 1.25), "2 diesels": (1.4, 0.55), "2 waterjets": (-0.15, 0.35),
    "navigation radar": (3.9, 4.75), "EO/IR gimbal": (4.2, 4.4), "AIS": (4.0, 2.3), "VHF": (4.2, 1.9),
    "microphone array": (3.9, 4.2), "MANET": (3.9, 4.5), "LEO broadband": (3.9, 4.3), "autonomy computer": (4.2, 1.8),
    "RTK GNSS": (4.2, 1.8), "long-range acoustic hailer": (4.9, 3.0), "LED searchlight": (4.9, 3.0),
    "white LED dazzler strobe": (4.9, 3.0), "water cannon": (5.8, 1.1), "navigation lights": (4.2, 3.0),
    "enclosed electronics console": (4.2, 2.0), "24 V LiFePO4": (2.6, 0.6), "bow recovery eye": (5.0, 0.9),
    "steering and bucket hydraulics": (0.8, 0.7), "fire suppression": (3.5, 0.6), "fuel system": (None, 0.45),
}


def _vessel(name, weights, structure_key, hull, notes, sources):
    """A catamaran: ``weights`` {item: (t, x, z)}; the item ``structure_key`` goes on the hull boxes (two demihulls,
    the cross deck, by volume), the rest are point masses."""
    p = hull
    L, b, H, T = p["length"] / 1000, p["hull_beam"] / 1000, p["hull_depth"] / 1000, p["draught"] / 1000
    y = (p["beam"] - p["hull_beam"]) / 2000
    m_s = weights[structure_key][0] * 1000
    v_h, v_d = L * 0.9 * b * H, L * 0.92 * p["beam"] / 1000 * 0.4
    geoms = [Geom(f"demihull_{s}", "box", (L * 0.45, b / 2, H / 2), pos=(-L * 0.05, sy * y, H / 2 - T),
                  mass=m_s * v_h / (2 * v_h + v_d), role="body", rgba=HULL_RGBA) for s, sy in (("port", 1), ("starboard", -1))]
    geoms.append(Geom("cross_deck", "box", (L * 0.46, p["beam"] / 2000, 0.2), pos=(-L * 0.04, 0.0, H - T + 0.2),
                      mass=m_s * v_d / (2 * v_h + v_d), role="body", rgba=(0.58, 0.65, 0.65, 1.0)))
    masses = [PointMass(k, m * 1000, (x, 0.0, z)) for k, (m, x, z) in weights.items() if k != structure_key]
    robot = Robot(name, Link("hull", log=True, geoms=geoms, masses=masses), notes=notes, sources=sources)
    robot.validate()
    return robot


def zarco() -> Robot:
    """Zarco loaded on the bed (lightship + deadweight), its weights at ``esmeralda.ZARCO_CENTRES``."""
    import zarco as zm
    from esmeralda import ZARCO_CENTRES
    w = {k: (m, *ZARCO_CENTRES[k]) for k, m in {**zm.LIGHTSHIP, **zm.DEADWEIGHT}.items()}
    return _vessel("Zarco", w, "hulls + cross deck (aluminium)", zm.Zarco.p(),
                   "Zarco (formerly Pinna), the unmanned drone-carrier catamaran, loaded on the clam bed: a rigid body",
                   {"masses": "zarco.LIGHTSHIP, zarco.DEADWEIGHT (class values)", "centres": "esmeralda.ZARCO_CENTRES (class estimates)",
                    "looks": "designs/zarco.py:Zarco"})


def esmeralda() -> Robot:
    """Esmeralda on station with Barracuda stowed, its weights from ``Esmeralda.weights``."""
    from esmeralda import Esmeralda
    w, _ = Esmeralda.weights()
    return _vessel("Esmeralda", w, "hulls + cross deck (aluminium)", Esmeralda.p(),
                   "Esmeralda, Zarco's guard-duty variant, on station with Barracuda stowed on its cradle: a rigid body",
                   {"masses and centres": "esmeralda.Esmeralda.weights (class values and estimates)", "looks": "designs/esmeralda.py:Esmeralda"})


def barracuda_weights(load: str = "half") -> tuple:
    """Barracuda's items as (kg, x, z) in the root frame (the waterline amidships of the hull), the fuel tank's x from
    the transom that puts the centre of gravity at ``lcg_frac`` of the waterline length, and the centre of gravity."""
    from barracuda import Barracuda
    p, s = Barracuda.p(), Barracuda.sizing()
    T = s["draught_m"] - 0.05
    mid = p["length"] / 2000
    items = {}
    for k, m in s["items"].items():
        c = next(v for pre, v in BARRACUDA_CENTRES.items() if k.startswith(pre))
        items[k] = (m * 1.08, c)                                           # the sizing's 8 % weight margin spread on every item
    fuel = s["fuel_kg"] * (0.5 if load == "half" else 1.0)
    water = s["mass_half_kg"] - s["lightship_kg"] - s["fuel_kg"] * 0.5
    lcg = p["lcg_frac"] * Barracuda.lwl(p)
    fixed = [(m, c[0]) for k, (m, c) in items.items() if c[0] is not None] + [(water, -0.1)]
    tank_items = [m for k, (m, c) in items.items() if c[0] is None] + [fuel]
    m_tank = sum(tank_items)
    m_all = sum(m for m, _ in fixed) + m_tank
    x_tank = (lcg * m_all - sum(m * x for m, x in fixed)) / m_tank
    out = {}
    for k, (m, (x, z)) in items.items():
        out[k] = (m, (x_tank if x is None else x) - mid, z - T)
    out["fuel"] = (fuel, x_tank - mid, 0.45 - T)
    out["entrained jet water"] = (water, -0.1 - mid, 0.3 - T)
    W = sum(m for m, _, _ in out.values())
    cg = (sum(m * x for m, x, _ in out.values()) / W, sum(m * z for m, _, z in out.values()) / W)
    return out, x_tank, {"mass_kg": W, "lcg_from_transom_m": cg[0] + mid, "vcg_above_wl_m": cg[1], "design_lcg_m": lcg}


def barracuda() -> Robot:
    from barracuda import Barracuda
    p = Barracuda.p()
    w, x_tank, cg = barracuda_weights()
    L, B, D = p["length"] / 1000, p["beam"] / 1000, p["depth"] / 1000
    T = Barracuda.sizing()["draught_m"] - 0.05
    hull_kg = w["hull (aluminium, framed)"][0] + w["foam collar (closed-cell PE, PU skin) + rubber D-fender"][0]
    geoms = [Geom("hull", "box", (L * 0.45, B / 2, D / 2), pos=(0.0, 0.0, D / 2 - T), mass=hull_kg, role="body", rgba=(0.5, 0.55, 0.55, 1.0)),
             Geom("collar_port", "cylinder", (p["collar_d"] / 2000, L * 0.45), pos=(0.0, B / 2, D - T - 0.15),
                  quat=(0.7071068, 0.0, 0.7071068, 0.0), role="visual", rgba=(0.9, 0.49, 0.13, 1.0)),
             Geom("collar_starboard", "cylinder", (p["collar_d"] / 2000, L * 0.45), pos=(0.0, -B / 2, D - T - 0.15),
                  quat=(0.7071068, 0.0, 0.7071068, 0.0), role="visual", rgba=(0.9, 0.49, 0.13, 1.0))]
    masses = [PointMass(k, m, (x, 0.0, z)) for k, (m, x, z) in w.items() if not k.startswith(("hull", "foam collar"))]
    robot = Robot("Barracuda", Link("hull", log=True, geoms=geoms, masses=masses),
                  notes=f"Barracuda, the unmanned interceptor boat, at half load ({cg['mass_kg']:.0f} kg): a rigid body; the fuel tank "
                        f"{x_tank:.2f} m from the transom puts the LCG at the planing model's {cg['design_lcg_m']:.2f} m",
                  sources={"masses": "barracuda.Barracuda.sizing items (class values) + 8 % margin, half fuel, the jets' water",
                           "centres": "sea_robots.BARRACUDA_CENTRES (class estimates)", "looks": "designs/barracuda.py:Barracuda"})
    robot.validate()
    return robot
