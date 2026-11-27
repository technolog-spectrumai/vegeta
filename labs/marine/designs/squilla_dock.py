"""Squilla's resident dock on a leg of the jacket (task 5 of the sea-game machines): does ``oil_platform.ResidentDock``
fit a leg of ``oil_platform.Jacket`` between -30 and -80 m, how is it clamped on, how is Squilla charged and linked
to the topside, and how long does it take to dock.

* **Fit**: the dock's envelope (the garage, its charger, the clamp, and a 12 m approach corridor in front of the
  mouth for Squilla) placed on each of the four legs, facing out from the jacket, at every depth from -30 to -80 m;
  its clearance to every other member (braces, horizontals, conductors, risers, the caisson) and to every anode,
  from the jacket's member list (``Jacket.members``). The legs are battered 1:10, so the clamp follows the leg's
  axis and a bracket levels the garage.
* **Mounting**: two split collars (bolted half shells with an elastomer liner) round the 2.0 m leg, sized by
  friction against the dock's weight in water, the current's drag and the moment of its offset (``mounting``).
* **Power and data**: the inductive charger (the card's 2 kW) against Squilla's battery and a guard patrol; a
  fibre and power cable up the leg to the topside; an acoustic modem and USBL head on the dock for Squilla away from
  it; an optical modem at the coupler when it is docked (``services``).
* **Docking**: Squilla homes on the dock's USBL, lines up into the current, backs in, latches and couples; the cross
  current it can hold while backing in from its lateral thrust (``docking``).

Class values (assumed) are marked; the jacket's geometry is the design's.
"""
from __future__ import annotations

import math

from oil_platform import Jacket, ResidentDock

RHO_SW = 1025.0
G = 9.81
LEGS = ("A1", "A2", "B1", "B2")
CORRIDOR_M = 12.0                         # Squilla's straight approach in front of the mouth
GROWTH_M = 0.05                           # marine growth on the leg below -40 m (North Sea class; 0.1 m above)


def _seg_dist(p, a, b):
    ax, ay, az = a
    d = [b[i] - a[i] for i in range(3)]
    L2 = sum(x * x for x in d) or 1e-9
    t = max(0.0, min(1.0, sum((p[i] - a[i]) * d[i] for i in range(3)) / L2))
    q = [a[i] + t * d[i] for i in range(3)]
    return math.dist(p, q)


def envelope(leg: str, z: float, over=None) -> dict:
    """The dock on ``leg`` at depth ``z`` facing radially out: the leg axis point, the outward unit vector, and sample
    points of the garage, the clamp and the approach corridor (each with the radius it needs)."""
    jp, dp = Jacket.p(), ResidentDock.p(over)
    c = Jacket.leg_at(jp, leg, z)
    r = math.hypot(c[0], c[1])
    u = (c[0] / r, c[1] / r, 0.0)                                    # out from the jacket's axis
    v = (-u[1], u[0], 0.0)
    R = jp["leg_d"] / 2 + GROWTH_M
    standoff = R + 0.7                                               # the collar, the bracket, the charger
    pts = []
    for i in range(7):                                               # the garage: its length along u
        s = standoff + dp["length"] * i / 6
        for w in (-dp["width"] / 2, 0.0, dp["width"] / 2):
            for h in (-dp["height"] / 2, dp["height"] / 2):
                pts.append(((c[0] + u[0] * s + v[0] * w, c[1] + u[1] * s + v[1] * w, z + h), 0.15, "garage"))
    for i in range(13):                                              # the approach corridor beyond the mouth
        s = standoff + dp["length"] + CORRIDOR_M * i / 12
        pts.append(((c[0] + u[0] * s, c[1] + u[1] * s, z), 1.0, "corridor"))
    for h in (-0.75, 0.75):                                          # the two collars on the leg
        pts.append(((c[0], c[1], z + h), R + 0.15, "collar"))
    return {"leg_axis": c, "out": u, "points": pts, "standoff_m": standoff}


def clearance(leg: str, z: float, over=None) -> dict:
    """The least clearance (m) from the dock's envelope to the jacket's other members and anodes, and what it is."""
    g = Jacket.members()
    env = envelope(leg, z, over)
    worst = (1e9, None, None)
    for m in g["members"]:
        if m["kind"] == "leg" and m["id"].startswith(f"L-{leg}-"):
            continue
        for p, rad, what in env["points"]:
            if what == "collar" and m["kind"] in ("brace", "horizontal") and _seg_dist(p, m["a"], m["b"]) - m["d"] / 2 - rad < 0:
                pass
            d = _seg_dist(p, m["a"], m["b"]) - m["d"] / 2 - rad
            if d < worst[0]:
                worst = (d, m["id"], what)
    for a in g["anodes"]:
        if a["member"].startswith(f"L-{leg}-"):
            continue
        for p, rad, what in env["points"]:
            d = math.dist(p, a["p"]) - 0.5 - rad
            if d < worst[0]:
                worst = (d, a["id"], what)
    own = [a for a in g["anodes"] if a["member"].startswith(f"L-{leg}-")]
    anode_gap = min(abs(a["p"][2] - z) for a in own) - 0.75 - 0.6 if own else 9.0     # the collars' half-span, the anode's half-length
    return {"leg": leg, "depth_m": z, "clearance_m": round(worst[0], 2), "nearest": worst[1], "of": worst[2],
            "own_anode_gap_m": round(anode_gap, 2), "fits": worst[0] > 0.3 and anode_gap > 0}


def survey(over=None) -> dict:
    """Every leg at every 5 m from -30 to -80 m: what fits, and the best place."""
    rows = [clearance(leg, -float(d), over) for leg in LEGS for d in range(30, 81, 5)]
    ok = [r for r in rows if r["fits"]]
    best = max(ok, key=lambda r: (r["clearance_m"] if r["clearance_m"] < 3 else 3, -r["depth_m"])) if ok else None
    return {"rows": rows, "fits": ok, "best": best, "at_80": [r for r in rows if r["depth_m"] == -80.0]}


def wave_orbital(depth: float, hs: float = 5.0, tp: float = 10.0) -> float:
    """The amplitude of the wave orbital velocity (m/s) at ``depth`` under a storm sea (deep water, linear)."""
    k = (2 * math.pi / tp) ** 2 / G
    return math.pi * hs / tp * math.exp(-k * abs(depth))


def mounting(depth: float = 80.0, over=None) -> dict:
    """Two split collars on the leg: the friction they need against the dock's loads (its weight in water, the
    drag of 1 m/s of current on it, the moment of its offset from the leg), the bolts' preload that gives it."""
    jp, dp = Jacket.p(), ResidentDock.p(over)
    steel = 7850.0
    w_water = dp["mass_kg"] * (1 - RHO_SW / steel) * G
    area = dp["length"] * dp["height"]
    drag = 0.5 * RHO_SW * 1.5 * area * 1.0 ** 2 + 0.5 * RHO_SW * 1.0 * area * wave_orbital(depth) ** 2
    lever = envelope("A1", -depth, over)["standoff_m"] + dp["length"] / 2
    spacing = 1.5
    couple = math.hypot(w_water, drag) * lever / spacing             # the moment as a pull on one collar, a push on the other
    mu, sf = 0.2, 2.0                                                 # elastomer liner on a cleaned leg (with growth left: 0.2), safety factor
    per_collar_N = sf * (math.hypot(w_water, drag) / 2 + couple) / mu
    bolt_kN = 280.0                                                   # M30 10.9, 70 % of proof (class)
    bolts = max(4, math.ceil(per_collar_N / 1000 / bolt_kN / 2) * 2)  # in pairs, half each side; 4 at least (one may be lost)
    R = jp["leg_d"] / 2
    return {"leg_d_m": jp["leg_d"], "liner_bore_m": round(jp["leg_d"] + 2 * 0.012, 3), "collars": 2, "collar_spacing_m": spacing,
            "collar_height_m": 0.45, "collar_wall_m": 0.03,
            "weight_in_water_kN": round(w_water / 1000, 1), "drag_kN": round(drag / 1000, 1), "lever_m": round(lever, 2),
            "normal_force_per_collar_kN": round(per_collar_N / 1000), "bolts_per_collar": bolts, "bolt": "M30 10.9, 280 kN preload (class)",
            "mu": mu, "safety_factor": sf,
            "collar_mass_kg": round(2 * 2 * math.pi * (R + 0.03) * 0.45 * 0.03 * steel + 2 * bolts * 4.0 + 2 * 120.0),   # rings, bolts, flanges
            "install": "two halves lowered from the DP vessel and closed round the leg by a work-class ROV (Homarus), studs "
                       "tensioned hydraulically after the leg is cleaned of growth under the liner; the garage on a bracket that "
                       "takes out the leg's 1:10 batter so it lies level",
            "cp": "bonded to the jacket's cathodic protection through the collars' bonding straps, plus two anodes on the garage",
            "no_welding": True}


def services(over=None) -> dict:
    """Charging and the data link: the inductive charger against Squilla's battery and a guard patrol; the cable
    up the leg; the acoustic and optical links."""
    from squilla import Squilla
    dp = ResidentDock.p(over)
    s = Squilla.sizing()
    batt = Squilla.p()["battery_kwh"] * 1000
    usable = batt * 0.85
    eta = 0.90
    charge_h = usable / (dp["charge_kw"] * 1000 * eta - 60.0)       # 60 W of hotel while docked (class)
    patrol_W = s["transit_W"]                                       # a slow guard patrol round the jacket: its transit power
    patrol_h = usable / patrol_W
    duty = patrol_h / (patrol_h + charge_h + 0.25)
    cable_m = abs(dp["depth"]) + 20.0 + 60.0                        # up the leg to the cellar deck, across to the equipment room
    return {"charger": {"kind": "inductive coupler at the garage's back (no wet-mate pins)", "kW": dp["charge_kw"], "efficiency": eta,
                        "squilla_battery_Wh": batt, "usable_Wh": round(usable), "charge_h": round(charge_h, 2)},
            "patrol": {"power_W": round(patrol_W), "hours_on_a_charge": round(patrol_h, 1), "duty_out_of_dock": round(duty, 2),
                       "note": "a guard patrol at ~1 m/s round the jacket; intervention work (s['work_W']) empties it ~3x faster"},
            "work_W": round(s["work_W"]),
            "cable": {"length_m": round(cable_m), "what": "armoured hybrid cable up the leg in clamps: 2 single-mode fibres + 2 x 6 mm2 copper",
                      "power": "400 V DC from the topside, 2.5 kW", "loss_W": round(2 * cable_m * 0.0031 * (2500 / 400) ** 2)},
            "data": {"topside": "1 Gbit/s Ethernet over fibre", "docked": "optical free-space modem at the coupler, ~100 Mbit/s over ~1 m: "
                     "video and logs off, mission plans on", "away": "acoustic modem + USBL head on the dock: ~5-10 kbit/s to ~2 km, "
                     "Squilla's position for the carrier and the topside", "latency": "fibre: ms; acoustic: ~1.5 s a km"}}


def docking(over=None) -> dict:
    """The docking cycle (class times) and the cross current Squilla can hold while backing in (its lateral thrust
    against its broadside drag, 70 % of it usable) - into the current it holds 1 m/s."""
    from squilla import Squilla
    s = Squilla.sizing()
    p = Squilla.p()
    lat_N = s["thrust_N"]["lateral"] * 0.7
    side = p["length"] / 1000 * 0.65
    u_cross = math.sqrt(lat_N / (0.5 * RHO_SW * 1.1 * side))
    steps = {"home on the dock's USBL to 30 m off the mouth": 0.0, "hover and line up 10 m off, tail to the mouth, into the current": 60.0,
             "back in along the guide lights at 0.15 m/s (the garage's 3 m and the last 7 m)": 10.0 / 0.15,
             "latch, seat on the coupler, optical link up": 30.0}
    undock = {"unlatch": 10.0, "out at 0.3 m/s and away": 4.0 / 0.3}
    return {"cycle_s": {k: round(v) for k, v in steps.items()}, "dock_s": round(sum(steps.values())), "undock_s": round(sum(undock.values())),
            "cross_current_limit_m_s": round(u_cross, 2), "head_current_limit_m_s": 1.0,
            "orbital_at_depth_m_s": {f"-{d}m": round(wave_orbital(d), 2) for d in (30, 50, 65, 80)},
            "note": "the mouth faces out from the jacket; a rotating tidal current crosses it twice a tide - docking waits for the "
                    "cross component under the limit (North Sea tides at 80 m: ~0.3-0.6 m/s, assumed)"}


def report(over=None) -> dict:
    sv = survey(over)
    return {"fits_at_80_on": [r["leg"] for r in sv["at_80"] if r["fits"]], "at_80": sv["at_80"], "best": sv["best"],
            "fitting_depths": {leg: [r["depth_m"] for r in sv["fits"] if r["leg"] == leg] for leg in LEGS},
            "fits_count": len(sv["fits"]), "checked": len(sv["rows"]), "mounting": mounting(80.0, over),
            "services": services(over), "docking": docking(over)}


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1, default=str))
