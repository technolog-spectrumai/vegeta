"""STERNA's mass budget and balance (notebook 51): every item, its mass, its station and where the number comes from;
the payload pallets and the battery modules as alternatives; the CG of each loading case.

Sources, marked per line (the lab's convention, see ``arigo_mass``):
- ``cell``: Molicel P42A 21700 cell, 70 g, 4.2 Ah, 3.6 V nominal;
- ``HAND``: a class value for the part, with its margin;
- ``geometry``: an areal or linear density (HAND, stated) times the CAD's areas and lengths.

Nothing here is weighed. Stations are x from the spinner tip [mm].
"""
from __future__ import annotations

import math

WING_KG_M2 = 1.18        # +-0.25: closed-cell foam core, thin glass skin, carbon tube spar (the lab's moulded-EPO value is 1.55)
TAIL_KG_M2 = 0.90        # +-0.2 (arigo_mass)
FUSE_KG_M = 0.30         # +-0.06 per metre: glass-epoxy shell 100 x 125 mm with frames and two hatches
CELL_G, CELL_AH, CELL_V = 70.0, 4.2, 3.6
MODULE_OVERHEAD = 1.10   # HAND: strip, wrap, lead, connector, sleeve (+-0.03)
MODULE_G = round(4 * CELL_G * MODULE_OVERHEAD)             # one 4S1P module
MODULE_WH = 4 * CELL_V * CELL_AH
MODULE_LEN = 78.0
PALLET_X = 250.0
PAYLOAD_LIMIT_G = 250.0
MTOW_G = 2450.0
BASE_AR = 7.19               # STERNA's aspect ratio: the wing's areal density is its value

FIXED = [  # (group, item, grams, station, source)
    ("structure", "wing joiner, root fittings, connectors", 60.0, 402.0, "HAND (+-15)"),
    ("structure", "belly rails, nose ring, tail bumper", 40.0, 330.0, "HAND (+-10)"),
    ("propulsion", "motor (35 mm outrunner class)", 110.0, 70.0, "HAND (+-20)"),
    ("propulsion", "folding propeller 11 in + spinner", 40.0, 20.0, "HAND (+-10)"),
    ("propulsion", "ESC 40 A, potted", 45.0, 125.0, "HAND (+-10)"),
    ("propulsion", "power wiring", 25.0, 220.0, "HAND (+-10)"),
    ("actuation", "2 aileron servos", 26.0, 460.0, "2 x 13 g (HAND +-3 each)"),
    ("actuation", "2 flap servos", 26.0, 480.0, "2 x 13 g (HAND +-3 each)"),
    ("actuation", "elevator and rudder servos", 26.0, 610.0, "2 x 13 g (HAND +-3 each)"),
    ("actuation", "pushrods and linkages", 22.0, 600.0, "HAND (+-8)"),
    ("avionics", "autopilot with IMU and barometer", 40.0, 290.0, "HAND (+-10)"),
    ("avionics", "power module and regulator", 25.0, 250.0, "HAND (+-8)"),
    ("avionics", "video downlink unit and antenna", 60.0, 215.0, "HAND (+-20)"),
    ("avionics", "GNSS and compass", 25.0, 660.0, "HAND (+-8)"),
    ("avionics", "airspeed sensor and pitot (left wing)", 15.0, 420.0, "HAND (+-5)"),
    ("avionics", "command/telemetry radio and fin antenna", 45.0, 620.0, "HAND (+-15)"),
    ("avionics", "backup control receiver", 10.0, 590.0, "HAND (+-4)"),
    ("avionics", "harness and connectors", 50.0, 420.0, "HAND (+-20)"),
    ("avionics", "strobe", 10.0, 700.0, "HAND (+-4)"),
]
PALLETS = {  # name -> (grams, what)
    "none": (0.0, "no pallet (a blank cover)"),
    "daylight": (120.0, "one daylight camera, fixed mount, flat window"),
    "stabilised": (200.0, "daylight camera on a roll/tilt mount, bay window, deeper skid shoes"),
    "dual": (220.0, "daylight camera and thermal core side by side, two windows"),
    "limit": (PAYLOAD_LIMIT_G, "the payload limit"),
}
CASES = {  # name -> (pallet, modules, tray centre station)
    "ferry": ("none", 1, 359.0),
    "daylight_1": ("daylight", 1, 397.0),
    "stabilised_1": ("stabilised", 1, 437.0),
    "dual_1": ("dual", 1, 445.0),
    "dual_1_unmoved": ("dual", 1, 397.0),
    "daylight_2": ("daylight", 2, 398.0),
    "dual_2": ("dual", 2, 415.0),
    "limit_2": ("limit", 2, 417.0),
}
DEFAULT_CASE = "daylight_1"


def empty(p: dict, L: dict) -> list:
    """The empty aircraft's items (no pallet, no battery): the structure from the CAD's areas, the fixed equipment,
    and a growth margin of 5 % of the rest at the items' own centroid."""
    S_w, S_t = L["S_mm2"] / 1e6, (L["S_h_mm2"] + L["S_v_mm2"]) / 1e6
    k = L.get("scale", 1.0)
    ar = math.sqrt(L["AR"] / BASE_AR) if L["AR"] > BASE_AR else 1.0          # a longer wing bends more: the spar grows with it (HAND)
    grow = {"structure": k * k, "propulsion": p.get("mtow_g", MTOW_G) / MTOW_G, "actuation": k, "avionics": 1.0}
    items = [("structure", "wing panels, pair", 1000 * WING_KG_M2 * S_w * ar, L["x_mac_le"] + 0.42 * L["mac"], f"geometry: {S_w:.3f} m2 x {WING_KG_M2 * ar:.2f} kg/m2 (HAND +-20 %)"),
             ("structure", "fuselage shell, frames, hatches", 1000 * FUSE_KG_M * L["length"] / 1000 * k, 445.0 * k, f"geometry: {L['length']:.0f} mm x {FUSE_KG_M * k:.2f} kg/m (HAND +-20 %)"),
             ("structure", "tailplane halves, fin and rudder", 1000 * TAIL_KG_M2 * S_t, 0.5 * (L["x_ac_h"] + L["x_ac_v"]) + 25.0, f"geometry: {S_t:.3f} m2 x {TAIL_KG_M2} kg/m2 (HAND +-20 %)")]
    items += [(g, n, grams * grow[g], x * k, src) for g, n, grams, x, src in FIXED]
    m = sum(i[2] for i in items)
    x = sum(i[2] * i[3] for i in items) / m
    items.append(("margin", "growth margin (5 % of the items above)", 0.05 * m, x, "HAND: paint, sealant, coatings"))
    return items


def budget(p: dict, L: dict, case=DEFAULT_CASE) -> dict:
    """``case``: a name in ``CASES`` (STERNA's own), or (payload grams, modules, tray centre station) for a variant;
    the payload at the pallet's station, scaled with the airframe."""
    k = L.get("scale", 1.0)
    if isinstance(case, str):
        pallet, n, x_tray = CASES[case]
        g_pay, x_tray = PALLETS[pallet][0], x_tray * k
    else:
        g_pay, n, x_tray = case
        pallet, case = f"{g_pay:.0f} g", f"{g_pay:.0f} g, {n} modules"
    items = empty(p, L)
    m_e = sum(i[2] for i in items)
    mom = sum(i[2] * i[3] for i in items)
    g_p = g_pay
    g_b = n * MODULE_G
    total = m_e + g_p + g_b
    x_cg = (mom + g_p * PALLET_X * k + g_b * x_tray) / total
    return {"case": case, "items": [{"group": g, "item": n_, "g": round(v, 1), "x_mm": round(x, 0), "source": s} for g, n_, v, x, s in items],
            "empty_g": round(m_e, 1), "empty_x_mm": round(mom / m_e, 1),
            "pallet": {"name": pallet, "g": g_p, "x_mm": PALLET_X * k}, "battery": {"modules": n, "g": g_b, "Wh_nominal": round(n * MODULE_WH, 1), "x_mm": x_tray},
            "total_g": round(total, 1), "x_cg_mm": round(x_cg, 1), "cg_mac": round((x_cg - L["x_mac_le"]) / L["mac"], 4),
            "within_mtow": total <= p.get("mtow_g", MTOW_G)}


def tray_for(p: dict, L: dict, pallet, modules: int, cg_mac: float = 0.30) -> float:
    """The tray centre station that puts the CG at ``cg_mac`` (may lie outside the bay: check against the travel).
    ``pallet``: a name in ``PALLETS`` or the payload's grams."""
    items = empty(p, L)
    m_e, mom = sum(i[2] for i in items), sum(i[2] * i[3] for i in items)
    g_p, g_b = (PALLETS[pallet][0] if isinstance(pallet, str) else float(pallet)), modules * MODULE_G
    x_t = L["x_mac_le"] + cg_mac * L["mac"]
    return (x_t * (m_e + g_p + g_b) - mom - g_p * PALLET_X * L.get("scale", 1.0)) / g_b
