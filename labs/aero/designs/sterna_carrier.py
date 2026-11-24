"""STERNA-C — STERNA made bigger to carry WASPs (the carrier of ``wasp.py``'s speed quadcopters).

STERNA (notebook 51) takes 250 g of payload; a WASP-1F is 290 g. The carrier is STERNA scaled geometrically by ``k``
until, with ``n`` WASPs on their racks, the camera pallet and the battery aboard, the **wing loading is back at
STERNA's own limit** (2.45 kg on 0.313 m2: 7.8 kg/m2) — so the stall speed and the handling it was designed for stay,
and what grows is the airframe.

How the mass grows with ``k`` (STERNA's budget, ``sterna_mass``, as the k = 1 point):

- wing, tail, fuselage shell, joiner and rails: with the area, k^2 (HAND: a constant areal density; a real larger wing
  needs a deeper spar and comes out a little heavier — the 5 % growth margin has to cover it);
- the drive (motor, ESC, propeller, wiring): with the take-off mass, 90 g per kg (STERNA: 220 g at 2.45 kg);
- servos and linkages: with k; avionics and radios: fixed;
- the battery: STERNA's 4S1P modules, by count; the camera pallet as STERNA's; each WASP with a 25 g rack and release.

The WASPs ride **on top**, at the CG station: one on the fuselage's back and one on each wing at the dihedral break.
On top they are clear of the belly landing, of the camera's view and of the ground if one fails to release; released
at the CG station they do not move the balance fore and aft. Nothing here is a release analysis.
"""
from __future__ import annotations

import math

import sterna as st
import sterna_mass as sm

LOADING_LIMIT = sm.MTOW_G / 1000.0 / 0.3129        # kg/m2: STERNA at its design limit
DRIVE_KG_PER_KG = 0.090
RACK_G = 25.0                                      # HAND: a printed cradle, a servo latch and its lead per WASP (+-10)
CL_MAX = 1.20                                      # notebook 51: Zephyr's section value x 0.9
STATIC_TW = 0.70


def empty_items(k: float, mtow_kg: float) -> list:
    """(item, grams) of the scaled airframe without battery, pallet and WASPs, before the margin."""
    p = st.resolve()
    L = st.Sterna.layout(p)
    S_w, S_t = L["S_mm2"] / 1e6, (L["S_h_mm2"] + L["S_v_mm2"]) / 1e6
    fixed = {n: g for _, n, g, _, _ in sm.FIXED}
    drive = sum(g for grp, _, g, _, _ in sm.FIXED if grp == "propulsion")
    avionics = sum(g for grp, _, g, _, _ in sm.FIXED if grp == "avionics")
    actuation = sum(g for grp, _, g, _, _ in sm.FIXED if grp == "actuation")
    return [("wing panels", 1000 * sm.WING_KG_M2 * S_w * k * k), ("fuselage shell, frames, hatches", 1000 * sm.FUSE_KG_M * L["length"] / 1000 * k * k),
            ("tailplane, fin and rudder", 1000 * sm.TAIL_KG_M2 * S_t * k * k),
            ("joiner, root fittings, rails, nose ring", (fixed["wing joiner, root fittings, connectors"] + fixed["belly rails, nose ring, tail bumper"]) * k * k),
            ("drive: motor, ESC, propeller, wiring", 1000 * DRIVE_KG_PER_KG * mtow_kg if mtow_kg else drive),
            ("servos and linkages", actuation * k), ("avionics and radios", avionics)]


def budget(k: float, n_wasps: int = 3, modules: int = 2, pallet: str = "daylight", wasp_g: float = 290.0) -> dict:
    """The take-off mass at scale ``k`` (the drive's mass depends on it: solved in closed form)."""
    base = sum(g for n, g in empty_items(k, 0.0) if not n.startswith("drive"))
    load = modules * sm.MODULE_G + sm.PALLETS[pallet][0] + n_wasps * (wasp_g + RACK_G)
    m = (1.05 * base + load) / (1 - 1.05 * 1000 * DRIVE_KG_PER_KG / 1000)          # grams; margin 5 % on the airframe and the drive
    items = empty_items(k, m / 1000)
    airframe = sum(g for _, g in items)
    S = st.Sterna.layout(st.resolve())["S_mm2"] / 1e6 * k * k
    return {"k": k, "items": [(n, round(g, 1)) for n, g in items] + [("growth margin (5 %)", round(0.05 * airframe, 1))],
            "empty_g": round(1.05 * airframe, 1), "battery_g": modules * sm.MODULE_G, "battery_Wh": round(modules * sm.MODULE_WH, 1), "pallet_g": sm.PALLETS[pallet][0],
            "wasps_g": n_wasps * wasp_g, "racks_g": n_wasps * RACK_G, "takeoff_g": round(m, 1), "S_m2": S, "loading_kg_m2": m / 1000 / S,
            "after_release_g": round(m - n_wasps * wasp_g, 1)}


def size(n_wasps: int = 3, modules: int = 2, pallet: str = "daylight", wasp_g: float = 290.0, loading: float = LOADING_LIMIT) -> dict:
    """The scale at which the loaded carrier is at ``loading`` [kg/m2], and what it is then."""
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        lo, hi = (k, hi) if budget(k, n_wasps, modules, pallet, wasp_g)["loading_kg_m2"] > loading else (lo, k)
    b = budget(hi, n_wasps, modules, pallet, wasp_g)
    p = st.resolve()
    L = st.Sterna.layout(p)
    W = b["takeoff_g"] / 1000 * 9.80665
    v = lambda m_g: math.sqrt(2 * m_g / 1000 * 9.80665 / (1.225 * b["S_m2"] * CL_MAX))
    y_wing = hi * p["centre_span"] / 2
    return {**b, "n_wasps": n_wasps, "modules": modules, "pallet": pallet, "span_mm": hi * p["span"], "length_mm": hi * L["length"], "height_mm": hi * L["height"],
            "mac_mm": hi * L["mac"], "x_cg_mm": hi * p["cg_x"], "panel_length_mm": hi * (p["span"] / 2 - 50.0), "prop_diameter_mm": hi * p["prop_diameter"],
            "stall_loaded_m_s": v(b["takeoff_g"]), "stall_after_release_m_s": v(b["after_release_g"]), "static_thrust_needed_N": STATIC_TW * W,
            "wasp_stations": {"x_mm": hi * p["cg_x"], "wing_y_mm": y_wing}, "roll_moment_one_wing_wasp_Nm": wasp_g / 1000 * 9.80665 * y_wing / 1000,
            "joiner_moment_6g_Nm": 6.0 * W / 2 * (0.43 * hi * p["span"] / 2 - hi * 50.0) / 1000}


def cad(k: float, with_wasps: bool = True, wasp_build: str = "WASP-1F"):
    """The carrier's solid: STERNA scaled by ``k``; with the WASPs on top at the CG station (pictures, not a rack design)."""
    import cadquery as cq
    import wasp
    import emerillon_flight as ef
    ship = st.Sterna().generate().shape.scale(k)
    if not with_wasps:
        return ship
    p = st.resolve()
    veh = wasp.vehicle(wasp.VARIANTS[wasp_build])
    w = wasp.cad(wasp.VARIANTS[wasp_build]).shape
    with wasp.small():
        g = ef.geometry(wasp.VARIANTS[wasp_build])
    drop = g["r_arm"] * 1e3 / math.sqrt(2) + g["pod_d"] * 1e3 / 2 + 12.0             # the WASP's axis above the surface it sits on
    x0 = k * p["cg_x"] - veh.x_cg * 1e3                                              # its CG over the carrier's
    parts = [ship]
    for y, z_top in ((0.0, k * 50.0), (k * p["centre_span"] / 2, k * (p["wing_z"] + 20.0)), (-k * p["centre_span"] / 2, k * (p["wing_z"] + 20.0))):
        parts.append(w.translate(cq.Vector(x0, y, z_top + drop)))
        parts.append(cq.Workplane("XY").box(60.0, 8.0, drop).translate((k * p["cg_x"], y, z_top + drop / 2 - 4.0)).val())   # the rack's post
    return cq.Compound.makeCompound(parts)
