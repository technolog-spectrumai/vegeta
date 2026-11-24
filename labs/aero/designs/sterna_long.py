"""STERNA-L — STERNA bigger, with a longer wing (a variant of notebook 51's aircraft).

STERNA is 1.5 m and 2 to 2.45 kg: one person throws it. STERNA-L gives that up for reach and load: the fuselage, fin
and rails 1.2 times the size, and a wing of 2.3 m — more span than the scale alone gives, at nearly the same chord, so
the aspect ratio goes from 7.2 to about 11. A longer, narrower wing loses less to induced drag at the low speeds a
survey flies at; the bigger airframe carries more battery and a heavier pallet at STERNA's own wing loading.

What changes, and what is kept:

- wing: 2.3 m span, 250 mm root chord over a 600 mm flat centre, 140 mm at the tips, the same 5 degrees on the outer
  panels and 2 degrees of washout; in three pieces now (the centre panels would be 1.1 m as two): see ``PANELS``;
- tail: the tailplane 580 mm span on the scaled arm — the same tail volume as STERNA's; the fin scaled with the body;
- propulsion: a 14x8 folding propeller on a slower, larger motor (the drive's mass grows with the design mass);
- the design mass limit is set by STERNA's wing loading limit (7.8 kg/m2) on the new area;
- the battery bay takes up to four of STERNA's 4S1P modules (two abreast in the wider body, two in line);
- everything else is STERNA: the shoulder wing, the tractor, the belly pallet ahead of the CG, the rails.

The masses follow ``sterna_mass`` with its growth rules: structure with the scale squared, the wing's areal density up
with the square root of the aspect ratio (a longer spar is a heavier one: HAND), the drive with the design mass, the
servos with the scale, the avionics fixed. Nothing here is weighed, and nothing is a solver result: the polar is the
hand polar for both aircraft (STERNA's CFD fit is not carried over to a wing it was not run on).
"""
from __future__ import annotations

import math

import arigo1_model as am
import sterna as st
import sterna_mass as sm
import sterna_model as mo

K = 1.20
LOADING_LIMIT = 7.83                       # kg/m2: STERNA at its design limit
GEOMETRY = {"scale": K, "span": 2300.0, "root_chord": 250.0, "tip_chord": 140.0, "centre_span": 600.0,
            "wing_x": 330.0 * K, "wing_z": 38.0 * K, "aileron_from": 700.0, "aileron_to": 1110.0, "flap_from": 70.0, "flap_to": 690.0,
            "htail_span": 580.0, "htail_root_chord": 150.0, "htail_tip_chord": 110.0, "htail_x": 948.0 * K, "htail_z": 18.0 * K,
            "prop_diameter": 356.0}
PANELS = {"centre": "one piece, 600 mm, on the fuselage", "outer": "two panels of 850 mm on a joiner each side"}
DRIVE = dict(kv=580.0, prop_in=(14.0, 8.0), max_a=55.0)
CL_MAX_2D = 1.33                           # Zephyr, notebook 51 (the same section)


def parameters() -> dict:
    """The variant's parameters: the geometry, the design mass limit from the wing loading, the CG at 30 % MAC."""
    p = st.resolve(GEOMETRY)
    L = st.Sterna.layout(p)
    mtow = LOADING_LIMIT * L["S_mm2"] / 1e6 * 1000.0
    return {**GEOMETRY, "mtow_g": round(mtow, 0), "cg_x": round(L["x_mac_le"] + 0.30 * L["mac"], 1)}


def case(payload_g: float, modules: int, over=None) -> tuple:
    """(payload, modules, the tray station that puts the CG at 30 % MAC) for ``sterna_mass.budget``."""
    p = st.resolve(over if over is not None else parameters())
    L = st.Sterna.layout(p)
    return (payload_g, modules, sm.tray_for(p, L, payload_g, modules))


def card(over: dict | None, payload_g: float, modules: int, drive: dict | None = None, V: float = 14.0) -> dict:
    """One aircraft (``over`` None: STERNA itself) in one loading: geometry, mass, speeds, the cruise point."""
    q = {} if over is None else over
    p = st.resolve(q)
    L = st.Sterna.layout(p)
    c = case(payload_g, modules, q)
    C = mo.coefficients(c, cl_max_2d=CL_MAX_2D, **q)
    sysn = mo.propulsion(modules=modules, **(drive or {}))
    tt = mo.thrust_table(sysn)
    sp = mo.speeds(C)
    W = C["mass"] * am.G
    best = max((mo.cruise_point(C, sysn, float(v)) for v in (11, 12, 13, 14, 15, 16, 18)), key=lambda r: r["minutes"] * r["V"])
    cr = mo.cruise_point(C, sysn, V)
    b = C["mass_budget"]
    return {"span_mm": p["span"], "length_mm": L["length"], "S_m2": L["S_mm2"] / 1e6, "AR": L["AR"], "mac_mm": L["mac"], "V_h": L["V_h"], "V_v": L["V_v"],
            "mtow_g": p["mtow_g"], "empty_g": b["empty_g"], "payload_g": payload_g, "modules": modules, "battery_Wh": b["battery"]["Wh_nominal"],
            "takeoff_g": b["total_g"], "within_mtow": b["within_mtow"], "loading_kg_m2": b["total_g"] / 1000 / (L["S_mm2"] / 1e6),
            "tray_mm": c[2], "cg_mac": b["cg_mac"], "static_margin": C["static_margin"],
            "stall_m_s": sp["V_stall"], "LD_max": sp["LD_max"], "V_best_glide": sp["V_best_glide"],
            "static_thrust_N": tt["T0_N"], "static_thrust_over_weight": tt["T0_N"] / (p["mtow_g"] / 1000 * am.G), "static_current_A": tt["table_A"][0],
            "cruise_V": V, "cruise_W": cr["total_W"], "cruise_LD": cr["L_D"], "minutes": cr["minutes"], "range_km": cr["minutes"] * 60 * V / 1000,
            "best_range_V": best["V"], "best_range_km": best["minutes"] * 60 * best["V"] / 1000, "best_range_minutes": best["minutes"],
            "joiner_moment_6g_Nm": 6.0 * p["mtow_g"] / 1000 * am.G / 2 * (0.43 * p["span"] / 2 - 50.0 * p["scale"]) / 1000}
