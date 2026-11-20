"""BUTEO's flight model: wing-borne on ARIGO-1's machinery (``arigo1_model.build``: lattice + hand) with BUTEO's wing
and H tail; and the lift system — four rotors from the propulsor library — for the hover.

Sources, marked: **lattice** (CL0, CL_alpha, neutral point, elevator); **CFD** (``data/buteo_cfd_screening.yaml`` when
``sims/buteo_cfd.sh`` has run: the airframe with booms and stopped rotors); **propulsor library** (the pusher's
thrust over speed, the lift rotors' static thrust and electrical power); **hand** (HAND): CD0 0.045 +-0.010 until the
CFD (the family's 0.032 + booms, four motors, feet and stopped blades), the inertias (the booms' length in pitch),
the rotors' torque-to-thrust ratio, no thrust loss in a crossflow.
"""
from __future__ import annotations

import math
from functools import lru_cache

import arigo1_model as am
import buteo as B

CD0 = 0.045
CRUISE = 22.0
KQ_M = 0.016                      # HAND: a 9-inch rotor's torque / thrust [m] (+-30 %)
LIFT_PROP = "prop-9x5-b2"


@lru_cache(maxsize=None)
def coefficients(pack: str | None = None) -> dict:
    import arigo_propulsion as ap
    q = B.variant()
    p = B.resolve(q)
    L = B.Buteo.layout(p)
    mb = B.budget(p, L, pack)
    m = mb["total_g"] / 1000.0
    sel = ap.select("pusher", round(m, 3), CD0, round(L["S_mm2"] / 1e6, 4), power_w=B.PUSHER_W)
    C = am.build(q, p, L, "buteo", "pusher", mass=m, CD0=CD0, thrust=ap.thrust_table(sel))
    Ixx, Iyy, Izz = C["inertia"]
    Iyy = m * (0.30 * (L["boom_x1"] - p["boom_x0"]) / 1e3) ** 2          # HAND: the booms carry the rotors and the tail
    C["inertia"] = (Ixx, Iyy, (Ixx + Iyy) * 0.95)
    u = ap._unit(LIFT_PROP, B.LIFT_W, "tractor")
    T_max = float(u.full(0.0)[0])
    W = m * am.G
    P_hover = 4 * float(u.electrical_power(0.0, W / 4)) + 15.0
    xcg = C["x_cg_m"] * 1e3
    C.update({"aircraft": "BUTEO", "mass_budget": mb, "pack_kind": "buteo", "aero_source": "model (lattice + hand)",
              "lift": {"prop": LIFT_PROP, "power_w_each": B.LIFT_W, "T_max_each_N": round(T_max, 2), "T_W": round(4 * T_max / W, 2),
                       "hover_power_W": round(P_hover, 0), "kq_m": KQ_M,
                       # rotor positions in the body frame (x forward, y right) [m] about the CG, and the spin (+1: its
                       # reaction yaws the nose right)
                       "rotors": [((xcg - x) / 1e3, y / 1e3, s) for (x, y), s in zip(L["lift"], (1, -1, -1, 1))],
                       "hover_endurance_min": round(mb["pack"]["Wh_nominal"] * 0.85 / P_hover * 60.0, 1),
                       "peak_power_W": 4 * B.LIFT_W + 15.0, "pack_limit_W": round(mb["pack"]["max_A"] * 4 * 3.4)},
              "equipment": {"product": "buteo", "launch": "pad", "recovery": "pad", "charging": True, "kits": [],
                            "pack": mb["pack"]["name"]}})
    t = cfd_table(C)
    if t is not None:
        am.calibrate(C, t, CRUISE)
        C["aero_source"] = f"cfd (screening RANS, {t.get('file', 'buteo')})"
    return C


def get(pack: str | None = None) -> dict:
    return dict(coefficients(pack))


def cfd_table(C):
    import yaml
    f = am.DATA / "buteo_cfd_screening.yaml"
    if not f.exists():
        return None
    d = yaml.safe_load(f.read_text()) or {}
    ref = d.get("reference", {})
    if abs(ref.get("S", 0) - C["S"]) > 1e-4:
        return None
    rows = am.clean_rows(d.get("rows", []))
    if len(rows) < 6:
        return None
    t = am.table(rows)
    dx = C["x_cg_m"] * 1e3 - ref.get("cg_mm", [0])[0]
    t = am.transfer(t, dx / 1000.0, C["c"]) if abs(dx) > 1.0 else t
    return {**t, "file": f.name}


def stall(C=None) -> float:
    import arigo_perf as P
    C = C or get()
    return math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * P.polar(C)[1]))
