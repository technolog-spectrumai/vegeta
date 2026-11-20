"""ARIGO-2's flight model: ARIGO-1's (``arigo1_model``, the twin layout's coefficients) with the two motors apart —
each its own throttle, the thrust lines at y = +-nacelle_y, so differential thrust yaws and an engine out yaws.

Still hand estimates (marked in ``arigo1_model``): the thrust curve (each motor half the twin's, +-25 %), the
mass (2.5 kg). Not modelled: the slipstream's extra lift on the blown inner wing and its effect on the stall —
the CFD's power-on cases (``arigo2_cfd``) measure it; paste them back to put it in.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import arigo1_model as am
import arigo2 as a2


@lru_cache(maxsize=None)
def coefficients(wing: str = "hybrid") -> dict:
    import arigo_mass
    C = dict(am.coefficients(wing, "twin"))
    p = a2.resolve(a2.variant(wing))
    mb = arigo_mass.budget("twin", p, a2.Arigo2.layout(p))
    k = mb["total_g"] / 1000.0 / C["mass"]                     # ARIGO-2's own budget (its nacelles and props) over the sketch's
    C.update({"mass": mb["total_g"] / 1000.0, "inertia": tuple(i * k for i in C["inertia"]), "mass_budget": mb})
    C.update({"aircraft": "ARIGO-2", "y_motor_m": p["nacelle_y"] / 1e3, "per_motor_T0_N": 0.5 * C["thrust"]["T0_N"]})
    t = cfd_table(wing, C)
    if t is not None:                                          # ARIGO-2's own CFD (nacelles, propellers off)
        for k in ("cfd", "Cm_rig", "stabiliser_rig_deg", "static_margin_cfd"):
            C.pop(k, None)
        am.calibrate(C, t, 18.0)
        C["aero_source"] = f"cfd (screening RANS, ARIGO-2's own, {t['file']})"
    C["thrust"] = am.propulsion_table("twin", C["mass"], C["CD0"], C["S"])     # ARIGO-2's mass
    return C


def cfd_table(wing, C):
    """ARIGO-2's propellers-off CFD (``data/arigo2_cfd_screening*.yaml``, ``<wing>_aero``), newest file first; the
    moment moved to the current CG if it differs."""
    import yaml
    for f in sorted(am.DATA.glob("arigo2_cfd_screening*.yaml"), key=lambda f: f.name, reverse=True):
        d = yaml.safe_load(f.read_text()) or {}
        key = f"{wing}_aero"
        if key in (d.get("stale") or {}):
            continue
        cfg = (d.get("configurations") or {}).get(key)
        if not cfg or abs(cfg.get("reference", {}).get("S", 0) - C["S"]) > 1e-4:
            continue
        rows = am.clean_rows(cfg.get("rows", []))
        if len(rows) < 6:
            continue
        t = am.table(rows)
        dx = C["x_cg_m"] * 1e3 - cfg["reference"]["cg_mm"][0]
        if abs(dx) > 1.0:
            t = am.transfer(t, dx / 1000.0, C["c"])
        return {**t, "file": f.name}
    return None


def thrusts(C: dict, u_air: float, t_left: float, t_right: float):
    """Each motor's thrust [N] at the forward airspeed u_air."""
    half = 0.5 * am.max_thrust(C, u_air)                       # each motor: half the pair's full thrust at this speed
    return (half * max(0.0, min(1.0, t_left)), half * max(0.0, min(1.0, t_right)))


def aero2(C: dict, vel_b, omega_b, de=0.0, da=0.0, dr=0.0, t_left=0.5, t_right=0.5):
    """Force and moment (body FRD, about the CG): ARIGO-1's aerodynamics without its thrust, plus the two motors."""
    F, M, info = am.aero(C, vel_b, omega_b, de, da, dr, throttle=0.0)
    tl, tr = thrusts(C, vel_b[0], t_left, t_right)
    y = C["y_motor_m"]
    F = F + np.array([tl + tr, 0.0, 0.0])
    M = M + np.array([0.0, 0.0, y * (tl - tr)])           # r x F: the left motor (y = -y_m) pushing yaws right (N > 0)
    return F, M, {**info, "T_left": tl, "T_right": tr}


def v_mc(C: dict, throttle: float = 1.0) -> float:
    """The speed below which full rudder no longer holds one motor at ``throttle`` [m/s] (bisection)."""
    def margin(V):
        T1 = thrusts(C, V, throttle, 0.0)[0]
        return abs(C["Cndr"]) * am.DR_MAX * 0.5 * am.RHO * V * V * C["S"] * C["b"] - T1 * C["y_motor_m"]
    lo, hi = 1.0, 40.0
    if margin(lo) > 0:
        return lo
    if margin(hi) < 0:
        return math.inf
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if margin(mid) > 0 else (mid, hi)
    return hi


def max_throttle_one_motor(C: dict, V: float, share: float = 0.8) -> float:
    """The good motor's throttle at which its yaw takes ``share`` of the full rudder's moment (the autopilot's limit
    with an engine out)."""
    N = share * abs(C["Cndr"]) * am.DR_MAX * 0.5 * am.RHO * V * V * C["S"] * C["b"]
    T_full = thrusts(C, V, 1.0, 0.0)[0]
    return 1.0 if T_full <= 0 else max(0.0, min(1.0, N / (T_full * C["y_motor_m"])))
