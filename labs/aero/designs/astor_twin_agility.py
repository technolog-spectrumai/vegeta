"""ASTOR-TWIN's turning at low and at high speed, and what each change to the aircraft would buy (2026-11-23).

A point-mass study on the flight model's own numbers (``astor_v_flight.get("twin")``: the lift and drag table from its
CFD, the thrust table of its two propellers). At each speed the load factor is the least of three limits:

- LIFT: the wing at the angle-of-attack limiter (the CFD's lift there, flaps up or down);
- STRUCTURE: the g limit (5 g now: ``astor_v_flight.SAS``; no structural rating exists);
- THRUST: the load factor whose drag the two motors can hold (a sustained turn).

The section (12 % thick, 4 % camber) and the flaps are the CFD's of 2026-11-23 and are in the flight model. The
propellers' slipstream over the inner wing and a higher g rating are still HAND estimates and are NOT in it.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import astor_twin as W
import astor_v_flight as VF

SPEEDS = (22.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0)
BLOWN_EFF = 0.8              # HAND (+-0.15): the share of the slipstream's extra dynamic pressure the wing behind a propeller turns into lift
OLD_THIN_WING = {"stall_m_s": 21.9, "radius_m": {25.0: 95.8, 30.0: 65.0, 40.0: 56.1, 50.0: 54.2, 60.0: 74.9, 80.0: 133.2},
                 "source": "this study on the 8 % thick, 2 % cambered wing's CFD (commit bbffdbe), 5 g limit"}


def _wing():
    p = W.resolve(W.variant(net_launchers=1, talon_deploy=-1.0, net_count=0))
    c = lambda y: W.WING["root_chord"] + (W.WING["tip_chord"] - W.WING["root_chord"]) * y / W.WING["half_span"]
    return p, c


def blown(C, V, T_total, p, c) -> float:
    """The extra wing area [m2 of free-stream dynamic pressure] from the two propellers' slipstreams at this thrust:
    momentum theory for the jet's speed and its contracted width, on the chord behind each propeller."""
    A = math.pi * (p["prop_diameter"] / 2000.0) ** 2
    vj = math.sqrt(V * V + 2 * (T_total / 2) / (am.RHO * A))
    width = p["prop_diameter"] / 1000.0 * math.sqrt((V + vj) / (2 * vj))
    return BLOWN_EFF * 2 * width * c(p["motor_y"]) / 1000.0 * ((vj / V) ** 2 - 1.0)


def envelope(flap=0.0, slipstream=False, g_max=None, mass_kg=0.0) -> list:
    """Per speed: the lift, structure and thrust limits, the load factor, the turn radius and rate, on the flight
    model's CFD table with the flaps a share ``flap`` down."""
    C = VF.with_flap(VF.get("twin"), flap)
    p, c = _wing()
    S, m = C["S"], C["mass"] + mass_kg
    t = C["cfd"]
    A, CLt, CDt = np.asarray(t["alpha_deg"]), np.asarray(t["CL"]), np.asarray(t["CD"])
    a_lim = math.degrees(C["alpha_stall"]) - C.get("alpha_margin_deg", VF.ALPHA_MARGIN_DEG)
    cl_lim = float(np.interp(a_lim, A, CLt))
    i0, i1 = int(np.argmin(np.abs(A + 4.0))), int(np.argmin(np.abs(A - a_lim)))
    th = C["thrust"]
    g_max = g_max or VF.SAS["twin"]["g_max"]
    rows = []
    for V in SPEEDS:
        q = 0.5 * am.RHO * V * V
        T = float(np.interp(V, th["table_V"], th["table_T"]))
        Se = S + (blown(C, V, T, p, c) if slipstream else 0.0)
        n_lift = q * Se * cl_lim / (m * am.G)
        drag = lambda n: q * S * float(np.interp(n * m * am.G / (q * Se), CLt[i0:i1 + 1], CDt[i0:i1 + 1]))
        lo, hi = 0.0, n_lift
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if drag(mid) <= T else (lo, mid)
        n_thrust = hi if drag(n_lift) <= T else lo
        n = min(n_lift, g_max, n_thrust)
        w = am.G * math.sqrt(max(n * n - 1, 0.0)) / V
        rows.append({"V_m_s": V, "n": round(n, 2), "limit": "lift" if n == n_lift else "structure" if n == g_max else "thrust",
                     "radius_m": round(V / w, 1) if n > 1.02 else None, "rate_deg_s": round(math.degrees(w), 1) if n > 1.02 else None})
    return [{"stall_m_s": round(VF.stall({**C, "mass": m}), 1), "lift_at_the_limiter": round(cl_lim, 3)}] + rows


def best(a, b) -> list:
    """Flaps as the flight controller would use them: down where they give the tighter turn, up elsewhere."""
    return [a[0]] + [({**y, "flaps": "down"} if y["n"] > x["n"] else {**x, "flaps": "up"}) for x, y in zip(a[1:], b[1:])]


def options() -> dict:
    up, dn = envelope(0.0), envelope(1.0)
    out = {"flaps up (CFD)": up, "flaps down 20 deg (CFD)": dn, "flaps as needed (CFD)": [dn[0]] + best(up, dn)[1:],
           "+ the propellers' slipstream at full throttle (HAND)": [dn[0]] + best(envelope(0.0, True), envelope(1.0, True))[1:],
           "+ slipstream and an 8 g structure (HAND: +100 g)": [dn[0]] + best(envelope(0.0, True, 8.0, 0.10), envelope(1.0, True, 8.0, 0.10))[1:]}
    return {"old_thin_wing": OLD_THIN_WING, "hand": {"slipstream_efficiency": BLOWN_EFF, "g_limit_now": VF.SAS["twin"]["g_max"]}, "options": {k: {"envelope": v} for k, v in out.items()}}


A_WING_KG = 1.9                # kg/m2 (astor.WING_KG_M2)

if __name__ == "__main__":
    import sys
    import yaml
    r = options()
    path = am.DATA / "astor_twin" / "astor_twin_agility.yaml"
    path.write_text(yaml.safe_dump(r, sort_keys=False, allow_unicode=True, width=160))
    for name, o in r["options"].items():
        e = o["envelope"]
        print(f"{name[:52]:52s} stall {e[0]['stall_m_s']:5.1f} | " + " | ".join(f"{x['n']:4.1f}g {str(x['radius_m']):>6s}m" for x in e[1:]))
    print("speeds", SPEEDS, "->", path)
