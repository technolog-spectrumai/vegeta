"""ASTOR-TWIN's turning at low and at high speed, and what each change to the aircraft would buy (2026-11-23).

A point-mass study on the flight model's own numbers (``astor_v_flight.get("twin")``: the lift and drag table from its
CFD, the thrust table of its two propellers). At each speed the load factor is the least of three limits:

- LIFT: the wing at the angle-of-attack limiter (the CFD's lift there), plus what an option adds;
- STRUCTURE: the g limit (5 g now: ``astor_v_flight.SAS``; no structural rating exists);
- THRUST: the load factor whose drag the two motors can hold (a sustained turn).

The options' increments are HAND estimates from textbook flap and section data (each line says which); the two that
change the shape (the section, the flaps) are drawn in ``astor_twin`` so that CFD can replace the estimates
(``sims/astor.sh --agility-cfd``). Nothing here changes the flight model.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import astor_twin as W
import astor_v_flight as VF

SPEEDS = (25.0, 30.0, 40.0, 50.0, 60.0, 80.0)
FLAP = {"chord": 0.30, "deg": 20.0}
# a plain flap: thin-aerofoil effectiveness for its chord share, x a viscous efficiency at 20 deg (HAND 0.65 +-0.1: the
# usual plain-flap chart), on the share of the wing it spans; the maximum lift rises by ~0.6 of the lift added at a fixed
# angle (HAND +-0.15)
_th = math.acos(2 * FLAP["chord"] - 1)
FLAP_TAU, FLAP_ETA, FLAP_CLMAX_SHARE = 1 - (_th - math.sin(_th)) / math.pi, 0.65, 0.6
SECTION_DCLMAX = 0.22        # HAND (+-0.1): a 12 % thick, 4 % cambered section against the 8 %, 2 % one at Re ~4e5 (2-D 1.35 against 1.10, x 0.87 for the wing)
BLOWN_EFF = 0.8              # HAND (+-0.15): the share of the slipstream's extra dynamic pressure the wing behind a propeller turns into lift


def _wing():
    p = W.resolve(W.variant(net_launchers=1, talon_deploy=-1.0, net_count=0))
    c = lambda y: W.WING["root_chord"] + (W.WING["tip_chord"] - W.WING["root_chord"]) * y / W.WING["half_span"]
    yj, b2 = W.WING["joint_y"], W.WING["half_span"]
    y0, y1 = yj + p["panel_aileron_from"] * (b2 - yj), yj + p["panel_aileron_to"] * (b2 - yj)
    outer = (c(y0) + c(y1)) * (y1 - y0) / 1e6                           # both sides: the wing area the flaperons span
    R = p["fuselage_diameter"] / 2
    inner = (c(R) + c(yj)) * (yj - R - p["nacelle_d"]) / 1e6            # the inner pieces' trailing edges, less the nacelles
    return p, c, outer, inner


def blown(C, V, T_total, p, c) -> float:
    """The extra wing area [m2 of free-stream dynamic pressure] from the two propellers' slipstreams at this thrust:
    momentum theory for the jet's speed and its contracted width, on the chord behind each propeller."""
    A = math.pi * (p["prop_diameter"] / 2000.0) ** 2
    vj = math.sqrt(V * V + 2 * (T_total / 2) / (am.RHO * A))
    width = p["prop_diameter"] / 1000.0 * math.sqrt((V + vj) / (2 * vj))
    return BLOWN_EFF * 2 * width * c(p["motor_y"]) / 1000.0 * ((vj / V) ** 2 - 1.0)


def envelope(opt: dict) -> list:
    """Per speed: the three limits, the load factor, the turn radius and rate. ``opt``: dCL (lift added at the limiter),
    dCLmax, dCD, area (x), mass_kg (+), g_max, blown (bool)."""
    C = VF.get("twin")
    p, c, _, _ = _wing()
    S = C["S"] * opt.get("area", 1.0)
    m = C["mass"] + opt.get("mass_kg", 0.0)
    t = C["cfd"]
    A, CLt, CDt = np.asarray(t["alpha_deg"]), np.asarray(t["CL"]), np.asarray(t["CD"])
    a_lim = math.degrees(C["alpha_stall"]) - VF.ALPHA_MARGIN_DEG
    cl_lim = float(np.interp(a_lim, A, CLt)) + opt.get("dCLmax", 0.0)
    k = int(np.argmax(CLt))
    th = C["thrust"]
    rows = []
    for V in SPEEDS:
        q = 0.5 * am.RHO * V * V
        T = float(np.interp(V, th["table_V"], th["table_T"]))
        Se = S + (blown(C, V, T, p, c) if opt.get("blown") else 0.0)
        n_lift = q * Se * cl_lim / (m * am.G)

        def drag(n):
            cl = n * m * am.G / (q * Se)
            base = cl - opt.get("dCL", 0.0)                             # the wing's own share: its drag from the table
            cd = float(np.interp(base, CLt[:k + 1], CDt[:k + 1])) if base <= CLt[k] else float(CDt[k] + C["k"] * (base ** 2 - CLt[k] ** 2))
            return q * S * (cd + opt.get("dCD", 0.0))
        lo, hi = 0.0, 30.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if drag(mid) <= T else (lo, mid)
        n_thrust = lo
        n = min(n_lift, opt.get("g_max", VF.SAS["twin"]["g_max"]), n_thrust)
        if n > 1.02:
            w = am.G * math.sqrt(n * n - 1) / V
            rows.append({"V_m_s": V, "n": round(n, 2), "limit": "lift" if n == n_lift else "thrust" if n == n_thrust else "structure",
                         "radius_m": round(V / w, 1), "rate_deg_s": round(math.degrees(w), 1), "n_lift": round(n_lift, 2), "n_thrust": round(n_thrust, 2)})
        else:
            rows.append({"V_m_s": V, "n": round(n, 2), "limit": "below the stall", "radius_m": None, "rate_deg_s": None, "n_lift": round(n_lift, 2), "n_thrust": round(n_thrust, 2)})
    stall = math.sqrt(2 * m * am.G / (am.RHO * S * (float(CLt[k]) + opt.get("dCLmax", 0.0))))
    out = [{"stall_m_s": round(stall, 1)}] + rows
    if opt.get("flap_dCL") is None and (opt.get("dCL") or 0.0) > 0.0 and "_no_flap" in opt:
        clean = envelope(opt["_no_flap"])                                # the flaps come up when they no longer help (their drag at speed)
        out = [out[0]] + [({**c, "flaps": "up"} if c["n"] > r["n"] else {**r, "flaps": "down"}) for r, c in zip(out[1:], clean[1:])]
    return out


def options() -> dict:
    C = VF.get("twin")
    p, c, S_out, S_in = _wing()
    S = C["S"]
    da0 = FLAP_TAU * FLAP_ETA * math.radians(FLAP["deg"])
    f = lambda Sf: {"dCL": C["CLa"] * da0 * Sf / S, "dCLmax": FLAP_CLMAX_SHARE * C["CLa"] * da0 * Sf / S,
                    "dCD": 0.9 * FLAP["chord"] ** 1.38 * Sf / S * math.sin(math.radians(FLAP["deg"])) ** 2}
    add = lambda *os: {k: (sum(o.get(k, 0.0) for o in os) if k in ("dCL", "dCLmax", "dCD", "mass_kg") else
                           max(o.get(k, 0) for o in os) if k in ("g_max", "blown") else float(np.prod([o.get(k, 1.0) for o in os])))
                       for k in {kk for o in os for kk in o}}
    flaperon, inboard = {**f(S_out), "_no_flap": {}}, {**f(S_in), "_no_flap": {}}
    section = {"dCLmax": SECTION_DCLMAX, "dCD": 0.0012}                  # HAND: + 10 % of the wing's profile drag
    area = {"area": 1.25, "mass_kg": 1.25 * 0.0 + A_WING_KG * 0.25 * S, "dCD": 0.0}
    sweep35 = {"dCLmax": -(1 - math.cos(math.radians(23.0))) * 0.884 * 0.6}       # the outer 60 % of the area swept 35 deg: its lift x cos (the flight model's rule)
    g8 = {"g_max": 8.0, "mass_kg": 0.10}                                 # HAND: 100 g of spar and joiner for an 8 g rating
    blownw = {"blown": True}
    light = {"mass_kg": -0.76}                                           # the 12S 2500 mAh pack (16.5 min in place of 28)
    out = {"as it is": {}, "outer panels swept to 35 deg": sweep35, "flaperons 20 deg (the ailerons drooped together)": flaperon,
           "inboard flaps 20 deg behind the propellers": inboard, "section 12 % thick, 4 % camber": section, "wing chord + 25 %": area,
           "propellers' slipstream counted (full throttle)": blownw, "8 g structure": g8, "12S 2500 mAh pack (-0.76 kg)": light,
           "ALL: section + flaperons + inboard flaps + slipstream + 8 g": {**add(section, f(S_out), f(S_in), blownw, g8), "_no_flap": add(section, blownw, g8)}}
    return {"flap_wing_area_m2": {"flaperons": round(S_out, 4), "inboard": round(S_in, 4), "wing": round(S, 4)},
            "hand": {"flap_effectiveness": round(FLAP_TAU, 3), "flap_efficiency_at_20_deg": FLAP_ETA, "flap_CLmax_share": FLAP_CLMAX_SHARE,
                     "section_dCLmax": SECTION_DCLMAX, "slipstream_efficiency": BLOWN_EFF},
            "options": {name: {"changes": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if not k.startswith("_")}, "envelope": envelope(o)} for name, o in out.items()}}


A_WING_KG = 1.9                # kg/m2 (astor.WING_KG_M2)

if __name__ == "__main__":
    import sys
    import yaml
    r = options()
    path = am.DATA / "astor_twin" / "astor_twin_agility.yaml"
    path.write_text(yaml.safe_dump(r, sort_keys=False, allow_unicode=True, width=160))
    for name, o in r["options"].items():
        e = o["envelope"]
        print(f"{name[:58]:58s} stall {e[0]['stall_m_s']:5.1f} | " + " | ".join(f"{x['n']:4.1f}g {str(x['radius_m']):>6s}m" for x in e[1:]))
    print("speeds", SPEEDS, "->", path)
