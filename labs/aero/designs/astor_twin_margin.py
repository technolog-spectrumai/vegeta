"""ASTOR-TWIN sized for MARGIN, not for the minimum (2026-11-24): what a larger wing, fewer nets, a lighter pack and a
propeller pitched for slow flight do to the numbers that decide the attack, against targets with room to spare.

A point-mass study on the flight model's CFD tables (``astor_v_flight.get("twin")``, flaps up and down). The wing is
scaled by area at the same section and flaps: the lift coefficients are kept, the drag is the body's part (kept) plus
the wing's (by area), the drag due to lift by the span. Masses: the wing by area (astor.WING_KG_M2), the launcher by
its nets (astor_twin.one_launcher), the pack as listed. HAND where marked. Nothing here changes the design.

The targets (each with its reason) are in TARGETS; a row's ``short`` lists the ones it misses.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import astor as A
import astor_twin as W
import astor_v_flight as VF

QUAD_M_S = (8.0, 14.0)          # astor_hunt's population
TARGETS = {"stall_flaps_m_s": ("<=", 15.5, "a pass at 1.3 x stall closes on a 14 m/s quadcopter at 6 m/s or less (now 11-17 m/s)"),
           "static_thrust_over_weight": (">=", 2.2, "a nose-high flare held on the propellers with a fifth of the thrust left for control (now 1.84)"),
           "radius_25_m": ("<=", 30.0, "inside a jinking quadcopter's ~30 m of swing (now 52 m)"),
           "g_at_50_m_s": (">=", 8.0, "the structure, not the wing, sets the turn from 50 m/s (now 6.4 g)"),
           "top_speed_m_s": (">=", 70.0, "a 50 m/s target is still overtaken with 20 m/s to spare"),
           "full_power_min": (">=", 2.0, "three passes with their turns back at full power (now 1.2 min)"),
           "gust_to_stall_at_loiter_m_s": (">=", 5.0, "the loiter survives a 5 m/s gust (now 2.7 m/s)")}
PROPS = {"11x13 (library, pitched for the dash)": {"static": 1.0, "fast": 1.0, "source": "propulsion library"},
         "11 in, pitched for slow flight (HAND)": {"static": 1.45, "fast": 0.78, "source": "HAND: momentum theory at a figure of merit 0.6 gives 190 N static for two; the fast end by the pitch ratio 10/13"},
         "13 in, pitched for slow flight (HAND)": {"static": 1.62, "fast": 0.78, "source": "HAND: the same, x (13/11)^(2/3); needs the motors 30 mm further out"}}
PACKS = {"12S 5000 mAh": (1.52, 222.0), "12S 2500 mAh": (0.76, 111.0)}


def evaluate(area=1.0, span=1.0, nets=3, pack="12S 5000 mAh", prop=next(iter(PROPS))) -> dict:
    C = VF.get("twin1")                                                  # HAWK1: the aircraft this study started from (its "wing x2" row became HAWK 2)
    F = VF.with_flap(C, 1.0)
    S0, b0 = C["S"], C["b"]
    S, b = S0 * area, b0 * span
    m = (C["mass"] + 1.1 * A.WING_KG_M2 * (S - S0) + (W.one_launcher(nets)["launcher_kg"] - W.one_launcher(3)["launcher_kg"])
         + (PACKS[pack][0] - PACKS["12S 5000 mAh"][0]))
    Wt = m * am.G
    t, tf = C["cfd"], F["cfd"]
    Aa = np.asarray(t["alpha_deg"])

    def polar(tab, lim_deg):
        i0, i1 = int(np.argmin(np.abs(Aa + 4.0))), int(np.argmin(np.abs(Aa - lim_deg)))
        cl, cd = np.asarray(tab["CL"])[i0:i1 + 1], np.asarray(tab["CD"])[i0:i1 + 1]
        cd_min = float(cd.min())
        # this wing's drag: the table's lowest drag split into the body (kept as an area) and the wing (by its area);
        # the rest (the drag due to lift) by the span squared
        body = cd_min * S0 - VF.WING_CD * 1.3 * S0
        # the drag due to lift: the same force at the same lift force (the table read at the lift coefficient this
        # lift would be on the original wing), less by the span squared
        return cl, lambda c: (body + VF.WING_CD * 1.3 * S) / S + float(np.interp(c * S / S0, cl, cd) - cd_min) * (S0 / S) * (b0 / b) ** 2
    lim_up = math.degrees(C["alpha_stall"]) - VF.ALPHA_MARGIN_DEG
    lim_dn = math.degrees(F["alpha_stall"])
    cl_up, cd_up = polar(t, lim_up)
    cl_dn, cd_dn = polar(tf, lim_dn)
    clmax_up, clmax_dn = float(C["CL_MAX"]), float(F["CL_MAX"])
    use_up, use_dn = float(np.interp(lim_up, Aa, t["CL"])), float(np.interp(lim_dn, Aa, tf["CL"]))
    th = C["thrust"]
    k = PROPS[prop]

    def thrust(V):
        w = min(1.0, V / 60.0)
        return float(np.interp(V, th["table_V"], th["table_T"])) * ((1 - w) * k["static"] + w * k["fast"])
    stall = lambda cl: math.sqrt(2 * Wt / (am.RHO * S * cl))

    def n_at(V):
        q = 0.5 * am.RHO * V * V
        best = 0.0
        for use, cd in ((use_up, cd_up), (use_dn, cd_dn)):
            n = q * S * use / Wt
            lo, hi = 0.0, n
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if q * S * cd(mid * Wt / (q * S)) <= thrust(V) else (lo, mid)
            best = max(best, min(n if q * S * cd(n * Wt / (q * S)) <= thrust(V) else lo, 8.0))
        return best
    radius = lambda V: (V * V / (am.G * math.sqrt(n_at(V) ** 2 - 1))) if n_at(V) > 1.02 else float("inf")
    drag = lambda V: 0.5 * am.RHO * V * V * S * cd_up(Wt / (0.5 * am.RHO * V * V * S))
    vs = np.arange(30.0, 110.0, 0.5)
    ok = [v for v in vs if thrust(v) >= drag(v)]
    v_loiter = max(1.3 * stall(clmax_up), 26.0)
    p_loiter = drag(v_loiter) * v_loiter / 0.6 + 15.0                    # HAND: 0.6 propulsive x electrical efficiency; 15 W of avionics
    a_trim = float(np.interp(Wt / (0.5 * am.RHO * v_loiter ** 2 * S), np.asarray(t["CL"])[8:24], Aa[8:24]))
    gust = v_loiter * math.tan(math.radians(max(math.degrees(C["alpha_stall"]) - a_trim, 0.0)))
    s_dn = stall(clmax_dn)
    out = {"wing_area_m2": round(S, 3), "span_m": round(b, 2), "nets": nets, "pack": pack, "propeller": prop, "mass_kg": round(m, 2),
           "wing_loading_kg_m2": round(m / S, 1), "stall_m_s": round(stall(clmax_up), 1), "stall_flaps_m_s": round(s_dn, 1),
           "closing_on_8_and_14_m_s_quadcopters_m_s": [round(1.3 * s_dn - v, 1) for v in QUAD_M_S],
           "static_thrust_over_weight": round(thrust(0.0) / Wt, 2), "radius_25_m": round(radius(25.0), 1), "radius_30_m": round(radius(30.0), 1),
           "g_at_50_m_s": round(n_at(50.0), 1), "top_speed_m_s": round(float(max(ok)), 1) if ok else None,
           "full_power_min": round(PACKS[pack][1] * 0.85 * 60.0 / 8015.0, 1), "loiter_m_s": round(v_loiter, 1),
           "endurance_min": round(PACKS[pack][1] * 0.85 * 60.0 / p_loiter, 0), "gust_to_stall_at_loiter_m_s": round(gust, 1)}
    out["short"] = [key for key, (op, val, _) in TARGETS.items() if out[key] is None or (out[key] > val if op == "<=" else out[key] < val)]
    return out


CASES = [("as it is", {}),
         ("wing +50 % by chord", {"area": 1.5}),
         ("wing +50 %: span 1.69 m", {"area": 1.5, "span": 1.225}),
         ("wing x2: span 1.95 m", {"area": 2.0, "span": 1.414}),
         ("as it is, slow-flight 11 in propellers", {"prop": list(PROPS)[1]}),
         ("wing +50 % span, slow 11 in propellers", {"area": 1.5, "span": 1.225, "prop": list(PROPS)[1]}),
         ("wing +50 % span, slow 13 in propellers", {"area": 1.5, "span": 1.225, "prop": list(PROPS)[2]}),
         ("wing +50 % span, slow 13 in, two nets", {"area": 1.5, "span": 1.225, "prop": list(PROPS)[2], "nets": 2}),
         ("wing x2 span, slow 13 in, three nets", {"area": 2.0, "span": 1.414, "prop": list(PROPS)[2]}),
         ("wing x2 span, slow 13 in, two nets", {"area": 2.0, "span": 1.414, "prop": list(PROPS)[2], "nets": 2}),
         ("wing x2 span, slow 13 in, two nets, small pack", {"area": 2.0, "span": 1.414, "prop": list(PROPS)[2], "nets": 2, "pack": "12S 2500 mAh"})]

if __name__ == "__main__":
    import yaml
    rows = {name: evaluate(**kw) for name, kw in CASES}
    path = am.DATA / "astor_twin" / "astor_twin_margin.yaml"
    path.write_text(yaml.safe_dump({"targets": {k: {"rule": f"{op} {v}", "why": why} for k, (op, v, why) in TARGETS.items()}, "propellers": PROPS, "cases": rows},
                                   sort_keys=False, allow_unicode=True, width=160))
    print(f"{'case':48s} {'kg':>5s} {'stall':>5s} {'flap':>5s} {'close':>11s} {'T/W':>5s} {'R25':>5s} {'R30':>5s} {'g50':>4s} {'top':>5s} {'full':>4s} {'end':>4s} {'gust':>4s}  misses")
    for n, r in rows.items():
        print(f"{n:48s} {r['mass_kg']:5.2f} {r['stall_m_s']:5.1f} {r['stall_flaps_m_s']:5.1f} {str(r['closing_on_8_and_14_m_s_quadcopters_m_s']):>11s} {r['static_thrust_over_weight']:5.2f} "
              f"{r['radius_25_m']:5.1f} {r['radius_30_m']:5.1f} {r['g_at_50_m_s']:4.1f} {str(r['top_speed_m_s']):>5s} {r['full_power_min']:4.1f} {r['endurance_min']:4.0f} {r['gust_to_stall_at_loiter_m_s']:4.1f}  {len(r['short'])}")
    print("->", path)
