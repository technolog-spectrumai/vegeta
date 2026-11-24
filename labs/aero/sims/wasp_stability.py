"""WASP's stability in fast level flight — a calculation, and the CFD's verdict when it exists. RUN BY YOU.

WASP flies nose-first as a body of revolution with four arms and four fins. Hovering, it is held by the motors alone;
this is about the fast flight, where the air's moment on the body grows with the square of the speed:

1. **By hand**: the pitching moment about the CG per radian of incidence — the slender body's destabilising moment
   (Munk: 2 q x the volume, with a fineness correction), the arms and the fins as low-aspect-ratio wings at their
   stations (the lift slope of ``emerillon_flight.normal_force_area`` for the arms, Helmbold's for the fins) — hence the
   neutral point and the static margin in body lengths. The propellers' own normal force is left out.
2. **From the CFD** (``<build>_cfd.yaml``, the ``alpha`` campaign, if it has been run): the same two slopes from the
   cases up to 10 degrees, and the neutral point they give. The CFD's body has no propellers either.
3. **What follows**: the short-period frequency (or the divergence time, if it is unstable) at 50, 75 and 100 % of
   the top speed from the moment slope and the pitch inertia of the mass budget, and the motors' authority: the moment
   a 10 % thrust difference between the front-up and front-down pairs gives, against the air's moment per degree.
4. **The CG**: where the budget puts it, and how far 10 mm of battery travel moves it.

Writes ``labs/aero/sims/output/<build>_stability.yaml``. No solver is run here; it takes seconds.

    python labs/aero/sims/wasp_stability.py
    python labs/aero/sims/wasp_stability.py --build WASP-1F
"""
import argparse
import math

import numpy as np

import wasp_common as wc
from wasp_common import fam

import emerillon as em  # noqa: E402
import emerillon_flight as ef  # noqa: E402
import emerillon_market as mk  # noqa: E402


def hand(veh, p: dict) -> dict:
    """Normal-force and moment slopes per radian over the dynamic pressure [m2, m3], about the CG (nose-up positive)."""
    g, L = veh.geo, em.Emerillon.layout(p)
    d, length = g["d"], g["length"]
    vol = math.pi * d * d / 4 * (0.5 * g["nose"] + g["cyl"] + 0.4 * g["tail"])              # ogive ~ 1/2, cone-ish tail ~ 0.4 of the cylinder
    f = length / d
    k_munk = 1.0 - 1.3 / f                                                                    # k2 - k1 for a fineness of 6 to 10 (HAND +-0.05)
    m_body = 2.0 * k_munk * vol                                                               # destabilising, a pure couple
    n_body = 2.0 * math.pi * (p["tail_end_diameter"] / 2e3) ** 2                              # a body's own lift ~ 2 x its base area: small here
    arms = 2.5 * 2 * g["arm_c"] * g["arm_l"]                                                  # emerillon_flight.normal_force_area's arm term
    x_arm = L["x_arm"] / 1e3 + 0.25 * g["arm_c"]
    span_f, chord_f = p["fin_span"] / 1e3, p["fin_chord"] / 1e3
    ar = (2 * span_f + d) ** 2 / ((2 * span_f + d) * chord_f)                                 # the horizontal pair and the body between as one wing
    a_fin = 2 * math.pi * ar / (2 + math.sqrt(ar * ar + 4))                                   # Helmbold
    fins = a_fin * (2 * span_f * chord_f)                                                     # of the four fins the horizontal pair lifts
    x_fin = L["x_fin"] / 1e3 + 0.25 * chord_f
    n = n_body + arms + fins
    m = m_body - arms * (x_arm - veh.x_cg) - fins * (x_fin - veh.x_cg)
    x_np = veh.x_cg - m / n
    return {"normal_slope_cm2_per_rad": wc.num(n * 1e4, 2), "moment_slope_cm3_per_rad": wc.num(m * 1e6, 2),
            "parts": {"body_moment_cm3": wc.num(m_body * 1e6, 2), "arms_cm2": wc.num(arms * 1e4, 2), "arms_x_mm": wc.num(x_arm * 1e3, 0),
                      "fins_cm2": wc.num(fins * 1e4, 2), "fins_x_mm": wc.num(x_fin * 1e3, 0)},
            "x_np_mm": wc.num(x_np * 1e3, 1), "static_margin_lengths": wc.num((x_np - veh.x_cg) / length, 3), "stable": bool(m < 0), "_n": n, "_m": m}


def from_cfd(veh, build: str) -> dict | None:
    doc = wc.load(wc.OUT / f"{wc.slug(build)}_cfd.yaml")
    rows = [r for r in (doc or {}).get("cases", {}).values() if r.get("campaign") == "alpha" and r.get("status") == "solved" and r["alpha_deg"] <= 10.0]
    if len(rows) < 3:
        return None
    a = np.radians([r["alpha_deg"] for r in rows])
    S, length = veh.geo["frontal"], veh.geo["length"]
    cn_a = float(np.polyfit(a, [r["CN"] for r in rows], 1)[0])
    cm_a = float(np.polyfit(a, [r["Cm_cg_length"] for r in rows], 1)[0])
    n, m = cn_a * S, cm_a * S * length
    return {"cases": len(rows), "quality": doc.get("quality"), "normal_slope_cm2_per_rad": wc.num(n * 1e4, 2), "moment_slope_cm3_per_rad": wc.num(m * 1e6, 2),
            "x_np_mm": wc.num((veh.x_cg - m / n) * 1e3, 1), "static_margin_lengths": wc.num(-m / n / length, 3), "stable": bool(m < 0), "_n": n, "_m": m}


def dynamics(veh, m_slope: float) -> list:
    """Per speed: the air's moment per degree, the short-period frequency or the time to double, the motors' moment."""
    Iyy = veh.inertia[1]
    r = veh.geo["r_arm"] / math.sqrt(2)
    out = []
    for f in (0.5, 0.75, 1.0):
        V = f * veh.V_top
        q = 0.5 * ef.RHO * V * V
        Ma = q * m_slope                                                                      # N m per rad
        w = math.sqrt(abs(Ma) / Iyy)
        with fam.ctx(veh.name):
            T = float(ef.balance(veh.config, V, veh.mass)["need"])
        out.append({"V_m_s": wc.num(V, 1), "air_moment_mNm_per_deg": wc.num(Ma * math.pi / 180 * 1e3, 2),
                    "short_period_Hz": wc.num(w / (2 * math.pi), 2) if Ma < 0 else None,
                    "time_to_double_s": wc.num(math.log(2) / w, 3) if Ma >= 0 else None,
                    "thrust_in_level_flight_N": wc.num(T, 2), "motor_moment_10pct_split_mNm": wc.num(0.10 * T * r * 1e3, 2),
                    "deg_of_incidence_a_10pct_split_holds": wc.num(0.10 * T * r / max(abs(Ma) * math.pi / 180, 1e-9), 2)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=wc.DEFAULT, choices=wc.BUILDS)
    a = ap.parse_args(argv)
    c = fam.config(a.build)
    veh = fam.vehicle(a.build)
    p = em.resolve(fam.cad_parameters(a.build))
    h = hand(veh, p)
    cfd = from_cfd(veh, a.build)
    use = cfd or h
    bat = mk.part(c.battery)
    doc = {"aircraft": a.build, "config": veh.label,
           "mass": {"kg": wc.num(veh.mass), "x_cg_mm": wc.num(veh.x_cg * 1e3, 1), "x_cg_over_length": wc.num(veh.x_cg / veh.geo["length"], 3),
                    "inertia_kg_m2": [wc.num(x, 7) for x in veh.inertia], "cg_shift_per_10mm_of_battery_mm": wc.num(10.0 * bat.mass_g / 1e3 / veh.mass, 2)},
           "by_hand": {k: v for k, v in h.items() if not k.startswith("_")},
           "from_cfd": ({k: v for k, v in cfd.items() if not k.startswith("_")} if cfd else "NOT RUN (labs/aero/sims/wasp_run.sh --cfd)"),
           "dynamics_from": "cfd" if cfd else "hand", "dynamics": dynamics(veh, use["_m"]),
           "note": "no propellers in either estimate (their normal force is forward of the CG: destabilising); hand slopes +-30 %"}
    path = wc.OUT / f"{wc.slug(a.build)}_stability.yaml"
    wc.save(path, doc)
    print(f"{a.build}: CG {doc['mass']['x_cg_mm']} mm ({doc['mass']['x_cg_over_length']} of the length)")
    print(f"  by hand:  neutral point {h['x_np_mm']} mm, static margin {h['static_margin_lengths']} lengths, {'stable' if h['stable'] else 'UNSTABLE'}")
    print(f"  from CFD: {('neutral point %s mm, static margin %s lengths, %s' % (cfd['x_np_mm'], cfd['static_margin_lengths'], 'stable' if cfd['stable'] else 'UNSTABLE')) if cfd else 'NOT RUN'}")
    for d in doc["dynamics"]:
        print(f"  {d}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
