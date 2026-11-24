"""ASTOR against GAVILAN-X, on GAVILAN's own scorecard (2026-11-24).

``gavilan_flight`` already compares hunters as ``Machine`` objects with one set of manoeuvre models (turn, brake,
accelerate, reverse, climb, endurance, a pass at a bandit). The two ASTOR aircraft are entered into it here as fixed
wings, from the flight model's own numbers (``astor_v_flight``): PROTOHAWK (ASTOR-TWIN, HAWK 2) and GOSHAWK. Their
polars take the lift slope, drag and aspect-ratio term of the flight model and the maximum lift WITH FLAPS (both have
them; GAVILAN's panels have none). Their thrust is the flight model's table; electrical power is thrust x speed over a
75 % propeller and an 85 % drive, never less than the share of full power that the thrust is of full thrust.

Both ASTOR aircraft fly on aerodynamics carried over from another aircraft's CFD, and their thrust at low speed is a
hand estimate: see their ``aero_source``. GAVILAN-X is as ``gavilan_flight`` has it.
"""
from __future__ import annotations

import math

import numpy as np

import astor_v_flight as VF
import gavilan_flight as GFm


def astor_machine(name: str, label: str, power_w: float, roll_deg_s: float, payload_kg: float, gun: str, length_m: float) -> GFm.Machine:
    C = VF.get(name)
    F = VF.with_flap(C, 1.0)
    ar = C["b"] ** 2 / C["S"]
    pol = GFm.Polar(cl_alpha=C["CLa"], cd0=C["CD0"], k=C["k"], cl_max=F["CL_MAX"], cd90=GFm.viterna_cd90(ar),
                    source="astor_v_flight: " + str(C["aero_source"])[:80] + "; maximum lift with every flap down")
    tv, tt = np.asarray(C["thrust"]["table_V"]), np.asarray(C["thrust"]["table_T"])
    t_max = lambda v: float(np.interp(max(v, 0.0), tv, tt))

    def power(T, v):
        return max(max(T, 0.0) * max(v, 0.0) / (0.75 * 0.85), power_w * max(T, 0.0) / max(t_max(v), 1e-6)) + 15.0
    return GFm.Machine(label, "wing", C["mass"], C["b"], length_m, C["S"], pol, 0.0, t_max, power, VF.SAS[name]["g_max"],
                       C["mass_budget"]["pack"]["Wh_nominal"], roll_rate=math.radians(roll_deg_s), fixed_w=15.0, payload_kg=payload_kg, gun=gun,
                       source="astor_v_flight (" + name + ")", notes={"aero": C["aero_source"]})


def machines() -> dict:
    import astor_twin as W
    import goshawk as GH
    return {"astor_twin": astor_machine("twin", "ASTOR-TWIN (PROTOHAWK): net launcher, three nets", 8000.0, 450.0, W.one_launcher(3)["launcher_kg"], "fixed", 1.27),
            "goshawk": astor_machine("goshawk", "GOSHAWK: two legs", 2 * GH.MOTOR_W, 240.0, 0.11, "talons", 1.32),
            "gavilan_x": GFm.gavilan_x(), "gavilan_x_net": GFm.gavilan_x_net(), "gavilan": GFm.gavilan_chosen()}


KEEP = ("aircraft", "mass [kg]", "payload [kg]", "span [m]", "hovers", "gun", "static thrust / weight", "wing stall [m/s]", "min speed [m/s]", "top speed [m/s]",
        "climb [m/s]", "radius inst @10 [m]", "radius inst @15 [m]", "radius inst @25 [m]", "rate inst @10 [deg/s]", "rate inst @15 [deg/s]", "rate inst @25 [deg/s]",
        "n inst @25", "stops", "brake distance [m]", "brake ends at [m/s]", "to 25 m/s [s]", "reverse at 15 m/s [s]", "reverse space [m]", "reverse how",
        "hover [min]", "cruise [min]", "cruise at [m/s]", "pass speed, hovering bandit [m/s]", "firing window [s]", "re-attack [s]", "source")


def compare() -> dict:
    ms = machines()
    rows = GFm.scorecard(ms)
    def clean(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (float, np.floating, int, np.integer)):
            return None if not math.isfinite(float(v)) else round(float(v), 2)
        return v
    return {r["key"]: {k: clean(r.get(k)) for k in KEEP} for r in rows}


if __name__ == "__main__":
    import yaml
    import arigo1_model as am
    out = compare()
    (am.DATA / "goshawk").mkdir(parents=True, exist_ok=True)
    (am.DATA / "goshawk" / "astor_vs_gavilan_x.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=160))
    keys = list(out)
    for k in KEEP[:-1]:
        print(f"{k[:34]:34s} " + " ".join(f"{str(out[m][k])[:17]:>17s}" for m in keys))
