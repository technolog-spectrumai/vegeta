"""The ARIGO family's propulsion: real propeller and ducted-fan maps instead of the hand thrust curve (notebook 39).

Each configuration's propulsor comes from the lab's propulsor library (``propulsor_maps``: Boreas BEMT maps over
airspeed x rpm, notebooks 25/25b, with the tractor / pusher installation factors), read at the battery's electrical
limit (``ogma.propulsion.Unit``: full throttle = the rpm where the electrical power limit times the drive efficiency is
reached, or the rpm limit):

- ARIGO-0 (tractor) and ARIGO-1 (pusher): one 10-inch two-blade propeller (the CAD's 254 mm), its pitch chosen here;
- ARIGO-2: two 9-inch two-blade propellers (the library's smallest; the CAD draws 8 in — a 9-inch disc needs the
  nacelles 13 mm further out), tractor installation;
- ARIGO-20: two 70 mm single-stage fans (catalogue duct quality), pitch and nozzle chosen here.

The choice (``select``): among the library entries that fit, the one needing least electrical power at the cruise
thrust, provided its full thrust is at least ``STATIC_TW`` of the weight at rest and covers the drag at
``V_DASH``. Electrical limits (HAND, from the packs): one motor 800 W (3S2P P42A: 90 A x 10.8 V, derated), ARIGO-2
2 x 400 W, ARIGO-20 2 x 1200 W (6S 4000 mAh at 30C: 2660 W).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

POWER_W = {"single": 800.0, "twin": 400.0, "edf": 1200.0}       # HAND: electrical limit per unit
STATIC_TW = 0.5                                                  # full static thrust / weight, at least (hand launch)
V_CRUISE = {"single": 18.0, "twin": 18.0, "edf": 30.0}
V_DASH = {"single": 28.0, "twin": 28.0, "edf": 45.0}


def kind_of(propulsion):
    return {"twin": "twin", "edf": "edf"}.get(propulsion, "single")


def layout_of(propulsion):
    return {"pusher": "pusher"}.get(propulsion, "tractor")


def _candidates(kind):
    import propulsor_maps as pm
    lib = pm.load()
    if kind == "edf":
        return [e for e in lib["entries"] if e["kind"] == "edf" and e["diameter_mm"] == 70.0 and e.get("stages") == 1
                and e.get("quality") == "catalogue"]
    d = 9.0 if kind == "twin" else 10.0
    return [e for e in lib["entries"] if e["kind"] == "propeller" and e["diameter_in"] == d and e.get("blades") == 2]


@lru_cache(maxsize=None)
def _unit(entry_id, power_w, layout):
    import propulsor_maps as pm
    e = pm.load()["by_id"][entry_id]
    return pm.unit(e, power_w, layout=layout if e["kind"] == "propeller" else None)


@lru_cache(maxsize=None)
def select(propulsion: str, mass_kg: float, cd0: float, S: float, power_w: float | None = None,
           v_dash: float | None = None) -> dict:
    """The propulsor for a configuration: entry id, layout, units, power per unit, and why. ``power_w`` (per unit)
    and ``v_dash`` override the class values (a product's smaller or bigger motor)."""
    k = kind_of(propulsion)
    n = 1 if k == "single" else 2
    lay = layout_of(propulsion)
    W = mass_kg * 9.80665
    Vc, Vd = V_CRUISE[k], (v_dash or V_DASH[k])
    PW = power_w or POWER_W[k]
    q = lambda V: 0.5 * 1.225 * V * V
    drag = lambda V: q(V) * S * (cd0 + 0.08 * (W / (q(V) * S)) ** 2)        # HAND polar for the choice only
    best = None
    for e in _candidates(k):
        u = _unit(e["id"], PW, lay)
        T0, Td = n * u.full(0.0)[0], n * u.full(Vd)[0]
        if T0 < STATIC_TW * W or Td < drag(Vd):
            continue
        Pc = n * u.electrical_power(Vc, drag(Vc) / n)
        if best is None or Pc < best[0]:
            best = (Pc, e["id"], T0, Td)
    if best is None:
        raise ValueError(f"no library propulsor meets the requirements for {propulsion}")
    Pc, eid, T0, Td = best
    return {"entry": eid, "layout": lay if k != "edf" else "fan", "units": n, "power_w_per_unit": PW,
            "static_thrust_N": round(T0, 1), "thrust_at_dash_N": round(Td, 1), "dash_speed_m_s": Vd,
            "cruise_power_W": round(Pc, 1), "cruise_speed_m_s": Vc,
            "why": f"least cruise power among the library's {'70 mm fans' if k == 'edf' else ('9' if k == 'twin' else '10') + '-inch 2-blade propellers'} "
                   f"with static thrust >= {STATIC_TW} W and thrust >= drag at {Vd} m/s"}


def thrust_table(sel: dict, V=np.linspace(0.0, 70.0, 36)) -> dict:
    """Full-throttle net thrust over airspeed (all units), for the flight models (``arigo1_model.aero``)."""
    u = _unit(sel["entry"], sel["power_w_per_unit"], "pusher" if sel["layout"] == "pusher" else "tractor")
    T = [max(sel["units"] * u.full(v)[0], 0.0) for v in V]
    return {"table_V": [float(v) for v in V], "table_T": [float(t) for t in T], "T0_N": float(T[0]),
            "V0_m_s": float(next((v for v, t in zip(V, T) if v > 0 and t <= 0.0), V[-1])), "propulsor": sel}


def electrical_power(sel: dict, T_total: float, V: float) -> float:
    """Battery power [W] (propulsion only) for a total net thrust at V; inf if beyond full throttle."""
    u = _unit(sel["entry"], sel["power_w_per_unit"], "pusher" if sel["layout"] == "pusher" else "tractor")
    if T_total <= 0:
        return 0.0
    return sel["units"] * u.electrical_power(V, T_total / sel["units"])
