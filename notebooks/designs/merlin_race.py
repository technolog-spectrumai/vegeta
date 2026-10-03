"""MERLIN's propulsor race (notebook 27, branch ``dev_rave``): can a reasonable ducted fan beat the propellers on the same
battery?

Nothing is solved here: every propeller and fan comes from notebook 25's propulsor library (``propulsor_maps``,
``data/propulsor_maps.json``: maps over airspeed x rpm, the installation on MERLIN, the unit masses), and the race only
interpolates in it.

Two races, both from the launch stand, still air, the same 6S 8000 mAh pack and its 15 % reserve for everyone:

- **5 km and back** (``round_trip``): climb to 150 m, dash out, a 60-degree-bank turn over the point, dash back; the
  clock stops over the launch site (the landing that follows is the same for all and is left out);
- **10 km** (``merlin_flight.race``): the reach — launch to overhead a fire 10 km away — keeping the energy to sample for
  three minutes and fly home; the 10 km out-and-back is reported as well.

What may change is the library's design space (``propulsor_maps.EDF_SPACE`` / ``PROP_SPACE``):

- the **ducted fan**: diameter 70–120 mm, blade pitch 1.4–2.2 diameters, nozzle exit 65–90 % of the fan annulus, the duct's
  build quality (the hobby catalogue fan's fitted loss, 0.34, or a well-made fan, 0.20), **one to three stages**;
- the **propellers**: tractor or pusher, 9–12 inch, pitch 0.6–1.2 diameters;
- **both**: the electrical power asked of the pack — 1900, 2700 or 3500 W (about 11, 15 and 20 C) — with motors and
  controllers that grow with it (``propulsor_maps.unit_mass_kg``).
"""
from __future__ import annotations

import math

import merlin_flight as mf
import propulsor_maps as pm

POWERS_W = (1900.0, 2700.0, 3500.0)
RACES = ("5 km and back [s]", "10 km reach [s]", "10 km and back [s]")


def candidates(lib):
    """(entry, layout) for every map: a fan once, a propeller as a tractor and as a pusher."""
    for e in lib["entries"]:
        if "error" in e:
            continue
        if e["kind"] == "edf":
            yield e, "edf"
        else:
            yield e, "tractor"
            yield e, "pusher"


def round_trip(af: mf.Airframe, unit: mf.Unit, distance_m: float, mission: mf.Mission = mf.Mission(), *, bank_deg=60.0,
               rho=mf.RHO) -> dict:
    """Out to a point ``distance_m`` away and back over the launch site: the launch and climb as ``merlin_flight.race``,
    a full-power acceleration, the dash out, a turn at ``bank_deg`` (radius ``V^2 / (g tan bank)``, load factor
    ``1 / cos bank``, at full power), the dash back. The dash speed is the highest (up to the top speed) the energy
    allows with the reserve kept. Returns the time [s] (launch to over the launch site), the dash speed, what limited it
    and the energy."""
    pf = mf.performance(af, unit, rho)
    W = af.mass_kg * mf.G
    v_c = pf["v_climb"]
    T_c, P_c, _ = unit.full(v_c)
    roc = (T_c - af.drag(v_c, rho)) * v_c / W
    if roc <= 0:
        return {"reachable": False, "why": "cannot climb"}
    t_climb = mission.sampling_agl_m / roc
    x_climb, e_climb = v_c * t_climb, P_c * t_climb / 3600
    t_l, x_l, e_l = mf._accelerate(af, unit, mission.launch_speed, v_c, rho)
    x_dash = 2 * distance_m - x_l - x_climb               # out from the end of the climb, and all the way back
    budget = mission.battery_wh * mission.usable_fraction * (1 - mission.reserve_fraction)
    tb = math.tan(math.radians(bank_deg))

    def lap(v):
        t_a, x_a, e_a = mf._accelerate(af, unit, v_c, v, rho, max_distance=x_dash)
        x = max(x_dash - x_a, 0.0)
        P = unit.electrical_power(v, float(af.drag(v, rho)))
        t_turn = math.pi * v / (mf.G * tb)
        P_turn = min(unit.electrical_power(v, float(af.drag(v, rho, 1.0 / math.cos(math.radians(bank_deg))))), unit.max_electrical_w)
        return t_l + t_climb + t_a + x / v + t_turn, e_l + e_climb + e_a + (P * x / v + P_turn * t_turn) / 3600

    v_hi = 0.995 * pf["v_top"]
    v_lo = min(pf["v_range"], v_hi)
    if lap(v_lo)[1] > budget:
        return {"reachable": False, "why": "not enough energy"}
    if lap(v_hi)[1] <= budget:
        v, limit = v_hi, "thrust"
    else:
        lo, hi = v_lo, v_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if lap(mid)[1] <= budget else (lo, mid)
        v, limit = lo, "energy"
    t, e = lap(v)
    return {"reachable": True, "time_s": t, "dash_speed": v, "top_speed": pf["v_top"], "limit": limit, "energy_wh": e,
            "budget_wh": budget, "turn_s": math.pi * v / (mf.G * tb)}


def study(lib, powers=POWERS_W, mission: mf.Mission = mf.Mission()):
    """Every map of the library, as every layout it can fly in, at every power, through the races: one row each (pandas)."""
    import pandas as pd
    rows = []
    for e, layout in candidates(lib):
        for P in powers:
            unit = pm.unit(e, P, layout=None if layout == "edf" else layout)
            af = pm.merlin_airframe(layout, P, e)
            pf = mf.performance(af, unit)
            r5 = round_trip(af, unit, 5000.0, mission)
            r10 = mf.race(af, unit, 10000.0, mission)
            r10b = round_trip(af, unit, 10000.0, mission)
            rows.append({"kind": layout, "config": unit.name, "id": e["id"], "power [W]": P, "mass [kg]": af.mass_kg,
                         "static thrust [N]": unit.full(0.0)[0], "top speed [m/s]": pf["v_top"], "energy [Wh/km]": pf["wh_per_km_best"],
                         "5 km and back [s]": r5["time_s"] if r5["reachable"] else math.nan,
                         "5 km and back: dash [m/s]": r5.get("dash_speed", math.nan), "5 km and back: limit": r5.get("limit", r5.get("why")),
                         "10 km reach [s]": r10["time_to_fire_s"] if r10["reachable"] else math.nan,
                         "10 km reach: limit": r10.get("limit", r10.get("why")),
                         "10 km and back [s]": r10b["time_s"] if r10b["reachable"] else math.nan,
                         "stages": e.get("stages", math.nan), "diameter": e.get("diameter_mm", e.get("diameter_in")),
                         "pitch ratio": e["pitch_ratio"], "exit area ratio": e.get("exit_area_ratio", math.nan),
                         "quality": e.get("quality", "")})
    return pd.DataFrame(rows)


def winners(table, race, by=("kind", "power [W]"), bound=False):
    """The fastest configuration of each kind (and power) in ``race``; the lossless bound fans only with ``bound``."""
    t = table.dropna(subset=[race])
    if not bound:
        t = t[t["quality"] != "lossless"]
    return t.loc[t.groupby(list(by))[race].idxmin()].set_index(list(by)).sort_index()


__all__ = ["POWERS_W", "RACES", "candidates", "round_trip", "study", "winners"]
