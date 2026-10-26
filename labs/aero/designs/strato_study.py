"""Strato — studies over the design space: seasons, sweeps, optimisation and Pareto fronts (Stage V8.13).

The design vector is scale-free, so one set of bounds serves a 15 m and an 80 m ship:

    fineness L/D · float altitude [m] · payload as a fraction of the float gross lift · battery [Wh per m³ of hull] ·
    propulsion [W per m² of V^(2/3)] · array half-angle [deg] · fin scale · night superpressure reserve [Pa]

``optimize`` maximises the payload under the feasibility gate (strato_feasibility) and the requirements of a
``Scenario`` (station-keeping availability, gas life) with SciPy's differential evolution — Vegeta has no generic
optimiser (the races' CMA/PSO strategies are bound to autopilot settings), so SciPy is the one dependency added.
Infeasible candidates get a penalty that grows with how far they miss (lift deficit, availability, gas life), so the
search is guided towards feasibility rather than told only "no". ``pareto`` keeps the non-dominated set of any
evaluated population for any choice of objectives. Every run is seeded and reproducible.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from ogma import lifting_gas as lg
from ogma import stratosphere as st

import strato_catalogue as cat
import strato_design as sd
import strato_feasibility as sf

# name, lower, upper
SPACE = (("fineness", 2.5, 6.0), ("altitude", 18000.0, 22000.0), ("payload_fraction", 0.0, 0.25),
         ("battery_per_m3", 0.1, 8.0), ("propulsion_per_m2", 2.0, 80.0), ("solar_half_angle_deg", 25.0, 85.0),
         ("fin_scale", 0.6, 1.0), ("night_superpressure", 0.0, 400.0))
NAMES = tuple(s[0] for s in SPACE)
BOUNDS = [(lo, hi) for _, lo, hi in SPACE]


@dataclass(frozen=True)
class Scenario:
    name: str
    months: tuple                 # months the ship must operate; the first is where the payload is judged
    lat: float = 50.0
    min_availability: float = 0.9 # fraction of days on station, in the worst month of ``months``
    min_gas_life: float = 60.0    # days
    min_payload: float = 0.0      # kg
    pv: str = "gaas"
    laminate: str = "zylon"
    battery: str = "li-ion"
    gas_system: str = "sealed"


SUMMER = Scenario("summer", (7, 6, 8))
YEAR = Scenario("year-round", (12, 3, 6, 9), min_availability=0.5)


def gross_lift_at(d_length: float, fineness: float, altitude: float) -> float:
    """Float net lift per the gas law alone [kg] (no superpressure, no superheat): the scale for the payload fraction."""
    hull = sd.ship_of(sd.Design(length=d_length, fineness=fineness)).hull
    a = st.state(altitude)
    return hull.volume * lg.net_lift_per_volume(a.pressure, a.temperature, gas=cat.GAS.gas, purity=cat.GAS.purity)


def decode(x, length: float, sc: Scenario, fixed: dict | None = None) -> sd.Design:
    """The design of vector ``x``; ``fixed`` overrides Design fields after decoding (e.g. a given payload)."""
    d = _decode(x, length, sc)
    return replace(d, **fixed) if fixed else d


def _decode(x, length: float, sc: Scenario) -> sd.Design:
    v = dict(zip(NAMES, map(float, x)))
    hull = sd.ship_of(sd.Design(length=length, fineness=v["fineness"])).hull
    lift = gross_lift_at(length, v["fineness"], v["altitude"])
    return sd.Design(length=length, fineness=v["fineness"], altitude=v["altitude"],
                     payload_mass=v["payload_fraction"] * lift, battery_energy=v["battery_per_m3"] * hull.volume,
                     propulsion_power=v["propulsion_per_m2"] * hull.volume ** (2 / 3),
                     solar_half_angle_deg=v["solar_half_angle_deg"], fin_scale=v["fin_scale"],
                     night_superpressure=v["night_superpressure"], pv=sc.pv, laminate=sc.laminate, battery=sc.battery,
                     gas_system=sc.gas_system)


def encode(d: sd.Design) -> np.ndarray:
    hull = sd.ship_of(d).hull
    lift = gross_lift_at(d.length, d.fineness, d.altitude)
    return np.array([d.fineness, d.altitude, d.payload_mass / lift, d.battery_energy / hull.volume,
                     d.propulsion_power / hull.volume ** (2 / 3), d.solar_half_angle_deg, d.fin_scale, d.night_superpressure])


def evaluate_season(d: sd.Design, sc: Scenario, n_days: int = 600) -> dict:
    """``d`` over the scenario's months: the envelope sized for the hottest month, availability in the worst.
    Returns the first month's verdict with ``availability`` = the worst month's and the per-month table."""
    first = [sf.evaluate(d, sd.Mission(lat=sc.lat, month=mo), n_days=n_days, passes=1) for mo in sc.months]
    if not all("superheat_day" in r for r in first):
        return first[0]
    hot = max(r["superheat_day"] for r in first)
    rs = [sf.evaluate(d, sd.Mission(lat=sc.lat, month=mo), n_days=n_days, passes=1, superheat_day_floor=hot)
          for mo in sc.months]
    out = dict(rs[0])
    out["months"] = {mo: {"availability": r["availability"], "feasible": r["feasible"], "problems": r["problems"],
                          "energy_solar_wh": r["energy_solar_wh"], "v_energy": r["v_energy"], "superheat_day": r["superheat_day"],
                          "wind_p90": r["wind_p90"]} for mo, r in zip(sc.months, rs)}
    out["availability"] = min(r["availability"] for r in rs)
    out["feasible"] = all(r["feasible"] for r in rs)
    out["problems"] = sorted({p for r in rs for p in r["problems"]})
    out["warnings"] = sorted({w for r in rs for w in r["warnings"]})
    out["system_mass"] = max(r["system_mass"] for r in rs)
    out["surplus_lift"] = min(r["surplus_lift"] for r in rs)
    out["gas_life_days"] = min(r["gas_life_days"] for r in rs)
    out["endurance_days"] = min((r["endurance_days"] if r["endurance_days"] == r["endurance_days"] else 0.0) for r in rs)
    out["scenario"] = sc.name
    return out


def meets(r: dict, sc: Scenario) -> bool:
    return (bool(r.get("feasible")) and r["availability"] >= sc.min_availability and r["gas_life_days"] >= sc.min_gas_life
            and r.get("payload", 0.0) >= sc.min_payload)


def penalty(r: dict, sc: Scenario) -> float:
    """0 for a design that meets the scenario; otherwise a positive measure of how far it misses."""
    if "system_mass" not in r:
        return 1e6
    p = 0.0
    p += max(0.0, -r["surplus_lift"]) / max(r["net_lift"], 1e-9) * 10
    p += max(0.0, sc.min_availability - r["availability"]) * 5
    p += max(0.0, sc.min_gas_life - r["gas_life_days"]) / max(sc.min_gas_life, 1.0)
    p += max(0.0, sc.min_payload - r.get("payload", 0.0)) / max(sc.min_payload, 1.0)
    other = [q for q in r["problems"] if not q.startswith("does not float")]
    return p + 2.0 * len(other)


def _cost(x, length, sc, objective="payload", fixed=None):
    d = decode(x, length, sc, fixed)
    r = evaluate_season(d, sc)
    pen = penalty(r, sc)
    if pen > 0:
        return 1.0 + pen
    if objective == "altitude":
        return -d.altitude / 20000.0
    if objective == "surplus":
        return -r["surplus_lift"] / max(r["net_lift"], 1e-9)
    # the payload as a share of the float lift keeps the costs of different sizes comparable
    return -d.payload_mass / gross_lift_at(length, d.fineness, d.altitude)


def optimize(length: float, sc: Scenario = SUMMER, *, seed: int = 0, maxiter: int = 25, popsize: int = 8,
             workers: int = 4, objective: str = "payload", bounds: dict | None = None, x0=None,
             fixed: dict | None = None) -> dict:
    """The best design of ``length`` [m] that meets ``sc``: the highest payload (``objective="payload"``) or the
    highest float altitude (``"altitude"``); ``bounds`` overrides entries of ``SPACE`` by name. ``result['best']`` is
    the seasonal verdict (``meets`` False when nothing in the bounds met the scenario: then it is the least-bad
    design found, the one that missed by the least). ``x0`` seeds the population with a known design (warm start).
    ``objective="surplus"`` maximises the spare lift (for a ``fixed`` payload: is there any design that carries it?);
    ``fixed`` sets Design fields the search does not touch."""
    from scipy.optimize import differential_evolution
    b = [tuple((bounds or {}).get(name, (lo, hi))) for name, lo, hi in SPACE]
    kw = dict(args=(length, sc, objective, fixed), seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
              updating="deferred", init="sobol", x0=None if x0 is None else np.clip(x0, *np.array(b).T))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = differential_evolution(_cost, b, workers=pool.map, **kw)
    else:
        res = differential_evolution(_cost, b, **kw)
    d = decode(res.x, length, sc, fixed)
    best = evaluate_season(d, sc)
    if objective == "payload" and not fixed and meets(best, sc):
        # polish: the search leaves some surplus lift; raise the payload (all else fixed) while the scenario holds
        mp = max_payload(d, sc, lo=d.payload_mass, tol=max(0.5, 1e-3 * d.payload_mass))
        if mp["payload"] == mp["payload"] and mp["payload"] > d.payload_mass:
            d, best = replace(d, payload_mass=mp["payload"]), mp["result"]
    return {"length": length, "scenario": sc.name, "objective": objective, "x": encode(d).tolist(), "design": asdict(d),
            "best": best, "meets": meets(best, sc), "cost": float(res.fun), "evaluations": int(res.nfev), "seed": seed,
            "bounds": b}


def smallest_length(sc: Scenario, fixed: dict, lo: float = 15.0, hi: float = 60.0, tol: float = 1.0, **kw) -> dict:
    """The shortest hull [m] in [lo, hi] for which the optimiser finds a design that carries ``fixed`` (a given
    payload) and meets ``sc``: bisection on the length, an optimisation (``objective="surplus"``) at each step.
    ``runs`` keeps every step. A length is "possible" only if the search found a design; "impossible" means none was
    found within the bounds and the budget (an optimiser cannot prove that none exists)."""
    runs = {}

    def ok(L):
        r = optimize(L, sc, objective="surplus", fixed=fixed, **kw)
        runs[L] = r
        print(f"  {L:6.2f} m  meets={r['meets']}  surplus={r['best'].get('surplus_lift', float('nan')):8.2f} kg", flush=True)
        return r["meets"]

    if not ok(hi):
        return {"length": float("nan"), "runs": runs}
    if ok(lo):
        return {"length": lo, "runs": runs}
    while hi - lo > tol:
        mid = round(0.5 * (lo + hi), 2)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return {"length": hi, "best": runs[hi], "runs": runs}


def sample(length: float, sc: Scenario, n: int, seed: int = 0, workers: int = 4) -> list[dict]:
    """``n`` designs of ``length`` from a scrambled Sobol sequence over the bounds, each evaluated (Pareto material)."""
    from scipy.stats import qmc
    u = qmc.Sobol(len(BOUNDS), scramble=True, seed=seed).random(n)
    lo, hi = np.array(BOUNDS).T
    xs = lo + u * (hi - lo)
    return _map(_sample_one, [(x, length, sc) for x in xs], workers)


def _sample_one(args):
    x, length, sc = args
    d = decode(x, length, sc)
    r = evaluate_season(d, sc, n_days=400)
    r["x"] = list(map(float, x))
    r["meets"] = meets(r, sc)
    return r


def _map(fn, items, workers):
    if workers <= 1:
        return [fn(i) for i in items]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, items, chunksize=max(1, len(items) // (4 * workers)))


def pareto(points: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of ``points`` (every column minimised)."""
    P = np.asarray(points, float)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        dominated = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if dominated.any():
            keep[i] = False
    return np.flatnonzero(keep)


def sweep(base: sd.Design, field: str, values, m: sd.Mission = sd.Mission()) -> list[dict]:
    """``base`` with ``field`` set to each of ``values``, each evaluated."""
    return [dict(sf.evaluate(replace(base, **{field: v}), m), swept=field, value=v) for v in values]


def max_payload(d: sd.Design, sc: Scenario, lo: float = 0.0, hi: float | None = None, tol: float = 0.5) -> dict:
    """The largest payload [kg] ``d`` can carry and still meet ``sc`` (bisection on the payload: more payload is more
    mass and more payload power, both monotone against meeting the scenario)."""
    hi = hi if hi is not None else gross_lift_at(d.length, d.fineness, d.altitude)
    r_lo = evaluate_season(replace(d, payload_mass=lo), sc)
    if not meets(r_lo, sc):
        return {"payload": float("nan"), "result": r_lo}
    best = r_lo
    while hi - lo > max(tol, 1e-3 * hi):
        mid = 0.5 * (lo + hi)
        r = evaluate_season(replace(d, payload_mass=mid), sc)
        if meets(r, sc):
            lo, best = mid, r
        else:
            hi = mid
    return {"payload": lo, "result": best}


def summary_row(r: dict) -> dict:
    """The columns the reports print for a verdict."""
    keys = ("length", "diameter", "volume", "envelope_area", "solar_area", "payload", "payload_power", "system_mass",
            "net_lift", "surplus_lift", "max_float_altitude", "plies", "dp_min", "dp_max", "superheat_day", "v_max",
            "v_energy", "availability", "wind_p90", "energy_solar_wh", "energy_at_p90_wh", "drag_p90", "cdv",
            "gas_life_days", "endurance_days", "stability_margin", "feasible")
    row = {k: r.get(k) for k in keys}
    d = r.get("design", {})
    row.update({"fineness": d.get("fineness"), "altitude": d.get("altitude"), "battery_wh": d.get("battery_energy"),
                "propulsion_w": d.get("propulsion_power"), "pv": d.get("pv"), "laminate": d.get("laminate"),
                "gas_system": d.get("gas_system"), "problems": "; ".join(r.get("problems", [])),
                "warnings": "; ".join(r.get("warnings", []))})
    return row


def isfinite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)
