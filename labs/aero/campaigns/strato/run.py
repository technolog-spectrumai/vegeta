"""Strato campaign: every Stage V8 study, its plots, and the feasibility report — reproducible from this one script.

    python labs/aero/campaigns/strato/run.py [--out DIR] [--only STAGE ...] [--quick] [--force]

Stages (each writes ``<out>/<stage>.json`` and its plots; a stage whose JSON exists is not re-run unless ``--force``):
atmosphere, wind, family (optimised 15–120 m ships, summer and year-round, and the demonstrator's best altitude),
pv (GaAs vs CIGS), detail (day/night cycles, endurance, ascent of the reference ship), sweeps (payload, altitude),
sensitivity (the estimates' effect on the payload), pareto, seasons (availability by month and latitude),
constraints (structure and thermal by month), cad (STEP and pictures of the optimised ships), report.
``--quick`` shrinks every search (for the tests): its numbers are not the study's. Seeds are fixed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import strato_catalogue as cat  # noqa: E402
import strato_design as sd  # noqa: E402
import strato_feasibility as sf  # noqa: E402
import strato_sim as ss  # noqa: E402
import strato_study as sst  # noqa: E402
import strato_wind as sw  # noqa: E402
from ogma import atmosphere as oa  # noqa: E402
from ogma import stratosphere as st  # noqa: E402

# the dataviz reference palette (light surface), categorical slots in fixed order; text in ink, never series colours
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb", "#b9b8b2"
LENGTHS = (15.0, 30.0, 50.0, 80.0, 100.0, 120.0)
REFERENCE = 80.0
DEMO = sst.Scenario("demonstrator", (7,), min_availability=0.5, min_gas_life=30.0, min_payload=1.0)


def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                         "axes.prop_cycle": matplotlib.cycler(color=SERIES)})


def save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def jdump(obj, path: Path):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return str(o)
    path.write_text(json.dumps(clean(obj), indent=1, default=default))


def clean(o):
    """JSON without NaN/inf (written as null) and without the simulation objects."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k not in ("configuration", "ship", "cycle")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


# ----------------------------------------------------------------------------------------------------------- stages
def stage_atmosphere(out: Path, quick: bool) -> dict:
    z = np.linspace(0.0, 30000.0, 301)
    s = [st.state(v) for v in z]
    tro = z <= 11000.0
    fig, ax = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    for a, (key, label) in zip(ax, (("temperature", "temperature [K]"), ("pressure", "pressure [kPa]"),
                                    ("density", "density [kg/m³]"), ("gravity", "gravity [m/s²]"))):
        v = np.array([getattr(x, key) for x in s]) / (1000.0 if key == "pressure" else 1.0)
        a.plot(v, z / 1000, color=SERIES[0], label="og.stratosphere (USSA 1976)")
        if key in ("temperature", "pressure", "density"):
            isa = [getattr(oa.isa(float(h)), key) / (1000.0 if key == "pressure" else 1.0) for h in z]
            a.plot(isa, z / 1000, color=SERIES[1], linestyle="--", linewidth=1.5, label="og.atmosphere (troposphere only)")
        a.axhspan(18, 22, color=GRID, alpha=0.6, linewidth=0)
        a.set_xlabel(label)
    ax[0].set_ylabel("altitude [km]")
    ax[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Standard atmosphere to 30 km (the HAPS band 18–22 km shaded)", color=INK, fontsize=11, x=0.01, ha="left")
    table = [st.state(v).as_dict() for v in (0.0, 11000.0, 18000.0, 20000.0, 22000.0)]
    return {"plots": [save(fig, out, "atmosphere.png")], "table": table,
            "isa_error_at_20km": float(oa.isa(20000.0).density / st.density(20000.0) - 1)}


def stage_wind(out: Path, quick: bool) -> dict:
    z = np.linspace(5000.0, 28000.0, 231)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, mo in enumerate((1, 4, 7, 10)):
        p = sw.profile(z, mo, 50.0)
        ax[0].plot(p["u"], z / 1000, color=SERIES[i], label=sw.MONTHS[mo - 1])
        ax[0].fill_betweenx(z / 1000, p["u"] - p["sigma"], p["u"] + p["sigma"], color=SERIES[i], alpha=0.08, linewidth=0)
    ax[0].axvline(0, color=MUTED, linewidth=1)
    ax[0].axhspan(18, 22, color=GRID, alpha=0.6, linewidth=0)
    ax[0].set(xlabel="mean zonal wind u [m/s] (east +), band ±σ", ylabel="altitude [km]", title="Wind profile at 50°N (estimate)")
    ax[0].legend(loc="upper right")
    months = np.arange(1, 13)
    rows = {}
    for i, lat in enumerate((40.0, 50.0, 60.0)):
        p90 = [sw.speed_percentile(20000.0, int(m), lat, 90, 4000) for m in months]
        rows[lat] = p90
        ax[1].plot(months, p90, marker="o", markersize=5, color=SERIES[i], label=f"{lat:.0f}°N")
    ax[1].set(xticks=months, xticklabels=[m[0] for m in sw.MONTHS], ylabel="daily wind speed, 90th percentile [m/s]",
              title="Wind at 20 km by month (estimate)")
    ax[1].legend()
    best = {sw.MONTHS[m - 1]: sw.best_altitude(m, 50.0) for m in (1, 4, 7, 10)}
    return {"plots": [save(fig, out, "wind.png")], "p90_20km": {str(k): v for k, v in rows.items()},
            "best_altitude_50N": best, "note": sw.CLIMATOLOGY.note}


def _opt(length, sc, quick, **kw):
    t = time.time()
    r = sst.optimize(length, sc, maxiter=3 if quick else 15, popsize=4 if quick else 8, **kw)
    r["seconds"] = time.time() - t
    print(f"  {sc.name:>12} {length:5.0f} m  meets={r['meets']}  payload={r['best'].get('payload', float('nan')):8.1f} kg"
          f"  ({r['evaluations']} evaluations, {r['seconds']:.0f} s)", flush=True)
    return r


def stage_family(out: Path, quick: bool) -> dict:
    lengths = (15.0, 50.0, 80.0) if quick else LENGTHS
    res = {"summer": {}, "year": {}}
    for L in lengths:
        res["summer"][str(L)] = _opt(L, sst.SUMMER, quick)
        res["year"][str(L)] = _opt(L, sst.YEAR, quick)
    res["demonstrator"] = _opt(15.0, DEMO, quick, objective="altitude", bounds={"altitude": (6000.0, 22000.0)})
    rows = []
    for scen in ("summer", "year"):
        for L, r in res[scen].items():
            rows.append({"scenario": scen, "meets": r["meets"], **sst.summary_row(r["best"])})
    res["table"] = rows
    res["plots"] = _family_plots(res, out)
    return res


def _family_plots(res, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for i, scen in enumerate(("summer", "year")):
        Ls = [float(L) for L in res[scen]]
        pay = [r["best"]["payload"] if r["meets"] else np.nan for r in res[scen].values()]
        miss = [L for L, r in zip(Ls, res[scen].values()) if not r["meets"]]
        ax[0].plot(Ls, pay, marker="o", markersize=6, color=SERIES[i], label=f"{scen}: meets the scenario")
        if miss:
            ax[0].scatter(miss, [0.0] * len(miss), marker="x", s=60, color=SERIES[i], label=f"{scen}: no feasible design")
    ax[0].set(xlabel="hull length [m]", ylabel="best payload [kg]", title="Payload of the optimised ships")
    ax[0].legend(fontsize=8)
    groups = ("envelope", "structure", "energy", "propulsion", "systems", "margin", "payload")
    summer = res["summer"]
    Ls = list(summer)
    names = [f"{float(L):g} m" for L in Ls]
    lift = np.array([summer[L]["best"].get("net_lift", np.nan) for L in Ls])
    bottom = np.zeros(len(Ls))
    for i, g in enumerate(groups):
        v = np.array([summer[L]["best"].get("masses", {}).get(g, 0.0) for L in Ls]) / lift * 100
        ax[1].bar(names, v, bottom=bottom, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, label=g)
        bottom += v
    ax[1].axhline(100, color=INK, linewidth=1.2, linestyle="--", label="net lift at float = 100 %")
    for k, (n_, top) in enumerate(zip(names, bottom)):
        ax[1].text(k, top + 3, f"{top:.0f} %", ha="center", color=INK2, fontsize=8)
    ax[1].set(ylabel="mass / net lift at float [%]", title="Where the lift goes (summer optimum, or least-bad design)")
    ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return [save(fig, out, "family.png")]


def _reference(out: Path) -> sd.Design:
    fam = json.loads((out / "family.json").read_text())
    r = fam["summer"].get(str(REFERENCE)) or list(fam["summer"].values())[-1]
    return sd.Design(**r["design"])


def stage_pv(out: Path, quick: bool) -> dict:
    res = {}
    for pv in ("gaas", "cigs"):
        for name, sc in (("summer", sst.SUMMER), ("year", sst.YEAR)):
            res[f"{pv}/{name}"] = _opt(REFERENCE, replace(sc, pv=pv), quick)
    return res


def stage_detail(out: Path, quick: bool) -> dict:
    d = _reference(out)
    res, plots = {}, []
    fig, ax = plt.subplots(2, 2, figsize=(15, 7))
    for i, mo in enumerate((7, 12)):
        r = sf.evaluate(d, sd.Mission(month=mo), detail=True)
        c, sh = r["configuration"], r["ship"]
        cyc = ss.day_cycle(c, min(r["wind_p90"], sh.v_max), ship=sh)
        t = cyc["t"] / 3600
        a = ax[0, i]
        a.plot(t, cyc["solar"] / 1e3, color=SERIES[3], label="array output")
        a.plot(t, (cyc["load"] + cyc["prop"]) / 1e3, color=SERIES[0], label="loads + propulsion")
        a.set(xlabel="hours from 00:00 UTC, 15th", ylabel="power [kW]", title=f"{sw.MONTHS[mo - 1]}: power at the 90th-percentile wind ({r['wind_p90']:.1f} m/s)")
        a.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        b = ax[1, i]
        b.plot(t, cyc["soc"] / sh.usable_wh * 100, color=SERIES[2], label="battery [% usable]")
        b.plot(t, cyc["superheat"], color=SERIES[1], label="gas superheat [K]")
        b.plot(t, cyc["dp"] / 10, color=SERIES[6], label="superpressure [daPa]")
        b.axhline(sh.reserve_wh / sh.usable_wh * 100, color=MUTED, linewidth=1, linestyle=":")
        b.set(xlabel="hours from 00:00 UTC, 15th", title=f"{sw.MONTHS[mo - 1]}: battery, gas and envelope")
        b.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        res[sw.MONTHS[mo - 1]] = {"verdict": r, "held": cyc["held"], "soc_min_pct": float(cyc["soc"].min() / sh.usable_wh * 100),
                                  "superheat_max": cyc["superheat_max"], "dp_max": float(cyc["dp"].max()), "dp_min": float(cyc["dp"].min()),
                                  "solar_kwh": cyc["energy_solar_wh"] / 1e3, "loads_kwh": cyc["energy_load_wh"] / 1e3}
    plots.append(save(fig, out, "day_cycle.png"))

    days = 5 if quick else 30
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for i, gas in enumerate(("sealed", "vented")):
        for j, mo in enumerate((7, 1)):
            dd = replace(d, gas_system=gas)
            r = sf.evaluate(dd, sd.Mission(month=mo), detail=True)
            if not r.get("configuration"):
                continue
            e = ss.endurance(r["configuration"], days=days, dt=1800.0, seed=j)
            key = f"{gas}/{sw.MONTHS[mo - 1]}"
            res[f"endurance/{key}"] = {k: e[k] for k in ("on_station", "lost_at_s", "gas_lost_kg", "vented_kg", "max_dist_km",
                                                         "min_altitude", "ballast_left")} | {"feasible": r["feasible"], "problems": r["problems"],
                                                                                              "days_simulated": float(e["t"][-1] / 86400)}
            tday = e["t"] / 86400
            ax[0, i].plot(tday, e["z"] / 1000, color=SERIES[j], label=f"{sw.MONTHS[mo - 1]}")
            ax[1, i].plot(tday, e["dist"] / 1000, color=SERIES[j], label=f"{sw.MONTHS[mo - 1]}")
        ax[0, i].set(title=f"{gas}: float altitude", ylabel="altitude [km]", xlabel="days")
        ax[1, i].set(title=f"{gas}: distance from the station", ylabel="km", xlabel="days")
        ax[1, i].axhline(sd.Mission().station_radius / 1000, color=MUTED, linestyle=":", linewidth=1)
        ax[0, i].legend(fontsize=8)
    plots.append(save(fig, out, "endurance.png"))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for i, gas in enumerate(("sealed", "vented")):
        r = sf.evaluate(replace(d, gas_system=gas), sd.Mission(month=7), detail=True)
        a = ss.ascent(r["configuration"], dt=4.0)
        res[f"ascent/{gas}"] = {k: a[k] for k in ("float_altitude", "max_rate", "time_to_99pct_s", "vented_kg", "dp_top", "drift_km")}
        ax[0].plot(a["t"] / 3600, a["z"] / 1000, color=SERIES[i], label=gas)
        ax[1].plot(a["t"] / 3600, a["w"], color=SERIES[i], label=gas)
        ax[2].plot(a["t"] / 3600, a["fill"] * 100, color=SERIES[i], label=gas)
    ax[0].set(xlabel="hours", ylabel="altitude [km]", title="Ascent (propellers off), July")
    ax[1].set(xlabel="hours", ylabel="vertical speed [m/s]", title="Ascent rate")
    ax[2].set(xlabel="hours", ylabel="gas volume [% of hull]", title="Helium fill (the rest is ballonet air)")
    for a_ in ax:
        a_.legend(fontsize=8)
    plots.append(save(fig, out, "ascent.png"))
    res["design"] = asdict(d)
    res["plots"] = plots
    return res


def stage_sweeps(out: Path, quick: bool) -> dict:
    d = _reference(out)
    n = 5 if quick else 13
    m = sd.Mission(month=7)
    pay = sst.sweep(d, "payload_mass", np.linspace(0.0, 2.0 * max(d.payload_mass, 50.0), n), m)
    alt = sst.sweep(d, "altitude", np.linspace(17000.0, 23000.0, n), m)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, rows, key, xl, xs in ((ax[0], pay, "payload", "payload [kg]", 1.0), (ax[1], alt, "altitude", "float altitude [km]", 1e-3)):
        x = np.array([r["value"] for r in rows]) * xs
        a.plot(x, [r["surplus_lift"] for r in rows], color=SERIES[0], marker="o", markersize=5, label="surplus lift [kg]")
        a.axhline(0, color=MUTED, linewidth=1)
        a.set(xlabel=xl, ylabel="surplus lift [kg]")
        b = a.twinx()
        b.plot(x, [r["availability"] * 100 for r in rows], color=SERIES[1], marker="s", markersize=5, label="availability [% of days]")
        b.set_ylim(0, 105)
        b.set_ylabel("station-keeping availability, July [% of days]", color=INK2)
        b.grid(False)
        a.legend(loc="lower left", fontsize=8)
        b.legend(loc="lower right", fontsize=8)
    ax[0].set_title("Payload sensitivity (reference ship, July)")
    ax[1].set_title("Altitude sensitivity (reference ship, July)")
    rows = lambda rs: [{"value": r["value"], "surplus_lift": r["surplus_lift"], "availability": r["availability"],  # noqa: E731
                        "feasible": r["feasible"], "wind_p90": r["wind_p90"], "problems": r["problems"]} for r in rs]
    return {"payload": rows(pay), "altitude": rows(alt), "design": asdict(d), "plots": [save(fig, out, "sweeps.png")],
            "note": "two y-axes on one plot only to share the swept x; each axis is labelled with its own measure"}


def _patched(name: str, factor: float):
    """A context that scales one catalogue estimate by ``factor`` (and puts it back)."""
    import contextlib
    import strato_thermal as sth

    @contextlib.contextmanager
    def ctx():
        saved = (dict(cat.LAMINATES), dict(cat.PVS), dict(cat.BATTERIES), cat.GAS, sth.H_INTERNAL, sw.CLIMATOLOGY,
                 cat.PAYLOAD_POWER_PER_KG)
        try:
            if name == "laminate areal mass":
                cat.LAMINATES.update({k: replace(v, areal_mass=v.areal_mass * factor) for k, v in saved[0].items()})
            elif name == "PV efficiency":
                cat.PVS.update({k: replace(v, efficiency=v.efficiency * factor) for k, v in saved[1].items()})
            elif name == "battery specific energy":
                cat.BATTERIES.update({k: replace(v, specific_energy=v.specific_energy * factor) for k, v in saved[2].items()})
            elif name == "helium leak rate":
                cat.GAS = replace(saved[3], leak_per_day=saved[3].leak_per_day * factor)
            elif name == "envelope-gas heat transfer":
                sth.H_INTERNAL = saved[4] * factor
            elif name == "wind variability":
                c = saved[5]
                sw.CLIMATOLOGY = replace(c, sigma_winter=c.sigma_winter * factor, sigma_summer=c.sigma_summer * factor)
                sw.profile.__defaults__ = (sw.CLIMATOLOGY,)
            elif name == "payload power":
                cat.PAYLOAD_POWER_PER_KG = saved[6] * factor
            yield
        finally:
            cat.LAMINATES.clear(); cat.LAMINATES.update(saved[0])  # noqa: E702
            cat.PVS.clear(); cat.PVS.update(saved[1])  # noqa: E702
            cat.BATTERIES.clear(); cat.BATTERIES.update(saved[2])  # noqa: E702
            cat.GAS, sth.H_INTERNAL, sw.CLIMATOLOGY, cat.PAYLOAD_POWER_PER_KG = saved[3], saved[4], saved[5], saved[6]
            sw.profile.__defaults__ = (saved[5],)
    return ctx()


SENSITIVITY = (("laminate areal mass", 0.7, 1.3), ("PV efficiency", 0.85, 1.15), ("battery specific energy", 0.8, 1.2),
               ("helium leak rate", 0.5, 2.0), ("envelope-gas heat transfer", 0.5, 2.0), ("wind variability", 0.7, 1.3),
               ("payload power", 0.5, 2.0))


def stage_sensitivity(out: Path, quick: bool) -> dict:
    """The estimates' effect on the reference ship's best payload (summer scenario), one at a time: each case re-runs
    the optimiser warm-started from the reference optimum (so battery, propulsion, fineness, altitude… re-size)."""
    fam = json.loads((out / "family.json").read_text())
    x0 = np.array(fam["summer"][str(REFERENCE)]["x"])
    kw = dict(maxiter=2 if quick else 8, popsize=3 if quick else 6, x0=x0, seed=2)

    def best(tag):
        r = sst.optimize(REFERENCE, sst.SUMMER, **kw)
        v = r["best"]["payload"] if r["meets"] else float("nan")
        print(f"  {tag:40s} payload {v:8.1f} kg", flush=True)
        return v, r

    base, _ = best("base")
    rows = []
    for name, lo, hi in (SENSITIVITY[:2] if quick else SENSITIVITY):
        vals = {}
        for fct in (lo, hi):
            with _patched(name, fct):
                vals[fct], r = best(f"{name} ×{fct:g}")
        rows.append({"estimate": name, "low_factor": lo, "high_factor": hi, "payload_low": vals[lo], "payload_high": vals[hi]})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = sorted(rows, key=lambda r: abs(np.nan_to_num(r["payload_high"]) - np.nan_to_num(r["payload_low"])))
    for i, r in enumerate(order):
        lo_v = np.nan_to_num(r["payload_low"], nan=0.0) - base
        hi_v = np.nan_to_num(r["payload_high"], nan=0.0) - base
        ax.barh(i, lo_v, color=SERIES[0], height=0.6, label="low factor" if i == 0 else None)
        ax.barh(i, hi_v, color=SERIES[1], height=0.6, label="high factor" if i == 0 else None)
    ax.set_yticks(range(len(order)), [f"{r['estimate']} (×{r['low_factor']:g} / ×{r['high_factor']:g})" for r in order])
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set(xlabel=f"change of the maximum payload [kg] (base {base:.0f} kg)", title=f"Which estimate matters: the {REFERENCE:g} m ship, summer")
    ax.legend(fontsize=8, loc="lower right")
    return {"base_payload": base, "rows": rows, "plots": [save(fig, out, "sensitivity.png")],
            "method": "re-optimised per case (differential evolution warm-started from the reference optimum)"}


def stage_pareto(out: Path, quick: bool) -> dict:
    n = 64 if quick else 512
    pts = sst.sample(REFERENCE, sst.SUMMER, n, seed=1)
    keys = ("payload", "system_mass", "availability", "drag_p90", "energy_at_p90_wh", "endurance_days", "meets", "feasible", "x")
    pts = [{k: p.get(k) for k in keys} for p in pts]
    feas = [p for p in pts if p["feasible"]]
    fronts = {}
    pairs = ((("payload", -1), ("system_mass", 1)), (("payload", -1), ("availability", -1)),
             (("payload", -1), ("drag_p90", 1)), (("payload", -1), ("energy_at_p90_wh", 1)))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for a, ((k1, s1), (k2, s2)) in zip(axs.ravel(), pairs):
        P = np.array([[s1 * p[k1], s2 * p[k2]] for p in feas]) if feas else np.zeros((0, 2))
        idx = sst.pareto(P) if len(P) else np.array([], int)
        fronts[f"{k1}|{k2}"] = [feas[i] for i in idx]
        a.scatter([p[k1] for p in pts if not p["feasible"]], [p[k2] if p[k2] is not None and math.isfinite(p[k2]) else np.nan for p in pts if not p["feasible"]],
                  s=10, color=GRID, label="rejected by the gate")
        a.scatter([p[k1] for p in feas], [p[k2] for p in feas], s=12, color=MUTED, label="feasible")
        f = sorted((feas[i] for i in idx), key=lambda p: p[k1])
        a.plot([p[k1] for p in f], [p[k2] for p in f], color=SERIES[0], marker="o", markersize=6, label="Pareto front")
        a.set(xlabel=k1.replace("_", " "), ylabel=k2.replace("_", " "))
        a.legend(fontsize=7)
    fig.suptitle(f"{REFERENCE:g} m ship, summer: {len(pts)} Sobol designs, {len(feas)} feasible", color=INK, x=0.01, ha="left")
    return {"n": len(pts), "n_feasible": len(feas), "n_meets": sum(bool(p["meets"]) for p in pts), "fronts": fronts,
            "plots": [save(fig, out, "pareto.png")]}


def stage_seasons(out: Path, quick: bool) -> dict:
    fam = json.loads((out / "family.json").read_text())
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    lats = (40.0, 50.0, 60.0) if quick else (40.0, 45.0, 50.0, 55.0, 60.0)
    months = (1, 7) if quick else tuple(range(1, 13))
    for a, scen in zip(axs, ("summer", "year")):
        r0 = fam[scen].get(str(REFERENCE)) or list(fam[scen].values())[-1]
        d = sd.Design(**r0["design"])
        hot = max(sf.evaluate(d, sd.Mission(month=7), passes=1, n_days=200)["superheat_day"], 0.0)
        grid = np.full((len(lats), len(months)), np.nan)
        for i, lat in enumerate(lats):
            for j, mo in enumerate(months):
                r = sf.evaluate(d, sd.Mission(lat=lat, month=mo), passes=1, n_days=400, superheat_day_floor=hot)
                grid[i, j] = r["availability"] * 100 if r["feasible"] else np.nan
        im = a.imshow(grid, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=100)
        for i in range(len(lats)):
            for j in range(len(months)):
                v = grid[i, j]
                a.text(j, i, "—" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=7,
                       color="white" if (not np.isnan(v) and v > 60) else INK)
        a.set_xticks(range(len(months)), [sw.MONTHS[m - 1][0] for m in months])
        a.set_yticks(range(len(lats)), [f"{v:.0f}°N" for v in lats])
        a.set_title(f"{scen} optimum ({REFERENCE:g} m): days on station [%]")
        a.grid(False)
        res[scen] = {"design": asdict(d), "lats": lats, "months": months, "availability_pct": grid.tolist()}
    fig.colorbar(im, ax=axs, shrink=0.8, label="% of days")
    fig.savefig(out / "seasons.png", dpi=130)
    plt.close(fig)
    res["plots"] = ["seasons.png"]
    res["note"] = "— = rejected by the gate in that month (does not close its energy, or does not float)"
    return res


def stage_constraints(out: Path, quick: bool) -> dict:
    fam = json.loads((out / "family.json").read_text())
    r0 = fam["year"].get(str(REFERENCE)) or list(fam["year"].values())[-1]
    d = sd.Design(**r0["design"])
    months = (1, 4, 7, 10) if quick else tuple(range(1, 13))
    rows = []
    for mo in months:
        r = sf.evaluate(d, sd.Mission(month=mo), passes=2, n_days=200)
        lam = cat.LAMINATES[d.laminate]
        rows.append({"month": mo, "superheat_day": r["superheat_day"], "dp_max": r["dp_max"], "dp_min": r["dp_min"],
                     "hoop_line_load": r["hoop_line_load"], "allowable": r["plies"] * lam.strength / cat.ENVELOPE_SAFETY_FACTOR,
                     "plies": r["plies"], "solar_kwh": r["energy_solar_wh"] / 1e3, "loads_kwh": r["energy_at_p90_wh"] / 1e3})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = [sw.MONTHS[r["month"] - 1][0] for r in rows]
    ax[0].plot(x, [r["hoop_line_load"] / 1e3 for r in rows], color=SERIES[0], marker="o", markersize=5, label="peak membrane line load")
    ax[0].plot(x, [r["allowable"] / 1e3 for r in rows], color=SERIES[7], linestyle="--", label=f"allowable (strength / {cat.ENVELOPE_SAFETY_FACTOR:g})")
    ax[0].set(ylabel="kN/m", title="Envelope: load vs allowable, by month")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, [r["superheat_day"] for r in rows], color=SERIES[1], marker="o", markersize=5, label="noon gas superheat [K]")
    ax[1].plot(x, [r["dp_max"] / 10 for r in rows], color=SERIES[6], marker="s", markersize=5, label="noon superpressure [daPa]")
    ax[1].set(title="Thermal: superheat and superpressure, by month (estimate)")
    ax[1].legend(fontsize=8)
    return {"design": asdict(d), "rows": rows, "plots": [save(fig, out, "constraints.png")]}


def stage_cad(out: Path, quick: bool) -> dict:
    """STEP files and pictures of the optimised ships (the summer optimum at each length that has one)."""
    import strato
    sys.path.insert(0, str(HERE))
    from cad import views
    fam = json.loads((out / "family.json").read_text())
    folder = out / "cad"
    folder.mkdir(exist_ok=True)
    made = []
    for L, r in fam["summer"].items():
        if not r["meets"] and float(L) != 15.0:
            continue
        d = sd.Design(**r["design"])
        ship = sd.ship_of(d)
        over = {"length": d.length, "diameter": ship.hull.diameter, "solar_half_angle_deg": d.solar_half_angle_deg,
                "fin_span": ship.layout.fin_span, "fin_root_chord": ship.layout.fin_root_chord}
        tag = f"{float(L):g}m"
        g = strato.StratoAirship().generate(**over)
        g.export_step(folder / f"strato_opt_{tag}.step")
        if not quick:
            views(strato.parts(**over), folder / f"views_opt_{tag}.png", f"Strato {tag} optimised (summer)")
        made.append(tag)
    return {"ships": made, "plots": [f"cad/views_opt_{t}.png" for t in made] if not quick else []}


STAGES = {"atmosphere": stage_atmosphere, "wind": stage_wind, "family": stage_family, "pv": stage_pv,
          "detail": stage_detail, "sweeps": stage_sweeps, "sensitivity": stage_sensitivity, "pareto": stage_pareto,
          "seasons": stage_seasons, "constraints": stage_constraints, "cad": stage_cad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--only", nargs="+", choices=list(STAGES) + ["report"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    for name, fn in STAGES.items():
        if a.only and name not in a.only:
            continue
        path = a.out / f"{name}.json"
        if path.exists() and not a.force:
            print(f"{name}: cached ({path})")
            continue
        t = time.time()
        print(f"{name} …", flush=True)
        res = fn(a.out, a.quick)
        res["seconds"] = time.time() - t
        res["quick"] = a.quick
        jdump(res, path)
        print(f"{name}: {res['seconds']:.0f} s", flush=True)
    if not a.only or "report" in a.only:
        import report
        p = report.write(a.out)
        print(f"report: {p}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
