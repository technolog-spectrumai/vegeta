"""Strato-Micro study: the cheapest maritime imaging airship at a few hundred metres, tethered vs free-flying
(Stage V8.19). Writes ``results/micro.json``, ``micro.png``, and the CAD of both sized ships (STEP, parts, views).

    python labs/aero/campaigns/strato/micro.py [--out DIR] [--quick]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs"))
sys.path.insert(0, str(HERE))

import strato  # noqa: E402
import strato_micro as mi  # noqa: E402
from run import INK2, MUTED, SERIES, jdump, plt, save, style  # noqa: E402

WINDS = np.arange(6.0, 20.01, 1.0)


def variant_overrides(r: dict) -> dict:
    """The Dedalus parameters of a sized Strato-Micro ship."""
    d = r["design"]
    ship = mi._ship(r["length"], mi.MicroDesign(**d))
    tethered = d["concept"] == "tethered"
    return {"length": r["length"], "diameter": ship.hull.diameter, "fin_span": ship.layout.fin_span,
            "fin_root_chord": ship.layout.fin_root_chord, "with_solar": False, "with_structure": False,
            "with_propellers": not tethered, "with_tether": tethered}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    out = a.out
    (out / "cad").mkdir(parents=True, exist_ok=True)
    style()
    m = mi.Mission()
    best = mi.cheapest(m)
    res = {"name": "Strato-Micro", "mission": asdict(m), "sea_wind": asdict(mi.SEA),
           "catalogue_notes": {k: v.note for k, v in mi.CAT.envelopes.items()}, "rows": best["rows"], "best": best["best"]}

    # cost and size against the design wind (how much weather it still works in), both concepts, TPU-nylon
    sweep = {}
    for concept, fin in (("tethered", 1.4), ("free", 0.7)):
        rows = []
        for w in (WINDS[::3] if a.quick else WINDS):
            r = mi.size(mi.MicroDesign(concept=concept, fin_scale=fin), replace(m, design_wind=float(w)))
            if r.get("feasible"):
                r["first_year_cost"] = r["build_cost"] + r["ground_cost"] + r["helium_cost_per_year"]
            rows.append({"design_wind": float(w), **{k: r.get(k) for k in ("length", "diameter", "volume", "system_mass",
                         "build_cost", "first_year_cost", "operable_fraction", "propulsion_w", "battery_wh", "feasible")}})
        sweep[concept] = rows
    res["sweep"] = sweep
    # altitude: 100, 200, 300 m for the cheapest concept
    res["altitude"] = [{"altitude": z, **{k: v for k, v in mi.size(mi.MicroDesign(**best["best"]["design"]),
                                                                          replace(m, altitude=z)).items()
                                          if k in ("length", "volume", "system_mass", "build_cost", "feasible")}}
                       for z in (100.0, 200.0, 300.0)]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (concept, rows) in enumerate(sweep.items()):
        w = [r["design_wind"] for r in rows if r["feasible"]]
        ax[0].plot(w, [r["first_year_cost"] / 1000 for r in rows if r["feasible"]], color=SERIES[i], marker="o", markersize=5, label=concept)
        ax[1].plot(w, [r["length"] for r in rows if r["feasible"]], color=SERIES[i], marker="o", markersize=5, label=concept)
    ax[0].set(xlabel=f"design wind at {m.altitude:g} m [m/s]", ylabel="first-year cost [k€] (estimate)", title="Cost vs the wind it works in")
    ax[1].set(xlabel=f"design wind at {m.altitude:g} m [m/s]", ylabel="hull length [m]", title="Size vs the wind it works in")
    sec = ax[0].secondary_xaxis("top", functions=(lambda v: np.interp(v, WINDS, [mi.SEA.fraction_below(x, m.altitude) * 100 for x in WINDS]),
                                                  lambda p: np.interp(p, [mi.SEA.fraction_below(x, m.altitude) * 100 for x in WINDS], WINDS)))
    sec.set_xlabel("hours it can work [%] (sea wind, estimate)", color=INK2)
    for a_ in ax[:2]:
        a_.legend(fontsize=8)
    b = best["best"]
    items = sorted(b["costs"], key=lambda k: -b["costs"][k])
    ax[2].barh(items, [b["costs"][k] for k in items], color=SERIES[0])
    for j, k in enumerate(items):
        ax[2].text(b["costs"][k], j, f" €{b['costs'][k]:,.0f}  ·  {b['masses'][k]:.2f} kg", va="center", color=INK2, fontsize=8)
    ax[2].invert_yaxis()
    ax[2].grid(axis="y", visible=False)
    ax[2].set(xlabel="parts cost [€] (estimate)", title=f"Cheapest: {b['concept']}, {b['design']['envelope']}, {b['length']:.1f} m")
    ax[2].set_xlim(0, max(b["costs"].values()) * 1.7)
    res["plots"] = [save(fig, out, "micro.png")]

    from cad import views
    for key, r in (("tethered", min((x for x in best["rows"] if x.get("feasible") and x["concept"] == "tethered"), key=lambda x: x["first_year_cost"])),
                   ("free", min((x for x in best["rows"] if x.get("feasible") and x["concept"] == "free"), key=lambda x: x["first_year_cost"]))):
        over = variant_overrides(r)
        tag = f"{key}_{r['length']:.1f}m"
        strato.StratoAirship().generate(**over).export_step(out / "cad" / f"strato_micro_{tag}.step")
        parts = strato.parts(**over)
        (out / "cad" / f"strato_micro_{tag}_parts").mkdir(exist_ok=True)
        for name, g in parts.items():
            g.export_step(out / "cad" / f"strato_micro_{tag}_parts" / f"{name}.step")
        if not a.quick:
            views(parts, out / "cad" / f"views_micro_{tag}.png", f"Strato-Micro, {key}, {r['length']:.1f} m")
            res["plots"].append(f"cad/views_micro_{tag}.png")
        res[f"cad_{key}"] = {"step": f"cad/strato_micro_{tag}.step", "overrides": over}
    res["quick"] = a.quick
    jdump(res, out / "micro.json")
    print(f"micro: best {b['concept']} {b['design']['envelope']} {b['length']:.2f} m, first year €{b['first_year_cost']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
