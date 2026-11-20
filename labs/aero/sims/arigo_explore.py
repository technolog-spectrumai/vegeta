"""ARIGO product exploration: from ARIGO-1H, how far can each product (BUTEO, ARPELLA, ASTOR, SPARFELL) get?

    labs/aero/sims/arigo_explore.sh                     # all four products, 1 process
    labs/aero/sims/arigo_explore.sh -j 8                # 8 processes
    labs/aero/sims/arigo_explore.sh buteo sparfell -j 8  # some products
    labs/aero/sims/arigo_explore.sh --fresh             # forget the cache

For each product it sweeps the design space in ``designs/arigo_products.SPACE`` around the product's base spec:
airframe scale, span, root chord, pack, motor rating, kits (perch, charging perch, rail, parachute, two-piece or
quick-release wing, lite avionics, rugged skin). Every spec gets a full performance card (``arigo_perf.card``: point
mass, MuJoCo responses, the three PN chases, launches, the mission metrics, landing) and is scored on its product's
mission (``arigo_charts.MISSIONS``: 0 at the client's threshold, 100 at the objective).

Airframes other than ARIGO-1H and ARIGO-20 have no CFD: the sweep runs on the lattice model, with ARIGO-1H on the
lattice as each product's baseline (like for like). The winners go to CFD next (arigo_cfd.sh).

No CFD here. About 13 s per spec on one core; about 90 specs. Restartable: finished specs are kept in
``output/explore/arigo_explore_cache.yaml`` (dropped when any ``designs/arigo*.py`` changes, or with ``--fresh``).
Each run writes ONE file ``output/runs/arigo_explore_<time>_<host>.yaml`` with every card, the scores, and per product
the baseline, the best spec and what it gained, plus charts next to it (``*_explore_scores.png``,
``*_explore_gains.png``, ``*_explore_mass.png``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs"))
sys.path.append(str(HERE))                                        # after designs: sims/arigo1_mujoco.py is a runner

import yaml                                                          # noqa: E402

import arigo1_resume as rs                                           # noqa: E402

rs.GLOBS = ["arigo*.py"]
OUT = HERE / "output"


def _work(job):
    """One spec's card, in a worker process (the rows are rebuilt here: the airframe registry is per process)."""
    product, key = job
    import arigo_perf as P
    import arigo_products as AP
    for pk, k, name, Cf, sf, man, V in AP.explore_rows([product]):
        if k == key:
            t = time.time()
            try:
                card = json.loads(json.dumps(P.card(name, Cf(), sf, man, V), default=float))
            except Exception as exc:                                  # noqa: BLE001  one bad spec must not stop the sweep
                card = {"aircraft": name, "error": f"{type(exc).__name__}: {exc}"[:300]}
            card["explore"] = {"product": pk, "seconds": round(time.time() - t, 1)}
            return key, card
    return key, {"error": f"spec {key} not found"}


def summarise(cards: dict) -> dict:
    import arigo_charts as AC
    import arigo_products as AP
    ok = {k: c for k, c in cards.items() if "summary" in c}
    MK = AC.mission_kpi(ok)
    PS = AP.product_scores(ok, MK)
    out = {}
    for pk, pr in PS.items():
        rows = pr["candidates"]
        if not rows:
            continue
        mission = AP.PRODUCTS[pk]["mission"]
        bkey = AP.spec_key(pk, {**AP.PRODUCTS[pk]["base"], "lattice": True})[0]     # the sweep's first spec
        base = next((r for r in rows if r["key"] == bkey), rows[-1])
        best = rows[0]
        items = AC.MISSIONS[mission][1]
        gains = []
        for key, lab, thr, obj in items:
            b, w = MK[base["key"]]["per_metric"].get(key), MK[best["key"]]["per_metric"].get(key)
            gains.append({"metric": lab, "key": key, "baseline_value": ok[base["key"]]["summary"].get(key),
                          "best_value": ok[best["key"]]["summary"].get(key), "baseline_score": b, "best_score": w})
        out[pk] = {"title": pr["title"], "role": pr["role"], "priorities": pr["priorities"],
                   "baseline": {"name": base["name"], "score": base["score"]}, "best": {"name": best["name"], "score": best["score"]},
                   "gain": round(best["score"] - base["score"], 1), "top": rows[:8], "metrics": gains,
                   "specs_scored": len(rows)}
    return {"mission_kpi": MK, "products": PS, "summary": out}


def charts(path: Path, doc: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import arigo_charts as AC
    INK, INK2, GRID, SURFACE = AC.INK, AC.INK2, AC.GRID, AC.SURFACE
    PCOL = {"buteo": "#2a78d6", "arpella": "#1baf7a", "astor": "#eda100", "sparfell": "#eb6834"}
    S = doc["summary"]
    stem = path.with_suffix("")
    out = []
    keys = [k for k in ("buteo", "arpella", "astor", "sparfell") if k in S]
    if not keys:
        return out
    # 1. the top specs per product, the baseline marked
    fig, axs = plt.subplots(len(keys), 1, figsize=(12, 3.0 * len(keys) + 0.8), facecolor=SURFACE, squeeze=False)
    for ax, pk in zip(axs[:, 0], keys):
        rows = doc["products"][pk]["candidates"][:12]
        base = S[pk]["baseline"]["name"]
        if base not in [r["name"] for r in rows]:
            rows = rows[:11] + [r for r in doc["products"][pk]["candidates"] if r["name"] == base]
        y = np.arange(len(rows))
        cols = [INK2 if r["name"] == base else PCOL[pk] for r in rows]
        ax.barh(y, [r["score"] for r in rows], color=cols, height=0.7, edgecolor=SURFACE, linewidth=2)
        for yi, r in zip(y, rows):
            ax.text(r["score"], yi, f" {r['score']:.0f}  ({r['mass_kg']:.2f} kg)", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in rows], fontsize=7); ax.invert_yaxis()
        ax.set_xlim(0, 112)
        AC._style(ax, f"{S[pk]['title']} — {S[pk]['role']}: {S[pk]['priorities']}  (grey: ARIGO-1H on the lattice)", "score")
    fig.suptitle("ARIGO exploration: the best specs per product (absolute mission score)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.append(stem.parent / f"{stem.name}_scores.png"); fig.savefig(out[-1], dpi=110, facecolor=SURFACE); plt.close(fig)
    # 2. where the gain comes from: per metric, the baseline's score and the best spec's
    fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5.2), facecolor=SURFACE, squeeze=False)
    for ax, pk in zip(axs[0], keys):
        m = S[pk]["metrics"]
        y = np.arange(len(m))
        ax.barh(y - 0.2, [x["baseline_score"] or 0 for x in m], height=0.38, color=INK2, label="ARIGO-1H (lattice)")
        ax.barh(y + 0.2, [x["best_score"] or 0 for x in m], height=0.38, color=PCOL[pk], label="best spec")
        ax.set_yticks(y); ax.set_yticklabels([x["metric"] for x in m], fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 105)
        AC._style(ax, f"{S[pk]['title']}: {S[pk]['baseline']['score']:.0f} -> {S[pk]['best']['score']:.0f}\n{S[pk]['best']['name']}", "score")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Where the gain comes from: each metric's score, baseline against the best spec", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.append(stem.parent / f"{stem.name}_gains.png"); fig.savefig(out[-1], dpi=110, facecolor=SURFACE); plt.close(fig)
    # 3. score against mass, every spec
    fig, axs = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.0), facecolor=SURFACE, squeeze=False)
    for ax, pk in zip(axs[0], keys):
        rows = doc["products"][pk]["candidates"]
        ax.scatter([r["mass_kg"] for r in rows], [r["score"] for r in rows], s=22, color=PCOL[pk], alpha=0.8)
        b = next(r for r in rows if r["name"] == S[pk]["baseline"]["name"])
        ax.scatter([b["mass_kg"]], [b["score"]], s=60, color=INK2, marker="D", label="ARIGO-1H (lattice)")
        AC._style(ax, f"{S[pk]['title']}: score against mass")
        ax.set_xlabel("take-off mass (kg)", fontsize=8, color=INK2); ax.set_ylabel("score", fontsize=8, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.append(stem.parent / f"{stem.name}_mass.png"); fig.savefig(out[-1], dpi=110, facecolor=SURFACE); plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="arigo_explore.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("products", nargs="*", help="bbuteo arpella astor sparfell (default: all)")
    ap.add_argument("-j", "--cpu", type=int, default=1, help="worker processes (default 1)")
    ap.add_argument("--fresh", action="store_true", help="forget the cache: score every spec again")
    ap.add_argument("--list", action="store_true", help="list the specs and exit")
    a = ap.parse_args(argv)
    import arigo_products as AP
    want = [p.lower() for p in a.products] or list(AP.PRODUCTS)
    bad = [p for p in want if p not in AP.PRODUCTS]
    if bad:
        ap.error(f"unknown product(s) {bad}; choose from {list(AP.PRODUCTS)}")
    rows = list(AP.explore_rows(want))
    if a.list:
        for pk, k, name, *_ in rows:
            print(f"{pk:9s} {k:60s} {name}")
        print(f"{len(rows)} specs")
        return 0
    (OUT / "explore").mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cache_p = OUT / "explore" / "arigo_explore_cache.yaml"
    cache = rs.load(cache_p, a.fresh) or {"what": "ARIGO exploration cache", "cards": {}}
    jobs = [(pk, k) for pk, k, *_ in rows if k not in cache["cards"] or "error" in cache["cards"][k]]
    print(f"{len(rows)} specs for {', '.join(want)}: {len(rows) - len(jobs)} cached, {len(jobs)} to score on {a.cpu} process(es)")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_p = OUT / "runs" / f"arigo_explore_{stamp}_{socket.gethostname()}.yaml"
    meta = {"phase": "explore", "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
            "products": want, "cpu": a.cpu, "specs": len(rows), "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()}
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(rows), initial=len(rows) - len(jobs), desc="explore", unit="spec")
    except ImportError:
        bar = None
    t0 = time.time()

    def done(key, card):
        cache["cards"][key] = card
        rs.save(cache_p, cache)
        if bar:
            bar.set_postfix_str(card.get("aircraft", key)[:40])
            bar.update(1)
        else:
            print(f"  {rs.eta(len(cache['cards']), len(rows), t0)} {card.get('aircraft', key)}")
    if a.cpu > 1 and jobs:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.cpu, maxtasksperchild=8) as pool:
            for key, card in pool.imap_unordered(_work, jobs):
                done(key, card)
    else:
        for job in jobs:
            done(*_work(job))
    if bar:
        bar.close()
    cards = {k: cache["cards"][k] for _, k, *_ in rows if k in cache["cards"]}
    doc = {"run": {**meta, "finished": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "errors": sum(1 for c in cards.values() if "error" in c)},
           **summarise(cards), "results": {"arigo_explore": {"cards": cards}}}
    run_p.write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=str)), sort_keys=False, width=140))
    for p in charts(run_p, doc):
        print("->", p)
    print("->", run_p)
    for pk, s in doc["summary"].items():
        print(f"  {s['title']:9s} baseline {s['baseline']['score']:5.1f} -> best {s['best']['score']:5.1f}: {s['best']['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
