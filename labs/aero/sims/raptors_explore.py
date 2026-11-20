"""RAPTORS exploration: from ARIGO-1H, how far can each product line (BUTEO, ARPELLA, ASTOR, SPARFELL) get?

    labs/aero/sims/raptors_explore.sh                       # all four lines, 1 process
    labs/aero/sims/raptors_explore.sh -j 16                 # 16 processes
    labs/aero/sims/raptors_explore.sh buteo sparfell -j 8   # some lines
    labs/aero/sims/raptors_explore.sh --samples 48 -j 16    # more Monte Carlo draws (default 24 launches, 6 landings)
    labs/aero/sims/raptors_explore.sh --no-robust           # skip the launch-success and operating-wind models (fast)
    labs/aero/sims/raptors_explore.sh --base-only           # each line's starting spec only
    labs/aero/sims/raptors_explore.sh --list                # the specs, then exit
    labs/aero/sims/raptors_explore.sh --fresh               # forget the cache

For each product it sweeps the design space in ``designs/arigo_products.SPACE`` around the product's base spec:
airframe scale, span, root chord, pack, motor rating, kits (perch, charging perch, rail, parachute, two-piece or
quick-release wing, lite avionics, rugged skin). Every spec gets a full performance card (``arigo_perf.card``: point
mass, MuJoCo responses, the three PN chases, launches, the mission metrics, landing) and is scored on its product's
mission (``arigo_charts.MISSIONS``: 0 at the client's threshold, 100 at the objective; the gates in ``GATES``).

Then the robustness models (``designs/raptors_robust.py``): **launch success** by Monte Carlo over wind, gusts and
the launch's own dispersion (KPIs B4, S8, P12), and the **operating wind** (B6, A10): the strongest steady wind (gusts
1.5x) in which launch and recovery both succeed 95 % of the time. Fixed stations take the wind from any direction;
the ``turning_station`` kit turns them into it; hand launches go into the wind.

Airframes other than ARIGO-1H and ARIGO-20 have no CFD: the sweep runs on the lattice model, with ARIGO-1H on the
lattice as each product's baseline (like for like). The winners go to CFD next (arigo_cfd.sh).

No CFD here. About 13 s per spec for the card and ~60 s for the robustness models on one core; ~100 specs: about
2 h on one core, ~10 min on 16. Restartable: finished specs are kept in
``output/explore/raptors_explore_cache.yaml`` (dropped when any ``designs/arigo*.py`` changes, or with ``--fresh``).
Each run writes ONE file ``output/runs/raptors_explore_<time>_<host>.yaml`` with every card, the scores, and per product
the baseline, the best spec and what it gained, plus charts next to it (``*_explore_scores.png``,
``*_explore_gains.png``, ``*_explore_mass.png``).
"""
from __future__ import annotations

import os

# one maths thread per worker: numpy's BLAS otherwise starts one per core in EVERY worker (12 workers x 16 threads
# froze a 16-thread laptop). Set before numpy loads; the shell launcher sets them too.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import json
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
WORKER_GB = 1.0          # measured: ~0.81 GB per worker (CadQuery/OCC ~0.48 GB on import, MuJoCo + one card), + margin
KEEP_GB = 3.0            # left for the desktop and the parent process
KEEP_CORES = 2


def mem_available_gb() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return None


def safe_workers(requested: int, force: bool = False) -> tuple[int, str]:
    """The number of worker processes this machine can carry: <= cores - KEEP_CORES and <= (free RAM - KEEP_GB) /
    WORKER_GB. ``force`` keeps the request."""
    cores = os.cpu_count() or 2
    cap_cpu = max(1, cores - KEEP_CORES)
    mem = mem_available_gb()
    cap_mem = max(1, int((mem - KEEP_GB) // WORKER_GB)) if mem is not None else cap_cpu
    cap = min(cap_cpu, cap_mem)
    why = f"{cores} cores, {mem:.1f} GB free" if mem is not None else f"{cores} cores"
    if force or requested <= cap:
        return requested, why
    return cap, why


def _init_worker():
    """Workers run at low priority: the desktop stays responsive however many there are."""
    try:
        os.nice(10)
    except OSError:
        pass


def _work(job):
    """One spec's card and robustness, in a worker process (the rows are rebuilt here: the airframe registry is per
    process)."""
    product, key, robust, samples = job
    import arigo_perf as P
    import arigo_products as AP
    for pk, k, name, Cf, sf, man, V in AP.explore_rows([product]):
        if k == key:
            t = time.time()
            try:
                card = json.loads(json.dumps(P.card(name, Cf(), sf, man, V), default=float))
            except Exception as exc:                                  # noqa: BLE001  one bad spec must not stop the sweep
                card = {"aircraft": name, "error": f"{type(exc).__name__}: {exc}"[:300]}
            if robust and "summary" in card:
                try:
                    import raptors_robust as RB
                    C = Cf()
                    eq = C.get("equipment") or {}
                    how = str(card["summary"].get("launch") or "")
                    mode = "perch" if how.startswith("perch") else ("rail" if how.startswith(("site", "rail")) else "hand")
                    rb = RB.robustness(sf, C, mode, eq.get("recovery", "belly"), n_launch=samples, n_land=max(4, samples // 4),
                                       turning_station=bool(eq.get("turning_station")),
                                       perch_push=eq.get("perch_push_m_s") or 4.0)
                    card["robust"] = rb
                    card["summary"].update(RB.summary(rb))
                except Exception as exc:                          # noqa: BLE001
                    card["robust"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            card["explore"] = {"product": pk, "seconds": round(time.time() - t, 1), "robust": robust, "samples": samples}
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
    fig.suptitle("RAPTORS exploration: the best specs per line (absolute mission score)", x=0.01, ha="left", fontsize=12, color=INK)
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
    ap = argparse.ArgumentParser(prog="raptors_explore.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("products", nargs="*", help="bbuteo arpella astor sparfell (default: all)")
    ap.add_argument("-j", "--cpu", type=int, default=1,
                    help="worker processes (default 1). Capped to the machine: cores - 2 and (free RAM - 3 GB) / 1 GB each")
    ap.add_argument("--force-cpu", action="store_true", help="use -j as given, even past the memory/core cap")
    ap.add_argument("--fresh", action="store_true", help="forget the cache: score every spec again")
    ap.add_argument("--list", action="store_true", help="list the specs and exit")
    ap.add_argument("--samples", type=int, default=24, help="Monte Carlo launches per envelope (landings: a quarter, min 4)")
    ap.add_argument("--no-robust", action="store_true", help="skip the launch-success and operating-wind models")
    ap.add_argument("--base-only", action="store_true", help="each line's starting spec only (plus ARIGO-20 for ASTOR)")
    a = ap.parse_args(argv)
    import arigo_products as AP
    want = [p.lower() for p in a.products] or list(AP.PRODUCTS)
    bad = [p for p in want if p not in AP.PRODUCTS]
    if bad:
        ap.error(f"unknown product(s) {bad}; choose from {list(AP.PRODUCTS)}")
    rows = list(AP.explore_rows(want))
    if a.base_only:
        base_keys = {AP.spec_key(pk, {**AP.PRODUCTS[pk]["base"], "lattice": True})[0] for pk in want}
        base_keys |= {AP.spec_key("astor", {"frame": "arigo-20", "kits": ("rail", "parachute"), "pack": "6S 6000 mAh LiPo"})[0]}
        rows = [r for r in rows if r[1] in base_keys]
    if a.list:
        for pk, k, name, *_ in rows:
            print(f"{pk:9s} {k:60s} {name}")
        print(f"{len(rows)} specs")
        return 0
    (OUT / "explore").mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cache_p = OUT / "explore" / "raptors_explore_cache.yaml"
    cache = rs.load(cache_p, a.fresh) or {"what": "RAPTORS exploration cache", "cards": {}}
    robust = not a.no_robust

    def stale(c):                                                 # an error, or scored with other robustness settings
        ex = c.get("explore") or {}
        return "error" in c or (robust and (not ex.get("robust") or ex.get("samples") != a.samples or "error" in (c.get("robust") or {})))
    jobs = [(pk, k, robust, a.samples) for pk, k, *_ in rows if k not in cache["cards"] or stale(cache["cards"][k])]
    n_req = a.cpu
    a.cpu, why = safe_workers(a.cpu, a.force_cpu)
    if a.cpu < n_req:
        print(f"-j {n_req} lowered to {a.cpu} ({why}; ~{WORKER_GB:.0f} GB per worker). --force-cpu to insist.")
    print(f"{len(rows)} specs for {', '.join(want)}: {len(rows) - len(jobs)} cached, {len(jobs)} to score on {a.cpu} process(es)")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_p = OUT / "runs" / f"raptors_explore_{stamp}_{socket.gethostname()}.yaml"
    meta = {"phase": "explore", "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
            "products": want, "cpu": a.cpu, "specs": len(rows), "robust": robust, "samples": a.samples, "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()}
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
        with mp.get_context("fork").Pool(a.cpu, initializer=_init_worker, maxtasksperchild=8) as pool:
            for key, card in pool.imap_unordered(_work, jobs, chunksize=1):
                done(key, card)
    else:
        for job in jobs:
            done(*_work(job))
    if bar:
        bar.close()
    cards = {k: cache["cards"][k] for _, k, *_ in rows if k in cache["cards"]}
    doc = {"run": {**meta, "finished": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "errors": sum(1 for c in cards.values() if "error" in c)},
           **summarise(cards), "results": {"raptors_explore": {"cards": cards}}}
    run_p.write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=str)), sort_keys=False, width=140))
    for p in charts(run_p, doc):
        print("->", p)
    print("->", run_p)
    for pk, s in doc["summary"].items():
        print(f"  {s['title']:9s} baseline {s['baseline']['score']:5.1f} -> best {s['best']['score']:5.1f}: {s['best']['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
