"""ASTOR-T's win rate against quadcopters across a grid of design choices (``designs/astor_hunt``).

    labs/aero/sims/astor_win.sh -j 14                  # the 64 designs x 30 sorties (1920 sorties, ~3 h of one core)
    labs/aero/sims/astor_win.sh -j 14 --sorties 60     # tighter win rates (+-6 % instead of +-9 % per design)
    labs/aero/sims/astor_win.sh -j 14 --only-baseline  # the current design alone, 100 sorties
    labs/aero/sims/astor_win.sh --dry-run

The requirement: ASTOR wins about 80 % of its sorties (the quadcopter carried off, or dropped with fouled rotors) and
is itself almost never lost. Every design meets the same quadcopters (the sortie's seed draws the quadcopter), so
designs are compared on equal terms. The six choices, two levels each (``astor_hunt.LEVELS``): motor power 2 / 4 kW;
wing chord x1.0 / x1.2; the catch's reach (the feet / a net frame between them); the line 1.2 / 2.0 m; the pass
speed 1.3 / 1.45 x stall; whether the line lets go when ASTOR is losing.

``-j N``: worker processes (lowered if the machine cannot carry them; ``--force-cpu`` insists). One tqdm bar; finished
sorties are kept, a restart continues (``--fresh`` starts over). Writes ONE file
``output/runs/astor_win_<time>_<host>.yaml``: per design the win rate with its 90 % interval, the share of sorties
ASTOR is lost, the outcomes, the win rate by the quadcopter's behaviour; each choice's main effect; the designs
that reach 80 %.
"""
import itertools
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(DESIGNS))
sys.path.append(str(HERE))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import astor_run as R                                                # noqa: E402

R.GLOBS_ACTIVE = ["arigo*.py", "astor*.py", "raptors*.py"]
FACTORS = ("power_kw", "chord", "reach", "line_m", "pass_speed", "release")


def _job(job):
    import astor_hunt as H
    t = time.time()
    design = dict(zip(FACTORS, job[1:7]))
    try:
        r = H.hunt(design, seed=job[7])
        out = {k: r[k] for k in ("outcome", "win", "astor_lost", "quadcopter", "caught", "t_catch_s", "after_the_catch", "astor")}
        out["passes"] = len(r["passes"])
        out["closest_pass_m"] = min((math.hypot(p["across_m"], p["height_m"]) for p in r["passes"]), default=None)
    except Exception as exc:                                          # noqa: BLE001
        import traceback
        out = {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-800:]}
    out["seconds"] = round(time.time() - t, 1)
    return "|".join(str(x) for x in job), out


def wilson(k, n, z=1.645):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="astor_win.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("-j", "--cpu", type=int, default=4, help="worker processes (default 4)")
    ap.add_argument("--force-cpu", action="store_true", help="use -j exactly as given, without the memory cap")
    ap.add_argument("--sorties", type=int, default=30, help="sorties per design (default 30)")
    ap.add_argument("--only-baseline", action="store_true", help="the current design alone (astor_hunt.DEFAULT_DESIGN), --sorties or 100")
    ap.add_argument("--fresh", action="store_true", help="forget the saved sorties")
    ap.add_argument("--dry-run", action="store_true", help="list the designs and stop")
    a = ap.parse_args(argv)
    import astor_hunt as H
    base = tuple(H.DEFAULT_DESIGN[f] for f in FACTORS)
    designs = [base] if a.only_baseline else list(itertools.product(*(H.LEVELS[f] for f in FACTORS)))
    n = (a.sorties if a.sorties != 30 else 100) if a.only_baseline else a.sorties
    jobs = [("hunt", *dz, seed) for dz in designs for seed in range(n)]
    if a.dry_run:
        for dz in designs[:8]:
            print(dict(zip(FACTORS, dz)))
        print(f"... {len(designs)} designs x {n} sorties = {len(jobs)} sorties (~5.5 s each on one core)")
        return 0
    tally = {"n": 0, "w": 0}

    def note(key, out):
        tally["n"] += 1
        tally["w"] += bool(out.get("win"))
        return f"wins {tally['w']}/{tally['n']}"
    res = R._run_jobs(jobs, a, "astor_win_cache.yaml", "astor win rate", _job, note=note, unit="sortie")
    rows = []
    for dz in designs:
        rs_ = [res["|".join(str(x) for x in ("hunt", *dz, s))] for s in range(n)]
        ok = [r for r in rs_ if "error" not in r and r["outcome"] != "sim_error"]
        k, lost = sum(r["win"] for r in ok), sum(r["astor_lost"] for r in ok)
        by = {}
        for r in ok:
            b = by.setdefault(r["quadcopter"]["behaviour"], [0, 0])
            b[0] += r["win"]; b[1] += 1
        outs = {}
        for r in ok:
            outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
        rows.append({"design": dict(zip(FACTORS, dz)), "sorties": len(ok), "wins": k, "win_rate": round(k / max(len(ok), 1), 3), "win_rate_90pct": wilson(k, len(ok)),
                     "astor_lost_rate": round(lost / max(len(ok), 1), 3), "outcomes": outs, "win_rate_by_behaviour": {b: round(v[0] / v[1], 2) for b, v in by.items()},
                     "caught_rate": round(sum(r["caught"] for r in ok) / max(len(ok), 1), 3), "errors": len(rs_) - len(ok),
                     "astor": ok[0]["astor"] if ok else None})
    rows.sort(key=lambda r: (-r["win_rate"], r["astor_lost_rate"]))
    effects = {}
    if len(designs) > 1:
        for i, f in enumerate(FACTORS):
            lv = {}
            for r in rows:
                lv.setdefault(str(r["design"][f]), []).append((r["win_rate"], r["astor_lost_rate"]))
            effects[f] = {k: {"win_rate": round(sum(x[0] for x in v) / len(v), 3), "astor_lost_rate": round(sum(x[1] for x in v) / len(v), 3)} for k, v in lv.items()}
    doc = {"run": {"what": "ASTOR-T win rate by design", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename,
                   "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(), "code": R.fingerprint(), "cpu": a.cpu, "sorties_per_design": n,
                   "population": H.POPULATION, "reach": H.REACH, "requirement": "win rate >= 0.80; ASTOR lost in at most a few per cent"},
           "summary": {"best": rows[:10], "reach_80_pct": [r["design"] for r in rows if r["win_rate"] >= 0.80],
                       "reach_80_pct_with_90pct_confidence": [r["design"] for r in rows if r["win_rate_90pct"][0] >= 0.80], "main_effects": effects},
           "results": {"astor_win": rows}}
    run = R._write("win", doc, "astor")
    print(f"{'win':>5s} {'90 %':>13s} {'lost':>5s}  design")
    for r in rows[:12]:
        print(f"{r['win_rate']:5.2f} {str(r['win_rate_90pct']):>13s} {r['astor_lost_rate']:5.2f}  {r['design']}  {r['win_rate_by_behaviour']}")
    for f, e in effects.items():
        print(f"  {f:10s} " + "   ".join(f"{k}: {v['win_rate']:.2f}" for k, v in e.items()))
    print("->", run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
