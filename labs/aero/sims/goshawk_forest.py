#!/usr/bin/env python3
"""GOSHAWK through a wood: the campaign. ``designs/goshawk_forest`` flies one flight; this flies them all.

    labs/aero/sims/goshawk_forest.sh -j 12                  # 4 densities x 3 speeds x 20 stands = 240 flights
    labs/aero/sims/goshawk_forest.sh -j 12 --flights 50     # 50 stands per case
    labs/aero/sims/goshawk_forest.sh --dry-run              # the cases, nothing flown
    labs/aero/sims/goshawk_forest.sh -j 12 --fresh          # forget the flights already flown

Each flight: a strip 150 m long of trunks at a stand density (stems per hectare), flown at a commanded speed with the
flaps down by a short-sighted planner. ``through``: the far end reached without touching a trunk. Flights already flown
are kept (a restart continues). One timestamped YAML per run in ``sims/output/runs``: per density and speed the share
of flights through, the mean distance before a hit, the closest pass. Uses the CFD installed for GOSHAWK if there is
one (``astor.sh --design goshawk --cfd-only``), else the carried-over model: the run records which.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path[:0] = [str(DESIGNS), str(HERE)]

import astor_run as R                                                # noqa: E402

R.GLOBS_ACTIVE = ["arigo*.py", "astor*.py", "raptors*.py", "goshawk*.py"]
DENSITIES = (150.0, 250.0, 400.0, 700.0)        # stems per hectare: mean distance 8.2, 6.3, 5.0, 3.8 m
SPEEDS = (10.0, 12.0, 14.0)


def _job(job):
    t = time.time()
    try:
        import goshawk_forest as GF
        out = GF.fly(job[1], job[2], job[3])
    except Exception as exc:                                         # noqa: BLE001
        import traceback
        out = {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-900:]}
    out["seconds"] = round(time.time() - t, 1)
    return "|".join(str(x) for x in job), out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--cpu", type=int, default=4, help="worker processes (default 4)")
    ap.add_argument("--force-cpu", action="store_true", help="use -j exactly as given")
    ap.add_argument("--flights", type=int, default=20, help="stands flown per density and speed (default 20)")
    ap.add_argument("--density", type=float, action="append", help=f"stems per hectare (repeat; default {DENSITIES})")
    ap.add_argument("--speed", type=float, action="append", help=f"commanded speed [m/s] (repeat; default {SPEEDS})")
    ap.add_argument("--fresh", action="store_true", help="forget the flights already flown")
    ap.add_argument("--dry-run", action="store_true", help="list the cases; fly nothing")
    a = ap.parse_args(argv)
    dens, speeds = a.density or list(DENSITIES), a.speed or list(SPEEDS)
    jobs = [("forest", d, v, s) for d in dens for v in speeds for s in range(a.flights)]
    if a.dry_run:
        for d in dens:
            for v in speeds:
                print(f"forest  {d:5.0f} stems/ha (stems {100 / d ** 0.5:4.1f} m apart)  {v:4.1f} m/s  x {a.flights} stands")
        print(len(jobs), "flights")
        return 0
    import astor_v_flight as VF
    import goshawk_forest as GF
    C = VF.get("goshawk")
    res = R._run_jobs(jobs, a, f"goshawk_forest_cache_{R.fingerprint()[:8]}.yaml", "goshawk forest", _job,
                      note=lambda k, o: f"{k.split('|')[1]}/ha {k.split('|')[2]} m/s: {o.get('outcome', '?')}", unit="flight")
    summary = {}
    for d in dens:
        for v in speeds:
            rows = [res[f"forest|{d}|{v}|{s}"] for s in range(a.flights) if "error" not in res.get(f"forest|{d}|{v}|{s}", {"error": 1})]
            if not rows:
                continue
            n = len(rows)
            summary[f"{d:.0f} stems/ha, {v:g} m/s"] = {
                "stems_apart_m": round(100 / d ** 0.5, 1), "flights": n, "through": sum(r["through"] for r in rows), "through_rate": round(sum(r["through"] for r in rows) / n, 2),
                "hit": sum(r["outcome"] == "hit" for r in rows), "strayed": sum(r["outcome"] == "strayed" for r in rows), "lost": sum(r["outcome"] == "lost" for r in rows),
                "mean_distance_m": round(sum(r["distance_m"] for r in rows) / n, 1), "mean_speed_m_s": round(sum(r["mean_speed_m_s"] for r in rows) / n, 1),
                "closest_pass_of_those_through_m": min([r["closest_m"] for r in rows if r["through"]], default=None),
                "peak_load_factor": max(r["peak_load_factor"] for r in rows)}
    errors = {k: v["error"] for k, v in res.items() if "error" in v}
    run = R._write("forest", {"run": {"aircraft": "GOSHAWK", "phase": "forest", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename,
                                      "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(), "cpu": a.cpu, "code": R.fingerprint(),
                                      "aero_source": C["aero_source"], "mass_kg": round(C["mass"], 2), "span_m": round(C["b"], 2),
                                      "stand": {"length_m": GF.LENGTH_M, "width_m": GF.WIDTH_M, "trunk_radius_m": list(GF.TRUNK_R_M), "jitter": GF.JITTER},
                                      "planner": {"sees_m": GF.SEE_M, "field_deg": GF.FOV_DEG, "horizon_s": GF.HORIZON_S, "margin_m": GF.MARGIN_M}},
                              "summary": summary, "errors": errors, "results": {"goshawk_forest": [res[k] for k in sorted(res)]}}, "goshawk")
    print("->", run)
    print(f"{'case':28s} {'through':>9s} {'mean dist':>10s} {'closest':>8s}")
    for k, v in summary.items():
        print(f"{k:28s} {v['through']:3d}/{v['flights']:<3d}   {v['mean_distance_m']:8.1f} m {str(v['closest_pass_of_those_through_m']):>8s}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
