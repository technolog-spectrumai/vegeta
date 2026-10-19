"""Tune the MuJoCo autopilot per course: a grid over the controller's knobs (``races.sim.run_mujoco``), every lap
refereed, the fastest *finished* lap kept per course. Writes ``races/runs/tune/<course>.csv`` (every try) and
``races/runs/tune/best.json`` (the winning settings per course, to pass to ``run_mujoco`` / the race notebooks'
``MUJOCO`` dict).

    python -m races.tune [--courses hangar farm] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

GRID = {"safety": (0.4, 0.5, 0.6, 0.7), "tilt_max_deg": (45.0, 60.0), "lookahead_s": (0.15, 0.25, 0.4), "omega_pos": (2.0, 2.5, 3.5)}
QUICK = {"safety": (0.4, 0.55), "tilt_max_deg": (60.0,), "lookahead_s": (0.15, 0.3), "omega_pos": (2.5,)}
OUT = Path(__file__).resolve().parent / "runs" / "tune"


def main(argv=None) -> int:
    from races import courses, sim
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--courses", nargs="*", default=list(courses.COURSES))
    ap.add_argument("--quick", action="store_true", help="a small grid (8 tries per course)")
    a = ap.parse_args(argv)
    grid = QUICK if a.quick else GRID
    OUT.mkdir(parents=True, exist_ok=True)
    best_f = OUT / "best.json"
    best = json.loads(best_f.read_text()) if best_f.exists() else {}
    for slug in a.courses:
        c = courses.COURSES[slug]()
        rows, keys = [], list(grid)
        for n, vals in enumerate(itertools.product(*grid.values()), 1):
            kw = dict(zip(keys, vals))
            t0 = time.time()
            try:
                st = sim.run_mujoco(c, **kw)["stats"]
            except Exception as e:                                   # a crash of one try is a result, not the end
                st = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
            row = {**kw, "status": st.get("status"), "reason": st.get("reason", ""), "lap time [s]": st.get("lap time [s]"),
                   "collisions": st.get("collisions"), "zones passed": st.get("zones passed"), "wall [s]": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{slug} {n:3d}  {kw}  → {row['status']} {row['lap time [s]'] or ''} {row['zones passed'] or ''} {row['collisions'] or 0} hits", flush=True)
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{slug}.csv", index=False)
        ok = df[df["status"] == "finished"].sort_values(["lap time [s]", "collisions"])
        if len(ok):
            w = ok.iloc[0]
            best[slug] = {"settings": {k: float(w[k]) for k in keys}, "lap time [s]": float(w["lap time [s]"]), "collisions": int(w["collisions"])}
            print(f"== {slug}: FINISHED {w['lap time [s]']:.1f} s with {best[slug]['settings']}")
        else:
            best[slug] = {"settings": None, "note": f"no finished lap in {len(df)} tries; closest: {df.iloc[0]['zones passed']}"}
            print(f"== {slug}: still DNF in all {len(df)} tries — see {OUT / (slug + '.csv')}")
        best_f.write_text(json.dumps(best, indent=1))
    print("written", best_f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
