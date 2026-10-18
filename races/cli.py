"""``python -m races``: list the courses, build the scene files, run a race (mock or MuJoCo), replay it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import courses, mock, refquad, replay
from .rules import Rules


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="races", description="drone race environments for the reference 5-inch quad")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the courses, their zones and lengths")
    sub.add_parser("quad", help="the reference quad's budget and drive")
    b = sub.add_parser("build", help="write the MJCF scene files (races/scenes/<course>.xml)")
    b.add_argument("--out", default=str(courses.SCENES))
    r = sub.add_parser("run", help="run one race")
    r.add_argument("--course", required=True, choices=list(courses.COURSES))
    r.add_argument("--aircraft", default="refquad", choices=["refquad"])
    r.add_argument("--mock", action="store_true", help="the kinematic mock flyer instead of MuJoCo")
    r.add_argument("--collision", default="reset", choices=["reset", "count", "stop"])
    r.add_argument("--safety", type=float, default=0.8, help="the schedule's speed safety factor")
    r.add_argument("--v-cap", type=float, default=None, help="cap the schedule's top speed [m/s]")
    r.add_argument("--out", default="races/runs", help="where the result (json + npz + png + gif) goes")
    r.add_argument("--gif", action="store_true", help="also write the replay GIF")
    r.add_argument("--param", action="append", default=[], help="course parameter, e.g. --param floor_gap=5")
    g = sub.add_parser("ingest", help="digest every run under workflows/runs/races into one ingest_race.json (statistics recomputed from the trajectories)")
    g.add_argument("--runs", default=None, help="the runs folder (default workflows/runs/races)")
    g.add_argument("--out", default="ingest_race.json")
    g.add_argument("--hz", type=float, default=2.0, help="the downsampled trajectory's rate")
    g.add_argument("--no-mock", action="store_true", help="skip the mock baseline")
    a = ap.parse_args(argv)
    if a.cmd == "ingest":
        from . import ingest
        doc = ingest.ingest(Path(a.runs) if a.runs else ingest.DEFAULT_RUNS, hz=a.hz, with_mock=not a.no_mock)
        out = ingest.write(doc, Path(a.out))
        got = [s for s, e in doc["courses"].items() if e.get("run")]
        print(f"{out} ({out.stat().st_size / 1e3:.0f} kB): MuJoCo laps for {got or 'no course'}; {len(doc['other_runs'])} other run(s) listed")
        for s, c in doc["comparison"].items():
            m = c.get("mujoco")
            print(f"  {s:11s}", (f"{m['status']}, lap {m['lap time [s]']} s, {m['collisions']} collision(s), {m['energy [Wh]']} Wh, max {m['max speed [m/s]']} m/s" if m else "no completed lap"))
        return 0
    if a.cmd == "list":
        for slug, fn in courses.COURSES.items():
            c = fn()
            print(f"{slug:11s} {c.kind:9s} {c.length():6.0f} m  {len(c.zones):2d} zones  — {c.description}")
        return 0
    if a.cmd == "quad":
        for k, v in refquad.summary().items():
            print(f"{k:32s} {v}")
        print("budget [g]:", json.dumps(refquad.BUDGET, indent=1))
        return 0
    if a.cmd == "build":
        for slug, p in courses.build_all(Path(a.out)).items():
            print(slug, "->", p)
        return 0
    params = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        params[k] = float(v)
    course = courses.build(a.course, **params)
    rules = Rules(collision=a.collision)
    if a.mock:
        res = mock.run_mock(course, rules=rules, safety=a.safety, v_cap=a.v_cap)
    else:
        from . import sim
        res = sim.run_mujoco(course, rules=rules, safety=a.safety, v_cap=a.v_cap)
    out = Path(a.out) / f"{a.course}_{a.aircraft}_{res['stats']['mode']}"
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez(out / "trajectory.npz", history=res["history"])
    (out / "stats.json").write_text(json.dumps({"stats": res["stats"], "gate times": res["gate times"], "events": [[float(t), s] for t, s in res["events"]]}, indent=1, default=float))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    replay.overview(course, res, ax=axes[0], view="plan"); replay.overview(course, res, ax=axes[1], view="elevation")
    fig.savefig(out / "map.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    if a.gif:
        replay.animate(course, res, out / "replay.gif")
    for k, v in res["stats"].items():
        print(f"{k:22s} {v}")
    print("results:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
