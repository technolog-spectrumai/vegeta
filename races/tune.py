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
QUICK = {"safety": (0.4, 0.55), "tilt_max_deg": (60.0,), "lookahead_s": (0.15, 0.3), "omega_pos": (2.5,)}   # 4 laps per course
HANGAR2 = {"safety": (0.25, 0.35, 0.45), "v_cap": (3.0, 5.0, None), "omega_pos": (3.5, 5.0, 7.0), "lookahead_s": (0.3, 0.5),
           "omega_att": (18.0, 28.0), "tilt_max_deg": (60.0,)}                   # round 2: slower and steadier for the stair descent


def edges(course: str) -> dict:
    """Round 2 for a course that finished: past the edges of round 1 (omega_pos and lookahead won at their maxima),
    the safety around its winner."""
    from races import autopilot
    w = autopilot.tuned(course) or {"safety": 0.6, "tilt_max_deg": 60.0}
    s = w.get("safety", 0.6)
    return {"safety": tuple(sorted({round(max(0.2, s - 0.1), 2), s, round(min(0.9, s + 0.1), 2)})), "omega_pos": (3.5, 5.0, 7.0),
            "lookahead_s": (0.4, 0.6), "tilt_max_deg": (w.get("tilt_max_deg", 60.0),)}
OUT = Path(__file__).resolve().parent / "runs" / "tune"


def main(argv=None) -> int:
    from races import autopilot, courses, sim
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--courses", nargs="*", default=list(courses.COURSES))
    ap.add_argument("--quick", action="store_true", help="a small grid (4 tries per course)")
    ap.add_argument("--method", choices=("pso+gps", "gps", "pso"), default="pso+gps",
                    help="pso+gps (default): a particle swarm over the whole box, then pattern search from its best; gps: pattern search from the stored settings only")
    ap.add_argument("--particles", type=int, default=12, help="PSO swarm size")
    ap.add_argument("--pso-iterations", type=int, default=15, help="PSO iterations at most")
    ap.add_argument("--delta-min", type=float, default=1 / 64, help="convergence: stop when the mesh size (a share of every knob's range) is below this")
    ap.add_argument("--max-laps", type=int, default=400, help="a safety budget of laps per course")
    ap.add_argument("--patience", type=int, default=0, help="optional heuristic early stop after this many iterations gaining < --tol s (0 = off)")
    ap.add_argument("--tol", type=float, default=0.2, help="see --patience")
    ap.add_argument("--round2", action="store_true", help="the second round: past the edges for the finished courses, a slow steady grid for the hangar")
    a = ap.parse_args(argv)
    print(f"tune: courses {a.courses}, {'quick' if a.quick else 'full'} grid", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    best_f = OUT / "best.json"
    best = json.loads(best_f.read_text()) if best_f.exists() else {}
    for slug in a.courses:
        c = courses.COURSES[slug]()
        if not (a.quick or a.round2):                                # the default: the converging search
            from races import converge
            print(f"== {slug}: {c.name}, {c.length():.0f} m, {len(c.zones)} zones — {a.method} search from {autopilot.tuned(slug) or 'the defaults'}", flush=True)

            def fly(kw):
                try:
                    res = sim.run_mujoco(c, **kw)
                    return res["stats"], res.get("events") or []
                except Exception as e:
                    return {"status": "error", "reason": f"{type(e).__name__}: {e}"}, []
            say = lambda m: print(f"{slug} {m}", flush=True)
            start, pso_rows = autopilot.tuned(slug), []
            if a.method in ("pso+gps", "pso"):
                gx, pso_rows = converge.pso(c, fly, start, particles=a.particles, iterations=a.pso_iterations, max_laps=a.max_laps, log=say)
                start = converge.to_settings(gx)
                say(f"pso best → {start}")
            if a.method == "pso":
                st, _ = fly(start)
                bs, bst, rows, why = start, st, [], "pso only (no pattern-search polish)"
            else:
                bs, bst, rows, why = converge.search(c, fly, start, delta_min=a.delta_min, tol=a.tol, patience=a.patience,
                                                     max_laps=max(20, a.max_laps - len(pso_rows)), log=say)
            rows = pso_rows + [{"stage": "gps", **r} for r in rows]
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(OUT / f"{slug}_converge.csv", index=False)
            if bst.get("status") == "finished":
                best[slug] = {"settings": bs, "lap time [s]": float(bst["lap time [s]"]), "collisions": int(bst.get("collisions") or 0), "search": why, "laps": len(df)}
                print(f"== {slug}: FINISHED {bst['lap time [s]']:.1f} s, {bst.get('collisions') or 0} collisions — {why} — {bs}", flush=True)
            else:
                far = df.sort_values("score").iloc[0]
                best[slug] = {"settings": None, "search": why, "laps": len(df),
                              "note": f"no finished lap; closest: {far['zones passed']}, last zone {far['last zone reached']!r}; collisions at {far['collisions at']}"}
                print(f"== {slug}: still DNF — {why}; {best[slug]['note']}", flush=True)
            autopilot.record(slug, best[slug])
            best_f.write_text(json.dumps(best, indent=1))
            continue
        grid = QUICK if a.quick else (HANGAR2 if slug == "hangar" else edges(slug)) if a.round2 else GRID
        rows, keys = [], list(grid)
        total = 1
        for v in grid.values():
            total *= len(v)
        print(f"== {slug}: {c.name}, {c.length():.0f} m, {len(c.zones)} zones — {total} laps to fly", flush=True)
        for n, vals in enumerate(itertools.product(*grid.values()), 1):
            kw = dict(zip(keys, vals))
            t0 = time.time()
            print(f"{slug} {n:3d}/{total}  flying {kw} …", flush=True)
            try:
                res = sim.run_mujoco(c, **kw)
                st, ev = res["stats"], res.get("events") or []
            except Exception as e:                                   # a crash of one try is a result, not the end
                st, ev = {"status": "error", "reason": f"{type(e).__name__}: {e}"}, []
            zones = [e[1] for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and str(e[1]).startswith("zone")]
            hits = [e for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and "collision" in str(e[1]).lower()]
            row = {**kw, "status": st.get("status"), "reason": st.get("reason", ""), "lap time [s]": st.get("lap time [s]"),
                   "collisions": st.get("collisions"), "zones passed": st.get("zones passed"),
                   "last zone reached": zones[-1] if zones else "", "collisions at": "; ".join(f"{e[0]:.1f} s {e[1]}" for e in hits[:6]),
                   "wall [s]": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{slug} {n:3d}/{total}  → {row['status']} {row['lap time [s]'] or ''} {row['zones passed'] or ''} {row['collisions'] or 0} hits, {row['wall [s]']} s{(' — ' + row['reason']) if row['reason'] else ''}", flush=True)
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{slug}.csv", index=False)
        ok = df[df["status"] == "finished"].sort_values(["lap time [s]", "collisions"])
        if len(ok):
            w = ok.iloc[0]
            best[slug] = {"settings": {k: (None if w[k] is None or w[k] != w[k] else float(w[k])) for k in keys}, "lap time [s]": float(w["lap time [s]"]), "collisions": int(w["collisions"])}
            print(f"== {slug}: FINISHED {w['lap time [s]']:.1f} s with {best[slug]['settings']}")
            autopilot.record(slug, best[slug])
        else:
            far = df.sort_values("zones passed", key=lambda z: z.fillna("0 of").str.split(" ").str[0].astype(int), ascending=False).iloc[0]
            best[slug] = {"settings": None, "note": f"no finished lap in {len(df)} tries; closest: {far['zones passed']}, last zone {far['last zone reached']!r}; collisions at {far['collisions at']}"}
            autopilot.record(slug, best[slug])
            print(f"== {slug}: still DNF in all {len(df)} tries — see {OUT / (slug + '.csv')}")
        best_f.write_text(json.dumps(best, indent=1))
    print("written", best_f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
