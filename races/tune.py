"""Tune the MuJoCo autopilot per course (``races.sim.run_mujoco``), every lap refereed. The default is a particle
swarm then a pattern-search polish (``races/converge.py``), checkpointed after every iteration to
``races/runs/tune/<course>.ckpt.json`` and resumed from there; ``--quick`` / ``--round2`` are the earlier fixed grids.
Writes ``races/runs/tune/<course>_converge.csv`` (every lap), ``best.json`` and ``races/autopilot.json`` after every
course.

    python -m races.tune [--courses hangar farm] [-j 8] [--fresh]
"""
from __future__ import annotations

import argparse
import os
import itertools
import json
import time
from pathlib import Path
from races.rules import STUCK

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


def _fly_worker(job):
    """One lap in a worker process (the course is built there; MuJoCo shares nothing between processes)."""
    slug, kw = job
    from races import courses, sim
    try:
        res = sim.run_mujoco(courses.COURSES[slug](), **kw)
        return res["stats"], [list(e) for e in (res.get("events") or [])]
    except Exception as e:                                       # a crash of one try is a result, not the end
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}, []


SCHEMA = 2                                                    # 2: the drop guard, the squared hangar window, the stuck label


def _write_atomic(path: Path, obj) -> None:
    """Write JSON so a crash leaves the old file or the new one, never half of one: a temporary file, fsync, rename."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Laps:
    """Laps flown side by side and remembered (MuJoCo is deterministic: a setting is flown once).

    ``prefetch(settings, stage, iteration, cut)`` flies the missing laps of a batch on the pool (``jobs`` processes)
    and returns how many it flew; ``score(settings)`` reads one from memory. With ``cut`` the batch's laps stop at
    1.25 × the best finished lap known when the batch starts + 5 s (the same for every lap of the batch, so the result
    does not depend on ``jobs``); a cut lap scores as a DNF and can never beat that best."""

    def __init__(self, slug: str, jobs: int, cut: bool = True, memo: dict | None = None, rows: list | None = None):
        import concurrent.futures as cf
        self.slug, self.cut = slug, cut
        self.memo, self.rows = dict(memo or {}), list(rows or [])
        self.pool = cf.ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None

    @staticmethod
    def key(s) -> str:
        return json.dumps(s, sort_keys=True)

    def best_finished(self):
        t = [r[0]["lap time [s]"] for r in self.memo.values() if r[0].get("status") == "finished"]
        return min(t) if t else None

    def prefetch(self, settings, stage, iteration, cut) -> int:
        todo = [s for k, s in {self.key(s): s for s in settings}.items() if k not in self.memo]
        if not todo:
            return 0
        b = self.best_finished()
        duration = 1.25 * b + 5.0 if (cut and self.cut and b) else None
        jobs = [(self.slug, {**s, **({"duration": duration} if duration else {})}) for s in todo]
        t0 = time.time()
        results = list(self.pool.map(_fly_worker, jobs)) if self.pool else [_fly_worker(j) for j in jobs]
        wall = (time.time() - t0) / len(todo)
        for s, (st, ev) in zip(todo, results):
            if duration and st.get("status") == STUCK:
                st["reason"] = f"cut at {duration:.0f} s (slower than the best, {b:.1f} s)"
            self.memo[self.key(s)] = [st, ev, round(wall, 2)]
            self.rows.append(_row(stage, iteration, s, st, ev, wall))
            print(f"{self.slug}   {stage} it {iteration} lap {len(self.memo):3d}  {s}  → {st.get('status')} {st.get('lap time [s]') or ''} "
                  f"{st.get('zones passed') or ''}, score {_score(st):.2f}{(' — ' + st['reason']) if st.get('reason') else ''}", flush=True)
        print(f"{self.slug}   batch of {len(todo)} lap(s) in {time.time() - t0:.1f} s wall ({wall:.1f} s per lap with {self.pool._max_workers if self.pool else 1} worker(s))", flush=True)
        return len(todo)

    def stats(self, s):
        return self.memo[self.key(s)][0]

    def score(self, s) -> float:
        return _score(self.stats(s))

    def close(self):
        if self.pool:
            self.pool.shutdown()


def _score(st):
    from races import converge
    return converge.score(st)


def _row(stage, iteration, s, st, ev, wall):
    zones = [e[1] for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and str(e[1]).startswith("zone")]
    hits = [e for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and "collision" in str(e[1]).lower()]
    return {"stage": stage, "iteration": iteration, **s, "score": round(_score(st), 3), "status": st.get("status"), "reason": st.get("reason", ""),
            "lap time [s]": st.get("lap time [s]"), "collisions": st.get("collisions"), "zones passed": st.get("zones passed"),
            "last zone reached": zones[-1] if zones else "", "collisions at": "; ".join(f"{e[0]:.1f} s {e[1]}" for e in hits[:6]),
            "wall per lap [s]": round(wall, 2)}


def fingerprint(a) -> dict:
    from races import converge
    return json.loads(json.dumps({"schema": SCHEMA, "method": a.method, "seed": a.seed, "particles": a.particles, "pso_iterations": a.pso_iterations,
            "delta_min": a.delta_min, "max_laps": a.max_laps, "polish_laps": a.polish_laps, "patience": a.patience, "tol": a.tol,
            "cut": not a.no_cut, "bounds": converge.BOUNDS}))                    # JSON-normalised (tuples → lists), as the checkpoint stores it


def converge_course(slug, c, a, best, best_f) -> int:
    """One course: PSO then polish (or one of them), checkpointed after every iteration, resumed when a compatible
    checkpoint exists. Returns 0, or 2 when a checkpoint is incompatible."""
    from races import autopilot, converge
    ckpt_f = OUT / f"{slug}.ckpt.json"
    fp = fingerprint(a)
    ck = json.loads(ckpt_f.read_text()) if ckpt_f.exists() and not a.fresh else None
    if ck is not None and ck.get("fingerprint") != fp:
        diff = sorted(k for k in set(fp) | set(ck.get("fingerprint", {})) if fp.get(k) != ck.get("fingerprint", {}).get(k))
        print(f"== {slug}: the checkpoint {ckpt_f.name} was made with other options ({', '.join(diff)}): "
              f"rerun with the same options to resume it, or with --fresh to start over", flush=True)
        return 2
    if ck and ck["phase"] == "done":
        print(f"== {slug}: already done (checkpoint {ckpt_f.name}): {ck['result']['line']}  (--fresh to redo)", flush=True)
        best[slug] = ck["result"]["entry"]
        return 0
    if ck is None:
        start = autopilot.tuned(slug)
        reserve = a.polish_laps if a.method != "pso" else 0
        ck = {"fingerprint": fp, "start": start, "phase": "pso" if a.method != "gps" else "gps",
              "pso": converge.PSO(start, particles=a.particles, iterations=a.pso_iterations, seed=a.seed, budget=max(a.particles, a.max_laps - reserve)).to_dict()
              if a.method != "gps" else None,
              "gps": None, "memo": {}, "rows": [], "time": {"pso_s": 0.0, "gps_s": 0.0}}
        print(f"== {slug}: {c.name}, {c.length():.0f} m, {len(c.zones)} zones — {a.method} from {start or 'the defaults'}", flush=True)
    else:
        print(f"== {slug}: resuming from {ckpt_f.name} — phase {ck['phase']}, {len(ck['memo'])} laps already flown "
              f"(PSO {ck['time']['pso_s'] / 60:.1f} min, polishing {ck['time']['gps_s'] / 60:.1f} min so far)", flush=True)
    print(f"   workers: {a.jobs} (this machine has {os.cpu_count()} CPUs; -j {os.cpu_count()} to use them all); "
          f"lap budget {a.max_laps}, {a.polish_laps if a.method != 'pso' else 0} of them reserved for polishing", flush=True)
    laps = Laps(slug, a.jobs, cut=not a.no_cut, memo=ck["memo"], rows=ck["rows"])
    say = lambda m: print(f"{slug} {m}", flush=True)

    def save():
        ck["memo"], ck["rows"] = laps.memo, laps.rows
        _write_atomic(ckpt_f, ck)

    try:
        if ck["phase"] == "pso":
            pso = converge.PSO.from_dict(ck["pso"])
            while not pso.done:
                t0 = time.time()
                pso.step(laps, log=say)
                ck["time"]["pso_s"] += time.time() - t0
                ck["pso"] = pso.to_dict()
                save()
            ck["phase"] = "gps" if a.method != "pso" else "final"
            save()
        if ck["phase"] == "gps":
            if ck["gps"] is None:
                origin = converge.PSO.from_dict(ck["pso"]).best() if ck["pso"] else (ck["start"] or {})
                used = len(laps.memo)
                g = converge.GPS(origin, delta_min=a.delta_min, budget=max(a.polish_laps, a.max_laps - used), patience=a.patience, tol=a.tol)
                say(f"polishing from {'the PSO best' if ck['pso'] else 'the stored settings'}: {origin}")
            else:
                g = converge.GPS.from_dict(ck["gps"])
            while not g.done:
                t0 = time.time()
                g.step(laps, log=say)
                ck["time"]["gps_s"] += time.time() - t0
                ck["gps"] = g.to_dict()
                save()
            ck["phase"] = "final"
            save()
    finally:
        laps.close()
    # ---- the result
    if ck["gps"]:
        g = converge.GPS.from_dict(ck["gps"])
        bs, why = g.best(), g.reason
    else:
        p = converge.PSO.from_dict(ck["pso"])
        bs, why = p.best(), f"PSO only: {p.reason}"
    bst = laps.stats(bs)
    s0 = converge.to_settings(converge.to_x(ck["start"] or {}))          # the start as both searches fly it
    start_sc = laps.score(s0) if laps.key(s0) in laps.memo else None
    end_sc = converge.score(bst)
    change = ("no improvement found over the starting settings" if start_sc is not None and end_sc >= start_sc - 1e-9
              else f"improved from {start_sc:.2f} to {end_sc:.2f}" if start_sc is not None else f"best score {end_sc:.2f}")
    t = ck["time"]
    nl = len(laps.memo)
    timing = {"pso [min]": round(t["pso_s"] / 60, 2), "polishing [min]": round(t["gps_s"] / 60, 2), "total [min]": round((t["pso_s"] + t["gps_s"]) / 60, 2),
              "laps": nl, "wall per lap [s]": round((t["pso_s"] + t["gps_s"]) / max(nl, 1), 2), "workers": a.jobs}
    import pandas as pd
    df = pd.DataFrame(laps.rows)
    df.to_csv(OUT / f"{slug}_converge.csv", index=False)
    if bst.get("status") == "finished":
        entry = {"settings": bs, "lap time [s]": float(bst["lap time [s]"]), "collisions": int(bst.get("collisions") or 0), "search": why, "change": change, "time": timing}
        line = f"FINISHED {bst['lap time [s]']:.1f} s, {bst.get('collisions') or 0} collisions — {change} — {why} — {bs}"
    else:
        far = df.sort_values("score").iloc[0]
        entry = {"settings": None, "search": why, "change": change, "time": timing,
                 "note": f"no finished lap; closest: {far['zones passed']}, last zone {far['last zone reached']!r}; collisions at {far['collisions at']}"}
        line = f"still DNF — {change} — {why}; {entry['note']}"
    print(f"== {slug}: {line}", flush=True)
    print(f"== {slug}: time — PSO {timing['pso [min]']} min, polishing {timing['polishing [min]']} min, total {timing['total [min]']} min, "
          f"{nl} laps, {timing['wall per lap [s]']} s wall per lap with {a.jobs} worker(s)", flush=True)
    best[slug] = entry
    autopilot.record(slug, entry)
    _write_atomic(best_f, best)                                   # saved right after the course
    ck["phase"], ck["result"] = "done", {"entry": entry, "line": line}
    save()
    return 0


def main(argv=None) -> int:
    from races import autopilot, courses, sim
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--courses", nargs="*", default=list(courses.COURSES))
    ap.add_argument("--quick", action="store_true", help="a small grid (4 tries per course)")
    ap.add_argument("--method", choices=("pso+gps", "gps", "pso"), default="pso+gps",
                    help="pso+gps (default): a particle swarm over the whole box, then pattern search from its best; gps: pattern search from the stored settings only")
    ap.add_argument("-j", "--jobs", type=int, default=1, help=f"worker processes flying laps side by side (default 1; this machine has {os.cpu_count()} CPUs)")
    ap.add_argument("--fresh", action="store_true", help="ignore and replace the courses' checkpoints (default: resume a compatible checkpoint, skip finished courses)")
    ap.add_argument("--seed", type=int, default=0, help="the swarm's random seed (a run with the same seed and options is repeated exactly)")
    ap.add_argument("--no-cut", action="store_true", help="polishing flies every lap to its end (default: laps slower than 1.25 × the best + 5 s are cut)")
    ap.add_argument("--particles", type=int, default=12, help="PSO swarm size")
    ap.add_argument("--pso-iterations", type=int, default=15, help="PSO hard maximum of iterations; then it always switches to polishing (default 15)")
    ap.add_argument("--polish-laps", type=int, default=120, help="laps of --max-laps reserved for polishing (default 120)")
    ap.add_argument("--delta-min", type=float, default=1 / 64, help="convergence: stop when the mesh size (a share of every knob's range) is below this")
    ap.add_argument("--max-laps", type=int, default=400, help="the lap budget per course, PSO and polishing together (default 400)")
    ap.add_argument("--patience", type=int, default=0, help="optional heuristic early stop after this many iterations gaining < --tol s (0 = off)")
    ap.add_argument("--tol", type=float, default=0.2, help="see --patience")
    ap.add_argument("--round2", action="store_true", help="the second round: past the edges for the finished courses, a slow steady grid for the hangar")
    a = ap.parse_args(argv)
    mode = "quick grid" if a.quick else "round-2 grids" if a.round2 else a.method
    print(f"tune: courses {a.courses}, {mode}; {a.jobs} worker(s) of {os.cpu_count()} CPUs", flush=True)
    rc = 0
    OUT.mkdir(parents=True, exist_ok=True)
    best_f = OUT / "best.json"
    best = json.loads(best_f.read_text()) if best_f.exists() else {}
    for slug in a.courses:
        c = courses.COURSES[slug]()
        if not (a.quick or a.round2):                                # the default: PSO, then the pattern-search polish
            rc = converge_course(slug, c, a, best, best_f) or rc
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
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
