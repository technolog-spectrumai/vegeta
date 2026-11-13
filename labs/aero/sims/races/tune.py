"""Tune the MuJoCo autopilot per course (``races.sim.run_mujoco``), every lap refereed: a search strategy
(``races.optimizers``: ``--cma``, the default, or ``--pso``) minimises the robust score of the candidates over the
nominal lap and K training scenarios (``races.laps``, ``races.validate.training``).

The driver knows no algorithm: it builds the strategy, calls ``step`` until it is done, times each phase, and
checkpoints after every step to ``races/runs/tune/<course>.ckpt.json`` (atomically). A rerun with the same options
resumes there and skips finished courses; other options are refused (``--fresh`` starts over). After every course:
``races/runs/tune/<course>_converge.csv`` (every lap), ``best.json`` and ``races/autopilot.json``.

    python -m races.tune [--courses hangar farm] [--cma | --pso] [-j 8] [--fresh]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

SCHEMA = 4                                # 2: the drop guard, the window, the stuck label; 3: training scenarios; 4: strategies
OUT = Path(__file__).resolve().parent / "runs" / "tune"


def _write_atomic(path: Path, obj) -> None:
    """Write JSON so a crash leaves the old file or the new one, never half of one: a temporary file, fsync, rename."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def fingerprint(a, cls, cfg) -> dict:
    """Everything a checkpoint's laps and decisions depend on (JSON-normalised, as the checkpoint stores it)."""
    from races import converge
    extra = {}
    if cls.name == "cma":
        import cma
        extra["cma"] = cma.__version__                            # a pickled CMA-ES state belongs to its package version
    return json.loads(json.dumps({"schema": SCHEMA, "strategy": cls.name, "config": dataclasses.asdict(cfg), "train_k": a.train_k,
                                  "train_seed": a.train_seed, "cut": not a.no_cut, "bounds": converge.BOUNDS, **extra}))


class Checkpoint:
    """One course's checkpoint: the fingerprint, the start, the strategy's state, the laps flown, the phase times."""

    def __init__(self, path: Path, fp: dict, start: dict):
        self.path = path
        self.data = {"fingerprint": fp, "start": start, "strategy": None, "memo": {}, "rows": [], "time": {}, "result": None}

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            return None
        ck = cls.__new__(cls)
        ck.path, ck.data = path, json.loads(path.read_text())
        return ck

    def differs(self, fp: dict) -> list:
        old = self.data.get("fingerprint", {})
        return sorted(k for k in set(fp) | set(old) if fp.get(k) != old.get(k))

    def save(self, strategy, laps) -> None:
        self.data.update(strategy=strategy.state(), memo=laps.memo, rows=laps.rows)
        _write_atomic(self.path, self.data)


def result(strategy, laps, start: dict, objective: str, times: dict, jobs: int) -> tuple[dict, str]:
    """The course's autopilot.json entry and its one-line summary."""
    from races import converge
    s0 = converge.to_settings(converge.to_x(start or {}))              # the start as the strategies fly it (CMA-ES samples around it)
    laps.prefetch([s0], "start", 0)
    start_sc, bs = laps.score(s0), strategy.best()
    if laps.score(bs) >= start_sc - 1e-9:
        bs, change = s0, "no improvement found over the starting settings"
    else:
        change = f"improved from {start_sc:.2f} to {laps.score(bs):.2f}"
    bst, end_sc = laps.stats(bs), laps.score(bs)
    total = sum(times.values())
    timing = {**{f"{k} [min]": round(v / 60, 2) for k, v in times.items()}, "total [min]": round(total / 60, 2), "laps": len(laps.memo),
              "evaluations": strategy.evals_used, "wall per lap [s]": round(total / max(len(laps.memo), 1), 2), "workers": jobs}
    search = f"{strategy.name}: {strategy.reason}"
    if bst.get("status") == "finished":
        entry = {"settings": bs, "lap time [s]": float(bst["lap time [s]"]), "collisions": int(bst.get("collisions") or 0), "objective": objective,
                 "score": round(end_sc, 3), "training clean rate": round(laps.clean_rate(bs), 3), "search": search, "change": change, "time": timing}
        line = (f"FINISHED {bst['lap time [s]']:.1f} s, {bst.get('collisions') or 0} collisions (nominal); robust score {end_sc:.2f}, "
                f"training clean rate {entry['training clean rate']:.2f} — {change} — {strategy.reason} — {bs}")
    else:
        import pandas as pd
        far = pd.DataFrame(laps.rows).query("scenario == 0").sort_values("score").iloc[0]
        entry = {"settings": None, "objective": objective, "search": search, "change": change, "time": timing,
                 "note": f"no finished nominal lap; closest: {far['zones passed']}, last zone {far['last zone reached']!r}; collisions at {far['collisions at']}"}
        line = f"still DNF — {change} — {strategy.reason}; {entry['note']}"
    return entry, line


def tune_course(slug: str, a, cls, cfg, best: dict, best_f: Path) -> int:
    """One course, resumable. Returns 0, or 2 when a checkpoint was made with other options."""
    from races import autopilot, courses, validate
    from races.laps import Laps
    c = courses.COURSES[slug]()
    path = OUT / f"{slug}.ckpt.json"
    fp = fingerprint(a, cls, cfg)
    ck = None if a.fresh else Checkpoint.load(path)
    if ck is not None and ck.differs(fp):
        print(f"== {slug}: the checkpoint {path.name} was made with other options ({', '.join(ck.differs(fp))}): "
              f"rerun with the same options to resume it, or with --fresh to start over", flush=True)
        return 2
    if ck is not None and ck.data["result"]:
        print(f"== {slug}: already done (checkpoint {path.name}): {ck.data['result']['line']}  (--fresh to redo)", flush=True)
        best[slug] = ck.data["result"]["entry"]
        return 0
    if ck is None:
        ck = Checkpoint(path, fp, autopilot.tuned(slug))
        strategy = cls(ck.data["start"], cfg)
        print(f"== {slug}: {c.name}, {c.length():.0f} m, {len(c.zones)} zones — {cls.summary} from {ck.data['start'] or 'the defaults'}", flush=True)
    else:
        strategy = cls.restore(ck.data["start"], cfg, ck.data["strategy"])
        print(f"== {slug}: resuming from {path.name} — phase {strategy.phase}, {len(ck.data['memo'])} laps already flown", flush=True)
    train = validate.training(a.train_k, a.train_seed)
    laps = Laps(slug, a.jobs, train, cut=not a.no_cut, memo=ck.data["memo"], rows=ck.data["rows"], log=lambda m: print(m, flush=True))
    objective = (f"robust: the mean of the worst {laps.m} of {len(train)} laps (the nominal lap and {a.train_k} training scenarios, seed {a.train_seed})"
                 if a.train_k else "nominal: the nominal lap alone")
    print(f"   objective — {objective}", flush=True)
    print(f"   workers: {a.jobs} (this machine has {os.cpu_count()} CPUs); budget {cfg.max_evals} evaluations of {len(train)} laps each", flush=True)
    times = ck.data["time"]
    try:
        while not strategy.done:
            t0, phase = time.time(), strategy.phase
            strategy.step(laps, log=lambda m: print(f"{slug} {m}", flush=True))
            times[phase] = times.get(phase, 0.0) + time.time() - t0
            ck.save(strategy, laps)
        entry, line = result(strategy, laps, ck.data["start"], objective, times, a.jobs)
    finally:
        laps.close()
    import pandas as pd
    pd.DataFrame(laps.rows).to_csv(OUT / f"{slug}_converge.csv", index=False)
    print(f"== {slug}: {line}", flush=True)
    print(f"== {slug}: time — " + ", ".join(f"{k} {v / 60:.2f} min" for k, v in times.items()) +
          f", total {sum(times.values()) / 60:.2f} min, {len(laps.memo)} laps, {strategy.evals_used} evaluations, {a.jobs} worker(s)", flush=True)
    best[slug] = entry
    autopilot.record(slug, entry)
    _write_atomic(best_f, best)                                   # saved right after the course
    ck.data["result"] = {"entry": entry, "line": line}
    ck.save(strategy, laps)
    return 0


def main(argv=None) -> int:
    from races import courses, optimizers, validate
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--courses", nargs="*", default=list(courses.COURSES), choices=list(courses.COURSES))
    ap.add_argument("-j", "--jobs", type=int, default=1, help=f"worker processes flying laps side by side (default 1; this machine has {os.cpu_count()} CPUs)")
    ap.add_argument("--fresh", action="store_true", help="ignore and replace the courses' checkpoints (default: resume a compatible checkpoint, skip finished courses)")
    ap.add_argument("--train-k", type=int, default=6, help="training scenarios per candidate besides the nominal lap (default 6, one per perturbation family; 0 = the nominal lap alone)")
    ap.add_argument("--train-seed", type=int, default=validate.TRAIN_SEED, help=f"the training scenarios' seed (default {validate.TRAIN_SEED}; never the held-out seed {validate.SEED})")
    ap.add_argument("--no-cut", action="store_true", help="with --train-k 0 only: fly every lap to its end (default: the PSO polish cuts laps slower than 1.25 × the best + 5 s)")
    optimizers.add_arguments(ap)
    a = ap.parse_args(argv)
    if a.train_seed == validate.SEED:
        ap.error(f"--train-seed {a.train_seed} is the held-out validation seed")
    cls, cfg = optimizers.config(a)
    print(f"tune: courses {a.courses}; {cls.summary} {dataclasses.asdict(cfg)}; {a.jobs} worker(s) of {os.cpu_count()} CPUs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    best_f = OUT / "best.json"
    best = json.loads(best_f.read_text()) if best_f.exists() else {}
    rc = 0
    for slug in a.courses:
        rc = tune_course(slug, a, cls, cfg, best, best_f) or rc
    print("written", best_f, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
