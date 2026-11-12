"""The objective the tuning strategies minimise (``races.optimizers``): a candidate's MuJoCo laps on the training set,
flown side by side and remembered (MuJoCo is deterministic: a setting is flown once per scenario).

Every candidate is flown on the training set (``races.validate.training``: the nominal lap, index 0, and K perturbed
scenarios, the same for every candidate) and scored on the robust score: the mean of the worst ⌈(K+1)/3⌉ of its
K+1 lap scores (``races.validate.worst_mean``; one lap scores as ``races.converge.score``). With K = 0 it is the
nominal lap's score.

``prefetch(settings, stage, iteration, cut, js)`` flies the missing laps of a batch (the scenarios ``js``, default
all) on the pool and returns how many candidates it evaluated for the first time (the strategies' budget unit);
``score(settings, js)`` reads them from memory — over part of the scenarios (at least ⌈(K+1)/3⌉ of them) it is a lower
bound of the full score. ``cut`` (only with K = 0) stops the batch's laps at 1.25 × the best finished lap known when
the batch starts + 5 s; a cut lap scores as a DNF and can never beat that best. With K > 0 nothing is cut: a lap's
score enters a tail average, where a cut could change a comparison."""
from __future__ import annotations

import json
import time

from races.rules import STUCK

NOMINAL = {"family": "nominal", "i": 0, "mult": {}, "scenario": {}}


def lap_score(st: dict) -> float:
    from races import converge
    return converge.score(st)


def fly_worker(job):
    """One lap in a worker process (the course is built there; MuJoCo shares nothing between processes)."""
    slug, kw, sc = job
    from races import courses, sim
    try:
        res = sim.run_mujoco(courses.COURSES[slug](), scenario=sc or None, **kw)
        return res["stats"], [list(e) for e in (res.get("events") or [])]
    except Exception as e:                                       # a crash of one try is a result, not the end
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}, []


def row(stage, iteration, s, st, ev, wall, j=0, family="nominal") -> dict:
    """One lap of the CSV."""
    zones = [e[1] for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and str(e[1]).startswith("zone")]
    hits = [e for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and "collision" in str(e[1]).lower()]
    return {"stage": stage, "iteration": iteration, **s, "scenario": j, "family": family, "score": round(lap_score(st), 3), "status": st.get("status"),
            "reason": st.get("reason", ""), "lap time [s]": st.get("lap time [s]"), "collisions": st.get("collisions"), "zones passed": st.get("zones passed"),
            "last zone reached": zones[-1] if zones else "", "collisions at": "; ".join(f"{e[0]:.1f} s {e[1]}" for e in hits[:6]),
            "wall per lap [s]": round(wall, 2)}


class Laps:
    def __init__(self, slug: str, jobs: int, train: list | None = None, cut: bool = True, memo: dict | None = None, rows: list | None = None,
                 log=print):
        import concurrent.futures as cf
        self.slug, self.train, self.log = slug, list(train or [NOMINAL]), log
        self.cut = cut and len(self.train) == 1
        self.m = -(-len(self.train) // 3)                          # the robust score's tail: the worst m of the K+1 laps
        self.memo, self.rows = dict(memo or {}), list(rows or [])
        self.jobs = jobs
        self.pool = cf.ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None

    @staticmethod
    def key(s, j: int = 0) -> str:
        return json.dumps([s, j], sort_keys=True)

    def seen(self, s) -> bool:
        return any(self.key(s, j) in self.memo for j in range(len(self.train)))

    def best_finished(self):
        t = [r[0]["lap time [s]"] for r in self.memo.values() if r[0].get("status") == "finished"]
        return min(t) if t else None

    def prefetch(self, settings, stage, iteration, cut=False, js=None) -> int:
        from races import validate
        js = range(len(self.train)) if js is None else js
        uniq = list({json.dumps(s, sort_keys=True): s for s in settings}.values())
        new = sum(1 for s in uniq if not self.seen(s))
        todo = [(s, j) for s in uniq for j in js if self.key(s, j) not in self.memo]
        if not todo:
            return new
        b = self.best_finished()
        duration = 1.25 * b + 5.0 if (cut and self.cut and b) else None
        jobs = [(self.slug, {**validate.apply(s, self.train[j]["mult"]), **({"duration": duration} if duration else {})}, self.train[j]["scenario"]) for s, j in todo]
        t0 = time.time()
        results = list(self.pool.map(fly_worker, jobs)) if self.pool else [fly_worker(j) for j in jobs]
        wall = (time.time() - t0) / len(todo)
        for (s, j), (st, ev) in zip(todo, results):
            if duration and st.get("status") == STUCK:
                st["reason"] = f"cut at {duration:.0f} s (slower than the best, {b:.1f} s)"
            self.memo[self.key(s, j)] = [st, ev, round(wall, 2)]
            self.rows.append(row(stage, iteration, s, st, ev, wall, j, self.train[j]["family"]))
            self.log(f"{self.slug}   {stage} it {iteration} lap {len(self.memo):4d}  {s} [{j} {self.train[j]['family']}]  → {st.get('status')} "
                     f"{st.get('lap time [s]') or ''} {st.get('zones passed') or ''}, score {lap_score(st):.2f}{(' — ' + st['reason']) if st.get('reason') else ''}")
        self.log(f"{self.slug}   batch of {len(todo)} lap(s) in {time.time() - t0:.1f} s wall ({wall:.1f} s per lap with {self.jobs} worker(s))")
        return new

    def lap(self, s, j: int = 0) -> dict:
        return self.memo[self.key(s, j)][0]

    def stats(self, s) -> dict:
        """The nominal lap's statistics (what autopilot.json and the notebooks report)."""
        return self.lap(s, 0)

    def scores(self, s, js=None) -> list:
        return [lap_score(self.lap(s, j)) for j in (range(len(self.train)) if js is None else js)]

    def score(self, s, js=None) -> float:
        from races import validate
        return validate.worst_mean(self.scores(s, js), self.m)

    def clean_rate(self, s) -> float:
        L = [self.lap(s, j) for j in range(len(self.train))]
        return sum(1 for st in L if st.get("status") == "finished" and not st.get("collisions")) / len(L)

    def close(self):
        if self.pool:
            self.pool.shutdown()
