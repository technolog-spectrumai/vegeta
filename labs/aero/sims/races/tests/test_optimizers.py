"""The tuner's strategies (races.optimizers) on a cheap stand-in objective — no MuJoCo: both improve, both repeat
for a seed, a resumed run (its state through JSON, as the checkpoint stores it) equals an uninterrupted one, and the
command line selects them."""
import argparse
import json

import numpy as np

import pytest

from races import converge, optimizers

TARGET = {"safety": 0.7, "lookahead_s": 0.5, "omega_pos": 4.0, "omega_att": 25.0, "tilt_max_deg": 60.0, "v_cap": 12.0}


class Bowl:
    """A smooth bowl in the normalised settings, minimum 30 at TARGET, flown on K + 1 "scenarios" (the bowl plus a
    scenario's own bump) and scored like Laps: the mean of the worst ⌈(K+1)/3⌉; remembers what it flew."""

    def __init__(self, k=6, jobs=1):
        from races import validate
        self.train, self.m, self.jobs, self.validate = [None] * (k + 1), -(-(k + 1) // 3), jobs, validate
        self.seen, self.order, self.laps = {}, [], 0
        self.t = converge.to_x(TARGET)

    def lap_value(self, s, j):
        x = converge.to_x(s)
        bowl = 30.0 + 100.0 * sum((x[q] - self.t[q]) ** 2 for q in converge.KEYS)
        return bowl + (0.0 if j == 0 else 3.0 * (1.0 + np.sin(7.0 * x["safety"] + 3.0 * x["omega_pos"] + j)))

    def prefetch(self, settings, stage, iteration, cut=False, js=None):
        new = 0
        for s in settings:
            k = json.dumps(s, sort_keys=True)
            if k not in self.seen:
                self.seen[k] = {}
                self.order.append(k)
                new += 1
            for j in (range(len(self.train)) if js is None else js):
                if j not in self.seen[k]:
                    self.seen[k][j] = self.lap_value(s, j)
                    self.laps += 1
        return new

    def score(self, s, js=None):
        d = self.seen[json.dumps(s, sort_keys=True)]
        return self.validate.worst_mean([d[j] for j in (range(len(self.train)) if js is None else js)], self.m)


START = {"safety": 0.4, "lookahead_s": 0.3, "omega_pos": 2.5, "omega_att": 18.0, "tilt_max_deg": 50.0, "v_cap": 8.0}
CFG = {"cma": optimizers.CMAConfig(seed=3, max_evals=200, generations=15),
       "pso": optimizers.PSOConfig(seed=3, max_evals=200, particles=6, pso_iterations=4, polish_evals=60, delta_min=1 / 32)}


def run(name, stop_after=None):
    cls, cfg, f = optimizers.STRATEGIES[name], CFG[name], Bowl()
    s, n = cls(START, cfg), 0
    while not s.done:
        s.step(f, log=lambda m: None)
        n += 1
        if n == stop_after:                                        # interrupt: rebuild from the saved state
            s = cls.restore(START, cfg, json.loads(json.dumps(s.state())))
    return s, f


@pytest.mark.parametrize("name", list(optimizers.STRATEGIES))
def test_strategy_improves_and_stops(name):
    s, f = run(name)
    f.prefetch([START], "start", 0)
    start_sc = f.score(START)
    assert s.done and s.reason and start_sc - f.score(s.best()) > 0.7 * (start_sc - 30.0) and s.evals_used <= CFG[name].max_evals + 20
    assert "global optimum" not in s.reason or "not" in s.reason


@pytest.mark.parametrize("name", list(optimizers.STRATEGIES))
def test_strategy_repeats_and_resumes_exactly(name):
    a, fa = run(name)
    b, fb = run(name)
    c, fc = run(name, stop_after=2)
    assert fa.order == fb.order == fc.order and a.best() == b.best() == c.best()


def test_command_line_selects_the_strategy():
    ap = argparse.ArgumentParser()
    optimizers.add_arguments(ap)
    cls, cfg = optimizers.config(ap.parse_args([]))
    assert cls is optimizers.CMAStrategy and cfg == optimizers.CMAConfig()
    cls, cfg = optimizers.config(ap.parse_args(["--pso", "--particles", "8", "--max-evals", "50"]))
    assert cls is optimizers.PSOStrategy and (cfg.particles, cfg.max_evals) == (8, 50)
    with pytest.raises(SystemExit):
        ap.parse_args(["--pso", "--cma"])


@pytest.mark.parametrize("jobs", [1, 3, 8])
def test_racing_finds_the_exact_best_and_saves_laps(jobs):
    rng = np.random.default_rng(5)
    S = [converge.to_settings({q: float(v) for q, v in zip(converge.KEYS, rng.random(6))}) for _ in range(12)]
    full = Bowl()
    full.prefetch(S, "all", 0)
    true = [full.score(s) for s in S]
    f = Bowl(jobs=jobs)
    F, new = optimizers.race(f, S, keep=4, stage="cma", iteration=0, log=lambda m: None)
    best = sorted(range(12), key=lambda i: true[i])[:4]
    assert new == 12 and all(F[i] == true[i] for i in best)                       # the best μ: exact
    assert min(F[i] for i in range(12) if i not in best) > sorted(true)[3]         # the others rank after them
    assert f.laps < full.laps
    ref = Bowl(jobs=1)
    assert optimizers.race(ref, S, 4, "cma", 0, log=lambda m: None)[0] == F         # the scores do not depend on -j


def test_racing_leaves_the_cma_es_run_unchanged(monkeypatch):
    cfg = optimizers.CMAConfig(seed=4, max_evals=400, generations=12, tolfun=1e-12, tolx=1e-12)

    def trajectory():
        f, s = Bowl(jobs=4), optimizers.CMAStrategy(START, cfg)
        means = []
        while not s.done:
            s.step(f, log=lambda m: None)
            means.append([round(v, 12) for v in s.es.mean])
        return means, s.best(), f.laps

    raced, best_r, laps_r = trajectory()
    monkeypatch.setattr(optimizers, "race", lambda o, S, keep, *a, **k: (o.prefetch(S, "cma", 0) and None) or ([o.score(s) for s in S], 0))
    full, best_f, laps_f = trajectory()
    assert raced == full and best_r == best_f and laps_r < laps_f
