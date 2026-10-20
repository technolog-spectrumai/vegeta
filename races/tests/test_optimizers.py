"""The tuner's strategies (races.optimizers) on a cheap stand-in objective — no MuJoCo: both improve, both repeat
for a seed, a resumed run (its state through JSON, as the checkpoint stores it) equals an uninterrupted one, and the
command line selects them."""
import argparse
import json

import pytest

from races import converge, optimizers

TARGET = {"safety": 0.7, "lookahead_s": 0.5, "omega_pos": 4.0, "omega_att": 25.0, "tilt_max_deg": 60.0, "v_cap": 12.0}


class Bowl:
    """A smooth bowl in the normalised settings, minimum 30 at TARGET; remembers what it evaluated, like Laps."""

    def __init__(self):
        self.seen, self.order = {}, []
        self.t = converge.to_x(TARGET)

    def prefetch(self, settings, stage, iteration, cut=False, js=None):
        new = 0
        for s in settings:
            k = json.dumps(s, sort_keys=True)
            if k not in self.seen:
                x = converge.to_x(s)
                self.seen[k] = 30.0 + 100.0 * sum((x[q] - self.t[q]) ** 2 for q in converge.KEYS)
                self.order.append(k)
                new += 1
        return new

    def score(self, s, js=None):
        return self.seen[json.dumps(s, sort_keys=True)]


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
    start_sc = 30.0 + 100.0 * sum((converge.to_x(START)[q] - f.t[q]) ** 2 for q in converge.KEYS)
    assert s.done and s.reason and f.score(s.best()) - 30.0 < 0.1 * (start_sc - 30.0) and s.evals_used <= CFG[name].max_evals + 20
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
