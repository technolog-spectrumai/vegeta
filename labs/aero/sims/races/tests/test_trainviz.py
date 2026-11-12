"""The training-history figures (races.trainviz) on a synthetic checkpoint and an old single-lap CSV — no MuJoCo."""
import csv
import json

import numpy as np
import pytest

from races import converge, trainviz, validate


def _lap(stage, it, s, j, score):
    return {"stage": stage, "iteration": it, **s, "scenario": j, "family": "x", "score": score, "status": "finished"}


def _ckpt(tmp_path, k=2):
    rng = np.random.default_rng(1)
    rows, hist = [], []
    for g in range(4):
        mean = np.full(6, 0.5) + 0.05 * g
        for n in range(5):
            x = np.clip(mean + 0.2 / (g + 1) * rng.standard_normal(6), 0, 1)
            s = converge.to_settings(dict(zip(converge.KEYS, x)))
            js = range(k + 1) if n < 3 else range(1)                       # two of five dropped by racing after one lap
            rows += [_lap("cma", g, s, j, 30 + 10 * float(np.sum((x - 0.7) ** 2)) + j) for j in js]
        hist.append({"gen": g, "mean": mean.tolist(), "sigma": 0.2 / (g + 1), "C": np.eye(6).tolist(), "best": 0, "median": 0})
    rows.append(_lap("start", 0, converge.to_settings(converge.to_x({})), 0, 99.0))
    f = tmp_path / "farm.ckpt.json"
    f.write_text(json.dumps({"fingerprint": {"strategy": "cma", "train_k": k}, "start": {}, "rows": rows, "strategy": {"history": hist}}))
    return f


def test_load_groups_laps_into_candidates(tmp_path):
    h = trainviz.load("farm", _ckpt(tmp_path))
    assert h.strategy == "cma" and h.k1 == 3 and len(h.steps) == 4 and len(h.candidates) == 20 and len(h.cma) == 4
    assert sum(not c.exact for c in h.candidates) == 8 and all(c.laps == 1 for c in h.candidates if not c.exact)
    c = next(c for c in h.candidates if c.exact)
    laps = [30 + 10 * float(np.sum((c.x - 0.7) ** 2)) + j for j in range(3)]
    assert c.score == pytest.approx(validate.worst_mean(laps, 1), abs=0.05)             # the worst of 3 (m = 1), as the tuner scored it
    best = [b for _, b in h.best_so_far]
    assert best == sorted(best, reverse=True)                                  # the best so far never rises


def test_pca_and_projected_ellipse():
    X = np.random.default_rng(2).random((40, 6))
    c, W, share = trainviz.pca(X)
    assert np.allclose(W.T @ W, np.eye(2)) and share.sum() == pytest.approx(1.0) and share[0] >= share[1]
    E = trainviz.ellipse(c, 0.1, np.eye(6), c, W)                             # isotropic: a circle of radius 2σ about the origin
    assert np.allclose(np.linalg.norm(E, axis=1), 0.2)


def test_figures_from_a_checkpoint_and_an_old_csv(tmp_path):
    files = trainviz.figures(trainviz.load("farm", _ckpt(tmp_path)), tmp_path / "train_hist", animate=True, dpi=40)
    assert [f.name for f in files] == ["farm_train_history.png", "farm_params.png", "farm_search.gif"] and all(f.stat().st_size > 0 for f in files)
    old = tmp_path / "quarry_converge.csv"                                    # the earlier format: one lap per candidate, no scenario column
    with old.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "iteration", *converge.KEYS, "score", "status"])
        w.writeheader()
        for it in range(3):
            for n in range(4):
                s = converge.to_settings(dict(zip(converge.KEYS, np.full(6, 0.3 + 0.1 * n + 0.01 * it))))
                w.writerow({"stage": "pso", "iteration": it, **{k: ("" if v is None else v) for k, v in s.items()}, "score": 40 - it - n, "status": "finished"})
    h = trainviz.load("quarry", old)
    assert h.k1 == 1 and h.strategy == "pso" and all(c.exact for c in h.candidates)
    assert len(trainviz.figures(h, tmp_path / "train_hist", dpi=40)) == 2 and "quarry" in trainviz.summary(h)
