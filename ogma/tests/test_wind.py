"""og.wind.Wind reproduces Vegeta's nisus_robot.Wind sample for sample (golden/wind.json); the OU statistics hold."""
import json
from pathlib import Path

import numpy as np

from ogma.wind import Wind

G = json.loads((Path(__file__).with_name("golden") / "wind.json").read_text())


def test_sequence_is_vegetas():
    w = Wind(steady=(3.0, -1.0, 0.0), sigma=1.5, tau=4.0, gust=(4.0, 2.0, 1.5, (0.0, 1.0, 0.0)), seed=7)
    w.reset()
    seq = []
    for k in range(400):
        w.step(0.01)
        seq.append(w.at(k * 0.01))
    assert np.allclose(seq, G["seq"], rtol=0, atol=1e-15)


def test_turbulence_statistics():
    w = Wind(sigma=2.0, tau=1.0, seed=1)
    w.reset()
    xs = []
    for _ in range(200000):
        w.step(0.05)
        xs.append(w.turb.copy())
    xs = np.array(xs)
    assert np.allclose(xs.std(0), [2.0, 2.0, 1.0], rtol=0.05)
