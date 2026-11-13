"""Terrains: analytic heights, seeds, height-field sampling order, spec round trips."""
import math

import numpy as np
import pytest

from vegeta.chiron import (AlternatingBumps, CrossSlope, Custom, Flat, LongitudinalBumps, Rough, Steps,
                           terrain_from_spec)

P = 0.17


def test_flat_and_scalar_array_io():
    t = Flat()
    assert t.height(0.3, -0.2) == 0.0 and isinstance(t.height(0.3, -0.2), float)
    assert t.height(np.zeros((2, 3)), 1.0).shape == (2, 3)
    assert t.is_flat


def test_longitudinal_bumps():
    t = LongitudinalBumps(0.03, P, P / 2, start=0.3)
    xs = np.linspace(-0.5, 0.3, 50)
    assert np.all(t.height(xs, 0.0) == 0.0)                                   # flat start
    for k in range(4):                                                         # crest h at the cell centres
        xc = 0.3 + (k + 0.5) * P
        assert t.height(xc, 0.0) == pytest.approx(0.03)
        assert t.height(xc, 0.4) == pytest.approx(0.03)                        # a ridge spans all y
        assert t.height(xc + P / 4 + 1e-6, 0.0) == 0.0                         # base width P/2
    j1 = LongitudinalBumps(0.03, P, P / 2, start=0.3, jitter=P / 4, seed=1)
    j1b = LongitudinalBumps(0.03, P, P / 2, start=0.3, jitter=P / 4, seed=1)
    j2 = LongitudinalBumps(0.03, P, P / 2, start=0.3, jitter=P / 4, seed=2)
    np.testing.assert_array_equal(j1.centres, j1b.centres)
    assert not np.allclose(j1.centres, j2.centres)
    k = np.arange(len(j1.centres))
    assert np.all(np.abs(j1.centres - (0.3 + (k + 0.5) * P)) <= P / 4 + 1e-12)
    x = np.linspace(0, 3, 3001)
    assert j1.height(x, 0).max() == pytest.approx(0.03, rel=1e-3)
    assert np.all(j1.height(x[x < 0.3], 0) == 0)


def test_alternating_bumps_left_then_right():
    t = AlternatingBumps(0.04, P, P / 2, track_y=0.13, start=0.3)
    x0, x1 = 0.3 + 0.5 * P, 0.3 + 1.5 * P
    assert t.height(x0, 0.13) == pytest.approx(0.04)
    assert t.height(x0, -0.13) == 0.0
    assert t.height(x1, -0.13) == pytest.approx(0.04)
    assert t.height(x1, 0.13) == 0.0
    assert t.height(x0, 0.0) == 0.0                                            # the centreline stays clear


def test_cross_slope():
    t = CrossSlope(10.0, start=0.3, transition=0.2)
    assert t.height(0.2, 0.4) == 0.0
    assert t.height(1.0, 0.4) == pytest.approx(math.tan(math.radians(10)) * 0.4)
    assert t.height(1.0, -0.4) == pytest.approx(-math.tan(math.radians(10)) * 0.4)
    assert t.height(0.4, 0.4) == pytest.approx(0.5 * math.tan(math.radians(10)) * 0.4)


def test_steps():
    t = Steps(0.02, 2 * P, start=0.3)
    assert t.height(0.29, 0) == 0.0
    assert t.height(0.31, 0) == pytest.approx(0.02)
    assert t.height(0.3 + 2 * P + 0.01, 0) == 0.0
    assert t.height(0.3 + 4 * P + 0.01, 0) == pytest.approx(0.02)
    s = Steps(0.02, 0.2, start=0.0, mode="stairs")
    assert s.height(0.45, 0) == pytest.approx(0.06)
    j = Steps(0.02, 2 * P, start=0.3, jitter=P / 4, seed=4)
    assert j.edges[0] >= 0.3 and np.all(np.diff(j.edges) > 0)


def test_rough_statistics_seed_and_start():
    lc = 0.25 * P
    t = Rough(0.02, lc, start=0.3, seed=7, extent=(-0.5, 3.5, -1.0, 1.0))
    g = t._field()
    assert np.sqrt(np.mean(g ** 2)) == pytest.approx(0.02, rel=1e-9)
    assert abs(g.mean()) < 1e-12
    # correlation C(r) = rms^2 exp(-r^2/lc^2): at r = lc the normalised correlation is 1/e
    lag = int(round(lc / t.cell))
    rho = np.mean(g[:, :-lag] * g[:, lag:]) / np.mean(g ** 2)
    assert rho == pytest.approx(math.exp(-((lag * t.cell) / lc) ** 2), abs=0.05)
    assert np.all(t.height(np.linspace(-0.5, 0.3, 20), 0.1) == 0.0)
    same = Rough(0.02, lc, start=0.3, seed=7, extent=(-0.5, 3.5, -1.0, 1.0))
    other = Rough(0.02, lc, start=0.3, seed=8, extent=(-0.5, 3.5, -1.0, 1.0))
    x, y = np.linspace(0.5, 3, 200), np.linspace(-0.4, 0.4, 200)
    np.testing.assert_array_equal(t.height(x, y), same.height(x, y))
    assert not np.allclose(t.height(x, y), other.height(x, y))
    # bilinear between grid points is exact at the grid points
    i, j = 300, 150
    xg, yg = -0.5 + i * t.cell, -1.0 + j * t.cell
    assert t.height(xg, yg) == pytest.approx(g[j, i] * min(1.0, max(0.0, (xg - 0.3) / lc)))


def test_heightfield_order_is_rows_along_y():
    t = Custom(lambda x, y: 0.1 * x + 0.01 * y ** 2 + 0.003 * x * y)
    Z, ext = t.heightfield((-1.0, 2.0), (-0.5, 0.75), 0.05)
    assert ext == (-1.0, 2.0, -0.5, 0.75)
    nrow, ncol = Z.shape
    assert (nrow, ncol) == (26, 61)
    for r, c in ((0, 0), (0, 60), (25, 0), (7, 33)):
        x, y = -1.0 + c * 0.05, -0.5 + r * 0.05
        assert Z[r, c] == pytest.approx(t.height(x, y))


@pytest.mark.parametrize("terrain", [
    Flat(),
    LongitudinalBumps(0.02, P, P / 2, start=0.3, jitter=0.04, seed=3),
    AlternatingBumps(0.03, P, P / 2, 0.13, start=0.3, jitter=0.04, seed=3, track_width=0.1),
    CrossSlope(12.0, start=0.3, transition=0.1),
    Steps(0.02, 2 * P, start=0.3, jitter=0.04, seed=3),
    Rough(0.015, 0.04, start=0.3, seed=3, extent=(-0.5, 2.5, -0.8, 0.8)),
])
def test_spec_round_trip(terrain):
    spec = terrain.spec()
    again = terrain_from_spec(dict(spec, level=0.15))                         # bookkeeping keys are ignored
    x, y = np.meshgrid(np.linspace(-0.2, 2.2, 97), np.linspace(-0.6, 0.6, 41))
    np.testing.assert_array_equal(terrain.height(x, y), again.height(x, y))
    nested = terrain_from_spec({"kind": spec["kind"], "params": {k: v for k, v in spec.items() if k != "kind"}})
    np.testing.assert_array_equal(terrain.height(x, y), nested.height(x, y))


def test_unknown_kind():
    with pytest.raises(ValueError):
        terrain_from_spec({"kind": "lava"})
