"""D3Q19 velocity set: the moment identities the second-order equilibrium relies on."""
import itertools

import numpy as np

from mancha import D3Q19


def test_shape_and_rest_first():
    lat = D3Q19()
    assert lat.c.shape == (19, 3) and lat.w.shape == (19,)
    assert not lat.c[0].any()
    assert len({tuple(v) for v in lat.c}) == 19


def test_opposites():
    lat = D3Q19()
    assert (lat.c[lat.opposite] == -lat.c).all()
    assert (lat.opposite[lat.opposite] == np.arange(19)).all()


def test_weight_moments_isotropic_to_fourth_order():
    lat = D3Q19()
    c, w, cs2 = lat.c.astype(float), lat.w, lat.cs2
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w @ c, 0.0)
    assert np.allclose(np.einsum("i,ia,ib->ab", w, c, c), cs2 * np.eye(3))
    assert np.allclose(np.einsum("i,ia,ib,ic->abc", w, c, c, c), 0.0)
    m4 = np.einsum("i,ia,ib,ic,id->abcd", w, c, c, c, c)
    d = np.eye(3)
    for a, b, g, h in itertools.product(range(3), repeat=4):
        expected = cs2 ** 2 * (d[a, b] * d[g, h] + d[a, g] * d[b, h] + d[a, h] * d[b, g])
        assert np.isclose(m4[a, b, g, h], expected)
