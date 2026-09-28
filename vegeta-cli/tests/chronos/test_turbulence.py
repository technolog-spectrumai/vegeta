"""Dryden turbulence: intensities and scales, series rms and spectrum, gust load factor, extremes, frozen field, mount."""
import math

import numpy as np
import pytest

from vegeta import chronos
from vegeta.chronos.turbulence import FT, KT, mass_ratio, upcrossing_rate


def test_low_altitude_dryden_parameters():
    t = chronos.Turbulence(120.0, 10.0)
    h = 120.0 / FT
    assert t.sigma_w == pytest.approx(1.0)
    assert t.sigma_u == pytest.approx(1.0 / (0.177 + 0.000823 * h) ** 0.4) and t.sigma_v == t.sigma_u
    assert t.length_w == pytest.approx(120.0)
    assert t.length_u == pytest.approx(h / (0.177 + 0.000823 * h) ** 1.2 * FT)
    assert t.sigma_u > t.sigma_w and t.length_u > t.length_w                     # horizontal gusts: stronger, longer near the ground
    m = chronos.Turbulence.from_severity("moderate", 120.0)
    assert m.wind_20ft_m_s == pytest.approx(30 * KT)
    assert chronos.Turbulence.from_severity("severe", 120.0).sigma_w == pytest.approx(1.5 * m.sigma_w)
    with pytest.raises(ValueError):
        chronos.Turbulence.from_severity("extreme", 120.0)


@pytest.mark.parametrize("comp", ["u", "v", "w"])
def test_spectrum_integrates_to_the_variance(comp):
    t = chronos.Turbulence(100.0, 12.0)
    f = np.linspace(0, 2000, 2_000_001)
    assert np.trapezoid(t.spectrum(comp, f, 30.0), f) == pytest.approx(t.sigma(comp) ** 2, rel=0.01)


def test_series_rms_spectrum_and_seed():
    t = chronos.Turbulence.from_severity("moderate", 120.0)
    g = chronos.gust_series(t, 35.0, 3600.0, dt=0.02, seed=3)
    rms = g.rms()
    for k in "uvw":
        assert rms[k] == pytest.approx(t.sigma(k), rel=0.1)
    f, S = g.spectrum("w", segment_s=120.0)
    for fk in (0.1, 1.0):                                                        # the Dryden form, within the Welch scatter
        i = np.argmin(abs(f - fk))
        band = slice(max(i - 5, 1), i + 6)
        assert S[band].mean() == pytest.approx(t.spectrum("w", f[band], 35.0).mean(), rel=0.3)
    again = chronos.gust_series(t, 35.0, 3600.0, dt=0.02, seed=3)
    other = chronos.gust_series(t, 35.0, 3600.0, dt=0.02, seed=4)
    assert np.array_equal(g.w, again.w) and not np.array_equal(g.w, other.w)
    assert abs(np.corrcoef(g.u, g.w)[0, 1]) < 0.1                                # independent components


def test_gust_load_factor_and_alleviation():
    # sharp-edged gust: dn = rho V a w / (2 W/S)
    n = chronos.load_factor([0.0, 2.0], 35.0, 2.5, 80.0)
    assert n[0] == pytest.approx(1.0) and n[1] - 1 == pytest.approx(1.225 * 35 * 2.5 * 2.0 / 160.0)
    mu = mass_ratio(80.0, 0.6, 2.5)
    assert mu == pytest.approx(2 * 80 / (1.225 * 0.6 * 2.5 * 9.81))
    kg = chronos.alleviation_factor(mu)
    assert 0 < kg < 0.88 and kg == pytest.approx(0.88 * mu / (5.3 + mu))
    assert chronos.load_factor([2.0], 35.0, 2.5, 80.0, mean_chord_m=0.6)[0] - 1 == pytest.approx(kg * (n[1] - 1))


def test_rice_extreme_of_a_gaussian_series():
    rng = np.random.default_rng(0)
    x = np.convolve(rng.standard_normal(200_000), np.ones(20) / math.sqrt(20), mode="same")     # smooth, unit-ish variance
    dt = 0.01
    nu = upcrossing_rate(x, dt)
    assert nu > 0
    e1 = chronos.rice_extreme(x, dt)
    assert e1 == pytest.approx(np.std(x) * math.sqrt(2 * math.log(nu * len(x) * dt)))
    assert 0.8 * e1 < x.max() - x.mean() < 1.25 * e1
    assert chronos.rice_extreme(x, dt, exposures=10) > e1


def test_frozen_field_and_mount():
    t = chronos.Turbulence.from_severity("moderate", 120.0)
    x, z, U, W = chronos.frozen_field(t, 4000.0, np.linspace(60, 180, 7), dx=2.0, seed=1)
    assert U.shape == W.shape == (7, len(x)) and x[1] - x[0] == pytest.approx(2.0)
    assert np.std(W) == pytest.approx(t.sigma_w, rel=0.3)
    c_near = np.corrcoef(W[0], W[1])[0, 1]; c_far = np.corrcoef(W[0], W[-1])[0, 1]
    assert c_near > c_far and c_near > 0.5
    # the mount: a low frequency passes, a high one is isolated, resonance amplifies
    tt = np.arange(0, 20, 0.001)
    for f, lo, hi in ((1.0, 0.98, 1.02), (25.0, 4.0, 6.0), (200.0, 0.0, 0.05)):
        out = chronos.through_mount(tt, np.sin(2 * math.pi * f * tt), 25.0, 0.1)
        assert lo < np.max(np.abs(out[len(tt) // 2:])) < hi
