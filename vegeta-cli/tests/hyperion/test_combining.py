"""Beam combining: power accounting (Parseval, losses, the coherent cross term), coherent interference (N² on axis,
fringes at λ/d, the π null), spectral separation, the limiting cases and the beam metrics."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy
from vegeta.hyperion import combining as cb

LAM, W0 = 1.07e-6, 25e-6


def _chans(n, layout="ring", **kw):
    return cb.channels(n, LAM, 1.0, W0, layout=layout, **kw)


def test_far_field_parseval_and_gaussian_divergence():
    ch = _chans(1)
    r = cb.IncoherentCombiner().combine(ch)
    assert r.conservation < 1e-12 and r.overlap == pytest.approx(0.0, abs=1e-12)
    assert r.power_near == pytest.approx(1.0, rel=1e-6)                     # the Gaussian's power on the grid
    assert r.sigma_x == pytest.approx(W0 / 2, rel=1e-3)                     # second moment of a Gaussian: σ = w0/2
    assert r.sigma_tx == pytest.approx(LAM / (2 * math.pi * W0), rel=1e-3)   # θ = λ/(π w0), σ_θ = θ/2
    assert r.m2_x == pytest.approx(1.0, abs=1e-3) and r.m2_y == pytest.approx(1.0, abs=1e-3)
    assert r.bpp_x == pytest.approx(LAM / (4 * math.pi), rel=1e-3)


def test_limiting_cases_single_channel_identical_across_methods():
    ch = _chans(1)
    ri = cb.IncoherentCombiner().combine(ch); rc = cb.CoherentCombiner().combine(ch)
    rs = cb.SpectralCombiner(efficiency_per_channel=1.0).combine(ch)
    for r in (rc, rs):
        assert np.allclose(r.I_near, ri.I_near) and np.allclose(r.I_far, ri.I_far)
        assert r.power_out == pytest.approx(1.0) and r.efficiency == pytest.approx(1.0, abs=1e-9)
    assert rc.extra["axis_gain"] == pytest.approx(1.0) and rc.extra["aperture"] == "filled"


@pytest.mark.parametrize("n", [2, 4, 8])
def test_incoherent_adds_power_and_etendue(n):
    r = cb.IncoherentCombiner().combine(_chans(n))
    assert r.power_out == pytest.approx(n) and r.power_near == pytest.approx(n, rel=1e-6) and r.conservation < 1e-12
    assert r.overlap == pytest.approx(0.0, abs=1e-9)                        # intensities add: no cross term
    assert r.sigma_tx == pytest.approx(LAM / (2 * math.pi * W0), rel=1e-3)   # the far field is each channel's
    assert r.m2_x > 2.0 and r.efficiency == pytest.approx(1.0)              # the near field is wider: the étendue grows
    one = cb.IncoherentCombiner().combine(_chans(1))
    assert r.brightness < one.brightness                                     # more power, less brightness


def test_losses_and_throughput_reduce_power_only():
    ch = _chans(4, losses=[0.1, 0.0, 0.2, 0.0])
    r = cb.IncoherentCombiner(throughput=0.9).combine(ch)
    assert r.power_in == pytest.approx(4.0) and r.power_out == pytest.approx(3.7 * 0.9) and r.power_near == pytest.approx(3.7 * 0.9, rel=1e-6)
    assert r.efficiency == pytest.approx(3.7 * 0.9 / 4)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_coherent_filled_in_phase_n_times_single_on_axis_and_power_conserved(n):
    r = cb.CoherentCombiner().combine(_chans(n, layout="overlap"))
    assert r.extra["aperture"] == "filled" and r.power_out == pytest.approx(n) and r.conservation < 1e-12 and r.overlap == pytest.approx(0.0, abs=1e-9)
    assert r.extra["axis_vs_single"] == pytest.approx(n, rel=1e-6)          # the output is one Gaussian of N × the power
    assert r.m2_x == pytest.approx(1.0, abs=1e-3) and r.efficiency == pytest.approx(1.0, abs=1e-9)
    one = cb.CoherentCombiner().combine(_chans(1))
    assert r.brightness == pytest.approx(n * one.brightness, rel=1e-6)      # brightness scales with the power: the étendue is one channel's


def test_coherent_filled_phase_error_goes_to_the_other_port():
    r = cb.CoherentCombiner().combine(_chans(2, layout="overlap", phases=[0.0, math.pi]))
    assert r.power_out == pytest.approx(0.0, abs=1e-9) and r.extra["port_loss"] == pytest.approx(2.0)
    r = cb.CoherentCombiner().combine(_chans(2, layout="overlap", phases=[0.0, math.pi / 2]))
    assert r.power_out == pytest.approx(1.0, rel=1e-6)                     # cos²(Δφ/2) = ½
    assert r.extra["axis_gain"] == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_coherent_tiled_n_squared_on_axis_narrower_lobe_power_kept(n):
    r = cb.CoherentCombiner().combine(_chans(n))
    assert r.extra["aperture"] == "tiled" and r.power_out == pytest.approx(n) and r.conservation < 1e-12
    assert abs(r.overlap) < 0.01                                             # neighbours 3.5 w0 apart: the cross term is 0.4 %
    assert r.extra["axis_vs_single"] == pytest.approx(n * n, rel=1e-3)      # the array factor
    assert r.extra["axis_gain"] == pytest.approx(1.0, rel=1e-9)
    assert 0 < r.efficiency < 1                                              # side lobes carry the rest
    inc = cb.IncoherentCombiner().combine(_chans(n))
    assert r.I_far_axis > inc.I_far_axis * (n - 0.5)                        # N² against N


def test_two_beam_interference_fringes_and_null():
    d = 4 * W0
    ch = cb.channels(2, LAM, 1.0, W0, pitch=d, layout="line")
    r = cb.CoherentCombiner().combine(ch)
    one = cb.IncoherentCombiner().combine(_chans(1), r.grid)
    th, I = r.cross_section("x", far=True)
    _, I1 = one.cross_section("x", far=True)
    assert np.allclose(I, 4 * I1 * np.cos(math.pi * d * th / LAM) ** 2, rtol=1e-6, atol=1e-9 * I.max())   # Young: the envelope × cos²(π d θ/λ)
    nulls = th[np.abs(np.cos(math.pi * d * th / LAM)) < 0.05]
    assert len(nulls) == 0 or np.all(I[np.abs(np.cos(math.pi * d * th / LAM)) < 0.05] < 0.01 * I.max())
    r_pi = cb.CoherentCombiner().combine(cb.channels(2, LAM, 1.0, W0, pitch=d, layout="line", phases=[0.0, math.pi]))
    assert r_pi.I_far_axis < 1e-6 * r.I_far_axis and r_pi.power_out == pytest.approx(2.0)   # the null on axis, the power in the side lobes


def test_coherent_raises_on_two_wavelengths_and_spectral_on_close_ones():
    ch = cb.channels(2, LAM, 1.0, W0, layout="overlap", wavelengths=[1.07e-6, 1.075e-6])
    with pytest.raises(ValueError):
        cb.CoherentCombiner().combine(ch)
    with pytest.raises(ValueError):
        cb.SpectralCombiner(min_separation=10e-9).combine(ch)


def test_spectral_combining_keeps_the_etendue_of_one_channel():
    lams = [1.060e-6, 1.065e-6, 1.070e-6, 1.075e-6]
    ch = cb.channels(4, LAM, 1.0, W0, wavelengths=lams)
    r = cb.SpectralCombiner(efficiency_per_channel=0.95).combine(ch)
    assert r.power_out == pytest.approx(4 * 0.95) and r.power_near == pytest.approx(4 * 0.95, rel=1e-6) and r.conservation < 1e-6
    assert math.isnan(r.m2_x) and len(set(r.wavelengths)) == 4            # no M² across wavelengths
    assert r.sigma_x == pytest.approx(W0 / 2, rel=1e-3)
    assert r.sigma_tx == pytest.approx(np.mean(lams) / (2 * math.pi * W0), rel=0.01)   # the far field is one channel's
    one = cb.IncoherentCombiner().combine(_chans(1))
    assert r.brightness == pytest.approx(4 * 0.95 * one.brightness, rel=0.02)
    r_d = cb.SpectralCombiner(efficiency_per_channel=0.95, dispersion=20e-3 / 1e-9 * 1e-3).combine(ch)   # 20 µrad per nm residual
    assert r_d.sigma_tx > r.sigma_tx and r_d.power_out == pytest.approx(r.power_out)


def test_equivalent_gaussian_and_phase_sensitivity():
    ch = _chans(4, layout="overlap")
    r = cb.CoherentCombiner().combine(ch)
    g = cb.equivalent_gaussian(4.0, LAM, W0, r.grid)
    assert np.allclose(g.I_far, r.I_far, rtol=1e-6, atol=1e-6 * g.I_far_axis)
    ps = cb.phase_sensitivity(ch, cb.CoherentCombiner(), [0.0, 0.5, 1.0], trials=40, seed=1)
    assert ps["axis_gain"][0] == pytest.approx(1.0) and ps["axis_gain"][2] < ps["axis_gain"][1] < 1.0
    assert ps["axis_gain"][1] == pytest.approx(ps["closed_form"][1], abs=0.05)


def test_m2_channels_widen_the_far_field_only_and_coherent_refuses_them():
    ch = cb.channels(1, LAM, 1.0, W0, m2=2.0)
    r = cb.IncoherentCombiner().combine(ch)
    assert r.sigma_x == pytest.approx(W0 / 2, rel=1e-3) and r.sigma_tx == pytest.approx(2 * LAM / (2 * math.pi * W0), rel=0.02)
    assert r.m2_x == pytest.approx(2.0, rel=0.02) and r.power_far == pytest.approx(1.0, rel=0.01)
    one = cb.IncoherentCombiner().combine(_chans(1))
    assert r.brightness == pytest.approx(one.brightness / 4, rel=0.05)                # the same power in M⁴ times the étendue
    rs = cb.SpectralCombiner(efficiency_per_channel=1.0).combine(cb.channels(2, LAM, 1.0, W0, m2=2.0, wavelengths=[1.065e-6, 1.075e-6]))
    assert rs.sigma_tx == pytest.approx(r.sigma_tx, rel=0.02) and rs.power_out == pytest.approx(2.0)
    with pytest.raises(ValueError):
        cb.CoherentCombiner().combine(ch)


def test_exports_and_table():
    assert hy.CoherentCombiner is cb.CoherentCombiner and hy.channels is cb.channels
    t = cb.power_table({"a": cb.IncoherentCombiner().combine(_chans(2))})
    assert t["a"]["power out [W]"] == pytest.approx(2.0) and "brightness [W/(cm²·sr)]" in t["a"]
