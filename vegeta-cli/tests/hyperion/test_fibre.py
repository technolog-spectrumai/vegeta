"""Fibres against the handbook (Bessel values, V and cutoff, Marcuse's mode radius, SMF-28's MFD, bend-loss behaviour)
and the power-evolution solver against its invariants (Rigrod's product, threshold, the quantum limit, energy
conservation, saturation, the optimum length)."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy
from vegeta.hyperion import fibre as fb


def test_bessel_by_quadrature():
    assert float(fb.bessel_j(0, 1.0)) == pytest.approx(0.7651976866, abs=1e-9)
    assert float(fb.bessel_j(1, 1.0)) == pytest.approx(0.4400505857, abs=1e-9)
    assert float(fb.bessel_j(0, fb.LP01_CUTOFF)) == pytest.approx(0.0, abs=1e-6)
    assert float(fb.bessel_k(0, 1.0)) == pytest.approx(0.4210244382, abs=1e-9)
    assert float(fb.bessel_k(1, 1.0)) == pytest.approx(0.6019072302, abs=1e-9)
    assert fb.bessel_j(1, np.array([0.5, 2.0])).shape == (2,)


def test_fibre_geometry_against_the_handbook():
    smf = hy.Fibre(4.1e-6, 0.12, 62.5e-6, name="SMF-28-like")
    assert smf.V(1550e-9) == pytest.approx(2 * math.pi * 4.1e-6 * 0.12 / 1550e-9)
    assert smf.single_mode(1550e-9) and not smf.single_mode(1200e-9)
    assert smf.cutoff_wavelength == pytest.approx(2 * math.pi * 4.1e-6 * 0.12 / 2.404826)
    assert 2 * smf.mode_radius(1550e-9) == pytest.approx(10.4e-6, abs=0.2e-6)       # the datasheet's MFD
    # Marcuse's fit at the cutoff: w/a = 1.10
    f = hy.Fibre(5e-6, 0.075, 62.5e-6)
    lam_c = f.cutoff_wavelength
    assert f.mode_radius(lam_c) / 5e-6 == pytest.approx(1.10, abs=0.01)
    assert f.overlap_clad == pytest.approx((5 / 62.5) ** 2)
    assert 0.7 < f.overlap_core(1070e-9) < 0.85
    b = f.beam(1070e-9)
    assert isinstance(b, hy.GaussianBeam) and b.w0 == f.mode_radius(1070e-9) and b.z0 == 0.0
    m = f.lp01(1070e-9)
    assert m["u"] ** 2 + m["w"] ** 2 == pytest.approx(m["V"] ** 2)
    assert 0 < m["u"] < fb.LP01_CUTOFF and m["beta"] < 2 * math.pi / 1070e-9 * m["n1"]
    assert hy.Fibre(**{k: v for k, v in f.to_dict().items()}) == f


def test_bend_loss_behaves_like_the_catalogue_curves():
    smf = hy.Fibre(4.1e-6, 0.12, 62.5e-6)
    r = np.array([0.008, 0.010, 0.016, 0.03])
    loss = smf.bend_loss_db_per_m(r, 1550e-9)
    assert np.all(np.diff(loss) < 0)                                               # less loss at a larger radius
    assert 5 < loss[1] < 100 and loss[3] < 1e-3                                     # tens of dB/m at R = 10 mm, nothing at 30 mm
    assert np.log(loss[0] / loss[2]) > 2                                           # exponential in R
    assert float(smf.bend_loss_db_per_m(0.01, 1550e-9, stress_factor=1.0)) > float(smf.bend_loss_db_per_m(0.01, 1550e-9))
    assert smf.bend_loss_db_per_m(0.01, 1310e-9) < smf.bend_loss_db_per_m(0.01, 1550e-9)   # the shorter wavelength is better confined


@pytest.fixture
def laser():
    f = hy.Fibre(5e-6, 0.075, 62.5e-6, 125e-6, alpha_signal=2.3e-3, alpha_pump=2.3e-3)
    yb = fb.YB_SILICA.with_(N=fb.ions_for_absorption(f, 1.7, 976e-9))
    return hy.FibreLaser(hy.ActiveFibre(f, yb, 8.0), 976e-9, 1070e-9, R_hr=0.99, R_oc=0.10)


def test_dopant_and_absorption(laser):
    act = laser.active
    assert act.absorption_db_per_m(976e-9) == pytest.approx(1.7)
    assert act.absorption_db_per_m(915e-9) < act.absorption_db_per_m(976e-9)
    assert act.absorbed_fraction(976e-9) == pytest.approx(1 - 10 ** (-1.7 * 8 / 10))
    sa, se = fb.YB_SILICA.cross_sections(976e-9)
    assert sa == pytest.approx(2.5e-24) and se == pytest.approx(2.5e-24)
    sa, se = fb.YB_SILICA.cross_sections(1070e-9)
    assert se / sa > 20                                                            # the signal line: emission wins
    with pytest.raises(ValueError):
        fb.YB_SILICA.cross_sections(1300e-9)
    assert hy.Dopant(**{k: (tuple(tuple(r) for r in v) if k == "table" else v) for k, v in fb.YB_SILICA.to_dict().items()}).N == fb.YB_SILICA.N


def test_oscillator_invariants(laser):
    s = laser.solve(20.0, n=200)
    assert s.output_W > 10 and s.note == ""
    prod = s.Ps_fwd * s.Ps_bwd
    assert prod == pytest.approx(np.full_like(prod, prod[0]), rel=1e-6)            # Rigrod: P⁺P⁻ is constant along the fibre
    assert s.Ps_bwd[-1] == pytest.approx(laser.R_oc * s.Ps_fwd[-1], rel=1e-6)       # the coupler's boundary
    assert s.Ps_fwd[0] == pytest.approx(laser.R_hr * s.Ps_bwd[0], rel=1e-9)         # the HR's
    assert s.output_W == pytest.approx((1 - laser.R_oc) * s.Ps_fwd[-1])
    assert s.absorbed_W + s.unabsorbed_W == pytest.approx(20.0)
    assert s.output_W <= s.absorbed_W * 976 / 1070 + 1e-9                           # the quantum limit
    assert s.heat_W > 0 and s.heat_W < s.absorbed_W - s.output_W + 0.1 * s.absorbed_W
    assert np.all((s.inversion >= 0) & (s.inversion <= 1)) and s.inversion[0] > s.inversion[-1]   # pumped from z = 0
    th = laser.threshold(n=150)
    assert 0.05 < th < 1.0
    assert laser.solve(0.8 * th, n=150).output_W == 0.0 and laser.solve(1.5 * th, n=150).output_W > 0
    pc = laser.power_curve([0.0, 5.0, 10.0, 20.0], n=150)
    assert 0.7 < pc["slope"] < 976 / 1070 and pc["output_W"][0] == 0.0
    assert pc["output_W"][3] == pytest.approx(s.output_W, rel=0.02)                 # n = 150 against n = 200
    assert np.all(np.diff(pc["output_W"]) > 0)


def test_optimum_length_and_signal_reabsorption(laser):
    sw = laser.length_sweep([2.0, 5.0, 10.0, 20.0], 20.0, n=120)
    assert sw["rows"][0]["unabsorbed_W"] > sw["rows"][-1]["unabsorbed_W"]
    assert sw["best_length"] in (10.0, 20.0)
    # at 1030 nm the signal is reabsorbed where the pump is gone: a long fibre loses, the optimum is short
    l1030 = hy.FibreLaser(laser.active, 976e-9, 1030e-9, 0.99, 0.10)
    sw2 = l1030.length_sweep([2.0, 5.0, 10.0, 20.0], 20.0, n=120)
    outs = [r["output_W"] for r in sw2["rows"]]
    assert sw2["best_length"] < 20.0 and outs[-1] < max(outs)


def test_amplifier_energy_and_saturation():
    f = hy.Fibre(10e-6, 0.06, 62.5e-6, 125e-6, alpha_signal=2.3e-3, alpha_pump=2.3e-3)
    yb = fb.YB_SILICA.with_(N=fb.ions_for_absorption(f, 5.0, 976e-9))
    act = hy.ActiveFibre(f, yb, 5.0)
    counter = hy.FibreAmplifier(act, 976e-9, 1070e-9, "counter")
    co = hy.FibreAmplifier(act, 976e-9, 1070e-9, "co")
    s = counter.solve(15.0, 40.0, n=200)
    assert s.output_W > 15.0 and s.output_W <= 15.0 + s.absorbed_W * 976 / 1070 + 1e-9
    assert s.Pp[-1] == pytest.approx(40.0, rel=1e-6) and s.Pp[0] == pytest.approx(s.unabsorbed_W)   # counter: launched at z = L
    c = co.solve(15.0, 40.0, n=200)
    assert c.Pp[0] == 40.0 and c.Pp[-1] == pytest.approx(c.unabsorbed_W)
    assert c.output_W == pytest.approx(s.output_W, rel=0.05)                        # the same energy, a different inversion profile
    assert c.inversion.max() > s.inversion.max()                                    # co-pumping: the strongest inversion at the weak-signal end
    weak, strong = counter.solve(0.5, 40.0, n=200), counter.solve(30.0, 40.0, n=200)
    assert weak.gain_db > s.gain_db > strong.gain_db                                # saturation
    assert s.gain_db == pytest.approx(10 * math.log10(s.output_W / 15.0))


def test_limits():
    f = hy.Fibre(10e-6, 0.06, 62.5e-6, 125e-6, alpha_signal=2.3e-3)
    srs = hy.srs_threshold(f, 5.0, 1070e-9)
    assert srs["P_threshold_W"] == pytest.approx(16 * f.mode_area(1070e-9) / (1e-13 * srs["L_eff"]))
    assert srs["L_eff"] < 5.0 and srs["stokes_wavelength"] > 1070e-9
    t = hy.core_temperature(f, 3.0, h=50.0)
    assert t["dT_core_to_clad"] == pytest.approx(3.0 * (1 + 2 * math.log(62.5 / 10)) / (4 * math.pi * 1.38))
    assert t["dT_surface_to_sink"] == pytest.approx(3.0 / (2 * math.pi * 125e-6 * 50))
    assert t["dT_core_to_sink"] == pytest.approx(t["dT_core_to_clad"] + t["dT_coating"] + t["dT_surface_to_sink"])
    assert hy.core_temperature(f, 3.0, h=1000.0)["dT_core_to_sink"] < t["dT_core_to_sink"]
