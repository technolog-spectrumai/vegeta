"""The four-level laser against the closed forms and Koechner's orders of magnitude; the thermal lens scaling."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy


@pytest.fixture
def laser():
    rod = hy.Rod(hy.ND_YAG, 1.5e-3, 0.01)
    return hy.Laser(rod, w_pump=200e-6, w_mode=200e-6, T=0.05, loss=0.01)


def test_medium_numbers():
    m = hy.ND_YAG
    assert m.quantum_defect == pytest.approx(1 - 808 / 1064)
    assert m.saturation_intensity == pytest.approx(6.626e-34 * 2.998e8 / 1064e-9 / (2.8e-23 * 230e-6), rel=1e-3)   # ≈ 2.9 kW/cm²
    assert m.saturation_intensity / 1e4 == pytest.approx(2.9e3, rel=0.03)
    assert hy.medium("nd:yag") is m and hy.medium(m) is m
    rod = hy.Rod(m, 1.5e-3, 0.01)
    assert rod.absorbed_fraction == pytest.approx(1 - math.exp(-4.0))
    assert rod.absorption_length == pytest.approx(2.5e-3)
    with pytest.raises(ValueError):
        hy.medium("ruby")


def test_threshold_slope_and_output(laser):
    """P_th = π hν_p (w_p² + w_l²)(T + L) / (4 σ τ η_abs η_q); η_s = η_abs η_q (λ_p/λ_l) T/(T + L); P_out = η_s (P − P_th)."""
    m = laser.medium
    hnu = 6.62607015e-34 * 299792458.0 / 808e-9
    eta = laser.rod.absorbed_fraction * 0.9
    pth = math.pi * hnu * (2 * 200e-6 ** 2) * 0.06 / (4 * 2.8e-23 * 230e-6 * eta)
    assert laser.threshold() == pytest.approx(pth, rel=1e-9)
    assert 0.1 < pth < 0.3                                                       # a diode-pumped Nd:YAG: a hundred-odd mW
    assert laser.slope_efficiency() == pytest.approx(eta * 808 / 1064 * 0.05 / 0.06)
    assert laser.output(pth) == 0.0 and laser.output(0.5 * pth) == 0.0
    assert laser.output(10.0) == pytest.approx(laser.slope_efficiency() * (10.0 - pth))
    assert laser.output(np.array([0.0, 1.0, 10.0])).tolist() == pytest.approx([0.0, laser.slope_efficiency() * (1 - pth), laser.slope_efficiency() * (10 - pth)])
    assert laser.intracavity(10.0) == pytest.approx(laser.output(10.0) / 0.05)
    # the threshold is where the small-signal gain equals the loss: g₀(P_th) = T + L
    assert laser.small_signal_gain(pth) == pytest.approx(0.06, rel=1e-9)
    # a larger mode raises the threshold, a larger coupler too
    assert hy.Laser(laser.rod, 200e-6, 400e-6, 0.05, 0.01).threshold() > pth
    assert laser.threshold(T=0.1) == pytest.approx(pth * 0.11 / 0.06)
    t = laser.table(10.0)
    assert t["optical_efficiency"] == pytest.approx(laser.output(10.0) / 10.0) and t["heat_W"] == pytest.approx(10 * laser.rod.absorbed_fraction * 0.3)


def test_optimum_coupler_is_rigrods(laser):
    """The scan's best T agrees with Rigrod's T_opt = √(g₀ L) − L in the low-gain limit, and the output there beats
    the design coupler's."""
    o = laser.optimum_coupler(10.0)
    assert o["T_opt"] == pytest.approx(o["T_opt_rigrod"], abs=0.002)
    assert o["P_out"] >= laser.output(10.0)
    assert o["g0"] == pytest.approx(laser.small_signal_gain(10.0))


def test_thermal_lens_and_temperature():
    """1/f = η_h P_abs (dn/dT) / (π κ w_p²): f ∝ 1/P, ∝ w_p², no lens without pump; the bulging factor adds to the power."""
    rod = hy.Rod(hy.ND_YAG, 1.5e-3, 0.01)
    f10 = hy.thermal_lens(rod, 10.0, 200e-6)
    assert isinstance(f10, hy.ThermalLens) and f10.pump_W == 10.0
    heat = 0.3 * rod.absorbed_fraction * 10.0
    assert f10.f == pytest.approx(math.pi * 13.0 * 200e-6 ** 2 / (heat * 7.3e-6))
    assert 0.03 < f10.f < 0.3                                                   # tens of mm to a few hundred: the known range at 10 W
    assert hy.thermal_lens(rod, 20.0, 200e-6).f == pytest.approx(f10.f / 2)
    assert hy.thermal_lens(rod, 10.0, 400e-6).f == pytest.approx(4 * f10.f)
    assert math.isinf(hy.thermal_lens(rod, 0.0, 200e-6).f)
    assert hy.thermal_lens(rod, 10.0, 200e-6, stress_and_bulge=0.2).f == pytest.approx(f10.f / 1.2)
    assert np.linalg.det(f10.matrix()) == pytest.approx(1.0)
    assert hy.thermal_lens(rod, 0.0, 200e-6).matrix() == pytest.approx(np.eye(2))
    t = hy.rod_temperature_rise(rod, 10.0, 200e-6)
    assert t["heat_W"] == pytest.approx(heat)
    assert t["dT_centre_to_barrel"] == pytest.approx(heat * (1 + 2 * math.log(1.5e-3 / 200e-6)) / (4 * math.pi * 13.0 * 0.01))
    assert t["dT_centre_to_barrel"] > t["dT_pump_edge_to_barrel"] > 0


def test_effective_pump_radius():
    rod = hy.Rod(hy.ND_YAG, 1.5e-3, 0.01)
    assert hy.effective_pump_radius(lambda z: np.full_like(z, 150e-6), rod) == pytest.approx(150e-6)
    fibre = hy.GaussianBeam.from_fibre(808e-9, 50e-6, 0.22)                     # the pump focused to a 150 µm waist at the entrance face
    pump = hy.GaussianBeam(808e-9, 150e-6, 0.0, fibre.m2, hy.YAG.n(808e-9))
    w_eff = hy.effective_pump_radius(pump.w, rod)
    assert 150e-6 < w_eff < pump.w(rod.length)                                   # between the waist and the exit-face spot
    assert w_eff < pump.w(rod.absorption_length) * 1.5                          # the weight sits in the first absorption length
