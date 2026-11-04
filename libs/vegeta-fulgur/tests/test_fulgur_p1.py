"""P1: the 0-D/1-D toolkit against the literature (benchmarks) and against the video (checks, looser)."""
import math

import pytest

from vegeta.fulgur import Air, DRY, Gap, HVSupply, MARK4, MARK4C, climate, thrust_per_watt
from vegeta.fulgur.breakdown import inception_voltage, propagation_voltage, spark_margin, surface_flashover, uniform_breakdown
from vegeta.fulgur.supply import resin_melt_energy

G = 9.81


# --- gas
def test_reference_air():
    assert DRY.density == pytest.approx(1.204, rel=0.01)
    assert DRY.delta == 1.0
    assert Air(T=273.15, p=101325).delta == pytest.approx(1.073, rel=0.01)


def test_peek_onset_field_of_a_1mm_wire():
    # the well-known ~55-60 kV/cm onset gradient of a 1 mm radius wire
    assert 5.4e6 < DRY.peek_field(1e-3) < 6.2e6


def test_net_ionisation_fit_anchors():
    assert DRY.alpha_eff(2.5e6) == 0.0
    assert DRY.alpha_eff(3.0e6) == pytest.approx(1100.0, rel=0.05)         # 11 /cm at 30 kV/cm


def test_humidity_and_temperature_lower_mobility():
    assert Air(rh=0.8).mobility() < DRY.mobility()
    assert Air(T=263.15).mobility() < DRY.mobility()                       # denser air, more collisions
    assert Air(T=293.15, rh=1.0).absolute_humidity == pytest.approx(17.3, rel=0.03)   # g/m^3 at 20 °C saturation


# --- breakdown benchmarks
def test_uniform_gap_paschen_limit():
    assert uniform_breakdown(0.01) == pytest.approx(31.4e3, rel=0.10)


def test_positive_needle_plane_breakdown_curve():
    assert 18e3 < propagation_voltage(0.01) < 22e3
    assert 28e3 < propagation_voltage(0.03) < 33e3                         # a lifter at 30 kV / 30 mm: at the edge
    assert 55e3 < propagation_voltage(0.10) < 75e3


def test_meek_inception_agrees_with_peek_onset_within_a_factor_two():
    g = Gap(d=0.02, r=50e-6, length=1.0)
    v_meek, v_peek = inception_voltage(50e-6, 0.02), g.onset_voltage()
    assert 0.5 < v_meek / v_peek < 2.0
    assert 1.5e3 < v_meek < 8e3                                          # a sharp needle starts at 2-4 kV


def test_surface_flashover_falls_with_humidity_nonlinearly():
    dry, mid, wet = (surface_flashover(0.014, Air(rh=r)) for r in (0.0, 0.5, 0.95))
    assert mid > 0.85 * dry and wet < 0.5 * dry


def test_spark_margin_names_its_limit():
    sm = spark_margin(20e3, 50e-6, 0.015, Air(rh=0.9), creepage=0.014)
    assert sm.limited_by == "surface" and sm.margin < 1.0
    sm = spark_margin(20e3, 50e-6, 0.015, DRY, creepage=0.014)
    assert sm.limited_by == "propagation" and sm.margin > 1.0


# --- corona
def test_thrust_per_watt_is_one_over_mu_E():
    # Masuyama & Barrett 2013: T/P = 1/(mu E); 5 mN/W at 1 MV/m, rising at low fields (their 110 N/kW at ~45 kV/m)
    assert thrust_per_watt(0.02, 20e3) == pytest.approx(5.0e-3, rel=0.02)
    assert thrust_per_watt(0.1, 4.5e3) == pytest.approx(0.11, rel=0.05)


def test_current_law_and_thrust_scale():
    g = Gap(d=0.02, r=50e-6, length=1.0)
    assert g.current(g.onset_voltage() * 0.9) == 0.0
    i1, i2 = g.current(20e3), g.current(30e3)
    assert i2 > 2 * i1                                                     # faster than linear
    assert g.thrust(20e3) == pytest.approx(i1 * 0.02 / DRY.mobility())


# --- the video (checks: the model is fitted on the Mark IV's 50 W; the 4C is then predicted)
def test_mark4_at_50W_matches_the_anemometer():
    e = MARK4.at_power(50.0, Air(rh=0.5))
    assert 3.3 < e["v_focal"] < 5.0                                        # 4.1 m/s at 50 % RH
    assert 12e3 < e["V"] < 30e3 and e["margin"] > 1.0


def test_mark4c_predicts_the_pendulum_thrust_at_its_spark_limit():
    e = MARK4C.at_margin(1.0)
    assert 0.018 * G < e["thrust"] < 0.040 * G                             # 25 gf measured
    assert 3.0 < e["v_focal"] < 4.5                                        # 3.7 m/s measured
    assert 40 < e["power"] < 120


def test_humidity_costs_thrust_as_in_the_video():
    dry, wet = MARK4.at_margin(1.0, Air(rh=0.5)), MARK4.at_margin(1.0, Air(rh=0.78))
    assert wet["thrust"] < dry["thrust"]
    assert wet["limited_by"] == "surface"


def test_warsaw_year_and_the_dry_room():
    c = climate.conditions()
    assert len(c) == 14
    worst = min(c, key=lambda k: MARK4C.at_margin(1.0, c[k])["thrust"])
    assert worst == "humid_day"
    assert MARK4C.at_margin(1.0, c["indoor"])["thrust"] > MARK4C.at_margin(1.0, c["dec"])["thrust"]


# --- supply
def test_supply_operating_point_and_fold_back():
    g = Gap(d=0.015, r=50e-6, length=0.8)
    s = HVSupply(v_set=25e3, i_max=5e-3, r_series=1e6)
    V, I = s.operating_point(g.current)
    assert V == pytest.approx(25e3 - 1e6 * I, rel=1e-3) and I < 5e-3
    s2 = HVSupply(v_set=25e3, i_max=1e-3, r_series=0.0)
    V2, I2 = s2.operating_point(g.current)
    assert I2 == pytest.approx(1e-3) and V2 < 25e3
    assert s.input_power(V, I) > V * I


def test_a_current_limited_supply_cannot_melt_the_frame_but_a_bare_flyback_can():
    gap = 0.015
    safe = HVSupply(v_set=25e3, i_max=2e-3).spark_energy(25e3, gap, t_trip=0.01)
    bare = HVSupply(v_set=25e3).spark_energy(25e3, gap, t_trip=30.0, p_short=50.0)   # a 50 W flyback, no trip
    assert safe < 1.0
    assert bare > resin_melt_energy(0.001)                                 # a gram of resin softens
