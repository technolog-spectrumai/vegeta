"""Timed parachute drop, crush pulse and shock response spectrum against closed forms."""
import math

import numpy as np
import pytest

from vegeta import chronos
from vegeta.chronos.parachute import Body, Parachute, Wind, crush_pulse, landing_scatter, simulate_drop

G = 9.81


def test_terminal_speed_and_canopy_sizing():
    ch = Parachute(area_m2=2.0, cd=0.75)
    assert ch.terminal_speed(10.0) == pytest.approx(math.sqrt(2 * 10 * G / (1.225 * 0.75 * 2.0)))
    assert Parachute(ch.area_for_descent(10.0, 5.0)).terminal_speed(10.0) == pytest.approx(5.0)
    assert Parachute(0.0).terminal_speed(10.0) == float("inf")


def test_drop_reaches_the_terminal_speed_and_the_opening_shock_is_bounded():
    body, ch = Body(mass_kg=10.0, cd_a_m2=0.08), Parachute(area_m2=4.0, cd=0.75, fill_time_s=1.0)
    r = simulate_drop(body, ch, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0, crush_stroke_m=0.05)
    assert r.landed and r.duration_s > 10
    assert r.descent_rate_m_s == pytest.approx(ch.terminal_speed(10.0), rel=0.05)
    assert r.touchdown_vertical_m_s == pytest.approx(ch.terminal_speed(10.0), rel=0.08)
    v_open = math.hypot(r.vx[int(2.0 / 0.01)], r.vz[int(2.0 / 0.01)])
    bound = 0.5 * 1.225 * v_open ** 2 * 0.75 * 4.0 / (10 * G)          # full canopy at the speed when the timer fires
    assert 1.0 < r.opening_g <= bound + 0.05 and 2.0 <= r.opening_time_s <= 3.0
    assert r.opening_altitude_m < 150.0
    assert r.touchdown_g == pytest.approx(crush_pulse(r.touchdown_vertical_m_s, 0.05)[2])
    assert (np.diff(r.z) <= 1e-9).all()                                # it only goes down
    assert np.allclose(r.speed, np.hypot(r.vx, r.vz)) and np.allclose(r.descent_speed, -r.vz)
    assert r.descent_speed[-1] == pytest.approx(ch.terminal_speed(10.0), rel=0.08)   # settled on the canopy
    assert r.descent_speed[:5].max() < 1.0 and r.speed[0] == pytest.approx(30.0)      # level and fast at the cut


def test_no_canopy_is_ballistic_and_later_timer_opens_lower_and_harder():
    body = Body(mass_kg=10.0, cd_a_m2=0.08)
    free = simulate_drop(body, Parachute(0.0), altitude_m=100.0, speed_m_s=30.0, timer_s=1.0)
    assert free.touchdown_vertical_m_s > 30.0 and math.isnan(free.descent_rate_m_s)
    early = simulate_drop(body, Parachute(4.0), altitude_m=150.0, speed_m_s=30.0, timer_s=1.0)
    late = simulate_drop(body, Parachute(4.0), altitude_m=150.0, speed_m_s=30.0, timer_s=4.0)
    assert late.opening_altitude_m < early.opening_altitude_m and late.opening_g > early.opening_g


def test_wind_drifts_and_gusts_are_seeded():
    body, ch = Body(10.0, 0.08), Parachute(4.0)
    calm = simulate_drop(body, ch, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0)
    windy = simulate_drop(body, ch, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0, wind=Wind(8.0))
    assert windy.drift_m > calm.drift_m + 100
    g1 = simulate_drop(body, ch, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0, wind=Wind(4.0, gust_rms_m_s=2.0), seed=3)
    g2 = simulate_drop(body, ch, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0, wind=Wind(4.0, gust_rms_m_s=2.0), seed=3)
    g3 = simulate_drop(body, ch, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0, wind=Wind(4.0, gust_rms_m_s=2.0), seed=4)
    assert g1.drift_m == g2.drift_m and g1.drift_m != g3.drift_m
    sc = landing_scatter(body, ch, n=8, altitude_m=150.0, speed_m_s=30.0, timer_s=2.0, wind=Wind(0, 1.0), dt=0.02)
    assert sc["drift_p95_m"] >= sc["drift_mean_m"] and len(sc["drift_m"]) == 8


def test_crush_pulse_and_srs_half_sine():
    a_p, tau, g = crush_pulse(5.0, 0.05)
    assert tau == pytest.approx(2 * 0.05 / 5.0) and a_p == pytest.approx(math.pi * 25 / (4 * 0.05)) and g == pytest.approx(a_p / G)
    assert crush_pulse(5.0, 0.0)[2] == float("inf")
    t, a = chronos.half_sine(100.0, 0.02)
    assert a.max() == pytest.approx(100.0) and t[-1] >= 0.02
    f = np.array([2.0, 5.0, 40.0, 60.0, 400.0])                       # f tau = 0.04 .. 8
    s = chronos.srs(t, a, f, damping_ratio=0.001)
    assert s[0] < 30.0                                                 # soft mount: isolated
    assert 1.70 * 100 < s[2:4].max() < 1.80 * 100                      # textbook half-sine maximum ~1.77 x near f tau ~ 0.8
    assert s[4] == pytest.approx(100.0, rel=0.1)                       # stiff mount: follows the base (a little overshoot)
    assert chronos.shock_at_mount(t, a, 40.0) == pytest.approx(chronos.srs(t, a, [40.0])[0])
