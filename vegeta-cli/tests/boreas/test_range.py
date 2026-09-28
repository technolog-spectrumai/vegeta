"""Mission range with the four powerplants: closed forms where they exist, sanity elsewhere."""
import math

import pytest

from vegeta import boreas
from vegeta.boreas.range import (Aircraft, ElectricFan, ElectricProp, Leg, PistonProp, Turbojet, breguet_jet, fly_mission,
                                 max_range)

G = 9.81


def aircraft(**over):
    kw = dict(name="delta", empty_kg=6.0, payload_kg=2.0, wing_area_m2=1.1, cd0=0.02, k=0.06, cl_max=1.0)
    kw.update(over)
    return Aircraft(**kw)


def test_polar_helpers():
    ac = aircraft()
    ld, cl = ac.best_ld()
    assert cl == pytest.approx(math.sqrt(0.02 / 0.06)) and ld == pytest.approx(cl / (2 * 0.02))
    d, c = ac.drag(30.0, 10.0)
    assert c == pytest.approx(10 * G / (0.5 * 1.225 * 900 * 1.1)) and d > 0
    assert ac.stall_speed(10.0) == pytest.approx(math.sqrt(2 * 10 * G / (1.225 * 1.1 * 1.0)))


def test_turbojet_matches_the_closed_form_when_only_induced_drag_burns_fuel():
    # cd0 = 0: D = k (m g)^2 / (q S), dm/dt = -c D  ->  1/m1 - 1/m0 = c k g^2 t / (q S)   (exact in t)
    ac = aircraft(cd0=0.0, k=1.0, empty_kg=8.0, payload_kg=0.0)          # a thirsty engine, so the tank empties in minutes
    jet = Turbojet(thrust_static_n=200.0, tsfc_kg_per_n_h=2.0, fuel_kg=3.0, lapse_per_m_s=0.0, idle_fraction=0.0)
    v = 40.0
    log = fly_mission(ac, jet, [Leg("cruise", v, duration_s=3600.0)], dt=0.25)
    m0, m1 = log.mass[0], log.mass[-1]
    c = 2.0 / 3600
    q_s = 0.5 * 1.225 * v * v * 1.1
    t_exact = q_s / (c * 1.0 * G * G) * (1 / m1 - 1 / m0)
    assert log.t[-1] == pytest.approx(t_exact, rel=0.01)
    assert log.range_km == pytest.approx(v * t_exact / 1000, rel=0.01)
    assert m0 - m1 == pytest.approx(3.0, abs=0.02)          # it flew until the fuel was gone
    assert log.stopped and "fuel" in log.stopped


def test_breguet_bound_for_a_jet_at_constant_ld():
    # at constant speed the Breguet formula assumes constant L/D; a mission at the best-L/D speed of the mean mass
    # comes within a few percent of it
    ac = aircraft(empty_kg=8.0, payload_kg=2.0)
    jet = Turbojet(thrust_static_n=200.0, tsfc_kg_per_n_h=0.15, fuel_kg=2.0, lapse_per_m_s=0.0, idle_fraction=0.0)
    ld, cl = ac.best_ld()
    m_mean = 8 + 2 + 1.0
    v = math.sqrt(2 * m_mean * G / (1.225 * 1.1 * cl))
    log = max_range(ac, jet, v, reserve=0.0, dt=1.0)
    assert log.range_km == pytest.approx(breguet_jet(v, ld, 0.15, 12.0, 10.0) / 1000, rel=0.03)


def test_electric_prop_range_is_energy_over_power():
    d, p = boreas.inches(13, 8)
    prop = boreas.Propeller.from_pitch("13x8", d, p, blades=2, chord_root_m=0.02, chord_max_m=0.03, chord_tip_m=0.008)
    af = boreas.Airfoil()
    motor = boreas.Motor("m", kv_rpm_per_volt=500, resistance_ohm=0.05, no_load_current_a=1.0, max_current_a=60)
    batt = boreas.Battery("6S", cells=6, capacity_ah=10.0, usable_fraction=0.8, mass_kg=1.3)
    pp = ElectricProp(boreas.Propulsion(prop, af, motor, batt))
    ac = aircraft(empty_kg=5.0, payload_kg=2.0)
    v = 25.0
    log = max_range(ac, pp, v, reserve=0.0, dt=10.0)
    assert log.mass[0] == pytest.approx(log.mass[-1])                        # nothing burns
    drag, _ = ac.drag(v, log.mass[0])
    p_w, f = pp.consume(drag, v)
    assert f == 0.0 and p_w > drag * v                                          # electrical power above the useful power
    assert log.endurance_min == pytest.approx(batt.usable_wh / p_w * 60, rel=0.03)
    assert log.range_km == pytest.approx(v * log.t[-1] / 1000, rel=1e-6)
    assert pp.thrust_max(v) > drag                                              # it can cruise here
    head = max_range(ac, pp, v, wind_m_s=-8.0, reserve=0.0, dt=10.0)
    assert head.range_km == pytest.approx(log.range_km * (v - 8) / v, rel=0.03)


def test_piston_prop_burns_fuel_and_fan_and_jet_lapse():
    d, p = boreas.inches(17, 10)
    prop = boreas.Propeller.from_pitch("17x10", d, p, blades=2, chord_root_m=0.025, chord_max_m=0.04, chord_tip_m=0.01)
    pp = PistonProp(prop, boreas.Airfoil(), power_max_w=1500.0, bsfc_kg_per_kwh=0.5, fuel_kg=0.8, rpm_max=8000)
    ac = aircraft(empty_kg=10.0, payload_kg=5.0)
    log = fly_mission(ac, pp, [Leg("cruise", 30.0, duration_s=1200.0)], dt=2.0)
    assert log.stopped is None and log.mass[0] - log.mass[-1] == pytest.approx(log.energy_used[-1], rel=1e-6)
    assert 0 < log.energy_used[-1] < 0.8
    _, f = pp.consume(0.5 * pp.thrust_max(30.0), 30.0)
    assert f > 0 and pp.thrust_max(30.0) > 10.0
    fan = ElectricFan(fan_area_m2=math.pi * 0.045 ** 2, power_max_w=3000.0, battery=boreas.Battery("6S", 6, 10.0, mass_kg=1.3))
    t0, t40 = fan.thrust_max(0.0), fan.thrust_max(40.0)
    assert t0 > t40 > 0                                                          # static thrust highest
    p_w, _ = fan.consume(0.5 * t40, 40.0)
    assert 0 < p_w < 3000.0 and fan.consume(t40, 40.0)[0] == pytest.approx(3000.0, rel=0.02)
    jet = Turbojet(60.0)
    assert jet.thrust_max(50.0) == pytest.approx(60.0 * 0.9) and jet.consume(0.0, 0)[1] > 0


def test_climb_is_flattened_when_thrust_runs_out():
    ac = aircraft(empty_kg=8.0, payload_kg=2.0)
    jet = Turbojet(thrust_static_n=25.0, fuel_kg=2.0, lapse_per_m_s=0.0)
    log = fly_mission(ac, jet, [Leg("climb", 30.0, duration_s=60.0, climb_m_s=8.0)], dt=1.0)
    assert log.thrust.max() <= 25.0 + 1e-9 and log.stopped is None
    with pytest.raises(ValueError):
        Leg("bad", 30.0)
