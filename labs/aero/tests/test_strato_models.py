"""Strato's models (Stage V8): aerodynamics, thermal, wind, design point, simulations, the feasibility gate, the study
tools — physics checks against closed forms and limits, and the rejections the gate must make."""
import math
from dataclasses import replace

import numpy as np
import pytest

import strato_aero as sa
import strato_design as sd
import strato_feasibility as sf
import strato_geometry as sg
import strato_sim as ss
import strato_study as sst
import strato_thermal as sth
import strato_wind as sw
from ogma import stratosphere as st

BIG = sd.Design(length=80.0, fineness=3.5, payload_mass=50.0, battery_energy=40000.0, propulsion_power=30000.0,
                laminate="zylon", fin_scale=0.7, night_superpressure=150.0)


# ------------------------------------------------------------------------------------------------------- aerodynamics
def test_lamb_coefficients():
    k1, k2, kr = sa.lamb_k(4.0)
    assert k1 == pytest.approx(0.082, abs=1e-3) and k2 == pytest.approx(0.860, abs=1e-3)
    k1, k2, _ = sa.lamb_k(10.0)
    assert k1 == pytest.approx(0.021, abs=1e-3) and k2 == pytest.approx(0.960, abs=1e-3)   # Munk's table, L/D = 10
    with pytest.raises(ValueError):
        sa.lamb_k(1.0)


def test_hull_drag_in_the_published_band():
    ship = sg.DEMONSTRATOR.scaled(80.0)
    d = sa.drag(ship, 20.0, 20000.0)
    q = st.state(20000.0).dynamic_pressure(20.0)
    cdv_hull = d.hull / (q * ship.hull.volume ** (2 / 3))
    assert 0.015 < cdv_hull < 0.035                   # streamlined hulls, Re ~1e7 (plausibility, not validation)
    assert d.total > d.hull and d.cdv > cdv_hull


def test_drag_scales_about_with_speed_squared():
    ship = sg.DEMONSTRATOR.scaled(50.0)
    d10, d20 = sa.drag(ship, 10.0, 20000.0).total, sa.drag(ship, 20.0, 20000.0).total
    assert 3.4 < d20 / d10 < 4.0                      # V² with a Cf that falls slowly with Re


def test_propeller_and_power_inverse():
    assert sa.propeller_efficiency(0.0, 10.0, 0.09, 10.0) == pytest.approx(0.85)
    assert sa.propeller_efficiency(100.0, 10.0, 0.09, 10.0) < sa.propeller_efficiency(10.0, 10.0, 0.09, 10.0)
    ship = sg.DEMONSTRATOR.scaled(50.0)
    p = sa.electric_power(ship, 15.0, 20000.0)["power"]
    assert sa.max_speed(ship, p, 20000.0) == pytest.approx(15.0, rel=1e-4)


# ------------------------------------------------------------------------------------------------------------ thermal
def test_cell_temperature_limits():
    air = st.state(20000.0)
    cold = sth.cell_temperature(0.0, 0.26, 0.9, 0.85, air, 10.0, 50.0, air.temperature)
    hot = sth.cell_temperature(1300.0, 0.26, 0.9, 0.85, air, 10.0, 50.0, air.temperature)
    assert cold < air.temperature < hot              # radiates to a cold sky at night, heats in the sun
    assert sth.cell_temperature(1300.0, 0.26, 0.9, 0.85, air, 30.0, 50.0, air.temperature) < hot   # wind cools


def test_envelope_step_is_stable_in_dense_air():
    c = sd.configure(BIG, sd.Mission())
    sh = ss.Ship(c)
    air = st.state(3000.0)
    s = sth.ThermalState(air.temperature + 40.0, air.temperature + 40.0)
    zero = np.zeros(len(sh.area))
    s2 = sth.envelope_step(sh.thermal, s, zero, 0.0, sh.normals, air, 20.0, 3600.0)
    assert abs(s2.T_envelope - air.temperature) < 40.0 and math.isfinite(s2.T_gas)


# --------------------------------------------------------------------------------------------------------------- wind
def test_wind_features():
    z = np.arange(8000.0, 26001.0, 500.0)
    summer, winter = sw.profile(z, 7, 50.0)["speed"], sw.profile(z, 1, 50.0)["speed"]
    assert 9000 <= z[np.argmax(winter)] <= 13000                   # the jet
    i20 = int(np.argmin(abs(z - 20000.0)))
    assert summer[i20] < winter[i20]                               # summer is calm aloft
    zmin = z[(z > 14000)][np.argmin(summer[z > 14000])]
    assert 16000 <= zmin <= 22000                                  # the summer minimum
    assert sw.speed_percentile(20000.0, 1, 60.0, 90) > sw.speed_percentile(20000.0, 1, 40.0, 90)   # the vortex
    with pytest.raises(ValueError):
        sw.profile(20000.0, 13, 50.0)


# ------------------------------------------------------------------------------------------------------- design point
def test_gas_fill_and_trim():
    c = sd.configure(BIG, sd.Mission(), superheat_day=40.0, superheat_night=-2.0)
    from ogma import lifting_gas as lg
    dp_night = lg.superpressure(c.gas_mass, c.ship.hull.volume, c.air.pressure, c.air.temperature - 2.0, "helium", 0.98)
    assert dp_night == pytest.approx(c.dp_min, rel=1e-9)
    assert c.dp_max > c.dp_min
    assert c.cg[0] == pytest.approx(c.x_cb, abs=1e-6)               # the gondola trims the CG over the CB
    assert c.cg[2] < 0.0                                            # pendulum stability
    assert c.system_mass == pytest.approx(sum(c.masses.values()))
    assert c.net_lift == pytest.approx(c.gross_lift - c.gas_mass)


def test_hotter_days_need_more_envelope():
    cool = sd.configure(BIG, sd.Mission(), superheat_day=10.0)
    hot = sd.configure(BIG, sd.Mission(), superheat_day=200.0)
    assert hot.dp_max > cool.dp_max and hot.plies >= cool.plies


@pytest.mark.parametrize("kw", [dict(battery_energy=-1.0), dict(pv="perovskite"), dict(gas_system="hot-air"),
                                dict(solar_x0=0.8, solar_x1=0.2), dict(altitude=40000.0), dict(fineness=20.0)])
def test_impossible_inputs_rejected(kw):
    with pytest.raises(ValueError):
        sd.configure(replace(BIG, **kw), sd.Mission())
    r = sf.evaluate(replace(BIG, **kw))
    assert not r["feasible"] and r["problems"][0].startswith("invalid design")


def test_demonstrator_does_not_float_at_20km():
    r = sf.evaluate(sd.Design(payload_mass=1.0, battery_energy=500.0, propulsion_power=300.0), passes=1, n_days=200)
    assert not r["feasible"]
    assert any(p.startswith("does not float at 20.0 km") for p in r["problems"])
    assert r["max_float_altitude"] < 20000.0


# -------------------------------------------------------------------------------------------------------- simulations
@pytest.fixture(scope="module")
def big():
    return sf.evaluate(BIG, sd.Mission(month=7), detail=True)


def test_big_ship_is_feasible_in_july(big):
    assert big["feasible"], big["problems"]
    assert big["surplus_lift"] > 0 and big["availability"] > 0.9


def test_day_cycle_physics(big):
    cyc = big["cycle"]
    night = cyc["sun_el"] < -4.6
    assert np.all(cyc["solar"][night] == 0.0) and cyc["solar"].max() > 0
    sh = big["ship"]
    assert np.all(cyc["soc"] <= sh.usable_wh + 1e-6) and np.all(cyc["soc"] >= sh.reserve_wh - 1e-6)
    assert cyc["dp"][~night].max() > cyc["dp"][night].min()          # the sun raises the superpressure


def test_availability_grows_with_battery(big):
    c = big["configuration"]
    small = sd.configure(replace(BIG, battery_energy=10000.0), sd.Mission(month=1), c.superheat_day, c.superheat_night)
    large = sd.configure(replace(BIG, battery_energy=80000.0), sd.Mission(month=1), c.superheat_day, c.superheat_night)
    assert ss.availability(small, n_days=300)["availability"] <= ss.availability(large, n_days=300)["availability"]


def test_gas_life_grows_with_superpressure_reserve():
    a = sd.configure(replace(BIG, night_superpressure=0.0), sd.Mission(), 40.0, 0.0)
    b = sd.configure(replace(BIG, night_superpressure=300.0), sd.Mission(), 40.0, 0.0)
    assert ss.gas_life_days(b) > ss.gas_life_days(a) > 0


def test_sealed_endurance_holds_altitude(big):
    e = ss.endurance(big["configuration"], days=3, dt=1800.0)
    assert e["lost_at_s"] is None
    assert e["min_altitude"] > 19700.0
    assert e["gas_lost_kg"] == pytest.approx(big["gas_mass"] * (1 - (1 - 0.001) ** 3), rel=0.02)


def test_ascent_reaches_float(big):
    a = ss.ascent(big["configuration"], dt=4.0)
    assert a["float_altitude"] == pytest.approx(20000.0, abs=1500.0)
    assert 0.2 < a["max_rate"] < 10.0
    assert a["vented_kg"] == 0.0                                    # sealed: nothing leaves


# -------------------------------------------------------------------------------------------------------------- study
def test_pareto_filter():
    P = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [2, 2.5]])
    assert sorted(sst.pareto(P).tolist()) == [0, 1, 2]


def test_encode_decode_round_trip():
    x = np.array([lo + 0.37 * (hi - lo) for lo, hi in sst.BOUNDS])
    d = sst.decode(x, 50.0, sst.SUMMER)
    assert np.allclose(sst.encode(d), x, rtol=1e-9)
