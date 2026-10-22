"""boreas.microjet: the turbojet cycle on its operating line, the datasheet calibration, the map and its export."""
import math

import numpy as np
import pytest

from vegeta.boreas import microjet as mj


@pytest.fixture(scope="module")
def engine():
    return mj.from_catalogue("140 N class")


def test_calibration_hits_the_datasheet(engine):
    c = mj.CATALOGUE["140 N class"]
    p = mj.solve(engine, engine.rpm_max)
    assert p.converged
    assert p.thrust == pytest.approx(c["thrust_N"], rel=1e-4)
    assert p.fuel_flow_g_min == pytest.approx(c["fuel_g_min"], rel=1e-4)
    assert p.t5 == pytest.approx(c["egt_K"], rel=1e-4)
    assert 1000 < p.t4 < 1300 and 3 < p.pressure_ratio < 4.5 and 0.6 < engine.eta_burner < 1


def test_energy_and_mass_balance(engine):
    p = mj.solve(engine, 0.9 * engine.rpm_max, 60.0)
    work = engine.slip * engine.power_input * p.tip_speed ** 2
    assert p.t3 - p.t2 == pytest.approx(work / mj.CP_AIR, rel=1e-9)
    f = p.fuel_flow / p.mass_flow
    assert (1 + f) * mj.CP_GAS * (p.t4 - p.t5) * engine.eta_mech == pytest.approx(work, rel=1e-9)
    assert engine.eta_burner * f * engine.lhv == pytest.approx((1 + f) * mj.CP_GAS * p.t4 - mj.CP_AIR * p.t3, rel=1e-9)


def test_thrust_and_fuel_rise_with_speed_and_fall_with_airspeed(engine):
    n = np.linspace(0.5, 1.0, 6) * engine.rpm_max
    pts = [mj.solve(engine, x) for x in n]
    assert np.all(np.diff([p.thrust for p in pts]) > 0) and np.all(np.diff([p.fuel_flow for p in pts]) > 0)
    full = [mj.solve(engine, engine.rpm_max, v).thrust for v in (0.0, 50.0, 100.0, 150.0)]
    assert full[0] > full[1] > full[3] > 0.7 * full[0]             # ram drag, partly recovered by ram pressure
    assert mj.solve(engine, engine.rpm_max, 150.0).overall_efficiency() > mj.solve(engine, engine.rpm_max, 50.0).overall_efficiency()


def test_altitude_lowers_thrust(engine):
    sl, hi = mj.solve(engine, engine.rpm_max), mj.solve(engine, engine.rpm_max, 0.0, mj.isa(3000.0))
    assert hi.thrust < 0.85 * sl.thrust and hi.mass_flow < sl.mass_flow


def test_rpm_for_thrust(engine):
    n = mj.rpm_for_thrust(engine, 60.0, 100.0)
    assert engine.rpm_idle < n < engine.rpm_max
    assert mj.solve(engine, n, 100.0).thrust == pytest.approx(60.0, abs=0.05)
    assert mj.rpm_for_thrust(engine, 1000.0, 0.0) is None


def test_flow_function_chokes():
    ff_crit, choked = mj._flow_function(2e5, 1000.0, 0.5e5, mj.G_GAS)
    ff_sub, sub = mj._flow_function(2e5, 1000.0, 1.5e5, mj.G_GAS)
    assert choked and not sub and ff_sub < ff_crit
    assert mj._flow_function(2e5, 1000.0, 2e5, mj.G_GAS)[0] == pytest.approx(0.0, abs=1e-9)


def test_export_round_trip(engine, tmp_path):
    path = mj.export(engine, tmp_path / "jet.json", [0.0, 100.0], [0.6 * engine.rpm_max, engine.rpm_max], extra={"note": "x"})
    d = mj.load(path)
    assert d["engine"] == engine and d["note"] == "x"
    assert d["map"]["thrust"].shape == (2, 2) and d["map"]["thrust"][0, 1] == pytest.approx(142.0, rel=1e-4)
    assert d["engine"].a4 == pytest.approx(engine.a4)


def test_metal_temperatures_order(engine):
    t = mj.metal_temperatures(mj.solve(engine, engine.rpm_max))
    p = mj.solve(engine, engine.rpm_max)
    assert p.t3 < t["bore_K"] < t["rim_K"] < t["blade_K"] < p.t4


def test_bad_design_is_rejected():
    with pytest.raises(ValueError):
        mj.Microjet(impeller_diameter=0.06, rpm_max=150000, rpm_idle=200000, design_mass_flow=0.2)
    assert math.isfinite(mj.isa(2000.0).density) and mj.isa(2000.0).density < mj.isa(0.0).density
