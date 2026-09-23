"""Physics sanity checks: momentum limits, scaling laws and one published static point."""
import json
import math

import numpy as np
import pytest

from vegeta import boreas
from vegeta.boreas.cli import main


@pytest.fixture
def apc10x47():
    d, p = boreas.inches(10, 4.7)
    return boreas.Propeller.from_pitch("APC 10x4.7 SF", d, p, chord_root_m=0.018, chord_max_m=0.026,
                                       chord_tip_m=0.008, mass_kg=0.012, rotor_mass_kg=0.040)


def test_static_thrust_close_to_published_data(apc10x47):
    # UIUC propeller database, APC 10x4.7 SF static: Ct ~ 0.11, Cp ~ 0.045 (a generic section model
    # is expected within ~35 %; use a fitted Airfoil for a specific propeller)
    op = boreas.solve(apc10x47, boreas.Airfoil(), 6000.0)
    assert op.converged and 0.07 < op.ct < 0.15 and 0.025 < op.cp < 0.06
    assert 0.4 < op.figure_of_merit < 0.85          # below the ideal, above a bad rotor
    assert op.tip_mach < 0.3


def test_scaling_and_limits(apc10x47):
    af = boreas.Airfoil()
    a, b = boreas.solve(apc10x47, af, 3000.0), boreas.solve(apc10x47, af, 6000.0)
    assert b.thrust / a.thrust == pytest.approx(4.0, rel=0.02)    # T ~ n^2 (constant Ct)
    assert b.power / a.power == pytest.approx(8.0, rel=0.02)      # P ~ n^3
    ideal = b.thrust * math.sqrt(b.thrust / (2 * 1.225 * apc10x47.disk_area))
    assert b.power > ideal                                        # never better than momentum theory
    fwd = boreas.solve(apc10x47, af, 6000.0, 8.0)
    assert 0 < fwd.efficiency < 1 and fwd.thrust < b.thrust and fwd.advance_ratio == pytest.approx(8 / 100 / 0.254)
    assert boreas.rpm_for_thrust(apc10x47, af, 2.0).thrust == pytest.approx(2.0, rel=1e-3)
    with pytest.raises(ValueError):
        boreas.rpm_for_thrust(apc10x47, af, 500.0, rpm_max=8000)


def test_motor_battery_system(apc10x47):
    motor = boreas.Motor("2212-920KV", kv_rpm_per_volt=920, resistance_ohm=0.12, no_load_current_a=0.6, max_current_a=20)
    bat = boreas.Battery("3S 2200", cells=3, capacity_ah=2.2)
    sys_ = boreas.Propulsion(apc10x47, boreas.Airfoil(), motor, bat)
    full = sys_.at_throttle(1.0)
    assert full.rpm < motor.kv_rpm_per_volt * bat.voltage and full.current > 0 and full.electrical_power > full.aero.power
    assert 0 < full.motor_efficiency < 1
    half = sys_.at_throttle(0.5)
    assert half.rpm < full.rpm and half.thrust < full.thrust
    pt = sys_.for_thrust(3.0)
    assert pt.thrust == pytest.approx(3.0, rel=2e-3) and 0 < pt.throttle < 1
    exc = boreas.excitations(apc10x47, 6000.0)
    assert exc["shaft_hz"] == 100 and exc["blade_pass_hz"] == 200
    assert exc["unbalance_force_n"] == pytest.approx(0.040 * 6.3e-3 * 6000 * 2 * math.pi / 60)


def test_export_and_load(tmp_path, apc10x47):
    af = boreas.Airfoil()
    grid = boreas.performance_map(apc10x47, af, [3000, 6000], [0, 5, 10])
    assert np.array(grid["thrust_n"]).shape == (2, 3) and grid["unconverged_points"] == 0
    res = boreas.export(tmp_path / "p.json", apc10x47, af, map=grid,
                        points={"hover": boreas.solve(apc10x47, af, 5000.0)}, notes="test")
    assert res.ok and res.metrics == {"points": 1, "map_points": 6}
    doc = boreas.load(res.artifacts["json"])
    assert doc["propeller"]["blades"] == 2 and doc["points"]["hover"]["excitation"]["blade_pass_hz"] == pytest.approx(166.67, rel=1e-3)
    assert set(doc) >= {"propeller", "airfoil", "map", "points", "rho"}


def test_validation():
    with pytest.raises(ValueError, match="increasing"):
        boreas.Propeller("bad", 2, (0.1, 0.05, 0.12), (0.01, 0.01, 0.01), (20, 15, 10))
    with pytest.raises(ValueError, match="throttle"):
        boreas.Propulsion(boreas.Propeller.from_pitch("x", 0.25, 0.1, chord_root_m=0.01, chord_max_m=0.02, chord_tip_m=0.005),
                          boreas.Airfoil(), boreas.Motor("m", 900, 0.1, 0.5, 20), boreas.Battery("b", 3, 2.0)).at_throttle(0.0)


def test_cli(tmp_path, capsys):
    assert main(["point", "--dp", "10x4.7", "--rpm", "6000", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["thrust"] > 0 and out["converged"]
    assert main(["for-thrust", "--dp", "10x4.7", "--thrust", "2"]) == 0
    assert "rpm" in capsys.readouterr().out
    assert main(["map", "--dp", "9x6", "--rpm", "3000", "9000", "3", "--speed", "0", "15", "4", "-o", str(tmp_path / "m.json")]) == 0
    assert np.array(boreas.load(tmp_path / "m.json")["map"]["thrust_n"]).shape == (3, 4)
    assert main(["point", "--dp", "10x4.7", "--rpm", "-5"]) == 2


def test_noise_and_cavitation(apc10x47):
    from vegeta import boreas

    tones = boreas.gutin_harmonics(apc10x47, thrust=5.0, torque=0.1, rpm=6000, distance=10.0, angle_deg=90.0)
    assert tones["blade_pass_hz"] == 200 and tones["frequency_hz"][:2] == [200.0, 400.0]
    assert tones["spl_db"][0] > tones["spl_db"][-1]                         # harmonics fall off
    louder = boreas.gutin_harmonics(apc10x47, thrust=10.0, torque=0.2, rpm=6000, distance=10.0, angle_deg=90.0)
    assert louder["total_tonal_db"] > tones["total_tonal_db"]
    far = boreas.gutin_harmonics(apc10x47, thrust=5.0, torque=0.1, rpm=6000, distance=20.0, angle_deg=90.0)
    assert tones["total_tonal_db"] - far["total_tonal_db"] == pytest.approx(6.02, abs=0.05)   # 1/r
    on_axis = boreas.gutin_harmonics(apc10x47, thrust=5.0, torque=0.1, rpm=6000, distance=10.0, angle_deg=0.0)
    assert on_axis["p_rms_pa"][0] < tones["p_rms_pa"][0]                    # Gutin: no loading noise on the axis
    assert boreas.spl(2e-5) == pytest.approx(0.0) and boreas.spl(1e-6, "water") == pytest.approx(0.0)
    bb = boreas.broadband_level(apc10x47, 5.0, 6000, 10.0)
    assert boreas.broadband_level(apc10x47, 5.0, 12000, 10.0) - bb == pytest.approx(60 * math.log10(2), abs=1e-6)
    cav = boreas.cavitation(apc10x47, rpm=3000, airspeed=2.0, depth_m=0.5, cp_min=-1.0)
    assert cav["cavitation_number"] > 0 and cav["relative_speed_m_s"] > 2.0
    deep = boreas.cavitation(apc10x47, rpm=3000, airspeed=2.0, depth_m=50.0, cp_min=-1.0)
    assert deep["cavitation_number"] > cav["cavitation_number"] and not deep["cavitates"]
    fast = boreas.cavitation(apc10x47, rpm=30000, airspeed=2.0, depth_m=0.5, cp_min=-1.0)
    assert fast["cavitates"] and fast["rpm_at_inception"] < 30000
