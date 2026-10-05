"""Pekari Rover (notebook 30): design/CAD consistency, the mass budget and CG, and the track, gear and terrain
calculations behind it (``tracks``, ``gears``, ``terramechanics``) against hand values and published examples.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_pekari_rover.py
"""
import math

import numpy as np
import pytest

import actuators as act
import gears
import pekari_rover_robot as prr
import terramechanics as tm
import tracks


# ----------------------------------------------------------------------------------------------- design and CAD
def test_design_defaults_match_the_dedalus_design():
    pekari_rover = pytest.importorskip("pekari_rover")
    defaults = {prm.name: prm.default for prm in pekari_rover.PekariRover.parameters if prm.name != "part"}
    assert defaults == prr.DESIGN


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = prr.cad_numbers(recompute=True)
    for k, v in prr.CAD.items():
        if isinstance(v, tuple):
            assert np.allclose(fresh[k], v, atol=0.5)
        else:
            assert fresh[k] == pytest.approx(v, rel=0.01)


def test_static_geometry_matches_the_cad():
    pekari_rover = pytest.importorskip("pekari_rover")
    D = pekari_rover.PekariRover
    p = D().resolve()
    dims = D().generate(part="rover").measure()["dimensions"]
    ov = D.overall(p)
    assert dims[0] == pytest.approx(ov["length"], abs=1.0)
    assert dims[1] == pytest.approx(ov["width"], abs=1.0)
    assert dims[2] == pytest.approx(ov["height"], abs=1.0)
    # the sprocket's pitch matches the links, and a whole number of links closes the loop on the idler's slide
    assert 2 * D.sprocket_pitch_radius(p) * math.sin(math.pi / p["sprocket_teeth"]) == pytest.approx(p["track_pitch"])
    n, take_up = D.link_count(p)
    assert 0 <= take_up < p["track_pitch"] and n * p["track_pitch"] == pytest.approx(D.belt_length(p) + take_up)


def test_impossible_geometry_raises():
    pekari_rover = pytest.importorskip("pekari_rover")
    from vegeta.dedalus import BuildError

    for bad in ({"road_wheel_spacing": 200.0}, {"idler_x": -300.0}, {"road_wheels": 3}, {"frame_offset": 20.0}):
        with pytest.raises(BuildError, match="ValueError"):
            pekari_rover.PekariRover().generate(**bad)


def test_pinion_matches_the_gear_module():
    pekari_rover = pytest.importorskip("pekari_rover")
    a = pekari_rover.PekariRover._tooth_points(0.8, 18)
    b = gears.involute_profile(0.8, 18)
    assert np.allclose(a, b)
    r_tip = max(math.hypot(x, y) for x, y in a)
    assert r_tip == pytest.approx(0.8 * 18 / 2 + 0.8)


# ----------------------------------------------------------------------------------------------- mass and CG
def test_mass_budget_and_cg():
    budget = prr.mass_budget()
    m = sum(budget.values())
    assert 15.0 < m < 21.0                                          # empty; ~23 kg loaded
    c = prr.cg()
    assert c["loaded"]["mass_kg"] == pytest.approx(m + prr.PAYLOAD_KG)
    assert abs(c["empty"]["x"]) < 0.005                               # the battery balances it over the patch
    assert 0.08 < c["empty"]["z"] < c["loaded"]["z"] < 0.25


def test_ground_pressure():
    g = prr.geometry()
    W = prr.total_mass(payload_kg=prr.PAYLOAD_KG) * prr.G
    p = tracks.ground_pressure(W, g["b"], g["L"])
    assert 2.5e3 < p < 5e3                                            # a person on foot is ~15-20 kPa
    mmp = tm.mmp_rowland(W, g["n_wheels"], g["b"], g["d_wheel"], g["pitch"])
    assert mmp > p


# ----------------------------------------------------------------------------------------------- gears
def test_planetary_stage():
    st = gears.planetary(18, 54, 3)
    assert st.ratio == 4.0 and st.z_planet == 18 and st.assembles and st.neighbours_clear
    assert not gears.planetary(14, 42, 3).assembles                    # (14 + 42) / 3 is not whole
    with pytest.raises(ValueError):
        gears.planetary(18, 55, 3)


def test_drive_catalogue_entry_is_the_motor_through_the_gearbox():
    motor, drive = act.get(prr.MOTOR), act.get(prr.DRIVE)
    i = gears.planetary(18, 54, 3).ratio ** 2
    eta = gears.chain_efficiency(["planetary stage", "planetary stage"])
    assert drive.stall_Nm == pytest.approx(motor.stall_Nm * i * eta, rel=0.01)
    assert drive.rated_Nm == pytest.approx(motor.rated_Nm * i * eta, rel=0.01)
    assert drive.no_load_rpm == pytest.approx(motor.no_load_rpm / i)


def test_lewis_and_hertz_hand_values():
    # Shigley ex. 14-1 style: F_t 1000 N, m 2 mm, b 20 mm, 20 teeth -> 1000 / (20 * 2 * 0.322) = 77.6 MPa
    assert gears.lewis_bending(1000.0, 2.0, 20.0, 20) == pytest.approx(77.64, rel=1e-3)
    assert gears.elastic_coefficient(210e3, 0.3, 210e3, 0.3) == pytest.approx(191.0, rel=0.01)
    s_ext = gears.contact_stress(1000.0, 40.0, 20.0, 2.0)
    s_int = gears.contact_stress(1000.0, 40.0, 20.0, 2.0, internal=True)
    assert s_int < s_ext                                              # the concave ring flank conforms
    I = math.cos(math.radians(20)) * math.sin(math.radians(20)) / 2 * 2 / 3
    assert s_ext == pytest.approx(191.0 * math.sqrt(1000.0 / (20.0 * 40.0 * I)))


def test_select_ratio_brackets_the_chosen_gearbox():
    g = prr.geometry()
    motor = act.get(prr.MOTOR)
    W = prr.total_mass(payload_kg=prr.PAYLOAD_KG) * prr.G
    r = g["r_sprocket"]
    climb = tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=30.0, v=0.5)["total"]
    flat = tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", v=1.5)["total"]
    sel = gears.select_ratio(stall_Nm=motor.stall_Nm, rated_Nm=motor.rated_Nm, no_load_rpm=motor.no_load_rpm,
                             tau_climb=climb / 2 * r, omega_climb=0.5 / r, tau_flat=flat / 2 * r, omega_top=1.5 / r,
                             efficiency=0.94)
    assert sel["i_min"] is not None and sel["i_min"] <= 16.0 <= sel["i_max"]


def test_sprocket_chordal_and_capstan():
    c = gears.sprocket_chordal(12, 31.0)
    assert c["pitch_radius_mm"] == pytest.approx(59.887, rel=1e-4)
    assert c["speed_variation"] == pytest.approx(1 - math.cos(math.pi / 12))
    assert gears.capstan(0.3, math.pi) == pytest.approx(math.exp(0.3 * math.pi))


def test_sn_cycles():
    assert gears.sn_cycles(100.0, 200.0, 3e6, -0.1) == math.inf
    assert gears.sn_cycles(400.0, 200.0, 3e6, -0.1) == pytest.approx(3e6 * 2 ** -10)


# ----------------------------------------------------------------------------------------------- terramechanics
def test_bekker_sinkage_round_trip():
    s = tm.SOILS["sandy loam"]
    z = 0.02
    p = tm.pressure_sinkage(z, 0.1, s)
    assert tm.sinkage(p, 0.1, s) == pytest.approx(z)


def test_track_on_sandy_loam_order_of_magnitude():
    # Wong's worked case shape: a track 0.5 m wide, 3 m long, 50 kN per track on LETE sandy loam — sinkage of
    # centimetres, compaction resistance a few % of the load, thrust ~ c A + W tan(phi) at full slip
    s = tm.SOILS["sandy loam"]
    z = tm.track_sinkage(50e3, 0.5, 3.0, s)
    Rc = tm.compaction_resistance_track(50e3, 0.5, 3.0, s)
    F = tm.max_thrust_track(50e3, 0.5, 3.0, s)
    assert 0.002 < z < 0.05
    assert 0.0002 < Rc / 50e3 < 0.10                                 # 42 N: the loam carries 33 kPa easily
    upper = 0.5 * 3.0 * s.c + 50e3 * math.tan(s.phi)
    assert 0.9 * upper < F < upper
    # thrust rises with slip
    i = np.linspace(0.02, 1.0, 20)
    assert np.all(np.diff(tm.thrust_track(50e3, 0.5, 3.0, i, s)) > 0)


def test_hard_ground_and_wheel():
    assert tm.compaction_resistance_track(100.0, 0.08, 0.43, "asphalt") == 0.0
    assert tm.max_thrust_track(100.0, 0.08, 0.43, "asphalt") == pytest.approx(0.8 * 100.0 * (1 - 0.01 / 0.43))
    # a narrow wheel sinks deeper than a track of the same load in loose sand
    z_w = tm.wheel_sinkage(56.0, 0.04, 0.11, "dry sand")
    z_t = tm.track_sinkage(112.0, 0.08, 0.43, "dry sand")
    assert z_w > 3 * z_t


# ----------------------------------------------------------------------------------------------- tracks
def test_road_wheel_loads_sum_and_bogie_split():
    xs = [-0.18, -0.06, 0.06, 0.18]
    rigid = tracks.road_wheel_loads(100.0, xs, 0.0)
    assert rigid.sum() == pytest.approx(100.0) and np.allclose(rigid, 25.0)
    shifted = tracks.road_wheel_loads(100.0, xs, 0.05)
    assert shifted.sum() == pytest.approx(100.0) and shifted[-1] > shifted[0]
    bog = tracks.road_wheel_loads(100.0, xs, 0.05, bogies=[(0, 1), (2, 3)])
    assert bog.sum() == pytest.approx(100.0) and bog[0] == bog[1] and bog[2] == bog[3]
    # two bogies are statically determinate: pivots at ±0.12, CG at 0.05 -> front pivot carries 100 * 0.17 / 0.24
    assert bog[2] + bog[3] == pytest.approx(100.0 * 0.17 / 0.24)


def test_skid_steer():
    W, L, B = 225.0, 0.43, 0.412
    t = tracks.skid_steer(W, L=L, B=B, R=1.0, mu_t_max=0.5, rolling=20.0, v=0.5)
    assert t["F_outer_N"] - t["F_inner_N"] == pytest.approx(2 * t["M_r_Nm"] / B)
    assert t["P_regenerative_W"] <= t["P_clutch_brake_W"]
    # a wider turn needs less steering force
    assert tracks.skid_steer(W, L=L, B=B, R=5.0, mu_t_max=0.5, rolling=20.0)["F_outer_N"] < t["F_outer_N"]
    assert tracks.min_turn_radius(W, L=L, B=B, mu_t_max=0.5, rolling=20.0, F_max=1e4) == 0.0


def test_tension_and_stability():
    assert tracks.sag(100.0, 30.0, 0.3) == pytest.approx(30.0 * 0.09 / 800.0)
    T = tracks.derail_tension(50.0, 0.12, 0.012)
    assert 50.0 * 0.12 / (4 * T) == pytest.approx(0.5 * 0.012)
    s = tracks.stability(x_cg=0.0, z_cg=0.15, x_front=0.36, x_rear=-0.36, half_gauge=0.206, half_width=0.246,
                         step_height=0.11, contact_front=0.215, contact_rear=-0.215)
    assert s["tip_uphill_deg"] == pytest.approx(s["tip_downhill_deg"])
    assert s["trench_m"] == pytest.approx(0.36)
