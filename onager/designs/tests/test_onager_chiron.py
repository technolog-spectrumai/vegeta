"""Onager Sentinel SX-1 (the Onager project, ``onager/01_onager_sentinel.ipynb``) in ChironLab: design/CAD consistency, mass budget, standing, the flat drive,
the heading hold, the partial failures and the three-wheel limp, the patrol scenario's bookkeeping.

Run: cd /home/user/vegeta && python3 -m pytest -q onager/designs/tests/test_onager_chiron.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

import actuators as act  # noqa: E402
import onager_controller as oc  # noqa: E402
import onager_robot as orb  # noqa: E402
import onager_scenario as osc  # noqa: E402
from vegeta import chiron as ch  # noqa: E402


@pytest.fixture(scope="module")
def lab():
    return orb.onager_lab(ch.Flat())


def drive(lab, failures=(), duration=8.0, **kw):
    ctrl = oc.Drive(3.0, failures=list(failures), **kw)
    ep = lab.run(ctrl, duration=duration, rules=None, settle=0.5)
    return ep, ctrl


# ----------------------------------------------------------------------------------------------- the design
def test_design_defaults_match_the_dedalus_design():
    onager = pytest.importorskip("onager")
    defaults = {prm.name: prm.default for prm in onager.OnagerSentinel.parameters if prm.name != "part"}
    assert defaults == orb.DESIGN


def test_envelope_matches_the_datasheet():
    onager = pytest.importorskip("onager")
    o = onager.OnagerSentinel.overall(orb.design_params())
    assert o["length"] == pytest.approx(2600.0, abs=50.0)
    assert o["width"] == pytest.approx(1600.0, abs=50.0)
    assert o["height"] == pytest.approx(2100.0, abs=20.0)


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = orb.cad_numbers(recompute=True)
    for k in ("hull_surface_area", "upper_leg_volume", "lower_leg_volume", "wheel_volume"):
        assert fresh[k] == pytest.approx(orb.CAD[k], rel=0.02)


def test_mass_budget_and_catalogue():
    robot = orb.onager()
    budget = orb.mass_budget()
    assert robot.total_mass() == pytest.approx(sum(budget.values()), abs=1e-6)
    assert 370.0 < robot.total_mass() < 420.0          # the datasheet says 380 kg; the budget lands ~7 % above
    la, wm = act.get(orb.LEG_ACTUATOR), act.get(orb.WHEEL_MOTOR)
    assert la.self_locking and la.stall_Nm == 800.0
    assert wm.stall_Nm * wm.no_load_rpm * 2 * math.pi / 60 / 4 == pytest.approx(3141.6, rel=1e-3)
    assert len(robot.actuated_joints()) == 12
    assert [j.tag for j in robot.joints()].count("wheel") == 4


def test_kinematic_conventions():
    qs, qk = orb.angles_to_q(40.0, 55.0)
    assert orb.q_to_angles(qs, qk) == pytest.approx((40.0, 55.0))
    g = orb.geometry()
    assert g["shoulder_height"] == pytest.approx(0.988, abs=1e-3)
    assert g["axle_x"] == pytest.approx(0.108, abs=1e-3)


# ----------------------------------------------------------------------------------------------- standing
def test_standing_with_gravity_feedforward(lab):
    ep = lab.run(oc.Stand(), duration=2.0, rules=None, settle=0.0)
    log = ep.log
    z = np.asarray(log["body_pos"])[:, 0, 2]
    assert abs(z[-1] - z[0]) < 0.05                    # sag under 5 cm
    fz = np.asarray(log["foot_force"])[-1, :, 2]
    assert fz.sum() == pytest.approx(lab.total_mass * orb.G, rel=0.03)
    assert fz[2:].sum() / fz.sum() == pytest.approx(0.53, abs=0.03)   # the rear pair carries ~53 %
    names = list(log["joints"])
    tau = np.asarray(log["tau"])[-1]
    knees = [abs(tau[names.index(f"{leg}_knee")]) for leg in orb.LEGS]
    assert max(knees) < act.get(orb.LEG_ACTUATOR).stall_Nm


# ----------------------------------------------------------------------------------------------- driving
def test_flat_drive_reaches_the_target_speed(lab):
    ep, _ = drive(lab)
    log = ep.log
    t, v, com = np.asarray(log["t"]), np.asarray(log["com_vel"])[:, 0], np.asarray(log["com"])
    late = t > 5.0
    assert v[late].mean() == pytest.approx(3.0, abs=0.1)
    assert np.abs(com[late, 1]).max() < 0.02
    assert osc.tilt_deg(np.asarray(log["body_quat"])[:, 0]).max() < 6.0
    assert com[-1, 0] > 18.0


def test_motor_off_keeps_the_heading(lab):
    ep, ctrl = drive(lab, [oc.Failure(3.0, "FL", "motor_off")], duration=9.0)
    log = ep.log
    t = np.asarray(log["t"])
    names = list(log["joints"])
    tau = np.asarray(log["tau"])[:, names.index("FL_wheel")]
    assert np.abs(tau[t > 3.1]).max() < 1e-6            # zero torque: freewheel
    yaw = osc.yaw_deg(np.asarray(log["body_quat"])[:, 0])
    assert np.abs(yaw).max() < 2.0
    assert np.asarray(log["com_vel"])[t > 6.0, 0].mean() == pytest.approx(3.0, abs=0.15)
    assert ctrl.events == [(pytest.approx(3.0, abs=0.02), "FL", "motor_off")]


def test_seized_wheel_drag_at_the_limp_speed(lab):
    ep, _ = drive(lab, [oc.Failure(3.0, "RR", "seized")], duration=10.0, lift_seized=0.0, v_limp=1.5)
    ts = osc.timeseries(ep)
    late = ts[ts.t > 6.0]
    assert late.v.mean() == pytest.approx(1.5, abs=0.2)   # slowed to the limp speed, dragging the braked tyre
    assert (-late.fx_RR).mean() > 300.0                 # hundreds of newtons of drag
    assert late.tilt_deg.max() < 4.0
    assert late.yaw_deg.abs().max() < 3.0
    names = list(ep.log["joints"])
    q = np.asarray(ep.log["q"])[-1, names.index("RR_shoulder")]
    assert abs(q - orb.nominal_qpos()["RR_shoulder"]) < 0.25   # the leg holds against the drag (force feed-forward)


def test_wheel_torque_is_clamped_to_what_the_legs_react(lab):
    ep, ctrl = drive(lab, duration=4.0)
    names = list(ep.log["joints"])
    tau = np.asarray(ep.log["tau"])[:, [names.index(f"{w}_wheel") for w in orb.LEGS]]
    assert np.abs(tau).max() <= ctrl.tau_wheel_max + 1e-6


def test_force_feedforward_holds_the_legs_against_a_push(lab):
    deflection = {}
    for ff in (False, True):
        lab.add_disturbance(ch.Disturbance("hull", 1.0, 3.0, force=600.0, direction=(-1, 0, 0)))
        ep = lab.run(oc.Drive(0.0, ramp_s=0.0, force_ff=ff), duration=4.0, rules=None, settle=0.5)
        lab.clear_disturbances()
        names = list(ep.log["joints"])
        q = np.asarray(ep.log["q"])[-1, names.index("FL_shoulder")]
        deflection[ff] = abs(q - orb.nominal_qpos()["FL_shoulder"])
    assert deflection[True] < 0.5 * deflection[False]


def test_three_wheel_limp_lifts_the_seized_wheel(lab):
    ep, ctrl = drive(lab, [oc.Failure(3.0, "RR", "seized")], duration=12.0, lift_seized=0.10, shift_x=0.4, v_limp=1.5)
    ts = osc.timeseries(ep)
    late = ts[ts.t > 6.0]
    assert (late.fz_RR < 1.0).mean() > 0.9              # airborne
    assert late.v.mean() == pytest.approx(1.5, abs=0.2)
    assert late.tilt_deg.max() < 6.0
    assert any("limp" in e[2] for e in ctrl.events)


def test_knee_locked_holds_the_angle(lab):
    ep, ctrl = drive(lab, [oc.Failure(3.0, "FR", "knee_locked")], duration=6.0)
    log = ep.log
    names = list(log["joints"])
    t, q = np.asarray(log["t"]), np.asarray(log["q"])[:, names.index("FR_knee")]
    held = q[np.argmax(t >= 3.0)]
    assert np.abs(q[t > 3.5] - held).max() < 0.1
    assert osc.tilt_deg(np.asarray(log["body_quat"])[:, 0]).max() < 8.0


# ----------------------------------------------------------------------------------------------- the scenario
def test_patrol_on_gravel_stays_upright_without_failures():
    lab_g = orb.onager_lab(osc.gravel_road())
    ep = osc.run(lab_g, "drag", duration=12.0)
    ts = osc.timeseries(ep)
    assert ts.x.iloc[-1] > 25.0
    assert ts[ts.t < 6.0].tilt_deg.max() < 8.0            # the speed bump at 3 m/s, before any failure
    assert ts.tilt_deg.max() < 30.0


def test_patrol_scenario_bookkeeping():
    road = osc.gravel_road()
    assert road.height(0.0, 0.0) == 0.0                 # flat before the gravel starts
    assert road.height(osc.BUMP["x"], 0.0) == pytest.approx(osc.BUMP["height"] + ch.Rough(
        osc.ROAD["rms"], osc.ROAD["correlation_length"], start=osc.ROAD["start"], seed=osc.ROAD["seed"],
        extent=osc.ROAD["extent"], cell=osc.ROAD["cell"]).height(osc.BUMP["x"], 0.0), abs=1e-9)
    ctrl = osc.controller("limp")
    assert ctrl.name == "patrol/limp" and ctrl.v_limp == 1.5 and [f.mode for f in ctrl.failures] == ["motor_off", "seized"]
    assert osc.controller("drag").lift_seized == 0.0 and osc.controller("drag").v_limp == 1.5
    with pytest.raises(ValueError):
        oc.Failure(1.0, "FL", "exploded")
