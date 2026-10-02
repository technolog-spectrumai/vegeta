"""Sikarian Lobster Nefri (notebook 24): design/CAD consistency, the mass and volume budget, trim, standing in water,
the tripod gait, the thrust line through the CG, the claw's inverse kinematics, and the three jobs (slow).

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_sikarian_lobster.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

import lobster_controller as lc  # noqa: E402
import lobster_robot as lr  # noqa: E402
import lobster_scenario as ls  # noqa: E402
from vegeta import chiron as ch  # noqa: E402


def test_design_defaults_match_the_dedalus_design():
    lobster = pytest.importorskip("lobster")
    defaults = {prm.name: prm.default for prm in lobster.SikarianLobster.parameters if prm.name != "part"}
    assert {k: defaults[k] for k in lr.NEFRI} == lr.NEFRI


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = lr.cad_numbers(recompute=True)
    for k in lr.CAD:
        assert fresh[k] == pytest.approx(lr.CAD[k], rel=0.01)


def test_budget_sinks_by_the_net_fraction_and_the_foam_fits():
    b = lr.budget()
    assert 5.5 < b["mass"] < 8.0                                     # datasheet: 4-8 kg
    assert b["net_kg"] == pytest.approx(lr.NET_FRACTION * b["mass"], rel=1e-6)
    assert 0 < b["foam_volume"] < b["foam_space"]
    r = lr.lobster()
    assert r.total_mass() == pytest.approx(b["mass"], abs=1e-6)
    assert sum(v for v, _ in r.volumes.values()) == pytest.approx(b["volume"], rel=1e-9)


def test_trimmed_it_stands_level_on_six_feet_in_water():
    r = lr.lobster()
    assert abs(r.trim["couple_trimmed_Nm"]) < 1e-6
    lab = lr.lobster_lab(ch.Flat(), robot=r)
    ep = lab.run(None, duration=2.0, rules=None, settle=0.0)
    fz = np.asarray(ep.log["foot_force"])[-1, :, 2]
    assert fz.sum() == pytest.approx(lr.budget()["wet_weight_N"], rel=0.03)
    assert fz.min() > 0.5 * fz.mean()                                # level: every foot carries a share
    q = np.asarray(ep.log["body_quat"])[-1, 0]
    assert abs(q[1]) < 0.01 and abs(q[2]) < 0.01


def test_tripod_gait_walks_straight():
    lab = lr.lobster_lab(ch.Flat())
    st, up = lc.stand(0.6)
    m = lc.Mission([lc.Phase("walk", lc.walk_to(0.6, v=0.1), timeout=20.0), lc.Phase("stand", up, st)])
    ep = lab.run(m, duration=9.0, rules=None, settle=0.0)
    pos = np.asarray(ep.log["body_pos"])[-1, 0]
    assert pos[0] > 0.55 and abs(pos[1]) < 0.05
    assert abs(math.degrees(lc.yaw_of(np.asarray(ep.log["body_quat"])[-1, 0]))) < 5


@pytest.mark.parametrize("theta", [10.0, 0.0, -10.0])
def test_thrust_line_passes_through_the_cg(theta):
    g = lr.geometry()
    q1, q2 = lc.thrust_line(math.radians(theta), (0.0, 0.0, 0.0), g)
    import lobster

    hub, d = lobster.SikarianLobster.thrust_axis(lr.design_params(), 0.0, math.degrees(q1))   # check with the CAD's frames
    (j1, R1), (j2, R2) = lobster.SikarianLobster.tail_frames(lr.design_params(), 0.0, 0.0)[0]
    x0, _, z0 = g["tail0"]
    L1, L2h = g["L1"], g["L2"] + g["shroud_L"] / 2
    hx = x0 - L1 * math.cos(q1) - L2h * math.cos(q1 + q2)
    hz = z0 + L1 * math.sin(q1) + L2h * math.sin(q1 + q2)
    t = math.radians(theta)
    assert abs((0 - hx) * math.sin(t) - (0 - hz) * math.cos(t)) < 1e-6
    assert -(q1 + q2) == pytest.approx(t)


def test_steep_thrust_needs_the_curled_tail():
    assert lc.thrust_line(math.radians(-60), (0.0, 0.0, 0.0)) is None           # not through the CG
    q = lc.thrust_line(math.radians(-60), (-0.08, 0.0, 0.0))                     # through a point inside the feet
    assert q is not None and q[0] > math.radians(90)                             # the tail curls over the back


def test_claw_ik_round_trip():
    g = lr.geometry()
    for target, elev in (((0.30, -0.055, 0.02), 0.0), ((0.28, 0.06, -0.05), -0.6)):
        q = lc.claw_ik(target, "R" if target[1] < 0 else "L", elev, g)
        p, e = lc.claw_fk(q, "R" if target[1] < 0 else "L", g)
        assert np.allclose(p, target, atol=1e-9) and e == pytest.approx(elev)


def test_water_hook_buoyancy_and_thrust_direction():
    lab = lr.lobster_lab(ch.Flat())
    lab.reset(seed=0)
    lab.thruster_rpm = 2000.0
    lab.water.rpm = 2000.0
    lab.water(lab)
    F = lab.data.xfrc_applied[:, :3].sum(axis=0)
    rho_g = lr.WATER["density"] * lr.WATER["g"]
    assert F[2] == pytest.approx(rho_g * lr.budget()["volume"], rel=1e-6)          # buoyancy, straight up
    assert F[0] == pytest.approx(lab.thrust, rel=1e-6) and lab.thrust > 5          # the thrust pushes forward


def test_rope_cut_force_and_scene():
    assert ls.F_CUT == 600.0
    sc = ls.Scene()
    props, welds = ls.rope_props(sc)
    assert [w.name for w in welds] == [sc.ROPE_WELD]
    assert ls.PIPE["inner_d"] == 0.60


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["swim", "cut_and_enter", "current"])
def test_jobs(kind):
    ep = ls.run(kind)
    assert ep.log["mission_finished"]
    ts = ls.timeseries(ep)
    if kind == "swim":
        assert ts.z.iloc[-1] < 0.2 and ts.x.iloc[-1] > 2.5 and ts.tilt_deg.iloc[-1] < 5
    elif kind == "cut_and_enter":
        assert ep.log["cut_at"] is not None
        assert ts.x.iloc[-1] > ls.Scene().x_mouth + 0.8                            # inside the pipe
    else:
        st = {r[1]: r[0] for r in ep.log["mission"] if r[2] == "start"}
        free = ts[(ts.t > 0.5) & (ts.t < st["curl the tail over the back"])]
        held = ts[(ts.t > st["press down with the thrust"] + 0.5) & (ts.t < st["walk across the current, pressing"])]
        v_free = (free.y.iloc[-1] - free.y.iloc[0]) / (free.t.iloc[-1] - free.t.iloc[0])
        v_held = (held.y.iloc[-1] - held.y.iloc[0]) / (held.t.iloc[-1] - held.t.iloc[0])
        assert v_free > 0.08 and v_held < 0.3 * v_free
    assert not [e for e in ep.log.get("events", []) if e[1] == "mujoco"]


def test_cfd_disk_calibration():
    """The tail case's rotor disk (no induction) runs at the rpm that gives the BEMT thrust of the open propeller."""
    import lobster_cfd as cfd

    for d in (0.0, 35.0):
        cal = cfd.disk_calibration(d)
        assert cal["uncalibrated_thrust_N"] > cal["target_thrust_N"] > 0          # no induction over-predicts
        assert cal["disk_rpm"] < cal["rpm"]
        assert abs(cfd.disk_thrust_fixed(cal["disk_rpm"], cal["axial_inflow_m_s"]) / cal["target_thrust_N"] - 1) < 1e-3
    c, axis = cfd.disk_geometry(35.0)
    assert abs(np.linalg.norm(axis) - 1) < 1e-9 and axis[0] < 0 and c[1] > 0      # thrust upstream; tail tip to robot right
    assert 1.3 < cfd.duct_gain() < 1.4
