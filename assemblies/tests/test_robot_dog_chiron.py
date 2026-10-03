"""(Copied from notebooks/designs/tests with only the imports changed, and files loaded by path loaded by module
name: the promoted copy must pass the same tests.)
Cerberus (notebook 16's robot dog) in ChironLab: mass budget, kinematic conventions, standing, flat walks and
trots, a small rough level, determinism and the episode-log contract.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_robot_dog_chiron.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from assemblies.components import gait  # noqa: E402
from assemblies.components import robot_dog_controller as rdc  # noqa: E402
from assemblies.components import robot_dog_robot as rdr  # noqa: E402
from vegeta import chiron as ch  # noqa: E402

G = 9.81
M_NOTEBOOK = 13.0652343780962          # notebook 16 §1 (robot_dog.json: mass_kg)
LEG_LENGTH = 0.41                      # L1 + L2 [m]: rough-terrain heights are h/L
CORRELATION = 0.25 * 0.38              # a quarter of the hip spacing [m]


def tilt_deg(log):
    q = log["body_quat"][:, 0]
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)))


# ----------------------------------------------------------------------------------------------- the robot
def test_mass_matches_notebook_16_budget():
    robot = rdr.dog_robot()
    budget = rdr.mass_budget()
    assert len(budget) == 8
    assert sum(budget.values()) == pytest.approx(M_NOTEBOOK, abs=1e-6)
    assert robot.total_mass() == pytest.approx(M_NOTEBOOK, abs=1e-6)
    assert budget["actuators 12x (hip roll, hip pitch, knee; 480 g each)"] == pytest.approx(5.76)
    lab = rdr.dog_lab()
    assert lab.total_mass == pytest.approx(M_NOTEBOOK, abs=1e-6)
    # notebook 16 §2: the CG is X_CG = 25 mm ahead of the body centre (standing pose)
    d = lab.data
    assert d.subtree_com[1][0] - d.qpos[0] == pytest.approx(0.025, abs=1e-6)
    s = robot.summary()
    assert s["n_actuated"] == 12 and s["n_passive"] == 0
    assert {j["servo"]["stall_torque"] for j in s["joints"]} == {24.0}


def test_geometry_matches_design():
    g = rdr.geometry()
    assert g["h_stand"] == pytest.approx(0.32469974, abs=1e-8)          # notebook 16: standing height 324.7 mm
    assert g["foot_x"] == pytest.approx(0.02027011, abs=1e-8)
    assert g["y_leg"] == pytest.approx(0.159)                             # 240/2 + 30 + 16/2 + 1 mm
    assert (g["L1"], g["L2"], g["hip_x"]) == (0.2, 0.21, 0.19)
    lab = rdr.dog_lab()
    # the lab's standing pose: hips h_stand + foot radius above the ground, feet under the CG's support
    assert lab.nominal_hip_height == pytest.approx(g["h_stand"] + g["foot_r"], abs=1e-9)
    obs = lab.observe(sync=True)
    assert np.allclose(obs.foot_pos[:, 2], g["foot_r"], atol=1e-9)


def test_design_parameters_match_robot_dog_py():
    pytest.importorskip("cadquery")
    from assemblies.components import robot_dog

    defaults = {p.name: p.default for p in robot_dog.RobotDog.parameters if p.name != "part"}
    assert defaults == rdr.DOG
    p = dict(defaults, part="dog")
    assert rdr.geometry()["h_stand"] * 1000 == pytest.approx(robot_dog.RobotDog.standing_height(p))
    assert rdr.geometry()["foot_x"] * 1000 == pytest.approx(robot_dog.RobotDog.foot_x(p))


def test_cad_numbers_rebuild():
    pytest.importorskip("cadquery")
    pytest.importorskip("vegeta.dedalus")
    fresh = rdr.cad_numbers(recompute=True)
    for k, v in rdr.CAD.items():
        assert np.allclose(fresh[k], v, rtol=1e-6, atol=1e-6), k


@pytest.mark.parametrize("rel_mm", [(20.3, 0.0, -324.7), (120.0, 0.0, -300.0), (-150.0, 30.0, -290.0),
                                    (60.0, -45.0, -260.0)])
def test_joint_conventions_follow_gait_ik_dog(rel_mm):
    """gait.ik_dog's (roll, a1, a2) through angles_to_q puts the simulated foot centre at the IK target."""
    (roll, a1, a2), ok = gait.ik_dog(np.array(rel_mm), 200.0, 210.0)
    assert ok
    q = rdr.angles_to_q(roll, a1, a2)
    assert rdr.q_to_angles(*q) == pytest.approx((roll, a1, a2))
    knee, foot = gait.fk_dog((roll, a1, a2), 200.0, 210.0)
    assert np.allclose(foot, rel_mm, atol=1e-6)
    lab = rdr.dog_lab()
    g = rdr.geometry()
    qpos = {}
    for leg in rdr.LEGS:
        qpos.update(dict(zip(rdr.leg_joints(leg), q)))
    lab.reset(base_pos=(0.0, 0.0, 1.0), qpos=qpos)
    obs = lab.observe(sync=True)
    for i, leg in enumerate(lab.feet):
        sx, sy = rdr.LEGS[leg]
        hip = np.array([sx * g["hip_x"], sy * g["y_leg"], 1.0 + g["hip_z"]])
        assert np.allclose(obs.foot_pos[i] - hip, np.array(rel_mm) / 1000.0, atol=1e-9), leg


def test_notebook_stride_frequencies():
    h = rdr.geometry()["h_stand"]
    stride, f = rdc.froude_stride(0.6, h)
    assert stride * 1000 == pytest.approx(538.877189053737, rel=1e-9)    # notebook 16 §9 (robot_dog.json)
    assert f == pytest.approx(1.1134262354908622, rel=1e-9)
    assert rdc.notebook_stride_hz("walk", h) == pytest.approx(f)
    assert rdc.notebook_stride_hz("trot", h) == pytest.approx(2.04, abs=0.01)  # notebook 16 §8 at 1.5 m/s
    walk = rdc.DogGait("walk", 0.3)
    assert walk.duty == 0.75 and walk.base_phases == {"RL": 0.0, "FL": 0.25, "RR": 0.5, "FR": 0.75}
    assert walk.stance_length == pytest.approx(0.3 * 0.75 / f)
    trot = rdc.DogGait("trot", 0.6)
    assert trot.duty == 0.5 and trot.base_phases["FL"] == trot.base_phases["RR"] == 0.0
    with pytest.raises(ValueError):
        rdc.DogGait("gallop", 1.0)


@pytest.mark.parametrize("profile", ["smooth", "linear", "cosine"])
def test_foot_path_is_continuous_and_lands_at_ground_speed(profile):
    ctrl = rdc.DogGait("walk", 0.6, swing_profile=profile)
    L = ctrl.stance_length
    cs = np.linspace(0.0, 1.0, 4001)[:-1]
    xz = np.array([ctrl.foot_target(c, L)[:2] for c in cs])
    assert np.max(np.abs(np.diff(xz, axis=0))) < 2e-3                   # no jumps
    assert xz[0, 0] == pytest.approx(L / 2) and xz[0, 1] == pytest.approx(-ctrl.depth)
    assert xz[:, 1].max() == pytest.approx(-ctrl.depth + ctrl.swing_lift, abs=1e-5)
    if profile == "smooth":                                              # dx/dc continuous at both ends of the swing
        dx = np.diff(xz[:, 0]) / np.diff(cs)
        i0 = int(np.searchsorted(cs, ctrl.duty))
        assert dx[i0] == pytest.approx(dx[i0 - 2], rel=0.05)
        assert dx[-1] == pytest.approx(dx[0], rel=0.05)


# ----------------------------------------------------------------------------------------------- in the lab
def test_stands_three_seconds():
    lab = rdr.dog_lab(ch.Flat())
    ep = lab.run(rdc.Stand(), duration=3.0, rules=ch.FailureRules(course_m=10.0, timeout=3.0), settle=0.5)
    log = ep.log
    assert ep.reason == "timeout"                                        # no fall, no off-course in 3 s
    assert tilt_deg(log).max() < 1.0
    late = log["t"] > 2.0
    fz = log["foot_force"][late, :, 2].sum(axis=1)
    assert np.allclose(fz, lab.total_mass * G, rtol=0.02)                # quiet: the feet carry the weight
    assert np.ptp(log["com"][:, 2]) < 0.003
    # the trunk settles a few mm back within ~1 s as the hip-pitch servos take the load (PD sag), then holds;
    # the feet do not slide
    assert np.abs(log["com"][-1, :2] - log["com"][0, :2]).max() < 0.010
    assert np.abs(log["com"][-1, :2] - log["com"][late][0, :2]).max() < 0.001
    assert np.abs(log["foot_pos"][-1] - log["foot_pos"][0]).max() < 0.001
    assert np.abs(log["tau"]).max() < 0.5 * 24.0


@pytest.mark.parametrize("gait_name,v", [("walk", 0.3), ("walk", 0.6), ("trot", 0.3), ("trot", 0.6)])
def test_flat_walk_success(gait_name, v):
    lab = rdr.dog_lab(ch.Flat())
    ep = lab.run(rdc.DogGait(gait_name, v), rules=ch.FailureRules(course_m=1.5, v_target=v), seed=0)
    assert ep.success, ep.outcome
    o, log = ep.outcome, ep.log
    assert o["distance_m"] >= 1.5 - log["com"][0, 0] - 1e-9
    assert o["distance_m"] / o["t_end"] > 0.6 * v                        # incl. the ramp from standstill
    assert np.percentile(tilt_deg(log), 95) < 10.0
    assert np.abs(log["com"][:, 1]).max() < 0.3
    assert np.abs(log["tau"]).max() < 24.0 + 1e-9                        # never beyond stall


def test_small_rough_level():
    """h/L = 0.01 (RMS 4.1 mm, correlation 95 mm): the terrain-blind trot crosses it."""
    hL = 0.01
    terrain = ch.Rough(rms=hL * LEG_LENGTH, correlation_length=CORRELATION, start=0.3, seed=0)
    lab = rdr.dog_lab(terrain)
    ep = lab.run(rdc.DogGait("trot", 0.3, random_phase=True), rules=ch.FailureRules(course_m=1.5, v_target=0.3),
                 seed=0, info={"terrain": {"level": hL}})
    assert ep.success, ep.outcome
    assert ep.log["terrain"]["kind"] == "rough" and ep.log["terrain"]["level"] == hL
    assert ep.log["terrain_height_under_com"].std() > 0.001


def test_determinism():
    def run(seed):
        lab = rdr.dog_lab(ch.Flat())
        return lab.run(rdc.DogGait("walk", 0.6, random_phase=True), duration=1.5, seed=seed).log

    a, b, c = run(3), run(3), run(4)
    for key in ("q", "qd", "tau", "com", "foot_force", "leg_phase"):
        assert np.array_equal(a[key], b[key]), key
    assert not np.allclose(a["q"], c["q"])
    assert not np.allclose(a["leg_phase"][0], c["leg_phase"][0])


def test_episode_log_contract_and_save(tmp_path):
    lab = rdr.dog_lab(ch.Flat())
    ctrl = rdc.DogGait("walk", 0.6)
    ep = lab.run(ctrl, duration=1.0, seed=1, info={"treatment": "cerberus", "v_target": 0.6})
    log = ep.log
    T = len(log["t"])
    assert log["robot"] == "Cerberus" and log["controller"] == "walk 0.6 m/s" and log["treatment"] == "cerberus"
    assert log["bodies"] == ["trunk"] and log["body_group"] == ["body"]
    assert log["feet"] == ["FL", "FR", "RL", "RR"] and list(log["foot_body"]) == [0, 0, 0, 0]
    assert log["foot_joints"] == [rdr.leg_joints(leg) for leg in log["feet"]]
    assert sorted(set(log["joint_kind"])) == ["hip_pitch", "hip_roll", "knee"]
    assert log["joint_active"].all() and np.allclose(log["tau_stall"], 24.0)
    assert np.allclose(log["qd_noload"], 300 * 2 * math.pi / 60)
    assert np.allclose(log["foot_mu"], 0.7)
    assert log["foot_jac"].shape == (T, 4, 3, 3)
    assert log["leg_phase"].shape == (T, 4) and log["leg_stance_cmd"].shape == (T, 4)
    assert log["leg_stance_cmd"].mean() == pytest.approx(0.75, abs=0.05)  # duty 0.75
    assert log["nominal_hip_height"] == pytest.approx(lab.nominal_hip_height)
    path = ep.save(tmp_path / "dog.npz")
    back = ch.Episode.load(path)
    assert np.array_equal(back.log["q"], log["q"]) and back.log["feet"] == log["feet"]
    res = ep.to_result()
    assert res.kind == "chiron.episode" and res.metrics["total_mass_kg"] == pytest.approx(M_NOTEBOOK, abs=1e-6)


def test_controller_settings_are_plain():
    s = rdc.DogGait("trot", 0.3, sigma=0.05).settings()
    assert s["gait"] == "trot" and s["sigma"] == 0.05 and s["stride_rule"] == "notebook"
    assert all(isinstance(v, (str, float, int, bool, dict)) for v in s.values())
