"""ChironLab: contact forces, height-field orientation, servos, impulses, determinism, failure rules, logs."""
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vegeta.chiron import (ChironLab, Command, Custom, Disturbance, Episode, FailureRules, Flat, FootSpec,  # noqa: E402
                           Geom, Joint, Link, Robot, Rough, Servo)

SERVO = Servo(stall_torque=6.0, rated_torque=2.0, no_load_speed=5.76, stall_current=3.0, voltage=12.0, kp=40.0,
              kd=0.8, source="test values (smart-servo class)")
G = 9.81


# ----------------------------------------------------------------------------------------------- toy robots
def toy_quadruped(trunk_mass=2.0):
    """A 2.68 kg four-legged toy: hip pitch + knee per leg, sphere feet, standing feet under the hips."""
    legs, feet, nominal = [], [], {}
    for leg, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        shank = Link(f"{leg}_shank", pos=(0, 0, -0.12),
                     joints=[Joint(f"{leg}_knee", axis=(0, 1, 0), range=(-2.5, 0.5), tag="knee", servo=SERVO, leg=leg)],
                     geoms=[Geom(f"{leg}_shank_g", "capsule", (0.01,), fromto=(0, 0, 0, 0, 0, -0.12), mass=0.05),
                            Geom(f"{leg}_foot", "sphere", (0.015,), pos=(0, 0, -0.12), mass=0.02,
                                 friction=(0.8, 0.005, 0.0001), role="foot")])
        thigh = Link(f"{leg}_thigh", pos=(0.12 * sx, 0.11 * sy, 0.0),
                     joints=[Joint(f"{leg}_hip", axis=(0, 1, 0), range=(-1.5, 1.5), tag="hip_pitch", servo=SERVO,
                                   leg=leg)],
                     geoms=[Geom(f"{leg}_thigh_g", "capsule", (0.012,), fromto=(0, 0, 0, 0, 0, -0.12), mass=0.1)],
                     children=[shank])
        legs.append(thigh)
        feet.append(FootSpec(leg, f"{leg}_foot", [f"{leg}_hip", f"{leg}_knee"], "trunk"))
        nominal[f"{leg}_hip"], nominal[f"{leg}_knee"] = 0.5, -1.0
    trunk = Link("trunk", geoms=[Geom("trunk_box", "box", (0.15, 0.1, 0.03), mass=trunk_mass,
                                      friction=(0.5, 0.005, 0.0001), role="body")], children=legs, log=True)
    return Robot("toy", trunk, feet=feet, nominal_qpos=nominal, sources={"all": "test toy"})


def box_on_feet(mass=3.0):
    """A rigid box with four sphere feet on the same body (no joints)."""
    geoms = [Geom("box", "box", (0.15, 0.1, 0.03), mass=mass, role="body")]
    feet = []
    for i, (x, y) in enumerate(((0.12, 0.08), (0.12, -0.08), (-0.12, 0.08), (-0.12, -0.08))):
        geoms.append(Geom(f"pad{i}", "sphere", (0.02,), pos=(x, y, -0.05), mass=0.05, role="foot",
                          friction=(0.8, 0.005, 0.0001)))
        feet.append(FootSpec(f"pad{i}", f"pad{i}", [], "box"))
    return Robot("box", Link("box", geoms=geoms, log=True), feet=feet)


def ball(radius=0.02, mass=0.1):
    return Robot("ball", Link("ball", geoms=[Geom("ball", "sphere", (radius,), mass=mass, role="foot")], log=True),
                 feet=[FootSpec("ball", "ball", [], "ball")])


class Hold:
    """Holds the standing pose; uses the seed for a tiny pose offset so determinism is meaningful."""

    name = "hold"

    def reset(self, lab, seed):
        rng = np.random.default_rng(seed)
        self.cmd = Command(q_target=lab.nominal_command().q_target + rng.normal(0, 0.02, len(lab.actuated_joints)))

    def __call__(self, obs):
        return self.cmd


# ----------------------------------------------------------------------------------------------- physics checks
@pytest.mark.parametrize("plane", [True, False])
def test_contact_force_on_feet_is_weight_upward(plane):
    robot = box_on_feet()
    lab = ChironLab(robot, Flat(), flat_as_plane=plane, course_extent=(-0.5, 0.5, -0.5, 0.5))
    lab.reset()
    for _ in range(1000):
        lab.step()
    obs = lab.observe(sync=True)
    total = obs.foot_force.sum(axis=0)
    mg = lab.total_mass * G
    assert total[2] == pytest.approx(mg, rel=0.01)
    assert abs(total[0]) < 0.01 * mg and abs(total[1]) < 0.01 * mg
    assert obs.foot_normal_force.sum() == pytest.approx(mg, rel=0.01)
    np.testing.assert_allclose(obs.foot_normal, np.tile([0, 0, 1.0], (4, 1)), atol=1e-5)
    assert np.all(np.abs(obs.foot_contact_pos[:, 2]) < 1e-3)
    assert obs.foot_in_contact.all() and not obs.belly_contact.any()


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_vectorised_contact_forces_equal_mj_contactForce(cone):
    lab = ChironLab(toy_quadruped(), Rough(0.01, 0.04, start=-1.0, seed=4, extent=(-1, 1, -1, 1)), cone=cone,
                    course_extent=(-0.6, 0.6, -0.5, 0.5))
    lab.reset()
    for _ in range(400):
        lab.step()
    m, d = lab.model, lab.data
    ref = np.zeros((4, 3))
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        foot = lab._foot_of_geom[c.geom2 if c.geom1 == lab._terrain_gid else c.geom1]
        if foot < 0 or c.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        f_world = c.frame.reshape(3, 3).T @ f6[:3]                            # force on geom2
        ref[foot] += f_world if c.geom2 == lab._foot_gid[foot] else -f_world
    force, normal, fn, _, inc, _ = lab._contacts()
    np.testing.assert_allclose(force, ref, atol=1e-9)
    np.testing.assert_allclose(fn, lab._foot_normal_forces(), atol=1e-9)
    assert d.ncon > 4 and inc.sum() >= 3


def test_heightfield_orientation_sphere_drop():
    """Spheres dropped on flat terraces of a non-symmetric surface rest at terrain.height + r."""
    def terraces(x, y):
        s = lambda u: 0.5 * (1 + np.tanh(u / 0.01))  # noqa: E731
        return 0.03 * s(x - 0.1) + 0.07 * s(y - 0.15) + 0.02 * s(-x - 0.35) * s(-y - 0.1)

    terrain = Custom(terraces, name="terraces")
    r = 0.02
    lab = ChironLab(ball(r), terrain, course_extent=(-1.0, 1.0, -0.8, 0.8), heightfield_cell=0.005)
    for x, y in ((0.5, -0.4), (-0.6, -0.4), (0.5, 0.5), (-0.6, 0.5), (-0.1, -0.5), (0.4, 0.05)):
        lab.reset(base_pos=(x, y, terrain.height(x, y) + 0.06))
        for _ in range(800):
            lab.step()
        z = lab.observe(sync=True).base_pos[2]
        assert z == pytest.approx(terrain.height(x, y) + r, abs=2e-3), (x, y)


def test_servo_torque_respects_the_torque_speed_line_in_simulation():
    lab = ChironLab(toy_quadruped())

    class Wave:
        def reset(self, lab, seed):
            self.q0 = lab.nominal_command().q_target

        def __call__(self, obs):
            return Command(q_target=self.q0 + 0.6 * math.sin(12 * obs.t))

    ep = lab.run(Wave(), duration=1.0, settle=0.1)
    log = ep.log
    act = log["joint_active"]
    lim = log["tau_stall"][act] * np.maximum(0.0, 1 - np.abs(log["qd"][:, act]) / log["qd_noload"][act])
    assert np.all(np.abs(log["tau"][:, act]) <= lim + 1e-9)
    assert np.any(np.abs(log["tau"][:, act]) >= 0.98 * lim)                  # the limit is actually reached
    lab.step(Command(q_target=lab.nominal_command().q_target + 1.0))
    np.testing.assert_allclose(lab.data.actuator_force, lab._tau, atol=1e-12)  # MuJoCo applied exactly the servo torque


def test_apply_impulse_gives_exactly_J_over_m():
    robot = Robot("brick", Link("brick", geoms=[Geom("brick", "box", (0.1, 0.05, 0.05), mass=2.5, role="body")],
                                log=True))
    lab = ChironLab(robot, gravity=(0, 0, 0))
    lab.reset(base_pos=(0.0, 0.0, 1.0))
    lab.apply_impulse("brick", (0.0, 3.0, -1.0))
    for _ in range(3):
        lab.step()
    obs = lab.observe(sync=True)
    np.testing.assert_allclose(obs.com_vel, [0.0, 3.0 / 2.5, -1.0 / 2.5], rtol=0.01, atol=1e-9)
    # a scheduled 50 ms push during a run delivers the same impulse
    lab.add_disturbance(Disturbance("brick", t_start=0.1, duration=0.05, impulse=2.0, direction=(0, 1, 0)))
    ep = lab.run(lambda obs: None, duration=0.3, settle=0.0, base_pos=(0, 0, 1.0))
    assert ep.log["com_vel"][-1, 1] == pytest.approx(2.0 / 2.5, rel=0.01)
    assert ep.log["com_vel"][5, 1] == 0.0                                     # before the push
    assert ep.log["disturbances"][0]["impulse_Ns"] == pytest.approx(2.0)
    lab.clear_disturbances()

    class Pusher:                                                             # a controller may push mid-run
        def __call__(self, obs):
            if obs.t == pytest.approx(0.05):
                lab.apply_impulse("brick", (0.0, 0.0, 0.5))

    ep = lab.run(Pusher(), duration=0.2, settle=0.0, base_pos=(0, 0, 1.0))
    assert ep.log["com_vel"][-1, 2] == pytest.approx(0.5 / 2.5, rel=0.01)
    assert ep.log["disturbances"][0]["t_start"] == pytest.approx(0.05)


def test_body_velocities_and_jacobians_match_mujoco():
    lab = ChironLab(toy_quadruped())
    lab.reset()
    d = lab.data
    rng = np.random.default_rng(3)
    d.qvel[:] = rng.normal(0, 1, lab.model.nv)
    obs = lab.observe(sync=True)
    res = np.zeros(6)
    b = lab._bid[0]
    mujoco.mj_objectVelocity(lab.model, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)     # at the COM, world
    np.testing.assert_allclose(obs.body_linvel[0], res[3:], atol=1e-10)
    mujoco.mj_objectVelocity(lab.model, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 1)    # body frame
    np.testing.assert_allclose(obs.body_angvel[0], res[:3], atol=1e-10)
    # foot Jacobian against finite differences of the foot centre
    J = obs.foot_jac[0]
    dofs = lab._foot_dofs[0]
    qadr = [lab._jq[lab._joint_index[j]] for j in lab.foot_joints[0]]
    m2, d2 = lab.model, mujoco.MjData(lab.model)
    d2.qpos[:] = d.qpos
    mujoco.mj_kinematics(m2, d2)
    p0 = d2.geom_xpos[lab._foot_gid[0]].copy()
    for c, a in enumerate(qadr):
        d2.qpos[:] = d.qpos
        d2.qpos[a] += 1e-6
        mujoco.mj_kinematics(m2, d2)
        np.testing.assert_allclose(J[:, c], (d2.geom_xpos[lab._foot_gid[0]] - p0) / 1e-6, atol=1e-5)
    assert J.shape == (3, len(dofs))
    for i in range(len(lab.feet)):                                            # the vectorised columns = mj_jac's
        np.testing.assert_allclose(obs.foot_jac[i], lab._foot_jacobian(i), atol=1e-12)


# ----------------------------------------------------------------------------------------------- the API
def test_reset_step_observe_on_a_toy_quadruped_that_stands():
    lab = ChironLab(toy_quadruped())
    obs = lab.reset(seed=0)
    assert obs.t == 0.0 and obs.q.shape == (8,) and obs.q_act.shape == (8,)
    np.testing.assert_allclose(obs.com[:2], 0.0, atol=1e-9)                  # standing COM above the origin
    assert lab.feet == ["FL", "FR", "RL", "RR"] and lab.bodies == ["trunk"]
    assert lab.nominal_hip_height == pytest.approx(lab.nominal_base_height)   # hips at trunk level
    for _ in range(1500):
        obs = lab.step(lab.nominal_command())
    assert obs.t == pytest.approx(1.5)
    assert obs.foot_normal_force.sum() == pytest.approx(lab.total_mass * G, rel=0.02)
    assert obs.foot_in_contact.all() and not obs.belly_contact.any()
    tilt = math.degrees(math.acos(1 - 2 * (obs.body_quat[0, 1] ** 2 + obs.body_quat[0, 2] ** 2)))
    assert tilt < 2.0
    assert obs.com[2] > 0.8 * lab.nominal_hip_height
    assert len(obs.foot_jac) == 4 and obs.foot_jac[0].shape == (3, 2)
    d = obs.as_dict()
    assert set(d) >= {"q", "qd", "tau", "body_pos", "com", "ang_mom", "foot_force", "foot_jac", "belly_contact"}
    fresh = lab.observe()
    obs2 = lab.step({"FL_hip": 0.6})                                          # dict: others keep their target
    assert obs.base_pos.shape == (3,)                                         # frozen by as_dict()
    with pytest.raises(RuntimeError, match="stale"):
        _ = fresh.base_pos                                                    # never read before stepping
    assert obs2.t == pytest.approx(1.501)
    assert lab._qt[lab.actuated_joints.index("FL_hip")] == 0.6
    assert lab._qt[lab.actuated_joints.index("FR_hip")] == 0.5
    with pytest.raises(KeyError):
        lab.step({"nope": 0.0})
    with pytest.raises(ValueError):
        lab.step(np.zeros(3))


def test_run_log_format_and_save_load(tmp_path):
    lab = ChironLab(toy_quadruped(), Rough(0.005, 0.05, start=0.4, seed=1), log_geoms=True,
                    course_extent=(-0.6, 1.0, -0.5, 0.5))

    class Phased(Hold):
        def __call__(self, obs):
            return Command(self.cmd.q_target, leg_phase=np.full(4, 0.25), leg_stance=np.ones(4, bool))

    ep = lab.run(Phased(), duration=1.0, settle=0.2, seed=3,
                 info={"treatment": "stiff", "terrain": {"level": 0.1}, "v_target": 0.2})
    log = ep.log
    T, B, F, J = len(log["t"]), 1, 4, 8
    assert T == 101 and log["t"][0] == 0.0 and log["t"][-1] == pytest.approx(1.0)
    shapes = {"body_pos": (T, B, 3), "body_quat": (T, B, 4), "body_angvel": (T, B, 3), "body_linvel": (T, B, 3),
              "com": (T, 3), "com_vel": (T, 3), "ang_mom": (T, 3), "foot_pos": (T, F, 3), "foot_force": (T, F, 3),
              "foot_normal": (T, F, 3), "foot_contact_pos": (T, F, 3), "foot_mu": (F,), "foot_jac": (T, F, 3, 2),
              "q": (T, J), "qd": (T, J), "tau": (T, J), "tau_stall": (J,), "tau_rated": (J,), "qd_noload": (J,),
              "i_stall": (J,), "voltage": (J,), "q_range": (J, 2), "belly_contact": (T, B),
              "terrain_height_under_com": (T,), "leg_phase": (T, F), "leg_stance_cmd": (T, F),
              "joint_active": (J,), "joint_leg": (J,), "foot_body": (F,), "gravity": (3,)}
    for k, s in shapes.items():
        assert np.shape(log[k]) == s, k
    assert log["bodies"] == ["trunk"] and log["body_group"] == ["trunk"] and log["foot_group"] == ["trunk"] * 4
    assert log["joint_kind"][:2] == ["hip_pitch", "knee"] and log["joint_leg"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert log["foot_joints"][0] == ["FL_hip", "FL_knee"] and log["foot_mu"][0] == pytest.approx(0.8)
    assert log["treatment"] == "stiff" and log["controller"] == "hold" and log["seed"] == 3
    assert log["terrain"]["kind"] == "rough" and log["terrain"]["level"] == 0.1 and log["v_target"] == 0.2
    assert log["total_mass"] == pytest.approx(2.68)
    np.testing.assert_allclose(np.linalg.norm(log["body_quat"], axis=2), 1.0)
    fz = log["foot_force"][:, :, 2]
    assert fz[-1].sum() == pytest.approx(2.68 * G, rel=0.05)                 # standing at the end
    touching = ~np.isnan(log["foot_normal"][:, :, 0])
    assert touching.mean() > 0.5 and np.all(fz[~touching] == 0.0)
    np.testing.assert_allclose(np.linalg.norm(log["foot_normal"][touching], axis=1), 1.0)
    assert np.all(np.isnan(log["foot_contact_pos"][~touching]))
    assert ep.outcome["reason"] == "completed"
    gp = log["geom_pose"]
    assert gp["pos"].shape[:2] == (T, len(gp["names"])) and gp["mat"].shape[2] == 9
    path = ep.save(tmp_path / "ep")
    again = Episode.load(path)
    assert again.outcome == ep.outcome and again.meta["timestep"] == 0.001
    for k, v in log.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(again.log[k], v, err_msg=k)
        elif isinstance(v, float) and math.isnan(v):
            assert math.isnan(again.log[k]), k
        elif k != "geom_pose":
            assert again.log[k] == v, k
    np.testing.assert_array_equal(again.log["geom_pose"]["pos"], gp["pos"])
    assert again.log["geom_pose"]["names"] == gp["names"]
    res = again.to_result()
    assert res.kind == "chiron.episode" and res.ok and res.metrics["outcome"] == "completed"
    assert res.artifacts["episode"] == path and res.to_dict()["metrics"]["n_samples"] == T


def test_determinism_same_seed_same_log():
    def once(seed):
        lab = ChironLab(toy_quadruped(), Rough(0.01, 0.04, start=0.0, seed=2, extent=(-1, 1, -1, 1)),
                        course_extent=(-0.6, 0.6, -0.5, 0.5))
        return lab.run(Hold(), duration=0.5, settle=0.2, seed=seed).log

    a, b, c = once(5), once(5), once(6)
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(v, b[k], err_msg=k)
    assert not np.array_equal(a["q"], c["q"])


def test_log_does_not_change_the_trajectory():
    lab = ChironLab(toy_quadruped())
    full = lab.run(Hold(), duration=0.5, settle=0.1, seed=1)
    light = lab.run(Hold(), duration=0.5, settle=0.1, seed=1, log=False)
    np.testing.assert_array_equal(full.log["com"], light.log["com"])
    assert "q" not in light.log and "foot_force" not in light.log


# ----------------------------------------------------------------------------------------------- failure rules
def test_upside_down_is_a_fall():
    lab = ChironLab(toy_quadruped())
    rules = FailureRules(course_m=1.5, v_target=0.2)
    ep = lab.run(Hold(), rules=rules, settle=0.3, base_pos=(0, 0, 0.3), base_quat=(0, 1, 0, 0))
    assert ep.outcome["reason"] == "fall" and not ep.success
    assert ep.outcome["t_end"] == 0.0 and "tilted" in ep.outcome["detail"]


def test_standing_still_is_a_stall():
    lab = ChironLab(toy_quadruped())
    ep = lab.run(Hold(), rules=FailureRules(course_m=1.5, v_target=0.2), settle=0.5, seed=0, log=False)
    assert ep.outcome["reason"] == "stall"
    assert ep.outcome["t_end"] == pytest.approx(5.0)                          # the first window after the grace
    assert abs(ep.outcome["distance_m"]) < 0.01


def test_low_com_is_a_fall_after_the_hold_time():
    lab = ChironLab(toy_quadruped())

    class Collapse(Hold):
        def __call__(self, obs):
            return Command(q_target=np.tile([1.4, -2.4], 4))                 # fold the legs: belly down

    rules = FailureRules(course_m=1.5, v_target=0.2, min_height_fraction=0.4)
    ep = lab.run(Collapse(), rules=rules, settle=0.2)
    assert ep.outcome["reason"] == "fall" and "above ground" in ep.outcome["detail"]
    low = ep.log["com"][:, 2] - ep.log["terrain_height_under_com"] < 0.4 * lab.nominal_hip_height
    n = int(round(0.5 / lab.log_dt))                                         # 0.5 s of consecutive low samples
    held = [i for i in range(n, len(low)) if low[i - n:i + 1].all()]
    assert ep.outcome["t_end"] == pytest.approx(ep.log["t"][held[0]], abs=1e-9)
    assert ep.log["t"][-1] == pytest.approx(ep.outcome["t_end"])             # the run stops at the outcome


def test_belly_contact_is_reported_per_body():
    shell = Link("shell", geoms=[Geom("shell", "box", (0.1, 0.06, 0.03), mass=1.0, role="body"),
                                 Geom("pad", "sphere", (0.01,), pos=(0, 0, 0.05), mass=0.01, role="foot")],
                 log=True)
    robot = Robot("belly", shell, feet=[FootSpec("pad", "pad", [], "shell")], nominal_base_height=0.04)
    lab = ChironLab(robot)
    lab.reset()
    for _ in range(300):
        lab.step()
    obs = lab.observe(sync=True)
    assert obs.belly_contact.tolist() == [True]
    assert not obs.foot_in_contact[0] and obs.foot_normal_force[0] == 0.0 and np.isnan(obs.foot_normal[0]).all()


def test_success_off_course_and_timeout_with_a_coasting_brick():
    robot = Robot("brick", Link("brick", geoms=[Geom("brick", "box", (0.1, 0.05, 0.05), mass=1.0, role="body")],
                                log=True))
    lab = ChironLab(robot, gravity=(0, 0, 0))
    rules = FailureRules(course_m=0.5, v_target=0.2, nominal_hip_height=0.1)
    lab.reset(base_pos=(0, 0, 0.3))
    lab.apply_impulse("brick", (1.0, 0.0, 0.0))
    ep = lab.run(lambda o: None, rules=rules, settle=0.0, reset=False)
    assert ep.success and ep.outcome["reason"] == "success"
    assert ep.outcome["t_end"] == pytest.approx(0.5, abs=0.011)
    assert ep.outcome["distance_m"] == pytest.approx(0.5, abs=0.011)
    lab.reset(base_pos=(0, 0, 0.3))
    lab.apply_impulse("brick", (0.0, 1.0, 0.0))
    ep = lab.run(lambda o: None, rules=rules, settle=0.0, reset=False)
    assert ep.outcome["reason"] == "off_course"
    ep = lab.run(lambda o: None, rules=FailureRules(course_m=0.5, timeout=0.4, nominal_hip_height=0.1),
                 settle=0.0, base_pos=(0, 0, 0.3))
    assert ep.outcome["reason"] == "timeout" and ep.outcome["t_end"] == pytest.approx(0.4)
    assert FailureRules(course_m=1.5, v_target=0.2).timeout_s() == pytest.approx(17.0)


def test_set_terrain_and_settle_command():
    lab = ChironLab(toy_quadruped())
    lab.set_terrain(Rough(0.004, 0.05, start=-1, seed=1, extent=(-1, 1, -1, 1)))
    assert lab.model.nhfield == 1
    calls = []

    class C(Hold):
        def settle_command(self, obs):
            calls.append(obs.t)
            return None

    lab.run(C(), duration=0.05, settle=0.01)
    assert len(calls) == 10 and calls[0] == pytest.approx(-0.01)


def test_performance_ten_seconds_of_the_toy(capsys):
    lab = ChironLab(toy_quadruped())
    t0 = time.perf_counter()
    ep = lab.run(Hold(), duration=10.0, settle=0.0, seed=0)
    wall = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[chiron] 10 s of the toy quadruped (1 kHz control, 100 Hz log): {wall:.2f} s wall, "
              f"real-time factor {10.0 / wall:.1f}")
    assert wall < 20.0
    assert len(ep.log["t"]) == 1001


# ----------------------------------------------------------------------------------------------- rendering
RENDER = textwrap.dedent("""
    import numpy as np, sys
    from vegeta.chiron import viz
    from vegeta.chiron.lab import Episode
    ep = Episode.load(sys.argv[1])
    imgs = viz.frames(ep, camera="follow", every=50, size=(160, 120))
    assert len(imgs) == 1 and imgs[0].shape == (120, 160, 3) and imgs[0].std() > 1.0
    out = viz.to_video(imgs * 3, sys.argv[2], fps=5)
    print("OK", out)
""")


def test_viz_renders_a_frame(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("cv2")
    lab = ChironLab(toy_quadruped(), log_geoms=True)
    ep = lab.run(Hold(), duration=0.4, settle=0.0, seed=0)
    path = ep.save(tmp_path / "ep.npz")
    cmd = [sys.executable, "-c", RENDER, str(path), str(tmp_path / "v.mp4")]
    env = dict(os.environ, PYVISTA_OFF_SCREEN="true")
    if not os.environ.get("DISPLAY"):
        if shutil.which("xvfb-run") is None:
            pytest.skip("no display and no xvfb-run")
        cmd = ["xvfb-run", "-a"] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout and (tmp_path / "v.mp4").stat().st_size > 0
