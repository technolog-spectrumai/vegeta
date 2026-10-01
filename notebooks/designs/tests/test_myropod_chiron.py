"""Cleopatra in ChironLab (myropod_robot.py, myropod_controller.py): the robot of protocol §1 and its body treatments
(§11.2), the fixed (§3) and adaptive (§9.1) controllers, and flat walking under the §5 failure rules.

Run: cd /home/user/vegeta/notebooks/designs && python3 -m pytest -q tests/test_myropod_chiron.py
"""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

import myropod_controller as mc  # noqa: E402
import myropod_robot as mr  # noqa: E402
from vegeta import chiron as ch  # noqa: E402

G = 9.81
P = mr.CleopatraParams()


def tilt_deg(quat):
    q = np.asarray(quat)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0)))


def _joint_dof(lab, name):
    m = lab.model
    return int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])


def _body_id(lab, name):
    return mujoco.mj_name2id(lab.model, mujoco.mjtObj.mjOBJ_BODY, name)


# ----------------------------------------------------------------------------------------------- the robot
@pytest.mark.parametrize("treatment", mr.TREATMENTS)
def test_mass_is_the_notebook_18_budget(treatment):
    robot = mr.cleopatra(treatment)
    budget = mr.mass_budget()
    assert robot.total_mass() == pytest.approx(6.17, abs=0.02)
    assert robot.total_mass() == pytest.approx(budget["total"], abs=1e-9)
    assert budget["per segment"] == pytest.approx(0.2592 + 0.1967 + 12 * 0.070 + 2 * 0.120 + 0.060 + 0.050)
    assert budget["head"] == pytest.approx(0.188 + 0.450)
    lab = ch.ChironLab(robot)                          # the compiled model carries the same mass
    assert lab.total_mass == pytest.approx(budget["total"], abs=1e-9)
    per_leg = P.femur_mass + P.tibia_mass + P.pad_mass
    assert per_leg == pytest.approx(0.1967 / 4)
    assert P.femur_mass / P.tibia_mass == pytest.approx(80 / 100)


def test_geometry_from_the_protocol():
    assert P.pitch == pytest.approx(0.170)
    assert P.hip_x == pytest.approx(0.0375) and P.hip_y == pytest.approx(0.061)
    assert P.foot_out == pytest.approx(0.0700, abs=5e-5) and P.foot_drop == pytest.approx(0.1510, abs=5e-5)
    assert P.hip_height == pytest.approx(0.163, abs=1e-4)
    assert P.head_offset == pytest.approx(0.160)
    assert mr.design_params(mr.CLEO_MM) == P                     # the same numbers as notebook 18's CLEO


@pytest.mark.parametrize("treatment,body", [("rigid", []),
                                            ("flexible", ["pitch", "roll"]),
                                            ("flexible+yaw", ["yaw", "pitch", "roll"])])
def test_treatments_have_the_right_joints(treatment, body):
    robot = mr.cleopatra(treatment, yaw_stiffness=4.0)
    lab = ch.ChironLab(robot)
    passive = [j for j in lab.joint_names if j not in lab.actuated_joints]
    assert passive == [f"body {i}-{i + 1} {a}" for i in (1, 2) for a in body]
    assert len(lab.actuated_joints) == 36 and len(lab.feet) == 12
    assert lab.bodies == ["head", "segment 1", "segment 2", "segment 3"]
    tags = dict(zip(lab.joint_names, lab.joint_tags))
    assert sorted(set(tags[j] for j in lab.actuated_joints)) == ["hip_pitch", "hip_yaw", "knee"]
    assert all(tags[j] == "body_" + j.rsplit(" ", 1)[1] for j in passive)
    m = lab.model
    for j in passive:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        axis = j.rsplit(" ", 1)[1]
        k = {"yaw": 4.0, "pitch": 8.0, "roll": 8.0}[axis]
        lim = {"yaw": 45.0, "pitch": 45.0, "roll": 20.0}[axis]
        assert m.jnt_stiffness[jid] == pytest.approx(k)
        assert m.dof_damping[m.jnt_dofadr[jid]] == pytest.approx(0.2)
        assert np.degrees(m.jnt_range[jid]) == pytest.approx([-lim, lim])
        assert m.dof_armature[m.jnt_dofadr[jid]] == 0.0 and not any(j in a for a in lab.actuated_joints)
    # servos: the smart servo 6 Nm with the protocol's gains, on every leg joint and nothing else
    s = robot.meta().servos
    assert set(s) == set(lab.actuated_joints)
    one = s[lab.actuated_joints[0]]
    assert (one.stall_torque, one.rated_torque, one.kp, one.kd) == (6.0, 2.0, 40.0, 0.8)
    assert one.no_load_speed == pytest.approx(55 * 2 * math.pi / 60)
    # friction: pads 0.8, shells and head 0.5; only pads, shells and head collide
    geoms = {g.name: g for g in robot.geoms()}
    assert {g.friction[0] for g in geoms.values() if g.role == "foot"} == {0.8}
    assert {g.friction[0] for g in geoms.values() if g.role == "body"} == {0.5}
    assert sorted(g.name for g in geoms.values() if g.role == "body") == [
        "head shell", "segment 1 shell", "segment 2 shell", "segment 3 shell"]
    assert {g.role for g in geoms.values()} == {"foot", "body", "visual"}


def test_treatment_names_and_design_parameters():
    assert mr.canonical_treatment("locked") == "rigid"                  # §2's names with the same physics
    assert mr.canonical_treatment("flexible+roll") == "flexible+yaw"
    assert mr.treatment_connection("flexible").hinged == ("pitch", "roll")  # §11.2: 'flexible' is pitch + roll
    with pytest.raises(ValueError):
        mr.cleopatra("wobbly")
    conn = mr.design_connection({"body_connection": "flexible", "body_yaw": True, "roll_stiffness": 4.0,
                                 "pitch_neutral_deg": 3.0, "yaw_limit_deg": 30.0})
    assert conn.hinged == ("yaw", "pitch", "roll")
    assert conn.roll.stiffness == 4.0 and conn.pitch.neutral_deg == 3.0 and conn.yaw.limits_deg == (-30.0, 30.0)
    assert mr.design_connection({"body_connection": "rigid"}).hinged == ()


def test_treatments_share_bodies_masses_inertias_and_actuators():
    """§11.2: both modes keep the same bodies, geometry, masses, inertias, leg actuators (compiled models)."""
    labs = [ch.ChironLab(mr.cleopatra(t)) for t in mr.TREATMENTS]
    ref = labs[0].model
    names = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(ref.nbody)]
    for lab in labs[1:]:
        m = lab.model
        assert [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)] == names
        np.testing.assert_allclose(m.body_mass, ref.body_mass, atol=1e-12)
        np.testing.assert_allclose(m.body_inertia, ref.body_inertia, atol=1e-12)
        np.testing.assert_allclose(m.body_ipos, ref.body_ipos, atol=1e-12)
        np.testing.assert_allclose(m.geom_size, ref.geom_size, atol=1e-12)
        np.testing.assert_allclose(m.geom_friction, ref.geom_friction, atol=1e-12)
        assert m.nu == ref.nu == 36
        np.testing.assert_allclose(m.actuator_biasprm, ref.actuator_biasprm)
        assert lab.actuated_joints == labs[0].actuated_joints
        assert lab.nominal_base_height == labs[0].nominal_base_height == pytest.approx(P.hip_height)


def test_joint_axes_follow_gait_ik_myropod():
    """MuJoCo's forward kinematics of gait.ik_myropod's angles puts every foot on its target (segment frame)."""
    lab = ch.ChironLab(mr.cleopatra("flexible"))
    rng = np.random.default_rng(5)
    qpos, targets = {}, {}
    for i, f in enumerate(lab.feet):
        sy = mr.LEG_SIDES[f.split(" ", 1)[1]][1]
        rel = (rng.uniform(-0.045, 0.045), sy * rng.uniform(0.05, 0.09), -rng.uniform(0.10, 0.16))
        angles, ok = mc.leg_ik(rel, P.femur, P.tibia, sy)
        assert ok
        targets[f] = rel
        qpos.update(dict(zip(lab.foot_joints[i], angles)))
    lab.reset(qpos=qpos)
    d = lab.data
    for f in lab.feet:
        seg = f"segment {f.split()[0][1:]}"
        b = _body_id(lab, seg)
        R = d.xmat[b].reshape(3, 3)
        g = mujoco.mj_name2id(lab.model, mujoco.mjtObj.mjOBJ_GEOM, f + " foot")
        rel = R.T @ (d.geom_xpos[g] - d.xpos[b]) - np.array(mr.hip_position(P, f.split(" ", 1)[1]))
        np.testing.assert_allclose(rel, targets[f], atol=1e-9)


def test_vectorised_ik_equals_gait_ik_myropod():
    rng = np.random.default_rng(1)
    n = 2000
    sy = rng.choice([-1.0, 1.0], n)
    x, y, z = rng.uniform(-0.06, 0.06, n), sy * rng.uniform(0.02, 0.13, n), rng.uniform(-0.19, -0.04, n)
    yaw, hip, knee, ok = mc.legs_ik(x, y, z, P.femur, P.tibia, sy)
    ref = [mc.leg_ik((a, b, c), P.femur, P.tibia, s) for a, b, c, s in zip(x, y, z, sy)]
    np.testing.assert_allclose(np.c_[yaw, hip, knee], [r[0] for r in ref], atol=1e-12)
    assert list(ok) == [r[1] for r in ref] and not ok.all()


def test_standing_pose_carries_the_weight():
    lab = mr.cleopatra_lab("flexible")
    obs = lab.reset()
    np.testing.assert_allclose(obs.foot_pos[:, 2], P.foot_diameter / 2, atol=1e-9)   # pads touching z = 0
    for _ in range(2000):                                                              # 0.5 s holding the pose
        obs = lab.step()
    obs = lab.observe(sync=True)
    assert obs.foot_normal_force.sum() == pytest.approx(lab.total_mass * G, rel=0.02)
    assert tilt_deg(obs.body_quat).max() < 2.0
    m = lab.model
    hips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in lab.actuated_joints if j.endswith("hip_yaw")]
    z = lab.data.xanchor[hips, 2]
    assert z.mean() == pytest.approx(P.hip_height, abs=0.004)                          # contact + servo sag
    assert z.min() > P.hip_height - 0.010


# ----------------------------------------------------------------------------------------------- §11.5 checks
def test_rigid_connection_holds_the_neutral_angles():
    conn = mr.BodyConnection(mode="rigid", pitch=mr.BodyAxis(neutral_deg=5.0))
    lab = mr.cleopatra_lab(robot=mr.cleopatra(connection=conn))
    lab.reset()
    for _ in range(2000):
        lab.step()
    d = lab.data
    R1, R2 = (d.xmat[_body_id(lab, f"segment {i}")].reshape(3, 3) for i in (1, 2))
    rel = R1.T @ R2
    assert math.degrees(math.atan2(rel[0, 2], rel[0, 0])) == pytest.approx(5.0, abs=1e-9)   # about +y, exact
    assert rel[1, 1] == pytest.approx(1.0, abs=1e-12)
    # and the default rigid robot keeps its segments exactly parallel under load
    lab = mr.cleopatra_lab("rigid")
    lab.reset()
    for _ in range(2000):
        lab.step()
    d = lab.data
    for i in (2, 3):
        rel = d.xmat[_body_id(lab, "segment 1")].reshape(3, 3).T @ d.xmat[_body_id(lab, f"segment {i}")].reshape(3, 3)
        np.testing.assert_allclose(rel, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("axis", ["pitch", "roll", "yaw"])
def test_body_spring_static_deflection_is_torque_over_k(axis):
    """A known torque on a body hinge (no gravity, the robot floating) settles at θ = θ₀ + τ/k."""
    k, tau = {"pitch": 8.0, "roll": 8.0, "yaw": 4.0}[axis], 0.3
    lab = mr.cleopatra_lab("flexible+yaw", robot_kw={"yaw_stiffness": 4.0}, gravity=(0.0, 0.0, 0.0))
    lab.reset(base_pos=(0.0, 0.0, 1.0))
    dof = _joint_dof(lab, f"body 1-2 {axis}")
    lab.data.qfrc_applied[dof] = tau
    for _ in range(16000):                           # 4 s; the hinge is damped (c = 0.2 N·m·s/rad)
        lab.step()
    assert lab.observe().joint(f"body 1-2 {axis}") == pytest.approx(tau / k, rel=0.01)
    assert abs(lab.data.qvel[dof]) < 1e-3


# ----------------------------------------------------------------------------------------------- controllers
def test_sigma_zero_is_the_fixed_controller():
    lab = mr.cleopatra_lab("flexible")
    ep_fixed = lab.run(mc.fixed(0.2), duration=1.0, seed=3)
    ep_zero = lab.run(mc.adaptive(0.2, sigma=0.0), duration=1.0, seed=3)
    for key in ("com", "q", "qd", "tau", "foot_force", "leg_phase", "leg_stance_cmd", "body_quat"):
        np.testing.assert_array_equal(ep_fixed.log[key], ep_zero.log[key], err_msg=key)
    # the fixed controller's phases are the §3 schedule: base + offset + t·v/stride (mod 1)
    c = mc.fixed(0.2)
    c.reset(lab, 3)
    base = np.array([(mc.BASE_PHASES[f.split(" ", 1)[1]] + (3 - int(f.split()[0][1:])) / 12) for f in lab.feet])
    expected = np.mod(base[None, :] + mc.phase_offset(3) + ep_fixed.log["t"][:, None] * 0.2 / 0.1, 1.0)
    diff = np.abs(np.mod(ep_fixed.log["leg_phase"] - expected + 0.5, 1.0) - 0.5)
    assert diff.max() < 1e-9
    np.testing.assert_array_equal(ep_fixed.log["leg_stance_cmd"], ep_fixed.log["leg_phase"] < 0.75)
    # load feedback (σ = 0.6) changes the phases
    ep_adapt = lab.run(mc.adaptive(0.2), duration=1.0, seed=3)
    assert np.abs(ep_adapt.log["leg_phase"] - ep_fixed.log["leg_phase"]).max() > 1e-3


def test_floored_generator_is_the_tegotae_rule_where_the_floor_is_inactive():
    legs = ["a", "b", "c"]
    base = {"a": 0.1, "b": 0.5, "c": 0.8}
    g0 = ch.PhaseGenerator(legs, base, 0.75, 2.0, sigma=0.6)
    g1 = mc.FlooredPhaseGenerator(legs, base, 0.75, 2.0, sigma=0.6, min_rate=0.25)
    g2 = mc.FlooredPhaseGenerator(legs, base, 0.75, 2.0, sigma=0.0, min_rate=0.25)
    g3 = ch.PhaseGenerator(legs, base, 0.75, 2.0, sigma=0.0)
    for _ in range(3000):                               # 6.7 N on every foot at 2 Hz: σN/ω ≈ 0.48, floor inactive
        np.testing.assert_array_equal(g0.step(0.001, [6.7, 6.7, 6.7])[0], g1.step(0.001, [6.7, 6.7, 6.7])[0])
        np.testing.assert_array_equal(g2.step(0.001, [30.0, 30.0, 30.0])[0], g3.step(0.001)[0])
    # a 30 N foot in late stance: the plain rule stops it, the floor keeps it moving at ≥ min_rate × the free rate
    g0.reset(phases={"a": 0.7, "b": 0.7, "c": 0.7})
    g1.reset(phases={"a": 0.7, "b": 0.7, "c": 0.7})
    assert g0.rate([30.0, 30.0, 30.0]).max() < 0
    g1.step(0.001, [30.0, 30.0, 30.0])
    assert g1.phases[0] == pytest.approx(0.7 + 0.25 * 2.0 * 0.001)


def test_start_pose_has_every_foot_on_the_ground():
    lab = mr.cleopatra_lab("flexible+yaw")
    c = mc.adaptive(0.2)
    c.reset(lab, seed=4)
    obs = lab.observe(sync=True)
    # the stance depth is the protocol's 0.151 m; the 40°/85° standing pose drops 0.15104 m: 42 µm apart
    np.testing.assert_allclose(obs.foot_pos[:, 2], P.foot_diameter / 2, atol=1e-4)
    assert obs.base_pos[2] == pytest.approx(P.hip_height)
    assert np.allclose([obs.joint(j) for j in mr.body_joints("flexible+yaw")], 0.0)
    assert np.all(obs.qd == 0)
    # the swinging leg of each segment is the one whose phase is past the duty
    assert (c.gen.phases >= 0.75).sum() == 3


def test_pairing_same_seed_same_terrain_and_initial_phase():
    terrain = ch.terrain_from_spec({"kind": "rough", "rms": 0.15 * 0.18, "correlation_length": 0.25 * 0.17,
                                    "start": 0.3, "seed": 11, "extent": (-0.6, 1.0, -0.4, 0.4)})
    labs = {t: mr.cleopatra_lab(t, ch.terrain_from_spec(terrain.spec()), course_extent=(-0.6, 1.0, -0.4, 0.4))
            for t in mr.TREATMENTS}
    data = [lab.model.hfield_data.copy() for lab in labs.values()]
    assert all(np.array_equal(data[0], x) for x in data[1:])
    starts = {}
    for t, lab in labs.items():
        for name, make in (("fixed", mc.fixed), ("adaptive", mc.adaptive)):
            c = make(0.2)
            c.reset(lab, seed=11)
            obs = lab.observe(sync=True)
            starts[t, name] = (c.offset, c.gen.phases, obs.q_act.copy(), obs.base_pos.copy())
    first = next(iter(starts.values()))
    for s in starts.values():
        assert s[0] == first[0]
        np.testing.assert_array_equal(s[1], first[1])
        np.testing.assert_array_equal(s[2], first[2])
        np.testing.assert_array_equal(s[3], first[3])
    assert mc.phase_offset(11) != mc.phase_offset(12)
    assert 0.0 <= mc.phase_offset(11) < 1.0 and mc.phase_offset(None) == 0.0


def test_determinism():
    lab = mr.cleopatra_lab("flexible+yaw")
    a = lab.run(mc.adaptive(0.2), duration=1.0, seed=7)
    b = lab.run(mc.adaptive(0.2), duration=1.0, seed=7)
    c = lab.run(mc.adaptive(0.2), duration=1.0, seed=8)
    for key in ("com", "q", "tau", "foot_force", "leg_phase"):
        np.testing.assert_array_equal(a.log[key], b.log[key], err_msg=key)
    assert not np.array_equal(a.log["q"], c.log["q"])


# ----------------------------------------------------------------------------------------------- walking
@pytest.mark.parametrize("controller", ["fixed", "adaptive"])
@pytest.mark.parametrize("treatment", mr.TREATMENTS)
def test_flat_course_at_0_2_m_s(treatment, controller):
    v = 0.2
    lab = mr.cleopatra_lab(treatment)
    ctrl = (mc.fixed if controller == "fixed" else mc.adaptive)(v)
    ep = lab.run(ctrl, rules=mr.failure_rules(v), seed=0, info={"treatment": treatment})
    assert ep.success, ep.outcome
    log = ep.log
    speed = ep.outcome["distance_m"] / ep.outcome["t_end"]
    assert speed > (0.85 if controller == "fixed" else 0.5) * v
    assert tilt_deg(log["body_quat"]).max() < 35.0
    assert np.abs(log["com"][:, 1]).max() < 0.25
    # the episode log carries what chiron.metrics needs
    assert log["bodies"] == ["head", "segment 1", "segment 2", "segment 3"]
    assert log["treatment"] == treatment and log["controller"] == controller
    assert log["leg_phase"].shape == (len(log["t"]), 12) and log["leg_stance_cmd"].dtype == bool
    assert log["total_mass"] == pytest.approx(6.1697, abs=1e-6)
    assert log["nominal_hip_height"] == pytest.approx(P.hip_height)
