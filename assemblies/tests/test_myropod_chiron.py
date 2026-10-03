"""(Copied from notebooks/designs/tests with only the imports changed, and files loaded by path loaded by module
name: the promoted copy must pass the same tests.)
Cleopatra in ChironLab (myropod_robot.py, myropod_controller.py): the robot of protocol §1 and its Amendment D body
treatments (§12: 'spring' and 'spring_damper'), the design parameters and the CLI, the fixed (§3) and load-feedback
(§9.1, §13.1) controllers, and flat walking under the §5 failure rules for treatment × controller.

The body-joint physics checks of §12.5 (model equivalence, c = 0 equivalence, restoring torque, damping dissipation,
numerical dissipation) are in test_body_joint_physics.py.

Run: cd /home/user/vegeta/notebooks/designs && python3 -m pytest -q tests/test_myropod_chiron.py
"""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from assemblies.components import myropod_controller as mc  # noqa: E402
from assemblies.components import myropod_robot as mr  # noqa: E402
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


@pytest.mark.parametrize("treatment", mr.TREATMENTS)
@pytest.mark.parametrize("roll", [True, False])
def test_treatments_have_the_right_joints(treatment, roll):
    """§12.1: pitch and yaw always, roll per body_roll_axis; per-axis k, q0, limits; c only in spring_damper."""
    k = {"yaw": 4.0, "pitch": 8.0, "roll": 6.0}
    c = {"yaw": 0.15, "pitch": 0.2, "roll": 0.25}
    lim = {"yaw": 0.6, "pitch": math.radians(45.0), "roll": math.radians(20.0)}
    robot = mr.cleopatra(treatment, body_roll_axis=roll, body_q0_pitch=0.05,
                         **{f"body_k_{a}": v for a, v in k.items()}, **{f"body_c_{a}": v for a, v in c.items()},
                         body_limit_yaw=lim["yaw"])
    lab = ch.ChironLab(robot)
    axes = ["yaw", "pitch", "roll"] if roll else ["yaw", "pitch"]
    passive = [j for j in lab.joint_names if j not in lab.actuated_joints]
    assert passive == [f"body {i}-{i + 1} {a}" for i in (1, 2) for a in axes] == mr.body_joints(robot.connection)
    assert len(lab.actuated_joints) == 36 and len(lab.feet) == 12
    assert lab.bodies == ["head", "segment 1", "segment 2", "segment 3"]
    tags = dict(zip(lab.joint_names, lab.joint_tags))
    assert sorted(set(tags[j] for j in lab.actuated_joints)) == ["hip_pitch", "hip_yaw", "knee"]
    assert all(tags[j] == "body_" + j.rsplit(" ", 1)[1] for j in passive)
    m = lab.model
    for j in passive:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        dof, a = m.jnt_dofadr[jid], j.rsplit(" ", 1)[1]
        assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE              # rotations only: translations constrained
        assert m.jnt_stiffness[jid] == k[a]
        assert m.qpos_spring[m.jnt_qposadr[jid]] == (0.05 if a == "pitch" else 0.0)      # q0 [rad]
        assert m.dof_damping[dof] == (c[a] if treatment == "spring_damper" else 0.0)
        np.testing.assert_allclose(m.jnt_range[jid], [-lim[a], lim[a]], rtol=0, atol=1e-10)  # MJCF: 10 digits
        assert m.dof_armature[dof] == 0.0 and m.dof_frictionloss[dof] == 0.0
        np.testing.assert_array_equal(m.jnt_axis[jid], {"yaw": [0, 0, 1], "pitch": [0, 1, 0], "roll": [1, 0, 0]}[a])
        np.testing.assert_allclose(m.jnt_pos[jid], [P.pitch / 2, 0, 0], atol=1e-15)   # the pin, mid-gap
    assert lab.nominal_q[[lab.joint_names.index(j) for j in passive]].tolist() == \
        [0.05 if j.endswith("pitch") else 0.0 for j in passive]                    # starts at q0: springs unloaded
    # no body actuator: servos on every leg joint and nothing else (smart servo 6 Nm, the protocol's gains)
    s = robot.meta().servos
    assert set(s) == set(lab.actuated_joints) and not set(s) & set(passive)
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
    # bookkeeping: what was built, and (spring) that body_c_* were ignored
    assert robot.treatment == treatment and robot.name == f"cleopatra {treatment}" and not robot.legacy
    assert robot.body_params["body_roll_axis"] is roll
    if treatment == "spring":
        assert "'spring': joint damping c = 0" in robot.notes and "body_c_pitch" in robot.notes
        assert all(robot.body_params[f"body_c_{a}"] == 0.0 for a in ("pitch", "yaw", "roll"))
    else:
        assert robot.body_params["body_c_roll"] == 0.25
    if not roll:
        assert "no roll hinge" in robot.notes


def test_treatment_names_parameters_and_legacy():
    assert mr.TREATMENTS == ("spring", "spring_damper")
    for bad in ("wobbly", "locked", "flexible+roll", "rigid", "flexible", "flexible+yaw"):
        with pytest.raises(ValueError):
            mr.cleopatra(bad)                       # Amendment C names only with legacy=True; §2's never
    with pytest.raises(ValueError, match="legacy"):
        mr.cleopatra("rigid")
    with pytest.raises(ValueError):
        mr.cleopatra("locked", legacy=True)
    with pytest.raises(TypeError):
        mr.cleopatra("spring", 4.0)                 # the old positional yaw_stiffness is gone: keywords only
    for bad in ({"body_roll_axis": "maybe"}, {"body_k_pitch": "stiff"}, {"body_k_yaw": -1.0}, {"body_c_roll": -0.1},
                {"body_limit_roll": 0.0}, {"body_q0_pitch": 1.0}, {"body_roll_axis": False, "body_q0_roll": 0.1},
                {"body_k_pitch": True}):
        with pytest.raises(ValueError):
            mr.cleopatra("spring_damper", **bad)
    # defaults (§12.1): the default robot is spring_damper with k 8, c 0.2, q0 0, ±45°/±45°/±20°
    assert mr.cleopatra().treatment == "spring_damper"
    assert mr.make_connection() == mr.BodyConnection() == mr.design_connection({})
    assert mr.connection_params(mr.make_connection()) == mr.BODY_DEFAULTS
    # the CLI's value forms (ints, numeric strings, 'false'/0) build the same joints as floats and bools
    a = mr.make_connection("spring", body_roll_axis="false", body_k_pitch=8, body_q0_yaw="0.05")
    b = mr.make_connection("spring", body_roll_axis=False, body_k_pitch=8.0, body_q0_yaw=0.05)
    assert a == b and a.hinged == ("yaw", "pitch") and a.yaw.q0 == 0.05
    assert mr.make_connection("spring", body_roll_axis=0, body_q0_yaw=0.05) == b
    assert mr.make_connection("spring_damper", body_limit_pitch=(-0.3, 0.5)).pitch.limits == (-0.3, 0.5)
    # a ready connection, or the keywords — not both
    conn = mr.make_connection("spring", body_k_yaw=4.0)
    assert mr.cleopatra(connection=conn).connection == conn
    with pytest.raises(ValueError):
        mr.cleopatra(connection=conn, body_k_yaw=4.0)
    # legacy (pre-Amendment D) treatments: labelled, never the default
    leg = mr.cleopatra("flexible+yaw", legacy=True)
    assert leg.treatment == "legacy:flexible+yaw" and leg.legacy and leg.notes.startswith("LEGACY (pre-Amendment D")
    assert mr.body_joints("flexible", legacy=True) == [f"body {i}-{i + 1} {a}" for i in (1, 2)
                                                       for a in ("pitch", "roll")]
    rigid = mr.cleopatra("rigid", legacy=True)
    assert [j.name for j in rigid.joints() if j.servo is None] == []
    with pytest.raises(ValueError):
        mr.cleopatra(connection=rigid.connection)  # a legacy connection still needs legacy=True


def test_design_parameters_are_the_robot_parameters():
    """§12.3: the Myropod design lists every body-joint parameter with the protocol's defaults, and the Chiron robot
    built from a design dict is the robot built from the same keywords."""
    myropod = pytest.importorskip("assemblies.components.myropod")                  # imports CadQuery
    design = myropod.Myropod()
    params = {p.name: p for p in design.params}
    assert set(mr.DESIGN_KEYS) <= set(params)
    for k, v in mr.BODY_DEFAULTS.items():
        assert params[k].default == v, k
        if k.startswith(("body_k_", "body_c_", "body_q0_", "body_limit_")):
            assert params[k].units == {"k": "N·m/rad", "c": "N·m·s/rad", "q0": "rad", "limit": "rad"}[k.split("_")[1]]
    assert params["body_connection"].choices == mr.TREATMENTS
    p = design.resolve(**mr.CLEO_MM, body_connection="spring", body_roll_axis=False, body_k_yaw=4.0,
                       body_q0_pitch=0.05)
    r1 = mr.robot_from_design(p)
    r2 = mr.cleopatra("spring", body_roll_axis=False, body_k_yaw=4.0, body_q0_pitch=0.05)
    assert r1.connection == r2.connection and r1.params == r2.params == P
    assert r1.to_mjcf() == r2.to_mjcf()
    assert mr.design_connection(p) == r2.connection
    with pytest.raises(ValueError):
        mr.robot_from_design(design.resolve())                # Persephone: 12 segments, not Cleopatra
    with pytest.raises(TypeError):
        mr.robot_from_design(p, body_k_yaw=8.0)


def test_controllers_walk_the_robots_own_leg_lengths():
    """A robot built from a design dict with other leg lengths gets them in the controller's IK (Chiron's runner
    and CLI pass ``robot`` to the controller factory)."""
    r = mr.robot_from_design(dict(mr.CLEO_MM, femur_length=90.0, body_connection="spring"))
    assert r.params.femur == pytest.approx(0.090)
    assert mc.fixed(0.2, robot=r).robot_params == r.params == mc.adaptive(0.2, robot=r).robot_params
    assert mc.fixed(0.2).robot_params == P and mc.fixed(0.2, robot=object()).robot_params == P


def test_cli_parses_the_body_parameters(tmp_path, capsys):
    """``chiron info|run designs/myropod_robot.py:cleopatra -p body_connection=spring -p body_k_pitch=8 ...``"""
    from vegeta.chiron import cli
    from vegeta.chiron.experiments import load_ref

    ref = "assemblies.components.myropod_robot:cleopatra"
    argv = ["run", ref, "-p", "body_connection=spring", "-p", "body_k_pitch=8", "-p", "body_roll_axis=false",
            "-p", "body_c_yaw=0.3", "-p", "body_q0_pitch=0.05", "-p", "body_limit_yaw=0.6",
            "--controller", "assemblies.components.myropod_controller:fixed", "--terrain", "flat", "--duration", "0.05", "--settle", "0.05",
            "--timestep", "0.00025", "--seed", "1", "--no-metrics", "--out", str(tmp_path / "run")]
    trial = cli.trial_from_args(cli.build_parser().parse_args(argv))
    assert trial.robot_kwargs == {"body_connection": "spring", "body_k_pitch": 8, "body_roll_axis": False,
                                  "body_c_yaw": 0.3, "body_q0_pitch": 0.05, "body_limit_yaw": 0.6}
    robot = load_ref(trial.robot)(**trial.robot_kwargs)
    ref_robot = mr.cleopatra("spring", body_k_pitch=8.0, body_roll_axis=False, body_c_yaw=0.3, body_q0_pitch=0.05,
                             body_limit_yaw=0.6)
    assert robot.connection == ref_robot.connection and robot.connection.hinged == ("yaw", "pitch")
    assert robot.connection.yaw.damping == 0.0                                   # 'spring': c = 0
    assert cli.main(argv) == 0
    out = tmp_path / "run"
    assert (out / "episode.npz").exists() and (out / "summary.json").exists()
    capsys.readouterr()
    assert cli.main(["info", ref, "-p", "body_connection=spring_damper", "-p", "body_roll_axis=0"]) == 0
    text = capsys.readouterr().out
    assert "body 1-2 pitch" in text and "body 2-3 roll" not in text
    assert cli.main(["info", ref, "-p", "body_connection=rigid"]) == 2           # legacy: invalid input


def test_standing_pose_carries_the_weight():
    lab = mr.cleopatra_lab("spring_damper")
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


def test_legacy_rigid_connection_holds_the_neutral_angles():
    """Legacy (Amendment C) 'rigid': the welded segments keep their neutral angles under load."""
    conn = mr.make_connection("rigid", body_q0_pitch=math.radians(5.0), legacy=True)
    lab = mr.cleopatra_lab(robot=mr.cleopatra(connection=conn, legacy=True))
    lab.reset()
    for _ in range(2000):
        lab.step()
    d = lab.data
    R1, R2 = (d.xmat[_body_id(lab, f"segment {i}")].reshape(3, 3) for i in (1, 2))
    rel = R1.T @ R2
    assert math.degrees(math.atan2(rel[0, 2], rel[0, 0])) == pytest.approx(5.0, abs=1e-9)   # about +y, exact
    assert rel[1, 1] == pytest.approx(1.0, abs=1e-12)


def test_joint_axes_follow_gait_ik_myropod():
    """MuJoCo's forward kinematics of gait.ik_myropod's angles puts every foot on its target (segment frame)."""
    lab = ch.ChironLab(mr.cleopatra("spring"))
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


# ----------------------------------------------------------------------------------------------- controllers
def test_sigma_zero_is_the_fixed_controller():
    lab = mr.cleopatra_lab("spring_damper")
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
    lab = mr.cleopatra_lab("spring")
    c = mc.adaptive(0.2)
    c.reset(lab, seed=4)
    obs = lab.observe(sync=True)
    # the stance depth is the protocol's 0.151 m; the 40°/85° standing pose drops 0.15104 m: 42 µm apart
    np.testing.assert_allclose(obs.foot_pos[:, 2], P.foot_diameter / 2, atol=1e-4)
    assert obs.base_pos[2] == pytest.approx(P.hip_height)
    assert np.allclose([obs.joint(j) for j in mr.body_joints("spring")], 0.0)     # q0 = 0: springs unloaded
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
    lab = mr.cleopatra_lab("spring")
    a = lab.run(mc.adaptive(0.2), duration=1.0, seed=7)
    b = lab.run(mc.adaptive(0.2), duration=1.0, seed=7)
    c = lab.run(mc.adaptive(0.2), duration=1.0, seed=8)
    for key in ("com", "q", "tau", "foot_force", "leg_phase"):
        np.testing.assert_array_equal(a.log[key], b.log[key], err_msg=key)
    assert not np.array_equal(a.log["q"], c.log["q"])


# ----------------------------------------------------------------------------------------------- walking
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("controller", ["fixed", "adaptive"])
@pytest.mark.parametrize("treatment", mr.TREATMENTS)
def test_flat_course_at_0_2_m_s(treatment, controller, seed):
    """§12.4's factorial on flat ground: both treatments × both controllers at 0.2 m/s with LAB_OPTIONS, paired
    seeds (same terrain, same initial gait phase); every trial must succeed under the §5 rules."""
    v = 0.2
    lab = mr.cleopatra_lab(treatment)
    assert lab.timestep == mr.LAB_OPTIONS["timestep"]
    ctrl = (mc.fixed if controller == "fixed" else mc.adaptive)(v)
    ep = lab.run(ctrl, rules=mr.failure_rules(v), seed=seed, info={"treatment": treatment})
    assert ep.success, ep.outcome
    log = ep.log
    speed = ep.outcome["distance_m"] / ep.outcome["t_end"]
    # sanity bounds, looser than the §5 rules but not protocol criteria. Measured (seeds 0, 1; a test, not a study
    # result): speed fixed 0.79-0.83 v spring, 0.87-0.88 v spring_damper, load-feedback 0.62-0.65 v; max tilt
    # 26-31° spring, 14-21° spring_damper; max |y| of the COM 0.12-0.26 m spring, 0.07-0.16 m spring_damper.
    assert speed > (0.7 if controller == "fixed" else 0.5) * v
    assert tilt_deg(log["body_quat"]).max() < 45.0
    assert np.abs(log["com"][:, 1]).max() < 0.4
    # the episode log carries what chiron.metrics needs
    assert log["bodies"] == ["head", "segment 1", "segment 2", "segment 3"]
    assert log["treatment"] == treatment and log["controller"] == controller and log["seed"] == seed
    assert log["robot"] == f"cleopatra {treatment}"
    assert [j for j in log["joints"] if j.startswith("body ")] == mr.body_joints(treatment)
    assert log["leg_phase"].shape == (len(log["t"]), 12) and log["leg_stance_cmd"].dtype == bool
    assert log["total_mass"] == pytest.approx(6.1697, abs=1e-6)
    assert log["nominal_hip_height"] == pytest.approx(P.hip_height)
