"""Robot descriptions: masses, validation, MJCF (Link trees and third-party files)."""
import math

import numpy as np
import pytest

from vegeta.chiron import (ChironLab, FootSpec, Geom, Joint, Link, PointMass, Robot, RobotMeta, Rough, Servo,
                           SimOptions)

mujoco = pytest.importorskip("mujoco")

SERVO = Servo(stall_torque=6.0, rated_torque=2.0, no_load_speed=5.76, stall_current=3.0, voltage=12.0, kp=40.0,
              kd=0.8, armature=0.002, source="test values")


def biped_like(servo=SERVO):
    """Two legs with one joint each on a box body — enough to exercise every building block."""
    legs, feet = [], []
    for side, y in (("L", 0.08), ("R", -0.08)):
        legs.append(Link(f"{side}_leg", pos=(0, y, 0),
                         joints=[Joint(f"{side}_hip", axis=(0, 1, 0), range=(-1.0, 1.0), tag="hip_pitch", servo=servo,
                                       leg=side)],
                         geoms=[Geom(f"{side}_shin", "capsule", (0.01,), fromto=(0, 0, 0, 0, 0, -0.2), mass=0.1),
                                Geom(f"{side}_foot", "sphere", (0.02,), pos=(0, 0, -0.2), mass=0.02, role="foot",
                                     friction=(0.8, 0.005, 0.0001))],
                         masses=[PointMass(f"{side}_servo", 0.07, (0, 0, 0))]))
        feet.append(FootSpec(side, f"{side}_foot", [f"{side}_hip"], "torso"))
    tail = Link("tail", pos=(-0.15, 0, 0), joints=[Joint("tail_pitch", axis=(0, 1, 0), range=(-0.5, 0.5),
                                                         stiffness=8.0, damping=0.2, tag="body_pitch")],
                geoms=[Geom("tail_shell", "box", (0.05, 0.04, 0.02), mass=0.3, role="body")], log=True,
                group="tail unit")
    torso = Link("torso", geoms=[Geom("torso_shell", "box", (0.1, 0.06, 0.03), mass=1.0, role="body"),
                                 Geom("torso_marker", "sphere", (0.01,), pos=(0.1, 0, 0.03), role="visual")],
                 masses=[PointMass("battery", 0.4, (0.02, 0, 0))], children=legs + [tail], log=True)
    return Robot("biped-like", torso, feet=feet, nominal_qpos={"L_hip": 0.1, "R_hip": 0.1},
                 sources={"geometry": "test"})


def test_mass_summary_and_meta():
    r = biped_like()
    expected = 1.0 + 0.4 + 2 * (0.1 + 0.02 + 0.07) + 0.3
    assert r.total_mass() == pytest.approx(expected)
    s = r.summary()
    assert s["n_actuated"] == 2 and s["n_passive"] == 1 and s["n_joints"] == 3
    assert s["link_mass_kg"]["torso"] == pytest.approx(1.4)
    meta = r.meta()
    assert meta.logged_bodies == ["torso", "tail"]
    assert meta.body_groups == {"torso": "torso", "tail": "tail unit"}
    assert meta.belly_geoms == {"torso_shell": "torso", "tail_shell": "tail"}
    assert set(meta.servos) == {"L_hip", "R_hip"} and meta.joint_tags["tail_pitch"] == "body_pitch"
    assert meta.leg_of_joint() == {"L_hip": "L", "R_hip": "R"}


def test_mjcf_compiles_with_the_right_masses_and_joints():
    r = biped_like()
    m = mujoco.MjModel.from_xml_string(r.to_mjcf())
    assert m.body_subtreemass[1] == pytest.approx(r.total_mass())
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert m.body_mass[torso] == pytest.approx(1.4)                           # shell + point mass, visual massless
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "tail_pitch")
    assert m.jnt_stiffness[j] == pytest.approx(8.0) and m.dof_damping[m.jnt_dofadr[j]] == pytest.approx(0.2)
    np.testing.assert_allclose(m.jnt_range[j], [-0.5, 0.5])
    h = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "L_hip")
    assert m.dof_armature[m.jnt_dofadr[h]] == pytest.approx(0.002)            # servo armature on the joint
    assert m.nu == 2 and m.actuator_biasprm[0][2] == pytest.approx(-0.8)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_marker")
    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "L_foot")
    assert m.geom_friction[f][0] == pytest.approx(0.8) and m.geom_priority[f] == 1


def test_inline_and_deferred_heightfields_agree():
    """The self-contained MJCF (elevation attribute, image-ordered) and the lab's model hold the same surface."""
    terrain = Rough(0.01, 0.05, start=0.2, seed=5, extent=(-1, 3, -1, 1))
    opt = SimOptions(course_extent=(-0.5, 1.0, -0.4, 0.3), heightfield_cell=0.02)
    lab = ChironLab(biped_like(), terrain, options=opt)
    inline = mujoco.MjModel.from_xml_string(lab.xml)
    np.testing.assert_allclose(inline.hfield_data, lab.model.hfield_data, atol=2e-6)
    np.testing.assert_allclose(inline.hfield_size, lab.model.hfield_size)


@pytest.mark.parametrize("breaker, match", [
    (lambda r: r.root.joints.append(Joint("bad", axis=(0, 0, 1))), "free joint"),
    (lambda r: r.root.children[0].geoms.append(Geom("torso_shell", "sphere", (0.01,))), "duplicate geom"),
    (lambda r: r.feet.append(FootSpec("X", "torso_shell", [], "torso")), "role 'foot'"),
    (lambda r: r.feet.append(FootSpec("X", "L_foot", ["L_hip"], "L_leg")), "not a logged link"),
    (lambda r: r.feet.append(FootSpec("X", "L_foot", ["nope"], "torso")), "unknown joint"),
    (lambda r: r.nominal_qpos.update(nope=1.0), "unknown joint"),
    (lambda r: r.root.children.append(Link("ghost", joints=[Joint("g", axis=(1, 0, 0))])), "no mass"),
])
def test_validation(breaker, match):
    r = biped_like()
    breaker(r)
    with pytest.raises(ValueError, match=match):
        r.validate()


def test_building_block_checks():
    with pytest.raises(ValueError):
        Geom("g", "mesh", (1,))
    with pytest.raises(ValueError):
        Geom("g", "box", (1, 1, 1), role="hand")
    with pytest.raises(ValueError):
        Joint("j", kind="ball")
    with pytest.raises(ValueError):
        Joint("j", range=(1.0, -1.0))
    with pytest.raises(ValueError):
        Servo(0.0, 1.0, 5.0, 1.0, 12.0, 10.0, 0.1)


def test_servo_helpers():
    s = Servo.from_rpm(stall_torque=6.0, rated_torque=2.0, no_load_rpm=55.0, stall_current=3.0, voltage=12.0,
                       kp=40.0, kd=0.8)
    assert s.no_load_speed == pytest.approx(5.7596, abs=1e-4)
    assert s.torque_constant == pytest.approx(2.0) and s.resistance == pytest.approx(4.0)
    assert s.torque_limit(s.no_load_speed / 2) == pytest.approx(3.0)

    class Act:  # duck-typed catalogue entry (notebooks/designs/actuators.py style)
        key, stall_Nm, rated_Nm, stall_A, voltage_V, no_load_rpm = "smart servo 6 Nm", 6.0, 2.0, 3.0, 12.0, 55
        source = "datasheet"

    a = Servo.from_actuator(Act(), kp=40.0, kd=0.8)
    assert a.stall_torque == 6.0 and a.no_load_speed == pytest.approx(s.no_load_speed) and "datasheet" in a.source


THIRD_PARTY = """
<mujoco model="hopper">
  <default><default class="vis"><geom contype="0" conaffinity="0" group="2" density="0"/></default></default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.4">
      <freejoint name="root"/>
      <geom name="base_box" type="box" size="0.1 0.08 0.03" mass="1.5"/>
      <geom name="base_vis" class="vis" type="sphere" size="0.02" pos="0.1 0 0.03"/>
      <body name="legA" pos="0.07 0 0">
        <joint name="hipA" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.1"/>
        <geom name="footA" type="sphere" pos="0 0 -0.2" size="0.02" mass="0.02" friction="0.9 0.005 0.0001"/>
      </body>
      <body name="legB" pos="-0.07 0 0">
        <joint name="hipB" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.1"/>
        <geom name="footB" type="sphere" pos="0 0 -0.2" size="0.02" mass="0.02" friction="0.9 0.005 0.0001"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="hipA" kp="5"/><position joint="hipB" kp="5"/></actuator>
</mujoco>
"""


def test_third_party_mjcf_robot_stands_in_the_lab():
    meta = RobotMeta(feet=[FootSpec("A", "footA", ["hipA"], "base"), FootSpec("B", "footB", ["hipB"], "base")],
                     logged_bodies=["base"], joint_tags={"hipA": "hip_pitch", "hipB": "hip_pitch"},
                     servos={"hipB": SERVO, "hipA": SERVO}, remove_geoms=("floor",), name="hopper")
    r = Robot.from_mjcf(THIRD_PARTY, meta)
    r.validate()
    assert r.total_mass() == pytest.approx(1.5 + 2 * 0.12)
    lab = ChironLab.from_mjcf(THIRD_PARTY, meta)
    assert lab.actuated_joints == ["hipA", "hipB"]                            # MuJoCo joint order
    m = lab.model
    assert m.nu == 2 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    np.testing.assert_allclose(m.jnt_range[1], [-math.radians(60), math.radians(60)])  # degrees resolved
    assert lab.nominal_base_height == pytest.approx(0.22, abs=1e-9)          # lowest foot on the ground
    for _ in range(1000):
        obs = lab.step(lab.nominal_command())
    assert obs.foot_normal_force.sum() == pytest.approx(lab.total_mass * 9.81, rel=0.01)
    assert obs.belly_contact.tolist() == [False]
    assert "chiron_terrain" in lab.xml
