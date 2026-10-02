"""Scenery in ChironLab: props (fixed, hinged, free), welds switched at run time, hooks, events, contact forces."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vegeta.chiron import ChironLab, Episode, Flat, Geom, Joint, Link, Prop, Robot, Servo, Weld  # noqa: E402

SERVO = Servo(stall_torque=20.0, rated_torque=8.0, no_load_speed=5.0, stall_current=5.0, voltage=24.0, kp=50.0,
              kd=1.0, source="test values")
G = 9.81


def toy_robot():
    """A 5.5 kg box on the ground with one actuated arm that swings about the vertical."""
    arm = Link("arm", pos=(0.2, 0, 0), joints=[Joint("arm_j", axis=(0, 0, 1), servo=SERVO)],
               geoms=[Geom("arm_g", "capsule", (0.02,), fromto=(0, 0, 0, 0.3, 0, 0), mass=0.5)])
    root = Link("base", log=True, geoms=[Geom("base_g", "box", (0.2, 0.15, 0.05), mass=5.0, role="body")],
                children=[arm])
    return Robot("toy", root, nominal_base_height=0.05)


def scenery():
    post = Prop(Link("post", pos=(1.0, 0.5, 0.0), geoms=[Geom("post_g", "box", (0.03, 0.03, 0.5), pos=(0, 0, 0.5))]))
    bar = Prop(Link("bar", pos=(1.06, 0.5, 1.0), joints=[Joint("bar_hinge", axis=(1, 0, 0))],      # beside the post
                    geoms=[Geom("bar_g", "capsule", (0.01,), fromto=(0, 0, 0, 0, -1.0, 0), mass=1.0)]))
    box = Prop(Link("box", pos=(0.6, -0.5, 0.1), geoms=[Geom("box_g", "box", (0.1, 0.1, 0.1), mass=2.0)]), free=True)
    return [post, bar, box], [Weld("bar_hold", "bar")]


@pytest.fixture()
def lab():
    props, welds = scenery()
    return ChironLab(toy_robot(), Flat(), props=props, welds=welds)


def bar_angle(log):
    q = log["prop_quat"][:, log["props"].index("bar")]
    return np.degrees(2 * np.arctan2(q[:, 1], q[:, 0]))


def test_props_are_not_robot(lab):
    assert lab.joint_names == ["arm_j"]                   # the bar's hinge and the box's free joint are scenery
    assert lab.actuated_joints == ["arm_j"]
    assert lab.total_mass == pytest.approx(5.5)
    assert lab.prop_bodies == ["post", "bar", "box"]
    assert lab.summary()["props"] == ["post", "bar", "box"]


def test_weld_holds_until_a_hook_releases_it(lab):
    calls = []

    def hook(l):
        calls.append(l.time)
        if l.time >= 0.5 and l.weld_active("bar_hold"):
            l.set_weld("bar_hold", False)
            l.log_event("hook", "released the bar")

    lab.add_hook(hook)
    ep = lab.run(lambda obs: None, duration=1.5, rules=None, settle=0.2)
    t, ang = ep.log["t"], bar_angle(ep.log)
    assert np.abs(ang[t < 0.5]).max() < 0.5                # welded: the bar stays horizontal
    assert ang.max() > 80.0                                 # released: it swings down on its hinge
    assert ep.log["events"] == [[pytest.approx(0.5), "hook", "released the bar"]]
    assert ep.log["welds"] == {"bar_hold": False}
    assert len(calls) == 1501 and min(calls) == 0.0         # every control step of walking time, not the settle
    lab.reset()
    assert lab.events == [] and lab.weld_active("bar_hold")  # reset restores the weld and clears the events


def test_free_prop_rests_and_contact_force_is_its_weight(lab):
    ep = lab.run(lambda obs: None, duration=0.5, rules=None, settle=0.0)
    z = ep.log["prop_pos"][:, ep.log["props"].index("box")][:, 2]
    assert abs(z[-1] - 0.1) < 2e-3
    lab.observe(sync=True)
    f, fn = lab.contact_force("chiron_terrain", "box_g")
    assert fn == pytest.approx(2.0 * G, rel=0.02)
    assert f[2] == pytest.approx(2.0 * G, rel=0.02)
    f2, fn2 = lab.contact_force("box_g", "chiron_terrain")
    assert f2[2] == pytest.approx(-f[2]) and fn2 == pytest.approx(fn)


def test_robot_pushes_a_prop():
    """The robot's geoms collide with props (not only with the terrain)."""
    box = Prop(Link("box", pos=(0.38, 0.2, 0.1), geoms=[Geom("box_g", "box", (0.05, 0.05, 0.1), mass=0.3)]), free=True)
    lab = ChironLab(toy_robot(), Flat(), props=[box])

    def swing(obs):
        from vegeta.chiron import Command
        return Command(q_target={"arm_j": 1.4})

    ep = lab.run(swing, duration=1.0, rules=None, settle=0.0)
    y = ep.log["prop_pos"][:, 0, 1]
    assert y[-1] > 0.3                                     # the arm swept the box aside


def test_episode_round_trip_keeps_scenery(lab, tmp_path):
    lab.add_hook(lambda l: l.log_event("t0", "start") if l.time == 0.0 else None)
    ep = lab.run(lambda obs: None, duration=0.2, rules=None, settle=0.0)
    back = Episode.load(ep.save(tmp_path / "ep.npz"))
    assert back.log["events"] == [[0.0, "t0", "start"]]
    assert back.log["welds"] == {"bar_hold": True}
    np.testing.assert_allclose(back.log["prop_pos"], ep.log["prop_pos"])


def test_scenery_validation():
    robot = toy_robot()
    clash = Prop(Link("base", geoms=[Geom("x", "box", (0.1, 0.1, 0.1))]))
    with pytest.raises(ValueError, match="duplicate link"):
        ChironLab(robot, props=[clash])
    driven = Prop(Link("gate", joints=[Joint("gate_j", servo=SERVO)], geoms=[Geom("gate_g", "box", (0.1, 0.1, 0.1),
                                                                                  mass=1.0)]))
    with pytest.raises(ValueError, match="passive"):
        ChironLab(robot, props=[driven])
    with pytest.raises(ValueError, match="no body"):
        ChironLab(robot, welds=[Weld("w", "nowhere")])
    lab = ChironLab(robot, props=scenery()[0], welds=scenery()[1])
    with pytest.raises(KeyError):
        lab.set_weld("missing", False)
    with pytest.raises(TypeError):
        lab.add_hook(42)
    assert "<equality>" in lab.xml and 'name="bar_hold"' in lab.xml
    assert math.isclose(lab.total_mass, 5.5)
