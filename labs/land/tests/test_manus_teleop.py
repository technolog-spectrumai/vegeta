"""Glove teleoperation of Manus's pincers (manus_teleop, docs/bionic.md P2): the bench builds with its wrist rolls and
straight jaws, the grasp height keeps the tips off the table, and two tasks succeed end to end at a low latency (the
simulated gloves, the downlink, the retargeting, the vibration feedback, the operator model and the MuJoCo physics)."""
import pytest

pytest.importorskip("mujoco")

import manus_teleop as mt
import onager_manus_robot as omr


def test_bench_has_wrist_rolls_and_straight_jaws():
    lab = mt.make_lab(mt.task_scene("pick_place"), mt.bench())
    assert {"L_roll", "R_roll"} <= set(lab.joint_names)
    import mujoco
    assert mujoco.mj_name2id(lab.model, mujoco.mjtObj.mjOBJ_GEOM, "L_jaw_upper_hook") == -1      # no hooked tips


def test_grasp_height_keeps_the_tips_off_the_table():
    Lj = omr.arm_geometry(omr.design_params(mt.BENCH_ARM))["Lj"]
    z = mt.grasp_pin_z(mt.TABLE_TOP + mt.BLOCK["size"] / 2)
    assert z - Lj > mt.TABLE_TOP


@pytest.mark.parametrize("task", ["pick_place", "catch"])
def test_task_succeeds_at_low_latency(task):
    r = mt.run_task(task, latency_s=0.02, seed=0, duration=30.0)
    assert r["success"], r
    assert r["steps_done"] == r["steps"]
    link = r["link"]["L"]
    assert link["lost"] == 0 and link["delivered"] > 100
