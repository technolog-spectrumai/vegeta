"""Reset, joint-limit enforcement, determinism, termination and the scenes."""
import numpy as np

from kerchak_rl.commands import Command


def test_reset_puts_the_frame_on_four(env):
    obs, info = env.reset(seed=5)
    pos, vel = env.base_state()
    assert abs(pos[2] - env.default["height"]) < 0.05
    assert np.allclose(vel, 0.0)
    assert set(info) >= {"critic_obs", "command", "sample"}


def test_reset_is_deterministic_in_eval(env):
    o1, _ = env.reset(seed=11, options={"command": Command(0.2)})
    o2, _ = env.reset(seed=11, options={"command": Command(0.2)})
    assert np.allclose(o1, o2)


def test_joint_targets_within_limits(env):
    env.reset(seed=3)
    j = env.joints
    for a in (np.ones(env.act_dim) * 5.0, -np.ones(env.act_dim) * 5.0):
        env.step(a)
        tgt = env.joint_targets()
        assert np.all(tgt >= j.lower - 1e-9) and np.all(tgt <= j.upper + 1e-9)
    assert np.all(np.abs(env.action) <= env.cfg.action.clip)


def test_torques_within_the_servo_law(env):
    env.reset(seed=4)
    env.step(np.ones(env.act_dim))
    tau = env.pd.last_torque
    assert np.all(np.abs(tau) <= env.joints.stall * env.cfg.action.torque_fraction + 1e-9)


def test_episode_ends_by_time_limit(env):
    env.reset(seed=6, options={"command": Command(0.0, stop=True)})
    done, steps = False, 0
    while not done and steps < env.max_steps + 5:
        _, _, term, trunc, _ = env.step(np.zeros(env.act_dim))
        done = term or trunc
        steps += 1
    assert done and steps <= env.max_steps


def test_scene_registry_builds(cfg):
    from kerchak_rl.scenes import build_scene, scene_names
    assert {"flat", "rough", "door", "window", "tree", "hull", "obstacles", "rope"} <= set(scene_names())
    s = build_scene("obstacles", cfg, {"count": 1})
    assert s.goal is not None and s.props
