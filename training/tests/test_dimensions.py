"""Observation and action dimensions, the layout, and the robot model's tables."""
import numpy as np

from kerchak_rl.commands import COMMAND_DIM
from kerchak_rl.env import PRIVILEGED_DIM, ObsLayout
from kerchak_rl.robot_model import CONTACT_GEOMS


def test_layout_matches_env(env):
    layout = ObsLayout(env.n, env.act_dim)
    assert env.obs_dim == layout.dim == 2 * env.n + env.act_dim + 3 + 3 + 3 + len(CONTACT_GEOMS) + COMMAND_DIM
    assert env.act_dim == len(env.act_index) == env.n - 26                    # the hands held (action.hands = held)
    obs, info = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    assert info["critic_obs"].shape == (PRIVILEGED_DIM,)
    assert np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= env.cfg.obs.clip)


def test_blocks_cover_the_vector(env):
    layout = ObsLayout(env.n, env.act_dim)
    covered = sum(s.stop - s.start for s in layout.slices.values())
    assert covered == layout.dim
    assert layout.slices["command"].stop == layout.dim


def test_robot_tables(robot):
    j = robot.joints
    assert j.n == robot.model.nu == len(robot.robot.actuated_joints())    # Talia's 21, the spine's 4, the hands' 32
    assert np.all(j.upper > j.lower) and np.all(j.stall > 0) and np.all(j.no_load_speed > 0)
    assert set(robot.contact_geoms) == set(CONTACT_GEOMS)
    assert robot.q_quad["q"].shape == (j.n,) and 0.5 < robot.q_quad["height"] < 0.8
    assert "sensors" in " ".join(robot.missing_requirements())            # stated, not invented


def test_step_shapes(env):
    env.reset(seed=2)
    obs, r, term, trunc, info = env.step(np.zeros(env.act_dim))
    assert obs.shape == (env.obs_dim,) and isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    assert info["critic_obs"].shape == (PRIVILEGED_DIM,) and "reward_terms" in info
