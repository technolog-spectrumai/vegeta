"""The tests' fixtures: the package on the path; a robot model and an environment built once per session (the
MuJoCo build takes a few seconds). Tests that need torch, gymnasium or onnx skip without them."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg():
    from belisario_rl.config import load_config
    c = load_config(ROOT / "configs" / "base.yaml")
    c.ppo.num_envs = 2
    c.ppo.steps_per_env = 4
    c.sim.episode_seconds = 1.0
    return c


@pytest.fixture(scope="session")
def robot(cfg):
    from belisario_rl.robot_model import RobotModel
    return RobotModel(sim_hz=cfg.sim.sim_hz)


@pytest.fixture(scope="session")
def env(cfg):
    from belisario_rl.env import BelisarioEnv
    e = BelisarioEnv(cfg, seed=0, eval_mode=True)
    yield e
    e.close()
