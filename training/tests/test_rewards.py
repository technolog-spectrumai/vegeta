"""The reward terms on hand-made states."""
import numpy as np
import pytest

from belisario_rl.config import RewardConfig
from belisario_rl.rewards import RewardComputer, StepState, TERMS


def _state(**kw):
    n = 4
    base = dict(dt=0.02, lin_vel=np.array([0.3, 0.0, 0.0]), ang_vel=np.zeros(3), cmd_vx=0.3, cmd_vy=0.0, cmd_yaw=0.0, cmd_height=0.62,
                stop=False, height=0.62, gravity_body=np.array([0.0, 0.0, -1.0]), pitch_err=0.0,
                contacts=np.array([True, True, True, False]), contacts_prev=np.array([True, True, True, False]),
                contact_forces=np.zeros(4), slip=np.zeros(4), tau=np.zeros(n), q=np.zeros(n), qd=np.zeros(n), qdd=np.zeros(n),
                lower=-np.ones(n), upper=np.ones(n), action=np.zeros(n), prev_action=np.zeros(n), body_collision=False, fell=False,
                saturation=np.zeros(n))
    base.update(kw)
    return StepState(**base)


def test_tracking_peaks_at_the_command():
    rc = RewardComputer(RewardConfig())
    total_good, terms_good = rc.compute(_state())
    total_bad, terms_bad = rc.compute(_state(lin_vel=np.array([-0.3, 0.0, 0.0])))
    assert terms_good["track_lin_vel"] > terms_bad["track_lin_vel"]
    assert total_good > total_bad


def test_stop_gives_no_credit_for_motion():
    rc = RewardComputer(RewardConfig())
    _, moving = rc.compute(_state(stop=True, cmd_vx=0.0, lin_vel=np.array([0.4, 0.0, 0.0]), contacts=np.ones(4, bool), contacts_prev=np.ones(4, bool)))
    _, still = rc.compute(_state(stop=True, cmd_vx=0.0, lin_vel=np.zeros(3), contacts=np.ones(4, bool), contacts_prev=np.ones(4, bool)))
    assert moving["track_lin_vel"] == 0.0 and still["stop_still"] > moving["stop_still"]


def test_penalties_are_negative():
    rc = RewardComputer(RewardConfig())
    _, t = rc.compute(_state(fell=True, body_collision=True, slip=np.ones(4), q=np.array([0.99, -0.99, 0.0, 0.0]), saturation=np.ones(4),
                             action=np.ones(4), qdd=np.ones(4) * 100))
    for k in ("fall", "body_collision", "slip", "joint_limits", "torque_saturation", "action_rate", "joint_accel"):
        assert t[k] < 0, k
    assert t["alive"] == 0.0


def test_coordination_and_contacts():
    assert TERMS["coordination"](_state(contacts=np.array([False, True, False, True]))) == 0.0   # the left side all in the air
    assert TERMS["coordination"](_state(contacts=np.array([False, True, True, True]))) == 1.0
    assert TERMS["contact_stability"](_state(contacts=np.ones(4, bool), contacts_prev=np.ones(4, bool))) == pytest.approx(1.0)
    assert TERMS["contact_stability"](_state(contacts=np.array([True, False, False, False]), contacts_prev=np.ones(4, bool))) < 0


def test_goal_progress():
    rc = RewardComputer(RewardConfig(), goal_weight=1.0)
    _, t = rc.compute(_state(goal_progress=0.1, goal_reached=True))
    assert t["goal"] == pytest.approx(5.1)
    rc0 = RewardComputer(RewardConfig(), goal_weight=0.0)
    _, t0 = rc0.compute(_state(goal_progress=0.1))
    assert "goal" not in t0
