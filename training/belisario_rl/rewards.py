"""The reward: weighted terms (``RewardConfig``) over one control step's state (``StepState``). Every term is a
plain function of the state so a test can call it; ``RewardComputer.compute`` sums them with their weights and
returns the breakdown for TensorBoard.

The terms: tracking of the commanded linear and angular velocity (Gaussian kernels), the pelvis's height against
the commanded one, orientation (the projected gravity's lateral part and the pitch against the commanded one),
contact stability (three or more contacts down while moving slowly, no contact flicker), slip (tangential speed
of a contact in contact), energy (|τ·q̇|), action rate, joint accelerations and velocities, limb coordination (the
lateral sequence: a hand and the foot of the same side not both in the air), joint limits (closeness to the
range's ends), body collisions (the trunk or the head touching the ground), the fall, torque saturation, an alive
bonus, the stop (when the command says stop: stillness rewarded, motion not credited), an optional imitation of
the programmed knuckle walk, and goal progress for the scenes with a goal."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["StepState", "RewardComputer", "TERMS"]


@dataclass
class StepState:
    dt: float
    lin_vel: np.ndarray                  #: the base's linear velocity in its own frame [m/s]
    ang_vel: np.ndarray                  #: the base's angular velocity in its own frame [rad/s]
    cmd_vx: float
    cmd_vy: float
    cmd_yaw: float
    cmd_height: float
    stop: bool
    height: float                        #: the pelvis above the ground
    gravity_body: np.ndarray             #: the gravity direction in the base frame (unit)
    pitch_err: float                     #: the base's pitch off the commanded one [rad]
    contacts: np.ndarray                 #: 4 bools: L foot, R foot, L knuckle, R knuckle
    contacts_prev: np.ndarray
    contact_forces: np.ndarray           #: 4 normal forces [N]
    slip: np.ndarray                     #: 4 tangential speeds of contacts in contact [m/s]
    tau: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    action: np.ndarray
    prev_action: np.ndarray
    body_collision: bool
    fell: bool
    saturation: np.ndarray               #: |τ| / τ_max(q̇) per joint
    q_ref: np.ndarray | None = None      #: the programmed gait's joints at this time, when imitation is on
    goal_progress: float = 0.0           #: metres closer to the goal this step (0 without a goal)
    goal_reached: bool = False
    speed_norm: float = 0.4              #: the command range's largest speed (for the stillness scale)
    extras: dict = field(default_factory=dict)


def _exp(err2, sigma):
    return float(np.exp(-err2 / max(sigma, 1e-6)))


def track_lin_vel(s: StepState):
    if s.stop:
        return 0.0
    err = (s.lin_vel[0] - s.cmd_vx) ** 2 + (s.lin_vel[1] - s.cmd_vy) ** 2
    return _exp(err, _sigma(s, "tracking_sigma"))


def _sigma(s, key):
    return s.extras.get(key, 0.25) ** 2


def track_ang_vel(s: StepState):
    if s.stop:
        return 0.0
    return _exp((s.ang_vel[2] - s.cmd_yaw) ** 2, _sigma(s, "tracking_sigma"))


def height(s: StepState):
    return _exp((s.height - s.cmd_height) ** 2, s.extras.get("height_sigma", 0.08) ** 2)


def orientation(s: StepState):
    """Lateral tilt (the gravity's y part in the base) squared, plus the pitch error squared: a penalty."""
    return float(s.gravity_body[1] ** 2 + s.pitch_err ** 2)


def contact_stability(s: StepState):
    moving_fast = abs(s.cmd_vx) > 0.25 or abs(s.cmd_vy) > 0.15
    n = int(np.sum(s.contacts))
    flicker = int(np.sum(s.contacts != s.contacts_prev))
    steady = 1.0 if (n >= 3 or (moving_fast and n >= 2)) else 0.0
    return steady - 0.25 * flicker


def slip(s: StepState):
    return float(np.sum(s.slip ** 2))


def energy(s: StepState):
    return float(np.sum(np.abs(s.tau * s.qd)))


def action_rate(s: StepState):
    return float(np.sum((s.action - s.prev_action) ** 2))


def joint_accel(s: StepState):
    return float(np.sum(s.qdd ** 2))


def joint_vel(s: StepState):
    return float(np.sum(s.qd ** 2))


def coordination(s: StepState):
    """The lateral sequence keeps a side's hand and foot out of the air together: 1 when neither side has both up."""
    lf, rf, lk, rk = s.contacts
    same_side_up = (not lf and not lk) or (not rf and not rk)
    if s.stop:
        return 1.0 if np.all(s.contacts) else 0.0
    return 0.0 if same_side_up else 1.0


def joint_limits(s: StepState):
    span = np.maximum(s.upper - s.lower, 1e-6)
    margin = 0.05 * span
    below = np.clip(s.lower + margin - s.q, 0.0, None)
    above = np.clip(s.q - (s.upper - margin), 0.0, None)
    return float(np.sum((below + above) / margin))


def body_collision(s: StepState):
    return 1.0 if s.body_collision else 0.0


def fall(s: StepState):
    return 1.0 if s.fell else 0.0


def torque_saturation(s: StepState):
    return float(np.mean(np.clip(s.saturation - 0.9, 0.0, None)))


def alive(s: StepState):
    return 0.0 if s.fell else 1.0


def stop_still(s: StepState):
    """When the command says stop: stillness of the base (speeds over the range's scale), nothing otherwise."""
    if not s.stop:
        return 0.0
    v = np.sqrt(s.lin_vel[0] ** 2 + s.lin_vel[1] ** 2) / max(s.speed_norm, 1e-6)
    w = abs(s.ang_vel[2]) / 0.5
    return _exp(v ** 2 + w ** 2, 0.3 ** 2)


def imitation(s: StepState):
    if s.q_ref is None:
        return 0.0
    return _exp(float(np.mean((s.q - s.q_ref) ** 2)), 0.3 ** 2)


def goal(s: StepState):
    return float(s.goal_progress) + (5.0 if s.goal_reached else 0.0)


TERMS = {"track_lin_vel": track_lin_vel, "track_ang_vel": track_ang_vel, "height": height, "orientation": orientation,
         "contact_stability": contact_stability, "slip": slip, "energy": energy, "action_rate": action_rate, "joint_accel": joint_accel,
         "joint_vel": joint_vel, "coordination": coordination, "joint_limits": joint_limits, "body_collision": body_collision, "fall": fall,
         "torque_saturation": torque_saturation, "alive": alive, "stop_still": stop_still, "imitation": imitation, "goal": goal}


class RewardComputer:
    """``compute(state)`` → (total, {term: weighted value}); weights from ``RewardConfig`` (a term with weight 0 is
    skipped; ``goal`` takes the ``goal`` weight, default 1 when the scene has a goal)."""

    def __init__(self, cfg, goal_weight: float = 1.0):
        self.weights = {k: float(getattr(cfg, k)) for k in TERMS if hasattr(cfg, k)}
        self.weights["goal"] = goal_weight
        self.sigmas = {"tracking_sigma": float(cfg.tracking_sigma), "height_sigma": float(cfg.height_sigma)}

    def compute(self, s: StepState):
        s.extras.update(self.sigmas)
        out, total = {}, 0.0
        for name, fn in TERMS.items():
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            v = w * fn(s) * (s.dt if name in ("energy", "joint_accel", "joint_vel", "slip") else 1.0)
            out[name] = v
            total += v
        return float(total), out
