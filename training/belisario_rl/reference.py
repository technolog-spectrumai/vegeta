"""The programmed gaits as reference trajectories: the knuckle walk of ``gorilla_motions`` (and the rise and the
drop) sampled at any time, in the environment's actuator order. Used by the optional imitation reward, by the
programmed controller (``controllers.ProgrammedGaitController``) and by ``compare_gaits``. The policy is never
forced onto these keyframes: the imitation weight is 0 by default."""
from __future__ import annotations

import sys

import numpy as np

from .robot_model import DESIGNS, RobotModel

__all__ = ["ReferenceGait"]


class ReferenceGait:
    def __init__(self, robot: RobotModel, speed: float | None = None):
        if str(DESIGNS) not in sys.path:
            sys.path.insert(0, str(DESIGNS))
        import gorilla_motions as gm
        import motion_kit as mk
        kin = mk.Kin(robot.robot)
        self.kin = kin
        self.q_quad = gm.quad_pose(kin)
        kw = {} if speed is None else {"speed": speed}
        self.walk = gm.knuckle_walk(kin, self.q_quad, **kw)
        self.rise = gm.rise(kin, self.q_quad)
        self.order = [kin.index[n] for n in robot.joints.names]
        self.speed = float(self.walk.base["velocity_m_s"][0])
        self.period = float(self.walk.period)
        self.height, self.pitch = gm.PELVIS_H, gm.PELVIS_PITCH

    def _frame(self, motion, t, loop=True):
        Q, keys = np.asarray(motion.Q), len(motion.t)
        if loop:
            s = (t % motion.period) / motion.period * keys
            i, f = int(s) % keys, s - int(s)
            j = (i + 1) % keys
        else:
            s = min(max(t / motion.period, 0.0), 1.0) * (keys - 1)
            i, f = min(int(s), keys - 2), min(s - int(s), 1.0)
            j = i + 1
        q = np.asarray(self.q_quad, dtype=float).copy()
        q[[self.kin.index[n] for n in motion.joints]] = (1 - f) * Q[i] + f * Q[j]
        return q

    def walk_at(self, t: float) -> np.ndarray:
        """The knuckle walk's joints at time ``t`` (periodic), actuator order."""
        return self._frame(self.walk, t)[self.order]

    def quad(self) -> np.ndarray:
        return np.asarray(self.q_quad)[self.order]

    def rise_at(self, t: float) -> np.ndarray:
        return self._frame(self.rise, t, loop=False)[self.order]
