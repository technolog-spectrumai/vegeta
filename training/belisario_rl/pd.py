"""The PD controller between the policy and the servos: normalised joint-position offsets → joint targets about
the default pose, clipped to the joint ranges → torques by Chiron's servo law (``vegeta.chiron.servo_torque``:
PD, clipped to the torque–speed line of each module, zero beyond its no-load speed), with an optional actuator
latency (a queue of targets) and the gains and limits scaled for domain randomization."""
from __future__ import annotations

from collections import deque

import numpy as np
from vegeta.chiron import servo_torque, torque_limit

from .robot_model import JointTable

__all__ = ["PDController"]


class PDController:
    def __init__(self, table: JointTable, default_pose: np.ndarray, *, action_scale: float = 0.5, action_clip: float = 1.0,
                 kp_fraction: float = 1.0, kd_fraction: float = 1.0, torque_fraction: float = 1.0, velocity_limit_fraction: float = 1.0,
                 latency_steps: int = 0):
        self.t = table
        self.default = np.asarray(default_pose, dtype=float).copy()
        self.scale, self.clip = float(action_scale), float(action_clip)
        self.kp = table.kp * kp_fraction
        self.kd = table.kd * kd_fraction
        self.stall = table.stall * torque_fraction
        self.w0 = table.no_load_speed * velocity_limit_fraction
        self.strength = np.ones(table.n)                                   # domain randomization's motor strength
        self.latency = int(latency_steps)
        self._queue = deque()
        self.last_target = self.default.copy()
        self.last_torque = np.zeros(table.n)

    def reset(self, latency_steps: int | None = None):
        if latency_steps is not None:
            self.latency = int(latency_steps)
        self._queue.clear()
        self.last_target = self.default.copy()
        self.last_torque[:] = 0.0

    def targets(self, action: np.ndarray) -> np.ndarray:
        """Joint targets [rad] from a normalised action: the default pose plus ``scale`` × the clipped action,
        kept within the joint ranges."""
        a = np.clip(np.asarray(action, dtype=float), -self.clip, self.clip)
        q = self.default + self.scale * a
        return np.clip(q, self.t.lower, self.t.upper)

    def command(self, action: np.ndarray) -> np.ndarray:
        """The target the servos see this control step, after the latency queue."""
        q = self.targets(action)
        if self.latency <= 0:
            self.last_target = q
            return q
        self._queue.append(q)
        while len(self._queue) > self.latency + 1:
            self._queue.popleft()
        self.last_target = self._queue[0] if len(self._queue) > self.latency else self.default
        return self.last_target

    def torques(self, q: np.ndarray, qd: np.ndarray, q_target: np.ndarray | None = None) -> np.ndarray:
        """The servo torques [N·m] for the joint state (Chiron's law), with the strength scale."""
        tgt = self.last_target if q_target is None else q_target
        tau = servo_torque(q, qd, tgt, kp=self.kp * self.strength, kd=self.kd, stall=self.stall * self.strength, no_load_speed=self.w0)
        self.last_torque = np.asarray(tau, dtype=float)
        return self.last_torque

    def saturation(self, qd: np.ndarray) -> np.ndarray:
        """|τ| over the torque bound at the current speed, per joint (1 = saturated)."""
        lim = torque_limit(qd, self.stall * self.strength, self.w0)
        return np.where(lim > 1e-9, np.abs(self.last_torque) / np.maximum(lim, 1e-9), 1.0)
