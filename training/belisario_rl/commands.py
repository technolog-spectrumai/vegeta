"""The command interface: what any controller of the robot, learned or programmed, is asked to do - a forward and
a lateral speed, a yaw rate, a torso (pelvis) height, or a stop. ``Command`` is the value, ``CommandRanges`` the
configured limits it is normalised by, ``CommandSampler`` draws training commands, and ``CommandSource``s feed a
controller at run time (constant, resampled, or from the keys)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Command", "CommandRanges", "CommandSampler", "CommandSource", "ConstantCommand", "SampledCommand", "COMMAND_DIM"]

COMMAND_DIM = 5                                                           #: vx, vy, yaw rate, height, stop


@dataclass
class Command:
    vx: float = 0.0                      #: forward [m/s]
    vy: float = 0.0                      #: left [m/s]
    yaw_rate: float = 0.0                #: [rad/s]
    height: float = 0.62                 #: the pelvis above the ground [m]
    stop: bool = False                   #: stop and hold the pose

    def vector(self) -> np.ndarray:
        """Raw values; ``stop`` as 1.0 / 0.0."""
        return np.array([self.vx, self.vy, self.yaw_rate, self.height, 1.0 if self.stop else 0.0], dtype=np.float32)

    @staticmethod
    def from_vector(v) -> "Command":
        v = np.asarray(v, dtype=float)
        return Command(float(v[0]), float(v[1]), float(v[2]), float(v[3]), bool(v[4] > 0.5))

    def effective(self) -> "Command":
        """The command as the controller should take it: a stop zeroes the speeds."""
        if self.stop:
            return Command(0.0, 0.0, 0.0, self.height, True)
        return self

    def is_still(self, tol: float = 0.03) -> bool:
        return self.stop or (abs(self.vx) < tol and abs(self.vy) < tol and abs(self.yaw_rate) < tol)


@dataclass
class CommandRanges:
    vx: tuple = (-0.3, 0.4)
    vy: tuple = (-0.15, 0.15)
    yaw_rate: tuple = (-0.5, 0.5)
    height: tuple = (0.5, 0.7)

    @staticmethod
    def from_config(cfg) -> "CommandRanges":
        return CommandRanges(tuple(cfg.vx), tuple(cfg.vy), tuple(cfg.yaw_rate), tuple(cfg.height))

    def _scale(self):
        return np.array([max(abs(self.vx[0]), abs(self.vx[1]), 1e-6), max(abs(self.vy[0]), abs(self.vy[1]), 1e-6),
                         max(abs(self.yaw_rate[0]), abs(self.yaw_rate[1]), 1e-6), 1.0, 1.0], dtype=np.float32)

    def normalize(self, c: Command) -> np.ndarray:
        """The command as the policy sees it: speeds scaled by their largest magnitude, the height as its offset
        from the range's middle over its half-span, stop as a flag."""
        v = c.effective().vector()
        out = v / self._scale()
        mid, half = 0.5 * (self.height[0] + self.height[1]), max(0.5 * (self.height[1] - self.height[0]), 1e-6)
        out[3] = (v[3] - mid) / half
        return out.astype(np.float32)

    def clip(self, c: Command) -> Command:
        lo = (self.vx[0], self.vy[0], self.yaw_rate[0], self.height[0])
        hi = (self.vx[1], self.vy[1], self.yaw_rate[1], self.height[1])
        return Command(float(np.clip(c.vx, lo[0], hi[0])), float(np.clip(c.vy, lo[1], hi[1])), float(np.clip(c.yaw_rate, lo[2], hi[2])),
                       float(np.clip(c.height, lo[3], hi[3])), c.stop)


class CommandSampler:
    """Training commands: uniform in the ranges, a stop with ``stop_probability``; ``hold_still_speed`` turns
    near-zero draws into stops too, so the policy learns to stand."""

    def __init__(self, ranges: CommandRanges, stop_probability: float = 0.15, hold_still_speed: float = 0.03, rng=None):
        self.ranges, self.p_stop, self.still = ranges, stop_probability, hold_still_speed
        self.rng = rng or np.random.default_rng()

    def sample(self) -> Command:
        r = self.ranges
        c = Command(float(self.rng.uniform(*r.vx)), float(self.rng.uniform(*r.vy)), float(self.rng.uniform(*r.yaw_rate)),
                    float(self.rng.uniform(*r.height)), False)
        if self.rng.random() < self.p_stop or c.is_still(self.still):
            c = Command(0.0, 0.0, 0.0, c.height, True)
        return c


class CommandSource:
    """What feeds a controller at run time: ``current(t)`` → Command."""

    def reset(self):
        pass

    def current(self, t: float) -> Command:
        raise NotImplementedError


class ConstantCommand(CommandSource):
    def __init__(self, command: Command):
        self.command = command

    def current(self, t: float) -> Command:
        return self.command


class SampledCommand(CommandSource):
    """A new sampled command every ``resample_seconds``."""

    def __init__(self, sampler: CommandSampler, resample_seconds: float = 4.0):
        self.sampler, self.period = sampler, resample_seconds
        self.reset()

    def reset(self):
        self._next, self._cmd = 0.0, self.sampler.sample()

    def current(self, t: float) -> Command:
        if t >= self._next:
            self._cmd = self.sampler.sample()
            self._next = t + self.period
        return self._cmd
