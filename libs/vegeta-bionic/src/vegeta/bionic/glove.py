"""A data glove: which joints carry a Hall angle sensor, the IMU on the back of the hand, and ``SimGlove``, which turns
a hand's motion into the samples a real glove would send.

``GloveSpec.default`` senses 14 joints: the thumb's four, MCP flexion of all four fingers, PIP of all four, and the
index's and little finger's abduction. The DIPs, and the middle's and ring's abduction, are not sensed: the estimate
infers them (DIP ~ 2/3 PIP, the coupling of the flexor tendons; ring abduction ~ half the little finger's).

Each sensor's magnet is mounted centred on its joint's range of motion and with enough poles that the joint's travel
stays within +-75 deg of the field's cycle: the sinusoid stays one-to-one, so an angle is recovered by arcsin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _quat as Q
from .hand import FINGERS, JOINTS, RANGES, HandPose, HandSize
from .sensors import HallAngle, Imu

DEFAULT_SENSED = ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
                  "index_mcp", "index_abd", "index_pip", "middle_mcp", "middle_pip",
                  "ring_mcp", "ring_pip", "little_mcp", "little_abd", "little_pip")
MAX_FIELD_HALF = math.radians(75.0)


def _neighbour(joint: str) -> str | None:
    """The joint whose magnet sits next to ``joint``'s sensor (cross-talk): MCP <-> PIP on a finger, CMC <-> MCP on the
    thumb."""
    if joint.startswith("thumb"):
        return {"thumb_cmc_flex": "thumb_mcp", "thumb_mcp": "thumb_cmc_flex"}.get(joint)
    f, part = joint.split("_", 1)
    other = {"mcp": "pip", "pip": "mcp"}.get(part)
    return f"{f}_{other}" if other else None


@dataclass(frozen=True)
class Mount:
    """How a sensor sits on its joint: the angle at the field's zero crossing (``centre``) and the poles ``k``."""

    centre: float
    k: float


def mount_for(joint: str) -> Mount:
    lo, hi = RANGES[joint]
    half = (hi - lo) / 2
    return Mount((lo + hi) / 2, min(1.0, MAX_FIELD_HALF / half))


@dataclass(frozen=True)
class GloveSpec:
    side: str = "R"
    sensed: tuple = DEFAULT_SENSED
    rate_hz: float = 100.0

    @classmethod
    def default(cls, side: str = "R") -> "GloveSpec":
        return cls(side=side)

    @property
    def mounts(self) -> dict:
        return {j: mount_for(j) for j in self.sensed}


@dataclass
class GloveSample:
    """One glove message: ``t`` [s, glove clock], ``seq``, ``hall`` (counts, in ``spec.sensed`` order), ``gyro``
    [rad/s], ``acc`` [m/s^2], ``mag`` [uT] in the IMU's frame (= the hand frame)."""

    t: float
    seq: int
    side: str
    hall: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    mag: np.ndarray


class SimGlove:
    """A simulated glove on a simulated hand: every unit's errors fixed by ``seed`` (``units``, ``imu``)."""

    def __init__(self, spec: GloveSpec | None = None, size: HandSize | None = None, seed: int = 0):
        self.spec = spec or GloveSpec.default()
        self.size = size or HandSize(side=self.spec.side)
        self.rng = np.random.default_rng(seed)
        self.units = {j: HallAngle.unit(self.rng) for j in self.spec.sensed}
        for j, m in self.spec.mounts.items():
            self.units[j].k = m.k
        self.imu = Imu.unit(self.rng)
        self.seq = 0
        self._prev = None

    def sample(self, t: float, pose: HandPose) -> GloveSample:
        """The glove's reading of ``pose`` at time ``t`` (the wrist's angular velocity and acceleration from the
        previous sample)."""
        dt = 1.0 / self.spec.rate_hz
        mounts = self.spec.mounts
        hall = np.array([self.units[j].read(pose.joints[j] - mounts[j].centre, t, self.rng,
                                            neighbour=pose.joints.get(_neighbour(j) or "", 0.0)) for j in self.spec.sensed])
        if self._prev is None:
            omega, acc = np.zeros(3), np.zeros(3)
        else:
            tp, qp, pp, vp = self._prev
            h = max(t - tp, 1e-9)
            omega = Q.angular_velocity(qp, pose.wrist_quat, h)
            v = (pose.wrist_pos - pp) / h
            acc = (v - vp) / h
        gyro, a, mag = self.imu.read(pose.wrist_quat, omega, acc, dt, self.rng)
        vel = np.zeros(3) if self._prev is None else (pose.wrist_pos - self._prev[2]) / max(t - self._prev[0], 1e-9)
        self._prev = (t, pose.wrist_quat.copy(), pose.wrist_pos.copy(), vel)
        s = GloveSample(t, self.seq, self.spec.side, hall, gyro, a, mag)
        self.seq += 1
        return s

    def run(self, motion, duration: float | None = None) -> tuple:
        """Sample ``motion`` (t -> HandPose) at the glove's rate: (samples, true poses)."""
        T = motion.duration if duration is None else duration
        n = int(T * self.spec.rate_hz) + 1
        samples, truth = [], []
        for k in range(n):
            t = k / self.spec.rate_hz
            p = motion(t)
            samples.append(self.sample(t, p))
            truth.append(p)
        return samples, truth
