"""Sensor models for gloves and puppet arms: what a real sensor would report, with its errors, so calibration and
filtering are designed against them before any hardware exists.

* ``HallAngle`` — a Hall-effect sensor over a magnet that turns with a joint: the field seen is a sinusoid of the
  angle (the magnet's rotation), read by a 12-bit ADC. Each unit has its own offset, gain and phase (assembly), a
  share of its neighbour's angle (cross-talk: the neighbour's magnet), white noise and a slow thermal drift.
* ``Imu`` — gyroscope (bias, bias random walk, white noise), accelerometer (gravity + linear acceleration, bias,
  noise) and magnetometer (the Earth's field, a hard-iron offset, noise), in the sensor's frame.
* ``Encoder`` — an absolute magnetic encoder (puppet arm joints): quantised to its resolution, small noise.

All parameters are datasheet-class values (stated, not measured); ``seed`` makes each unit's errors reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _quat as Q

GRAVITY = np.array([0.0, 0.0, -9.81])
EARTH_FIELD = np.array([20.0, 0.0, -45.0])          # uT, mid-latitude northern hemisphere, x north (approximate)


@dataclass
class HallAngle:
    """counts = offset + amplitude * sin(k * (angle + phase) + c * neighbour) + noise, quantised to ``bits``."""

    amplitude: float = 1500.0                        # counts at the sinusoid's peak
    k: float = 1.0                                   # magnet poles: field cycles per joint revolution
    offset: float = 2048.0
    phase: float = 0.0
    crosstalk: float = 0.0
    noise: float = 2.0                               # counts rms
    drift_per_s: float = 0.0                         # counts per second (thermal)
    bits: int = 12

    @classmethod
    def unit(cls, rng: np.random.Generator, **over) -> "HallAngle":
        """One manufactured unit: offset, gain, phase and cross-talk scattered as on a hand-assembled glove."""
        p = dict(amplitude=1500.0 * rng.uniform(0.85, 1.15), offset=2048.0 + rng.normal(0, 60.0),
                 phase=rng.normal(0, math.radians(4.0)), crosstalk=rng.uniform(0.0, 0.05), noise=2.0,
                 drift_per_s=rng.normal(0, 0.02))
        p.update(over)
        return cls(**p)

    def read(self, angle: float, t: float, rng: np.random.Generator, neighbour: float = 0.0) -> int:
        v = self.offset + self.drift_per_s * t + self.amplitude * math.sin(self.k * (angle + self.phase) + self.crosstalk * neighbour)
        v += rng.normal(0, self.noise)
        return int(np.clip(round(v), 0, 2 ** self.bits - 1))


@dataclass
class Imu:
    """A 9-axis IMU (MEMS class). Units: rad/s, m/s^2, uT."""

    gyro_noise: float = 0.003                        # rad/s rms per sample (~0.17 deg/s)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_walk: float = 2e-5                     # rad/s per sqrt(s)
    acc_noise: float = 0.02                          # m/s^2 rms
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag_noise: float = 0.4                           # uT rms
    mag_hard_iron: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def unit(cls, rng: np.random.Generator, **over) -> "Imu":
        p = dict(gyro_bias=rng.normal(0, math.radians(1.0), 3), acc_bias=rng.normal(0, 0.05, 3),
                 mag_hard_iron=rng.normal(0, 3.0, 3))
        p.update(over)
        return cls(**p)

    def read(self, q, omega_body, acc_lin_world, dt: float, rng: np.random.Generator) -> tuple:
        """(gyro, accel, mag) in the sensor frame for orientation ``q`` (sensor to world), body angular velocity and
        world linear acceleration. Advances the gyro bias by its random walk over ``dt``."""
        self.gyro_bias = self.gyro_bias + rng.normal(0, self.gyro_bias_walk * math.sqrt(max(dt, 0.0)), 3)
        Rt = Q.to_R(q).T
        gyro = np.asarray(omega_body, float) + self.gyro_bias + rng.normal(0, self.gyro_noise, 3)
        acc = Rt @ (np.asarray(acc_lin_world, float) - GRAVITY) + self.acc_bias + rng.normal(0, self.acc_noise, 3)
        mag = Rt @ EARTH_FIELD + self.mag_hard_iron + rng.normal(0, self.mag_noise, 3)
        return gyro, acc, mag


@dataclass
class Encoder:
    """An absolute magnetic encoder: ``bits`` per revolution, a little noise in counts."""

    bits: int = 14
    noise: float = 0.5
    offset: float = 0.0

    def read(self, angle: float, rng: np.random.Generator) -> int:
        n = 2 ** self.bits
        c = (angle + self.offset) / (2 * math.pi) * n + rng.normal(0, self.noise)
        return int(round(c)) % n

    def angle(self, counts: int) -> float:
        a = counts / 2 ** self.bits * 2 * math.pi - self.offset
        return (a + math.pi) % (2 * math.pi) - math.pi
