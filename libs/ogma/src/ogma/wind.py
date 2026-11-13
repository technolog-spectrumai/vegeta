"""og.wind — the air's velocity [m/s, world frame] a vehicle flies or drives through.

``Wind``: a steady vector plus turbulence — an Ornstein–Uhlenbeck process per axis with standard deviation ``sigma``
(horizontal; half of it vertical) and time constant ``tau`` (exact discretisation, so any step size gives the same
statistics) — plus an optional discrete 1−cos gust (``gust`` = (amplitude [m/s], start [s], duration [s], direction
unit vector)). Seeded: a trial repeats exactly. Terrain or obstacle effects (slope winds, lee turbulence, wakes)
subclass it and override ``at``. Moved from Vegeta's ``nisus_robot.Wind`` (reproduced exactly: tests/test_wind.py).

Building blocks for other wind models (Velutina's): ``from_direction`` (a meteorological "from" direction → the
vector), ``linear_shear`` (+x per km above a base), ``power_law`` (the boundary-layer profile), ``ou_euler_step``
(an Ornstein–Uhlenbeck process stepped by Euler–Maruyama, per-axis scale, for sims that vary sigma in space).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Wind:
    steady: tuple = (0.0, 0.0, 0.0)
    sigma: float = 0.0
    tau: float = 12.5
    gust: tuple | None = None
    seed: int = 0

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.turb = np.zeros(3)

    def step(self, dt):
        if self.sigma > 0:
            s = np.array([self.sigma, self.sigma, 0.5 * self.sigma])
            a = math.exp(-dt / self.tau)
            self.turb = a * self.turb + math.sqrt(1 - a * a) * s * self.rng.standard_normal(3)

    def at(self, t) -> np.ndarray:
        w = np.asarray(self.steady, float) + self.turb
        if self.gust is not None:
            amp, t0, dur, direction = self.gust
            if t0 <= t <= t0 + dur:
                w = w + amp * 0.5 * (1 - math.cos(2 * math.pi * (t - t0) / dur)) * np.asarray(direction, float)
        return w


def from_direction(speed: float, direction_deg: float) -> np.ndarray:
    """The horizontal wind vector [m/s] of a wind blowing FROM ``direction_deg`` (meteorological, x = 0°, y = 90°)."""
    a = math.radians(direction_deg)
    return -speed * np.array([math.cos(a), math.sin(a), 0.0])


def linear_shear(altitude: float, base: float, per_km: float) -> float:
    """The speed factor 1 + ``per_km`` × (height above ``base``) / 1000 (no reduction below the base)."""
    return 1.0 + per_km * max(0.0, (altitude - base)) / 1000.0


def power_law(speed_ref: float, z: float, z_ref: float, alpha: float = 0.14) -> float:
    """The boundary-layer profile u(z) = u_ref (z / z_ref)^α (α ≈ 0.14 over open land)."""
    return speed_ref * (z / z_ref) ** alpha


def ou_euler_step(x, dt: float, sigma: float, tau: float, rng, axes=(1.0, 1.0, 0.4)) -> np.ndarray:
    """One Euler–Maruyama step of an Ornstein–Uhlenbeck process: x + (−x/τ) dt + σ √(2 dt/τ) N(0, 1) · ``axes``
    (Velutina's gusts, axes (1, 1, 0.4); its navigation error, τ = 20 s, axes (1, 1, 0.5)). Draws 3 normals."""
    return x + ((-x / tau) * dt + sigma * math.sqrt(2 * dt / tau) * rng.standard_normal(3) * np.asarray(axes, dtype=float))
