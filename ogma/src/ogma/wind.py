"""og.wind — the air's velocity [m/s, world frame] a vehicle flies or drives through.

``Wind``: a steady vector plus turbulence — an Ornstein–Uhlenbeck process per axis with standard deviation ``sigma``
(horizontal; half of it vertical) and time constant ``tau`` (exact discretisation, so any step size gives the same
statistics) — plus an optional discrete 1−cos gust (``gust`` = (amplitude [m/s], start [s], duration [s], direction
unit vector)). Seeded: a trial repeats exactly. Terrain or obstacle effects (slope winds, lee turbulence, wakes)
subclass it and override ``at``. Moved from Vegeta's ``nisus_robot.Wind`` (reproduced exactly: tests/test_wind.py).
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
