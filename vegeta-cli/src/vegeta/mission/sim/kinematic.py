"""A kinematic fixed-wing aircraft that follows ``GuidanceCmd`` (course, height, airspeed) within a bank limit, a roll
rate, a climb rate and a speed time constant: fast enough to test the whole mission software in seconds. The
MuJoCo aircraft (notebooks/designs/nisus_*) replaces it for the full missions."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..bus import wrap
from ..messages import NavState

__all__ = ["KinematicAircraft"]

G = 9.81


@dataclass
class KinematicAircraft:
    pos: np.ndarray
    course: float = 0.0
    V: float = 15.0
    bank: float = 0.0
    bank_max_deg: float = 40.0
    roll_rate_deg: float = 90.0
    climb_max: float = 3.0
    tau_V: float = 3.0
    k_course: float = 1.6
    wind: tuple = (0.0, 0.0, 0.0)
    vz: float = 0.0
    p_rate: float = 0.0
    r_rate: float = 0.0

    def step(self, cmd, dt):
        if cmd is not None and cmd.valid:
            err = wrap(cmd.course - self.course)
            bank_cmd = float(np.clip(self.k_course * err, -1, 1)) * math.radians(self.bank_max_deg)
            self.vz = float(np.clip(0.35 * (cmd.height - self.pos[2]), -self.climb_max, self.climb_max))
            self.V += (cmd.airspeed - self.V) * dt / self.tau_V
        else:
            bank_cmd = math.radians(20.0)
        db = float(np.clip(bank_cmd - self.bank, -math.radians(self.roll_rate_deg) * dt, math.radians(self.roll_rate_deg) * dt))
        self.p_rate = db / dt
        self.bank += db
        self.r_rate = G * math.tan(self.bank) / max(self.V, 5.0)
        self.course = wrap(self.course + self.r_rate * dt)
        w = np.asarray(self.wind, float)
        self.pos = self.pos + (np.array([self.V * math.cos(self.course), self.V * math.sin(self.course), self.vz]) + w) * dt

    def R_wb(self):
        gamma = math.atan2(self.vz, self.V)
        th = gamma + math.radians(3.0)
        cy, sy = math.cos(self.course), math.sin(self.course)
        cp, sp = math.cos(th), math.sin(th)
        cr, sr = math.cos(self.bank), math.sin(self.bank)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])          # nose up: body x tilts towards +z
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])          # positive bank: left wing down, a left turn (course increases)
        return Rz @ Ry @ Rx.T

    def nav(self, t, margin_wh=float("inf")):
        v = np.array([self.V * math.cos(self.course), self.V * math.sin(self.course), self.vz]) + np.asarray(self.wind, float)
        return NavState(t, self.pos.copy(), v, self.R_wb(), np.array([self.p_rate, 0.0, self.r_rate]), self.V, margin_wh)
