"""Blade section aerodynamics: an explicit linear-lift / parabolic-drag model with a stall cap.

No Reynolds-number dependence and no polar database: the coefficients are engineer inputs. Use
measured polars by fitting these four numbers, or subclass and override ``coefficients``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Airfoil:
    name: str = "thin cambered section (generic)"
    cl_alpha: float = 2 * math.pi * 0.9   # lift slope [1/rad]; ~0.9 x thin-airfoil for low Re, finite thickness
    alpha0_deg: float = -2.0              # zero-lift angle [deg] (cambered: negative)
    cl_max: float = 1.1                   # lift is capped at +/- cl_max (flat post-stall)
    cd0: float = 0.02                     # drag at cl = 0
    k: float = 0.04                       # cd = cd0 + k * cl^2
    source: str = "generic values, not a measured polar"

    def __post_init__(self):
        if self.cl_alpha <= 0 or self.cl_max <= 0 or self.cd0 < 0 or self.k < 0:
            raise ValueError("airfoil coefficients must be positive (cd0, k may be zero)")

    def coefficients(self, alpha_rad):
        """``(cl, cd)`` arrays for angle of attack in radians (element-wise)."""
        a = np.asarray(alpha_rad, dtype=float)
        cl = np.clip(self.cl_alpha * (a - math.radians(self.alpha0_deg)), -self.cl_max, self.cl_max)
        cd = self.cd0 + self.k * cl**2
        stalled = np.abs(self.cl_alpha * (a - math.radians(self.alpha0_deg))) > self.cl_max
        cd = np.where(stalled, cd + 0.5 * (np.abs(a - math.radians(self.alpha0_deg)) - self.cl_max / self.cl_alpha), cd)
        return cl, cd
