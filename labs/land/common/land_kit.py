"""What the land studies repeated (``from land_kit import add_rocks``): terrain profiles and pin stresses."""
from __future__ import annotations

import math

import numpy as np


def add_rocks(x, z, height_m, width_m, spacing_m, seed):
    rng = np.random.default_rng(seed)
    z = z.copy()
    centres = np.arange(spacing_m, x[-1] - spacing_m, spacing_m)
    for xc in centres + rng.uniform(-0.3, 0.3, len(centres)):
        mask = np.abs(x - xc) < width_m / 2
        z[mask] += height_m * np.cos(math.pi * (x[mask] - xc) / width_m)
    return z


def add_drop(x, z, at_m, depth_m):
    z = z.copy(); z[x > at_m] -= depth_m
    return z


def pin_bending(F, lever_mm, d_mm): return F * lever_mm / (math.pi * d_mm**3 / 32)


def pin_shear(F, d_mm, n=2): return F / (n * math.pi * d_mm**2 / 4)
