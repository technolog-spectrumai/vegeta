"""Leg kinematics shared by the walking and the wheel-legged machines."""
from __future__ import annotations

import math


def ik_two_link_planar(x_foot, z_foot, L1, L2):
    """Dog leg in its plane: upper-leg angle behind vertical and lower-leg angle ahead of vertical (knee back) [deg]
    for a foot at (x ahead, z below the hip); None when out of reach."""
    r = math.hypot(x_foot, z_foot)
    if r > L1 + L2 - 1e-6 or r < abs(L1 - L2) + 1e-6:
        return None
    knee_int = math.acos(max(-1, min(1, (L1**2 + L2**2 - r**2) / (2 * L1 * L2))))
    phi = math.atan2(x_foot, -z_foot)
    alpha = math.acos(max(-1, min(1, (L1**2 + r**2 - L2**2) / (2 * L1 * r))))
    a1 = -(phi - alpha)
    a2 = math.pi - knee_int - a1
    return math.degrees(a1), math.degrees(a2)
