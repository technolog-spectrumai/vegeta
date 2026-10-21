"""The rotor law the races fly: the desaturating mixer, the rotors' wrench about the centre of mass, the frame's drag.

A local copy of Vegeta's parts library (``ogma.rotors``: ``mix``, ``wrench``, ``quadratic_drag``, copied unchanged on
2026-10-21). Piccolo never imports Vegeta; designs reach it as standard files, and the few functions a hatched machine
flies with are duplicated here (the rule: duplicate rather than import).
"""
from __future__ import annotations

import numpy as np


def mix(Ainv: np.ndarray, T: float, M, lo: float, hi: float) -> tuple:
    """Rotor thrusts [N] for a collective thrust ``T`` and body torques ``M`` within [lo, hi] per rotor; out of range,
    the yaw torque is dropped first, then roll and pitch are scaled down (bisection). Returns (thrusts, saturated)."""
    n = Ainv.shape[0]
    T = float(np.clip(T, n * lo, n * hi))
    u = Ainv @ np.array([T, M[0], M[1], M[2]])
    if lo - 1e-9 <= u.min() and u.max() <= hi + 1e-9:
        return u, False
    u = Ainv @ np.array([T, M[0], M[1], 0.0])
    if not (lo - 1e-9 <= u.min() and u.max() <= hi + 1e-9):
        k_lo, k_hi = 0.0, 1.0
        for _ in range(20):
            k = 0.5 * (k_lo + k_hi)
            v = Ainv @ np.array([T, k * M[0], k * M[1], 0.0])
            if lo - 1e-9 <= v.min() and v.max() <= hi + 1e-9:
                k_lo = k
            else:
                k_hi = k
        u = Ainv @ np.array([T, k_lo * M[0], k_lo * M[1], 0.0])
    return np.clip(u, lo, hi), True


def wrench(thrusts, R, pos, spin, kappa: float, com_offset=(0.0, 0.0, 0.0)) -> tuple:
    """World force (thrust only) and torque about the COM [N, N m] of rotors at ``pos`` (body frame, from the body
    origin) on a body with rotation ``R`` (body → world) whose COM sits at ``com_offset`` [m, world] from the origin
    (MuJoCo: ``xipos − xpos``)."""
    T = np.asarray(thrusts, dtype=float)
    zb = R[:, 2]
    arms = np.asarray(pos, dtype=float) @ R.T - np.asarray(com_offset, dtype=float)
    torque = np.cross(arms, np.outer(T, zb)).sum(axis=0) + zb * float(np.dot(spin, T)) * kappa
    return zb * T.sum(), torque


def quadratic_drag(v, cda: float, rho: float) -> np.ndarray:
    """The air's force on a body moving at ``v`` [m/s, relative to the air]: −½ ρ CdA |v| v."""
    v = np.asarray(v, dtype=float)
    return -0.5 * rho * cda * float(np.linalg.norm(v)) * v
