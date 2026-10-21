"""og.rotors — a multirotor's rotors as forces: layout, allocation and the desaturating mixer, thrust lag, the wrench
about the centre of mass, frame drag, and Glauert's induced velocity at a disk.

Conventions: body frame x forward, y left, z up; spin +1 when the propeller's drag torque on the frame is +z;
κ = torque per thrust [m]. Consolidates Vegeta's Drongo (``drongo_robot.rotor_layout`` / ``Rotors``,
``drongo_controller.Flight.mix``), the races' copies, and ``kestrel_flight.induced_velocity`` (byte-identical to
``aeromant.transitions``); reproduced exactly (tests/test_rotors.py). The torques act about the COM — the races'
copy took them about the body origin (a few-mm lever error, fixed by using this).
"""
from __future__ import annotations

import math

import numpy as np

X_SPIN = np.array([1.0, -1.0, 1.0, -1.0])


def x_layout(radius: float, z: float = 0.0, spin=X_SPIN) -> tuple:
    """Hub positions (4, 3) [m] of an X — arms at 45°, 135°, 225°, 315° — and the spin signs."""
    pos = np.array([[radius * math.cos(math.radians(45 + 90 * k)), radius * math.sin(math.radians(45 + 90 * k)), z] for k in range(4)])
    return pos, np.asarray(spin, dtype=float)


def allocation(pos, spin, kappa: float) -> np.ndarray:
    """The matrix A with [T, Mx, My, Mz] = A @ thrusts (thrust along +z at each hub; κ the reaction torque per thrust)."""
    pos = np.asarray(pos, dtype=float)
    return np.vstack([np.ones(len(pos)), pos[:, 1], -pos[:, 0], np.asarray(spin, dtype=float) * kappa])


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


def lag_alpha(dt: float, spin_up_s: float) -> float:
    """The first-order step factor of the rotors' spin-up: T += (cmd − T) α."""
    return 1.0 - math.exp(-dt / spin_up_s)


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


def induced_velocity(thrust, V, alpha_rad, A, rho=1.225, iterations=40):
    """Glauert's induced velocity at a disk [m/s] in an inflow V at α to its axis (axial component V cos α)."""
    if thrust <= 0:
        return 0.0
    va, vn = V * math.cos(alpha_rad), V * math.sin(alpha_rad)
    vh = math.sqrt(thrust / (2 * rho * A))
    vi = vh
    for _ in range(iterations):
        new = thrust / (2 * rho * A * math.sqrt(max(va + vi, 0.0) ** 2 + vn ** 2 + 1e-9))
        vi = 0.5 * vi + 0.5 * new
    if va < -2 * vh:                                # windmill-brake state: momentum theory fails; keep the hover value
        vi = vh
    return vi


def hover_induced_velocity(thrust: float, A: float, rho: float = 1.225) -> float:
    """Momentum theory's induced velocity of a disk of area ``A`` in hover: √(T / (2 ρ A))."""
    return math.sqrt(thrust / (2 * rho * A))


def ideal_hover_power(thrust: float, A: float, rho: float = 1.225) -> float:
    """Momentum theory's ideal (induced) hover power of one disk: T √(T / (2 ρ A))."""
    return thrust * math.sqrt(thrust / (2 * rho * A))
