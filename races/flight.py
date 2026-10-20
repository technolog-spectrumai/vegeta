"""The racing quad's flight controller: Drongo's geometric controller, as it was when the racing quad hatched.

Copied from Vegeta's ``designs/drongo_controller.py`` (history kept: ``git log --follow races/flight.py``) when the
racing drones graduated to Piccolo; Drongo itself has not hatched and keeps its own controller (missions, the pincer,
the ground hold) in Vegeta. Only what the races fly is here: ``Ref``, ``quat_to_R`` and ``Flight``.

**Flight controller** (``Flight``). It cascades two loops and a mixer:

* *position* — the force the rotors must make, ``F = m (a_ref + g ẑ) − k_x e_p − k_v e_v − k_i ∫e_p``: a PD loop on the
  position and velocity errors to a reference (``Ref``: position, velocity, acceleration, yaw) with the reference
  acceleration fed forward and a small integral term (it absorbs what the mass estimate misses); the tilt is limited
  to ``tilt_max_deg``;
* *attitude* — the frame's z axis turned onto ``F`` (a geometric controller on SO(3), Lee, Leok & McClamroch 2010):
  torques ``M = −k_R e_R − k_Ω Ω`` with gains from the frame's inertia;
* *mixer* — collective thrust and the three torques to the four rotor thrusts of the X layout (arms at 45°); out of
  range, the yaw torque is dropped first, then the roll/pitch torques are scaled until every rotor is between its
  idle and maximum thrust.

A subclass provides ``reset`` (``m0``, ``J``, ``kR``, ``kW``, the mixer ``A``/``Ainv``, ``integral``, ``last``):
``races.sim`` builds it from the MuJoCo model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81

__all__ = ["Ref", "Flight", "quat_to_R", "G"]


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class Ref:
    """The reference the flight controller follows (world, SI)."""

    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0


class Flight:
    """Position → attitude → rotor thrusts (see the module). ``omega_pos`` [rad/s] and ``zeta_pos`` set the position
    loop (k_x = m ω², k_v = 2 ζ m ω), ``omega_att`` / ``zeta_att`` the roll-pitch loop and ``omega_yaw`` the yaw
    loop on the frame's inertia (``reset`` sets it: the races' ``QuadFlight`` reads it from the MuJoCo model); ``ki`` [1/s] the integral's rate relative to k_x, ``i_max`` [N] its bound;
    ``tilt_max_deg`` the largest tilt the position loop may ask for; ``idle`` the thrust share every rotor keeps in
    the air. ``prop`` needs ``max_thrust_N`` and ``torque_per_thrust_m`` (``races.refquad``)."""

    def __init__(self, prop, *, omega_pos: float = 2.4, zeta_pos: float = 1.0, ki: float = 0.4,
                 i_max: float = 2.0, omega_att: float = 16.0, zeta_att: float = 0.8, omega_yaw: float = 5.0,
                 tilt_max_deg: float = 35.0, idle: float = 0.03):
        self.prop = prop
        self.omega_pos, self.zeta_pos, self.ki, self.i_max = omega_pos, zeta_pos, ki, i_max
        self.omega_att, self.zeta_att, self.omega_yaw = omega_att, zeta_att, omega_yaw
        self.tilt_max = math.radians(tilt_max_deg)
        self.idle = idle

    def reset_integral(self):
        self.integral[:] = 0.0

    def mix(self, T: float, M: np.ndarray) -> tuple:
        """Rotor thrusts [N] for a collective thrust ``T`` and body torques ``M``; True when they had to be limited."""
        lo, hi = self.idle * self.prop.max_thrust_N, self.prop.max_thrust_N
        T = float(np.clip(T, 4 * lo, 4 * hi))
        u = self.Ainv @ np.array([T, M[0], M[1], M[2]])
        if lo - 1e-9 <= u.min() and u.max() <= hi + 1e-9:
            return u, False
        u = self.Ainv @ np.array([T, M[0], M[1], 0.0])                        # drop the yaw torque first
        if not (lo - 1e-9 <= u.min() and u.max() <= hi + 1e-9):
            k_lo, k_hi = 0.0, 1.0                                             # then scale roll and pitch
            for _ in range(20):
                k = 0.5 * (k_lo + k_hi)
                v = self.Ainv @ np.array([T, k * M[0], k * M[1], 0.0])
                if lo - 1e-9 <= v.min() and v.max() <= hi + 1e-9:
                    k_lo = k
                else:
                    k_hi = k
            u = self.Ainv @ np.array([T, k_lo * M[0], k_lo * M[1], 0.0])
        return np.clip(u, lo, hi), True

    def attitude(self, R: np.ndarray, omega: np.ndarray, b3: np.ndarray, yaw: float) -> np.ndarray:
        """Body torques [N·m] that turn the frame's z axis onto ``b3`` at heading ``yaw``."""
        b1c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        return -self.kR * eR - self.kW * omega

    def thrusts(self, obs, ref: Ref, m_payload: float, dt: float) -> np.ndarray:
        """The four rotor thrusts [N] to follow ``ref`` carrying ``m_payload`` [kg]."""
        m = self.m0 + m_payload
        p = np.asarray(obs.base_pos, dtype=float)
        v = np.asarray(obs.body_linvel[0], dtype=float)
        R = quat_to_R(obs.base_quat)
        omega = np.asarray(obs.body_angvel[0], dtype=float)
        wp = self.omega_pos
        kx, kv = m * wp * wp, 2 * self.zeta_pos * m * wp
        ep, ev = p - ref.p, v - ref.v
        self.integral = np.clip(self.integral + kx * self.ki * ep * dt, -self.i_max, self.i_max)
        F = m * (ref.a + np.array([0.0, 0.0, G])) - kx * ep - kv * ev - self.integral
        F[2] = max(F[2], 0.25 * m * G)
        h = math.hypot(F[0], F[1])
        if h > F[2] * math.tan(self.tilt_max):                                # tilt limit: shorten the horizontal part
            F[:2] *= F[2] * math.tan(self.tilt_max) / h
        b3 = F / np.linalg.norm(F)
        M = self.attitude(R, omega, b3, ref.yaw)
        T, sat = self.mix(float(F @ R[:, 2]), M)
        self.last = {"F": F, "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))), "T": T, "saturated": sat}
        return T
