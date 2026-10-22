"""Capture and coupled-body estimates (lightweight, numpy only). A MuJoCo model of the same scenario is *prepared* in
``external/mujoco_capture.py`` and is not run here.

Frames: all vectors in a common inertial-aligned frame; positions relative to the servicer centre of mass."""
from __future__ import annotations

import math

import numpy as np


def cylinder_inertia(m: float, radius: float, length: float) -> np.ndarray:
    """Uniform solid cylinder, axis along x: diag(Ixx, Iyy, Izz)."""
    return np.diag([0.5 * m * radius ** 2, m * (3 * radius ** 2 + length ** 2) / 12, m * (3 * radius ** 2 + length ** 2) / 12])


def box_inertia(m: float, a: float, b: float, c: float) -> np.ndarray:
    return np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])


def combine(m1, I1, m2, I2, r2) -> tuple[float, np.ndarray, np.ndarray]:
    """Rigid combination of body 1 (CoM at origin) and body 2 (CoM at ``r2``): total mass, CoM, inertia about the new CoM."""
    r2 = np.asarray(r2, float)
    m = m1 + m2
    com = m2 * r2 / m
    def shift(I, mass, d):
        return I + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return m, com, shift(I1, m1, -com) + shift(I2, m2, r2 - com)


def capture_rate(m_sat, I_sat, m_deb, I_deb, w_deb, r_deb, v_rel=(0.0, 0.0, 0.0)):
    """Rigid, instantaneous capture. Servicer initially non-rotating. Returns (omega_after [rad/s], H [N m s], m, I_comb, com)."""
    r_deb, v_rel, w_deb = (np.asarray(x, float) for x in (r_deb, v_rel, w_deb))
    m, com, I = combine(m_sat, I_sat, m_deb, I_deb, r_deb)
    # angular momentum about the combined CoM: servicer at -com (at rest), debris at r_deb - com with velocity v_rel - v_com
    v_com = m_deb * v_rel / m
    H = (np.cross(r_deb - com, m_deb * (v_rel - v_com)) + I_deb @ w_deb + np.cross(-com, m_sat * (-v_com)))
    return np.linalg.solve(I, H), H, m, I, com


def detumble_propellant(H: float, lever_arm: float, isp: float) -> float:
    """Propellant for RCS torque couples that remove angular momentum H: m = H / (arm Isp g0)."""
    return H / (lever_arm * isp * 9.80665)


def impulsive_capture(m1: float, m2: float, v_rel: float, restitution: float = 0.0) -> dict:
    """1-D contact: impulse J = (1+e) mu v, with mu the reduced mass; each body's velocity change."""
    mu = m1 * m2 / (m1 + m2)
    J = (1 + restitution) * mu * v_rel
    return {"impulse_Ns": J, "dv_sat": J / m1, "dv_debris": J / m2}


def contact_1d(m1: float, m2: float, v_rel: float, k: float, c: float, t_end: float = 5.0, dt: float = 1e-4) -> dict:
    """Two masses joined by the arm's compliance (spring k [N/m], damper c [N s/m]) at first touch with closing speed v_rel.
    Semi-implicit Euler. Returns peak force, time of peak and the settling time to 5 % of the peak oscillation.
    Undamped analytic peak: F = v_rel sqrt(k mu)."""
    mu = m1 * m2 / (m1 + m2)
    x, v, peak, tpk = 0.0, v_rel, 0.0, 0.0
    hist = []
    for i in range(int(t_end / dt)):
        f = k * x + c * v
        a = -f / mu
        v += a * dt
        x += v * dt
        hist.append((i * dt, f))
        if f > peak:
            peak, tpk = f, i * dt
    h = np.array(hist)
    settle = next((t for t, f in h[::-1] if abs(f) > 0.05 * peak), 0.0)
    return {"peak_force_N": peak, "t_peak_s": tpk, "settling_s": float(settle), "analytic_undamped_N": v_rel * math.sqrt(k * mu)}
