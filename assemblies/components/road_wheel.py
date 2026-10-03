"""The road wheel: the road it rolls on, the quarter car that carries it, and what a driven wheel pushes and loses.

Lifted as they were from notebooks 11 (rover), 20 (Onager Sentinel) and 21 (Onager Atlas), where each notebook had
its own copy with its own globals; here the globals are arguments. The wheel's CAD and its Chiron model are the
machines' own (``onager.OnagerSentinel`` part ``wheel``, ``onager_robot.wheel_servo``; the rover's wheel inside
``rover.Rover``). (``components/wheels.py`` is the microjet's turbine and impeller, not this.)
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81
C_RR = {"asphalt": 0.015, "gravel": 0.03, "soft soil": 0.08}                    # notebook 20 cell 10


def iso8608_profile(length_m, dx, gd_n0, seed, n0=0.1, n_min=0.02, n_max=8.0, n_waves=400):
    """An ISO 8608 road: displacement PSD ``Gd(n) = Gd(n0) (n/n0)^-2`` as a sum of sinusoids with random phase.
    Returns ``(x [m], z [m])``."""
    rng = np.random.default_rng(seed)
    x = np.arange(0, length_m, dx)
    ns = np.linspace(n_min, n_max, n_waves)
    dn = ns[1] - ns[0]
    amps = np.sqrt(2 * gd_n0 * (ns / n0) ** -2 * dn)
    phases = rng.uniform(0, 2 * math.pi, n_waves)
    z = (amps[None, :] * np.sin(2 * math.pi * ns[None, :] * x[:, None] + phases[None, :])).sum(axis=1)
    return x, z


def add_rocks(x, z, height_m, width_m, spacing_m, seed):
    """Cosine bumps ``height_m`` high, ``width_m`` wide, every ``spacing_m`` (± 0.3 m) (notebook 11 cell 8)."""
    rng = np.random.default_rng(seed)
    z = z.copy()
    centres = np.arange(spacing_m, x[-1] - spacing_m, spacing_m)
    for xc in centres + rng.uniform(-0.3, 0.3, len(centres)):
        mask = np.abs(x - xc) < width_m / 2
        z[mask] += height_m * np.cos(math.pi * (x[mask] - xc) / width_m)
    return z


def add_drop(x, z, at_m, depth_m):
    """A step down of ``depth_m`` from ``at_m`` on (notebook 11 cell 8)."""
    z = z.copy()
    z[x > at_m] -= depth_m
    return z


def quarter_car(x, z_road, speed, *, k_susp, c_susp, k_tyre, m_sprung, m_unsprung, static_corner_N=None, dt=5e-4,
                g=G) -> dict:
    """One corner over the road at ``speed``: the sprung mass on the suspension, the wheel on the tyre, the tyre only
    pushing. ``static_corner_N`` is the corner's static load (default ``(m_sprung + m_unsprung) g``; notebooks 11 and
    20 use a quarter of the whole weight). Returns the time series ``t, zr, Fs`` (suspension force with the static
    load), ``Ft`` (tyre force with the static load), ``As`` (body acceleration), ``travel``, ``zs``."""
    static = (m_sprung + m_unsprung) * g if static_corner_N is None else static_corner_N
    t = np.arange(0, x[-1] / speed, dt)
    zr = np.interp(t * speed, x, z_road)
    zs = zw = vs = vw = 0.0
    Fs, Ft, As, travel, Zs = (np.zeros_like(t) for _ in range(5))
    for i, z_r in enumerate(zr):
        f_susp = k_susp * (zw - zs) + c_susp * (vw - vs)
        f_tyre = max(k_tyre * (z_r - zw), -static)
        a_s = f_susp / m_sprung
        a_w = (f_tyre - f_susp) / m_unsprung
        vs += a_s * dt; vw += a_w * dt; zs += vs * dt; zw += vw * dt
        Fs[i], Ft[i], As[i], travel[i], Zs[i] = f_susp + m_sprung * g, f_tyre + static, a_s, zw - zs, zs
    return {"t": t, "zr": zr, "Fs": Fs, "Ft": Ft, "As": As, "travel": travel, "zs": Zs}


def rock_strike_force(run: dict, speed: float) -> np.ndarray:
    """The longitudinal load of rock strikes: the tyre force times the road's slope, up to 1.5 (notebook 11 cell 11)."""
    slope = np.gradient(run["zr"], run["t"] * speed)
    return np.clip(run["Ft"] * np.clip(np.abs(slope), 0, 1.5), 0, None)


def rolling_resistance(v, weight_N, c_rr, grade_pct, *, rho_air=1.2, cd=1.0, frontal_area_m2=1.9):
    """Rolling, grade and air resistance [N] (notebook 20 cell 10)."""
    th = math.atan(grade_pct / 100)
    return weight_N * (c_rr * math.cos(th) + math.sin(th)) + 0.5 * rho_air * cd * frontal_area_m2 * v ** 2


def hub_tractive(v, *, stall_Nm, wheel_radius_m, no_load_omega, n_motors=4):
    """The hub motors' tractive force [N] on their linear torque-speed line (notebook 20 cell 10)."""
    return n_motors * stall_Nm * max(0.0, 1 - abs(v) / wheel_radius_m / no_load_omega) / wheel_radius_m
