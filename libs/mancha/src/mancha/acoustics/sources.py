"""Exact linear-acoustic fields (for verifying the FW-H solver) and rigid-rotation kinematics.

A source with velocity potential ``φ(x, t)`` gives ``u = ∇φ``, ``p' = −ρ0 ∂φ/∂t`` and ``ρ' = p'/c²``. The
monopole ``φ = (A/r) cos ω(t − r/c)`` and its z derivative (a dipole) are exact solutions of the wave equation;
their fields are evaluated here by fourth-order central differences of ``φ`` (relative error ~1e-10).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def monopole_potential(amplitude: float, omega: float, c: float, centre=(0.0, 0.0, 0.0)) -> Callable:
    x0 = np.asarray(centre, dtype=np.float64)

    def phi(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(x - x0, axis=-1)
        return amplitude / r * np.cos(omega * (t - r / c))
    return phi


def dipole_potential(amplitude: float, omega: float, c: float, axis=(0.0, 0.0, 1.0), centre=(0.0, 0.0, 0.0),
                     h: float = 1e-4) -> Callable:
    """``axis · ∇`` of the monopole potential (by a fourth-order difference with step ``h``)."""
    mono = monopole_potential(amplitude, omega, c, centre)
    e = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)

    def phi(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (-mono(x + 2 * h * e, t) + 8 * mono(x + h * e, t) - 8 * mono(x - h * e, t)
                + mono(x - 2 * h * e, t)) / (12 * h)
    return phi


def acoustic_field(phi: Callable, x: np.ndarray, t: np.ndarray, rho0: float, c: float, h: float = 1e-3,
                   ht: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ρ, u, p')`` of the potential ``phi`` at points ``x[..., 3]`` and times ``t[...]`` (broadcast)."""
    ht = ht if ht is not None else h / c

    def d(f0, f1, f2, f3, step):
        return (-f3 + 8 * f2 - 8 * f1 + f0) / (12 * step)

    u = np.empty(np.broadcast_shapes(x.shape, np.shape(t) + (3,)))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        u[..., k] = d(phi(x - 2 * h * e, t), phi(x - h * e, t), phi(x + h * e, t), phi(x + 2 * h * e, t), h)
    dphidt = d(phi(x, t - 2 * ht), phi(x, t - ht), phi(x, t + ht), phi(x, t + 2 * ht), ht)
    p = -rho0 * dphidt
    return rho0 + p / (c * c), u, p


def rotate_x(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate ``points[..., 3]`` about the x axis by ``angle[nt]`` (right hand: +y towards +z) → ``[nt, ..., 3]``."""
    a = np.asarray(angle, dtype=np.float64).reshape((-1,) + (1,) * (points.ndim - 1))
    ca, sa = np.cos(a), np.sin(a)
    out = np.empty(a.shape[:1] + points.shape)
    out[..., 0] = points[..., 0]
    out[..., 1] = ca * points[..., 1] - sa * points[..., 2]
    out[..., 2] = sa * points[..., 1] + ca * points[..., 2]
    return out


def rigid_rotation(points: np.ndarray, normals: np.ndarray | None, omega: float, tau: np.ndarray,
                   centre=(0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    """Kinematics of body points (and normals) turning at ``omega`` about the x axis through ``centre``:
    positions ``y``, velocities ``v = Ω×r``, accelerations ``Ω×(Ω×r)``, normals ``n`` and ``ṅ = Ω×n``
    (each ``[nt, ne, 3]``). At τ = 0 the body is as given."""
    c0 = np.asarray(centre, dtype=np.float64)
    rel = rotate_x(points - c0, omega * tau)
    om = np.array([omega, 0.0, 0.0])
    out = {"y": rel + c0, "v": np.cross(om, rel), "vdot": np.cross(om, np.cross(om, rel))}
    if normals is not None:
        n = rotate_x(normals, omega * tau)
        out.update(n=n, ndot=np.cross(om, n))
    return out


def sphere_panels(radius: float, n_theta: int, n_phi: int, centre=(0.0, 0.0, 0.0)):
    """A latitude–longitude panelling of a sphere: centres, outward normals and areas (they sum to 4πR²)."""
    th_e = np.linspace(0.0, np.pi, n_theta + 1)
    ph_e = np.linspace(0.0, 2 * np.pi, n_phi + 1)
    th = 0.5 * (th_e[1:] + th_e[:-1])
    ph = 0.5 * (ph_e[1:] + ph_e[:-1])
    T, P = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1).reshape(-1, 3)
    area = (radius ** 2 * (np.cos(th_e[:-1]) - np.cos(th_e[1:]))[:, None] * np.diff(ph_e)[None, :]).reshape(-1)
    return np.asarray(centre, dtype=np.float64) + radius * n, n, area
