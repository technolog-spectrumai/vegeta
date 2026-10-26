"""Reference data and closed-form solutions used by the validation cases. Each comes with its source."""
from __future__ import annotations

import numpy as np

# Ghia, Ghia & Shin (1982), "High-Re solutions for incompressible flow using the Navier-Stokes equations and a
# multigrid method", J. Comput. Phys. 48, 387–411. Re = 100, 2D lid-driven square cavity (129×129 grid).
# Table I: u/U along the vertical centreline x = 0.5; Table II: v/U along the horizontal centreline y = 0.5.
GHIA_RE100_U = np.array([
    (1.0000, 1.00000), (0.9766, 0.84123), (0.9688, 0.78871), (0.9609, 0.73722), (0.9531, 0.68717),
    (0.8516, 0.23151), (0.7344, 0.00332), (0.6172, -0.13641), (0.5000, -0.20581), (0.4531, -0.21090),
    (0.2813, -0.15662), (0.1719, -0.10150), (0.1016, -0.06434), (0.0703, -0.04775), (0.0625, -0.04192),
    (0.0547, -0.03717), (0.0000, 0.00000)])
GHIA_RE100_V = np.array([
    (1.0000, 0.00000), (0.9688, -0.05906), (0.9609, -0.07391), (0.9531, -0.08864), (0.9453, -0.10313),
    (0.9063, -0.16914), (0.8594, -0.22445), (0.8047, -0.24533), (0.5000, 0.05454), (0.2344, 0.17527),
    (0.2266, 0.17507), (0.1563, 0.16077), (0.0938, 0.12317), (0.0781, 0.10890), (0.0703, 0.10091),
    (0.0625, 0.09233), (0.0000, 0.00000)])
# Ghia et al. (1982) Table V: primary vortex centre at Re = 100 (x, y), lid at y = 1 moving in +x.
GHIA_RE100_PRIMARY_VORTEX = (0.6172, 0.7344)


def poiseuille_force_profile(y: np.ndarray, height: float, force: float, viscosity: float,
                             density: float = 1.0) -> np.ndarray:
    """Body-force-driven plane Poiseuille flow between walls at ``y = 0`` and ``y = height``:
    ``u(y) = F / (2 ρ ν) · y (h − y)``."""
    return force / (2.0 * density * viscosity) * y * (height - y)


def poiseuille_mean_profile(y: np.ndarray, height: float, mean_velocity: float) -> np.ndarray:
    """Plane Poiseuille profile with a given mean velocity: ``u = 6 U y (h − y) / h²`` (peak 1.5 U)."""
    return 6.0 * mean_velocity * y * (height - y) / height ** 2


def taylor_green_decay_rate(viscosity: float, kx: float, ky: float) -> float:
    """Kinetic-energy decay rate of the 2D Taylor–Green vortex, ``E(t) = E0 exp(−2 ν (kx² + ky²) t)``."""
    return 2.0 * viscosity * (kx * kx + ky * ky)


def schiller_naumann_cd(re: float) -> float:
    """Drag coefficient of an unconfined sphere, Schiller & Naumann (1933): ``24/Re (1 + 0.15 Re^0.687)``,
    a correlation of experiments within a few percent for Re < 800."""
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)
