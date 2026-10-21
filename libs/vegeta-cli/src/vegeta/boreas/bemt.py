"""Blade element momentum theory (BEMT) for a propeller/rotor in axial flow, hover included.

Per radial station the blade-element thrust is balanced against the momentum-theory thrust of the
annulus (with Prandtl tip loss) by iterating on the induced velocity; swirl is taken from the torque
balance. Assumptions: axial inflow (no yaw/edgewise), rigid blades, no Reynolds effects, no
compressibility (tip Mach kept below ~0.6 for these small propellers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .airfoil import Airfoil
from .propeller import Propeller


@dataclass
class OperatingPoint:
    """One (rpm, airspeed) solution with integral quantities and radial distributions."""

    rpm: float
    airspeed: float
    rho: float
    thrust: float          # N
    torque: float          # N m
    power: float           # W (shaft)
    efficiency: float      # T V / P (0 in hover)
    figure_of_merit: float # ideal hover power / shaft power (meaningful in hover only)
    ct: float              # T / (rho n^2 D^4)
    cp: float              # P / (rho n^3 D^5)
    advance_ratio: float   # J = V / (n D)
    tip_mach: float
    converged: bool
    r: np.ndarray = field(repr=False)
    dT_dr: np.ndarray = field(repr=False)
    dQ_dr: np.ndarray = field(repr=False)
    alpha_deg: np.ndarray = field(repr=False)
    induced_velocity: np.ndarray = field(repr=False)
    cl: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        d = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in self.__dict__.items()
             if not isinstance(v, np.ndarray)}
        d["radial"] = {k: np.asarray(getattr(self, k)).round(6).tolist()
                       for k in ("r", "dT_dr", "dQ_dr", "alpha_deg", "induced_velocity", "cl")}
        return d


def _tip_loss(blades: int, r: np.ndarray, R: float, phi: np.ndarray) -> np.ndarray:
    f = 0.5 * blades * (R - r) / (r * np.maximum(np.sin(phi), 1e-3))
    return np.clip(2 / math.pi * np.arccos(np.clip(np.exp(-f), 0.0, 1.0)), 0.05, 1.0)


def solve(prop: Propeller, airfoil: Airfoil, rpm: float, airspeed: float = 0.0, rho: float = 1.225, *,
          n_stations: int = 40, iterations: int = 300, relaxation: float = 0.3, tolerance: float = 1e-5,
          speed_of_sound: float = 340.0) -> OperatingPoint:
    """Steady operating point at ``rpm`` and axial ``airspeed`` [m/s]."""
    if rpm <= 0:
        raise ValueError("rpm must be > 0")
    if airspeed < 0:
        raise ValueError("airspeed must be >= 0 (axial inflow only)")
    omega = rpm * 2 * math.pi / 60
    r, c, beta, dr = prop.stations(n_stations)
    R, B = prop.radius, prop.blades
    V = airspeed
    vi = np.full_like(r, 0.05 * omega * R)   # induced velocity guess
    b = np.zeros_like(r)                     # tangential induction factor
    converged = False
    for _ in range(iterations):
        vt = omega * r * (1 - b)
        va = V + vi
        phi = np.arctan2(va, vt)
        alpha = beta - phi
        cl, cd = airfoil.coefficients(alpha)
        w2 = va**2 + vt**2
        F = _tip_loss(B, r, R, phi)
        dT = 0.5 * rho * w2 * B * c * (cl * np.cos(phi) - cd * np.sin(phi))   # per unit radius
        dQ = 0.5 * rho * w2 * B * c * (cl * np.sin(phi) + cd * np.cos(phi)) * r
        # momentum balance of the annulus: dT = 4 pi r rho F (V + vi) vi
        rhs = np.maximum(dT, 0.0) / (4 * math.pi * r * rho * F)
        vi_new = 0.5 * (-V + np.sqrt(V**2 + 4 * rhs))
        # swirl: dQ = 4 pi r^3 rho F (V + vi) omega b
        b_new = np.clip(dQ / (4 * math.pi * r**3 * rho * F * np.maximum(va, 1e-6) * omega), 0.0, 0.5)
        change = float(np.max(np.abs(vi_new - vi)) / max(float(np.max(np.abs(vi_new))), 1e-9))
        vi = (1 - relaxation) * vi + relaxation * vi_new
        b = (1 - relaxation) * b + relaxation * b_new
        if change < tolerance:
            converged = True
            break
    T = float(np.sum(dT * dr))
    Q = float(np.sum(dQ * dr))
    P = Q * omega
    n = rpm / 60
    D = prop.diameter
    ideal_hover_power = T * math.sqrt(max(T, 0.0) / (2 * rho * prop.disk_area))
    return OperatingPoint(
        rpm=rpm, airspeed=V, rho=rho, thrust=T, torque=Q, power=P,
        efficiency=(T * V / P) if P > 0 and V > 0 else 0.0,
        figure_of_merit=(ideal_hover_power / P) if P > 0 and V == 0 else 0.0,
        ct=T / (rho * n**2 * D**4), cp=P / (rho * n**3 * D**5), advance_ratio=V / (n * D),
        tip_mach=omega * R / speed_of_sound, converged=converged,
        r=r, dT_dr=dT, dQ_dr=dQ, alpha_deg=np.degrees(alpha), induced_velocity=vi, cl=cl,
    )


def rpm_for_thrust(prop: Propeller, airfoil: Airfoil, thrust: float, airspeed: float = 0.0, rho: float = 1.225,
                   rpm_max: float = 60000.0) -> OperatingPoint:
    """The rpm that produces ``thrust`` [N] at ``airspeed`` (bisection; raises if unreachable)."""
    lo, hi = 100.0, rpm_max
    if solve(prop, airfoil, hi, airspeed, rho).thrust < thrust:
        raise ValueError(f"{thrust:.2f} N is not reachable below {rpm_max:.0f} rpm at {airspeed} m/s")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if solve(prop, airfoil, mid, airspeed, rho).thrust < thrust:
            lo = mid
        else:
            hi = mid
    return solve(prop, airfoil, hi, airspeed, rho)
