"""Terramechanics for small ground vehicles (notebook 30 and the drive-type comparison of todo 5j.6): Bekker's
pressure–sinkage law, Janosi–Hanamoto shear, Wong's closed forms for a track with uniform pressure, Rowland's mean
maximum pressure, and the rigid-wheel counterparts. SI throughout: N, m, Pa, rad.

Bekker: p = (k_c / b + k_φ) zⁿ with b the smaller side of the contact patch; shear τ = (c + p tan φ)(1 − e^(−j/K))
with j the shear displacement. A track of contact length l, width b and normal load W at slip i develops (Wong,
*Theory of Ground Vehicles*, uniform pressure)

    F = (A c + W tan φ) [1 − K / (i l) (1 − e^(−i l / K))],     A = b l,

and meets the compaction resistance R_c = b (k_c / b + k_φ) z₀^(n+1) / (n + 1) with z₀ = (W / A / (k_c / b + k_φ))^(1/n).
Drawbar pull = thrust − resistances.

``SOILS`` holds the soil values (Bekker–Wong; the published values are in kN/m^(n+1), kN/m^(n+2), kPa — converted to
N and Pa here) and, for every terrain, the rigid-surface numbers a hard ground needs (μ peak traction, C_rr rolling
resistance of a small rubber tyre or band). The deformable soils use the Bekker model; the hard ones (asphalt,
gravel, grass) only μ and C_rr (``rigid=True``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

__all__ = ["Soil", "SOILS", "soil", "table", "bekker_k", "pressure_sinkage", "sinkage", "track_sinkage",
           "compaction_resistance_track", "shear_stress", "thrust_track", "max_thrust_track", "drawbar_pull_track",
           "mmp_rowland", "wheel_sinkage", "wheel_compaction_resistance", "thrust_wheel", "rigid_traction",
           "rigid_rolling_resistance"]


@dataclass(frozen=True)
class Soil:
    """One terrain. Bekker: ``n`` [-], ``k_c`` [N/m^(n+1)], ``k_phi`` [N/m^(n+2)], ``c`` cohesion [Pa], ``phi_deg``
    internal friction [deg], ``K`` shear deformation modulus [m]. Hard ground: ``mu`` peak traction coefficient,
    ``c_rr`` rolling resistance of a small rubber tyre/band (used on every terrain where the soil does not sink: the
    hysteresis of the rubber). ``rigid`` True = no Bekker sinkage (asphalt, gravel, grass)."""

    name: str
    n: float
    k_c: float
    k_phi: float
    c: float
    phi_deg: float
    K: float
    mu: float
    c_rr: float
    rigid: bool = False
    source: str = ""

    @property
    def phi(self) -> float:
        return math.radians(self.phi_deg)

    def as_dict(self) -> dict:
        return asdict(self)


_WONG = "Wong, Theory of Ground Vehicles, Table 2.3 (Bekker values); K: Wong's typical shear moduli"
SOILS = {s.name: s for s in [
    Soil("dry sand", 1.1, 0.99e3, 1528.43e3, 1.04e3, 28.0, 0.025, 0.6, 0.10, source=_WONG + " (LLL dry sand)"),
    Soil("sandy loam", 0.7, 5.27e3, 1515.04e3, 1.72e3, 29.0, 0.025, 0.6, 0.06, source=_WONG + " (LETE sandy loam)"),
    Soil("clayey soil", 0.5, 13.19e3, 692.15e3, 4.14e3, 13.0, 0.010, 0.4, 0.08, source=_WONG + " (Thailand, 38 % moisture)"),
    Soil("heavy clay", 0.13, 12.70e3, 1555.95e3, 68.95e3, 34.0, 0.006, 0.6, 0.05, source=_WONG + " (WES heavy clay)"),
    Soil("lean clay", 0.2, 16.43e3, 1724.69e3, 68.95e3, 20.0, 0.006, 0.5, 0.05, source=_WONG + " (WES lean clay)"),
    Soil("snow", 1.6, 4.37e3, 196.72e3, 1.03e3, 19.7, 0.04, 0.3, 0.06, source=_WONG + " (US snow)"),
    Soil("mud", 0.5, 13.19e3, 692.15e3, 2.0e3, 6.0, 0.010, 0.25, 0.10,
         source="assumed: Wong's wet clayey soil with half the cohesion and φ 6° (saturated)"),
    Soil("asphalt", 1.0, 0.0, 1e12, 0.0, 0.0, 0.01, 0.8, 0.015, rigid=True, source="handbook: rubber on dry asphalt"),
    Soil("gravel", 1.0, 0.0, 1e12, 0.0, 0.0, 0.01, 0.6, 0.03, rigid=True, source="handbook: rubber on loose gravel"),
    Soil("grass", 1.0, 0.0, 1e12, 0.0, 0.0, 0.01, 0.5, 0.06, rigid=True, source="handbook: rubber on dry short grass"),
]}


def soil(name_or_soil) -> Soil:
    return name_or_soil if isinstance(name_or_soil, Soil) else SOILS[name_or_soil]


def table():
    """``SOILS`` as a pandas DataFrame in the published units (kN/m^(n+1), kN/m^(n+2), kPa, deg, mm)."""
    import pandas as pd

    rows = {s.name: {"n": s.n, "k_c [kN/m^(n+1)]": s.k_c / 1e3, "k_phi [kN/m^(n+2)]": s.k_phi / 1e3 if not s.rigid else np.inf,
                     "c [kPa]": s.c / 1e3, "phi [deg]": s.phi_deg, "K [mm]": s.K * 1e3, "mu": s.mu, "C_rr": s.c_rr,
                     "rigid": s.rigid, "source": s.source} for s in SOILS.values()}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------------------------- pressure–sinkage
def bekker_k(b: float, s) -> float:
    """k_c / b + k_φ [N/m^(n+2)] for a plate of width ``b`` [m]."""
    s = soil(s)
    return s.k_c / b + s.k_phi


def pressure_sinkage(z, b: float, s):
    """Pressure [Pa] at sinkage ``z`` [m] under a plate of width ``b`` [m]."""
    s = soil(s)
    return bekker_k(b, s) * np.power(np.maximum(z, 0.0), s.n)


def sinkage(p, b: float, s):
    """Sinkage [m] under pressure ``p`` [Pa] (0 on rigid ground)."""
    s = soil(s)
    if s.rigid:
        return np.zeros_like(np.asarray(p, dtype=float)) + 0.0
    return np.power(np.maximum(p, 0.0) / bekker_k(b, s), 1.0 / s.n)


def track_sinkage(W: float, b: float, l: float, s) -> float:
    """Static sinkage [m] of one track carrying ``W`` [N] on a ``b`` × ``l`` [m] patch, uniform pressure."""
    return float(sinkage(W / (b * l), b, s))


def compaction_resistance_track(W: float, b: float, l: float, s) -> float:
    """Motion resistance [N] of one track from compacting the soil (a rut of depth z₀ and width b)."""
    s = soil(s)
    if s.rigid:
        return 0.0
    z0 = track_sinkage(W, b, l, s)
    return b * bekker_k(b, s) * z0 ** (s.n + 1) / (s.n + 1)


# ----------------------------------------------------------------------------------------------- shear and thrust
def shear_stress(p, j, s):
    """Janosi–Hanamoto shear stress [Pa] at normal pressure ``p`` [Pa] and shear displacement ``j`` [m]."""
    s = soil(s)
    return (s.c + np.asarray(p) * math.tan(s.phi)) * (1.0 - np.exp(-np.asarray(j) / s.K))


def thrust_track(W: float, b: float, l: float, slip, s):
    """Thrust [N] of one track at ``slip`` (0…1), uniform pressure (Wong's closed form). Rigid ground: the
    traction coefficient μ reached exponentially with the slip displacement (the same law, c = 0, tan φ = μ)."""
    s = soil(s)
    i = np.maximum(np.asarray(slip, dtype=float), 1e-9)
    shape = 1.0 - s.K / (i * l) * (1.0 - np.exp(-i * l / s.K))
    if s.rigid:
        return s.mu * W * shape
    return (b * l * s.c + W * math.tan(s.phi)) * shape


def max_thrust_track(W: float, b: float, l: float, s) -> float:
    """Thrust [N] at 100 % slip (the shear strength of the patch for a soil, μW on hard ground)."""
    return float(thrust_track(W, b, l, 1.0, s))


def drawbar_pull_track(W: float, b: float, l: float, slip, s, other_resistance: float = 0.0):
    """Drawbar pull [N] of one track: thrust − compaction resistance − ``other_resistance`` (internal, grade)."""
    return thrust_track(W, b, l, slip, s) - compaction_resistance_track(W, b, l, s) - other_resistance


def mmp_rowland(W: float, n_wheels: int, b: float, d: float, pitch: float, n_tracks: int = 2) -> float:
    """Rowland's mean maximum pressure [Pa] of a tracked vehicle with rigid links: MMP = 1.26 W / (2 m b √(p d)) in
    kN, m → kPa, with m road wheels per track of diameter ``d``, track width ``b`` and link pitch ``pitch``
    (Rowland 1972; ``n_tracks`` = 2)."""
    return 1.26 * (W / 1e3) / (n_tracks * n_wheels * b * math.sqrt(pitch * d)) * 1e3


# ----------------------------------------------------------------------------------------------- rigid wheel
def wheel_sinkage(W: float, b: float, D: float, s) -> float:
    """Bekker's static sinkage [m] of a rigid wheel (load ``W`` [N], width ``b``, diameter ``D`` [m])."""
    s = soil(s)
    if s.rigid:
        return 0.0
    k = s.k_c + b * s.k_phi
    return (3.0 * W / ((3.0 - s.n) * k * math.sqrt(D))) ** (2.0 / (2.0 * s.n + 1.0))


def wheel_compaction_resistance(W: float, b: float, D: float, s) -> float:
    """Bekker's compaction resistance [N] of a rigid wheel: b k z₀^(n+1) / (n+1), k = k_c/b + k_φ."""
    s = soil(s)
    if s.rigid:
        return 0.0
    z0 = wheel_sinkage(W, b, D, s)
    return b * bekker_k(b, s) * z0 ** (s.n + 1) / (s.n + 1)


def thrust_wheel(W: float, b: float, D: float, slip, s):
    """Thrust [N] of a rigid wheel at ``slip``: the track formula over the contact chord l = √(D z₀) (a common
    approximation; on hard ground a tyre patch of 0.15 D)."""
    s = soil(s)
    l = 0.15 * D if s.rigid else max(math.sqrt(D * wheel_sinkage(W, b, D, s)), 0.05 * D)
    return thrust_track(W, b, l, slip, s)


# ----------------------------------------------------------------------------------------------- hard ground
def rigid_traction(W: float, s) -> float:
    """Peak traction [N] on hard ground: μ W."""
    return soil(s).mu * W


def rigid_rolling_resistance(W: float, s) -> float:
    """Rolling resistance [N] of the rubber on any ground: C_rr W."""
    return soil(s).c_rr * W
