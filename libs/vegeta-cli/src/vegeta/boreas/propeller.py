"""Propeller geometry: radial stations with chord and blade angle, plus mass properties."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Propeller:
    """A propeller/rotor given at radial stations.

    ``r`` are radii [m] from hub to tip, ``chord`` [m] and ``beta_deg`` (blade angle from the rotation
    plane) at those stations. ``mass_kg`` is the propeller alone; ``rotor_mass_kg`` (motor bell +
    propeller) is what an unbalance acts on.
    """

    name: str
    blades: int
    r: tuple[float, ...]
    chord: tuple[float, ...]
    beta_deg: tuple[float, ...]
    mass_kg: float = 0.0
    rotor_mass_kg: float = 0.0
    notes: str = ""

    def __post_init__(self):
        r, c, b = map(np.asarray, (self.r, self.chord, self.beta_deg))
        if self.blades < 1:
            raise ValueError("blades must be >= 1")
        if not (len(r) == len(c) == len(b) >= 3):
            raise ValueError("r, chord and beta_deg need the same length (>= 3 stations)")
        if np.any(np.diff(r) <= 0) or r[0] <= 0:
            raise ValueError("r must be increasing and start above the hub centre")
        if np.any(c <= 0):
            raise ValueError("chord must be positive at every station")

    @property
    def radius(self) -> float:
        return float(self.r[-1])

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def hub_radius(self) -> float:
        return float(self.r[0])

    @property
    def disk_area(self) -> float:
        return math.pi * self.radius**2

    @property
    def solidity(self) -> float:
        r, c = np.asarray(self.r), np.asarray(self.chord)
        return float(self.blades * np.trapezoid(c, r) / self.disk_area)

    def stations(self, n: int = 40):
        """Interpolated ``(r, chord, beta_rad)`` at ``n`` mid-points plus ``dr`` for integration."""
        edges = np.linspace(self.hub_radius, self.radius, n + 1)
        r = 0.5 * (edges[1:] + edges[:-1])
        dr = np.diff(edges)
        c = np.interp(r, self.r, self.chord)
        b = np.radians(np.interp(r, self.r, self.beta_deg))
        return r, c, b, dr

    def describe(self) -> dict:
        return {"name": self.name, "blades": self.blades, "diameter_m": self.diameter,
                "hub_radius_m": self.hub_radius, "solidity": self.solidity, "mass_kg": self.mass_kg,
                "rotor_mass_kg": self.rotor_mass_kg, "stations": [
                    {"r_m": float(r), "chord_m": float(c), "beta_deg": float(b)}
                    for r, c, b in zip(self.r, self.chord, self.beta_deg)], "notes": self.notes}

    @classmethod
    def from_pitch(cls, name: str, diameter_m: float, pitch_m: float, blades: int = 2, *,
                   chord_root_m: float, chord_max_m: float, chord_tip_m: float, hub_radius_m: float | None = None,
                   n_stations: int = 12, mass_kg: float = 0.0, rotor_mass_kg: float = 0.0, notes: str = "") -> "Propeller":
        """A constant-geometric-pitch propeller (beta = atan(P / 2 pi r)) with a simple planform:
        chord grows from the root to ``chord_max_m`` at 45 % radius and tapers to the tip."""
        R = diameter_m / 2
        r0 = hub_radius_m if hub_radius_m is not None else 0.12 * R
        r = np.linspace(r0, R, n_stations)
        x = (r - r0) / (R - r0)
        chord = np.where(x < 0.35, chord_root_m + (chord_max_m - chord_root_m) * x / 0.35,
                         chord_max_m + (chord_tip_m - chord_max_m) * np.clip((x - 0.35) / 0.65, 0, 1) ** 1.5)
        beta = np.degrees(np.arctan(pitch_m / (2 * math.pi * r)))
        return cls(name, blades, tuple(map(float, r)), tuple(map(float, chord)), tuple(map(float, beta)),
                   mass_kg=mass_kg, rotor_mass_kg=rotor_mass_kg,
                   notes=notes or f"constant geometric pitch {pitch_m * 1000:.0f} mm, generic planform")


def inches(d_in: float, p_in: float) -> tuple[float, float]:
    """``(diameter_m, pitch_m)`` from the usual inch designation, e.g. ``inches(10, 4.7)``."""
    return d_in * 0.0254, p_in * 0.0254
