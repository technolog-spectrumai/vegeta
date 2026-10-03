"""The wing: its planform numbers, its structure cases and the gust it must carry.

The geometry is ``components/fixed_wing.FixedWing`` (unchanged): a constant-chord centre section and two ruled lofts with
dihedral, NACA 4-digit sections. The planform types it makes are rectangular (``taper=1``) and straight-tapered with an
unswept quarter-chord line (``0.3 <= taper < 1``); swept and delta wings are not available. ``WingSpec`` holds those
numbers and refuses others.

The functions are the cells repeated across notebooks 09a (cell 10, 12), 09b (cell 8), 26 (cell 12) and 29 (cell 22,
``components/aguya_wing``), lifted as they were.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from vegeta import talos

# notebook 09a cell 12 = 26 cell 12
LW_PLA = dict(name="LW-PLA printed wing (solid-equivalent)", youngs_modulus=400.0, poissons_ratio=0.35, density=0.6e-9,
              yield_strength=12.0, source="assumed; replace with coupon tests")
TYPES = ("rectangular", "tapered")


@dataclass(frozen=True)
class WingSpec:
    """The wing's numbers in ``FixedWing``'s parameters (mm, degrees, NACA fractions)."""

    span: float
    root_chord: float
    taper: float = 1.0
    dihedral_deg: float = 0.0
    camber: float = 0.02
    camber_pos: float = 0.4
    thickness: float = 0.12

    def __post_init__(self):
        if not 0.3 <= self.taper <= 1.0:
            raise ValueError(f"taper {self.taper}: the wing geometry makes 0.3 <= taper <= 1 (straight, unswept quarter "
                             "chord); swept and delta wings are not available")
        if self.span <= 0 or self.root_chord <= 0:
            raise ValueError("span and root_chord must be positive")

    @classmethod
    def from_params(cls, p: dict) -> "WingSpec":
        """From a ``FixedWing`` (or Merlin, Aguya) parameter dict; other keys are ignored."""
        return cls(**{k: p[k] for k in ("span", "root_chord", "taper", "dihedral_deg", "camber", "camber_pos", "thickness")
                      if k in p})

    @property
    def type(self) -> str:
        return "rectangular" if self.taper == 1.0 else "tapered"

    @property
    def naca(self) -> str:
        return f"NACA {round(self.camber * 100)}{round(self.camber_pos * 10)}{round(self.thickness * 100):02d}"

    def planform(self) -> dict:
        """Area [m²] (centre section ~ root chord), aspect ratio, mean chord [m] (notebook 09a cell 10)."""
        b, c0, lam = self.span / 1000, self.root_chord / 1000, self.taper
        S = b * c0 * (1 + lam) / 2
        return {"area_m2": S, "aspect_ratio": b ** 2 / S, "mean_chord_m": c0 * (1 + lam) / 2, "span_m": b,
                "type": self.type, "section": self.naca}

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------- loads
def lift_slope(aspect_ratio: float, section_factor: float = 1.0) -> float:
    """The wing's lift-curve slope per radian, ``2 pi k AR / (AR + 2)`` (k = 0.85 in notebook 26, 1 in notebook 29)."""
    return 2 * math.pi * section_factor * aspect_ratio / (aspect_ratio + 2)


def pratt_gust(weight_N: float, wing_area_m2: float, mean_chord_m: float, lift_slope_per_rad: float, speed_m_s: float, *,
               gust_m_s: float = 7.5, rho: float = 1.225, limit_load_factor: float | None = None, g: float = 9.80665) -> dict:
    """Pratt's gust formula: mass ratio, alleviation, load factor at ``speed_m_s``, and (with a limit) the gust
    penetration speed (notebook 26 cell 12, notebook 29 cell 22)."""
    W, S, a = weight_N, wing_area_m2, lift_slope_per_rad
    mu = 2 * W / S / (rho * mean_chord_m * a * g)
    k_g = 0.88 * mu / (5.3 + mu)
    out = {"weight_N": W, "wing_area_m2": S, "mean_chord_m": mean_chord_m, "lift_slope": a, "mass_ratio": mu,
           "alleviation": k_g, "speed_m_s": speed_m_s, "gust_m_s": gust_m_s,
           "load_factor": 1 + k_g * rho * speed_m_s * gust_m_s * a * S / (2 * W)}
    if limit_load_factor is not None:
        out["limit_load_factor"] = limit_load_factor
        out["penetration_speed"] = (limit_load_factor - 1) * 2 * W / (k_g * rho * gust_m_s * a * S)
    return out


def lift_pressure_MPa(load_factor: float, weight_N: float, wing_area_m2: float) -> float:
    """``n`` times the weight spread over the planform, in MPa (N/mm²)."""
    return load_factor * weight_N / (wing_area_m2 * 1e6)


# ------------------------------------------------------------------------------------------------- regions and models
def lower_skins(step: str | Path) -> list[int]:
    """The two lower skin surfaces of a ``FixedWing`` wing STEP (the lift pressure goes on them)."""
    info = talos.inspect_step(step, units="mm-N-MPa")
    skins = [s for s in info.surfaces if s.kind == "BSpline surface" and s.area > 5e4]
    return [min((s for s in skins if (s.centroid[1] > 0) == right), key=lambda s: s.centroid[2]).tag
            for right in (False, True)]


def root_region(fuselage_diameter: float) -> talos.SurfacesInBox:
    """The centre section's faces at the fuselage: the clamp."""
    yc = fuselage_diameter / 2 + 5.0
    return talos.SurfacesInBox("root", (-1.0, -yc - 0.1, -100.0, 400.0, yc + 0.1, 100.0))


def motor_regions(nacelle_y: float, nacelle_diameter: float, nacelle_forward: float) -> list[talos.SurfacesInBox]:
    """The two nacelle front faces of the twin-motor wing (notebook 09a): where the motors' thrust goes in."""
    ny, d, f = nacelle_y, nacelle_diameter / 2 + 1, nacelle_forward
    return [talos.SurfacesInBox("motor_left", (-f - 0.1, -ny - d, -d, -f + 0.1, -ny + d, d)),
            talos.SurfacesInBox("motor_right", (-f - 0.1, ny - d, -d, -f + 0.1, ny + d, d))]


def wing_model(step: str | Path, regions, loads, *, element_mm: float, material: dict | None = None,
               name: str = "wing") -> talos.StructuralModel:
    """The wing clamped at its root region with ``loads`` (09a cell 12, 26 cell 12)."""
    return talos.StructuralModel(step, "mm-N-MPa", talos.Material(**(material or LW_PLA)), list(regions),
                                 [talos.FixedSupport("root")], list(loads), talos.MeshSettings(element_size=element_mm),
                                 name=name)
