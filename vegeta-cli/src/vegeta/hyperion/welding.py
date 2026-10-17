"""Laser welding on paper: the beam a multimode delivery fibre gives (beam parameter product, M²), the welding head's
spot and depth of focus, the keyhole threshold (the surface reaching its boiling point under the spot), Rosenthal's
moving point source for the conduction-mode pool, and the keyhole penetration from an energy balance with a stated
melting efficiency — penetration against speed, the speed that just fully penetrates a thickness, the process window.
Numbers good to the tens of per cent a process window is drawn with; the weld trial decides the parameters."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .beam import GaussianBeam
from .thermal import Solid, T_ROOM, gaussian_steady_rise, solid as _solid


@dataclass(frozen=True)
class DeliveryFibre:
    """A multimode process fibre: ``core_radius`` [m], ``NA`` — its beam parameter product r·NA [m·rad] and M² = π BPP/λ."""
    core_radius: float
    NA: float
    length: float = 20.0
    name: str = "process fibre"

    @property
    def bpp(self) -> float:
        return self.core_radius * self.NA

    def m2(self, wavelength: float) -> float:
        return math.pi * self.bpp / wavelength

    def beam(self, wavelength: float, name: str = "") -> GaussianBeam:
        """The beam at the fibre's end: waist = core radius, M² from the BPP (a top-hat core as its embedded Gaussian)."""
        return GaussianBeam(wavelength, self.core_radius, 0.0, self.m2(wavelength), 1.0, name or self.name)

    def to_dict(self) -> dict:
        return {"core_radius": self.core_radius, "NA": self.NA, "length": self.length, "name": self.name, "bpp_mm_mrad": self.bpp * 1e6}


@dataclass(frozen=True)
class WeldHead:
    """Collimator ``f_collimator`` and focusing lens ``f_focus``: the spot is the fibre core imaged at f_focus/f_collimator,
    the focusing cone's half-angle NA·f_collimator/f_focus, the collimated beam's radius NA·f_collimator."""
    f_collimator: float = 0.100
    f_focus: float = 0.200
    name: str = "welding head"

    @property
    def magnification(self) -> float:
        return self.f_focus / self.f_collimator

    def spot_radius(self, fibre: DeliveryFibre) -> float:
        return fibre.core_radius * self.magnification

    def cone_half_angle(self, fibre: DeliveryFibre) -> float:
        return fibre.NA / self.magnification

    def collimated_radius(self, fibre: DeliveryFibre) -> float:
        return fibre.NA * self.f_collimator

    def rayleigh_length(self, fibre: DeliveryFibre) -> float:
        """z_R of the multimode beam: w₀/θ (the depth of focus is twice it)."""
        return self.spot_radius(fibre) / self.cone_half_angle(fibre)

    def focused_beam(self, fibre: DeliveryFibre, wavelength: float) -> GaussianBeam:
        return GaussianBeam(wavelength, self.spot_radius(fibre), 0.0, fibre.m2(wavelength), 1.0, f"{self.name} focus")

    def to_dict(self) -> dict:
        return {"f_collimator": self.f_collimator, "f_focus": self.f_focus, "name": self.name}


# ----------------------------------------------------------------------------------------------------- the keyhole
def keyhole_threshold(material: Solid | str, spot_radius: float, T0: float = T_ROOM, k_hot: float | None = None) -> dict:
    """The power (and the mean irradiance) at which a Gaussian spot's centre reaches the boiling point on a semi-infinite
    solid in steady conduction: P = √(2π) k w (T_vap − T0) / α; above it the vapour opens a keyhole. ``k_hot`` the
    conductivity to use (default the solid's)."""
    m = _solid(material)
    k = k_hot if k_hot is not None else m.k
    p = math.sqrt(2 * math.pi) * k * spot_radius * (m.T_fail - T0) / m.absorptivity
    return {"P_threshold_W": p, "I_threshold_W_per_cm2": p / (math.pi * spot_radius ** 2) / 1e4, "spot_radius": spot_radius, "material": m.name}


def rosenthal_pool(material: Solid | str, absorbed_power: float, speed: float, T0: float = T_ROOM, n: int = 2000) -> dict:
    """Rosenthal's moving point source on a thick plate: T − T0 = (q/2πk r) exp(−v(x + r)/2κ); the melt isotherm's
    length (ahead and behind), width and depth (the width's half: a hemispherical source). Conduction-mode welding."""
    m = _solid(material)
    kap = m.diffusivity
    dT = m.T_melt - T0
    q = absorbed_power
    # along the axis behind the source (x < 0, r = −x): T − T0 = q/(2πk r) → r_back = q/(2πk ΔT)
    r_back = q / (2 * math.pi * m.k * dT)
    # ahead (x > 0, r = x): q/(2πk x) exp(−v x/κ) = ΔT, solved by bisection
    lo, hi = 1e-9, r_back
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if q / (2 * math.pi * m.k * mid) * math.exp(-speed * mid / kap) > dT:
            lo = mid
        else:
            hi = mid
    r_front = 0.5 * (lo + hi)
    # the width: the largest y with T = T_melt over x
    xs = np.linspace(-3 * r_back, r_front, n)
    ys = np.zeros_like(xs)
    for i, x in enumerate(xs):
        lo, hi = 0.0, r_back
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            r = math.hypot(x, mid)
            if r > 0 and q / (2 * math.pi * m.k * r) * math.exp(-speed * (x + r) / (2 * kap)) > dT:
                lo = mid
            else:
                hi = mid
        ys[i] = 0.5 * (lo + hi)
    half_width = float(ys.max())
    return {"length": r_front + r_back, "length_ahead": r_front, "length_behind": r_back, "width": 2 * half_width, "depth": half_width,
            "absorbed_power": q, "speed": speed, "material": m.name, "mode": "conduction"}


def penetration(material: Solid | str, power: float, speed: float, spot_radius: float, absorption: float = 0.7, melting_efficiency: float = 0.4,
                T0: float = T_ROOM) -> dict:
    """Keyhole-mode penetration from the energy balance: d = η_abs η_melt P / (v b ρ (c ΔT_m + L_m)) with the weld width
    b = 2w₀ + 2√(2κ·2w₀/v) (the spot plus the heat's spread during the beam's dwell); ``absorption`` the keyhole's
    (0.6–0.8 for steel, ~0.5 for aluminium: the Fresnel share raised by the multiple reflections), ``melting_efficiency``
    the share of the absorbed energy that ends in the melt (Swift-Hook & Gick's bound 0.48 at speed; 0.4 is a keyhole
    weld's usual). Below the keyhole threshold the mode is conduction and the depth is Rosenthal's."""
    m = _solid(material)
    kap = m.diffusivity
    b = 2 * spot_radius + 2 * math.sqrt(2 * kap * 2 * spot_radius / speed)
    e_vol = m.rho * (m.c * (m.T_melt - T0) + m.L_melt)
    d = absorption * melting_efficiency * power / (speed * b * e_vol)
    th = keyhole_threshold(m, spot_radius, T0)
    mode = "keyhole" if power >= th["P_threshold_W"] else "conduction"
    out = {"depth": d, "width": b, "mode": mode, "keyhole_threshold_W": th["P_threshold_W"], "heat_input_J_per_m": power / speed,
           "material": m.name, "power": power, "speed": speed, "spot_radius": spot_radius}
    if mode == "conduction":
        pool = rosenthal_pool(m, m.absorptivity * power, speed, T0)
        out.update({"depth": pool["depth"], "width": pool["width"], "pool_length": pool["length"]})
    return out


def penetration_curve(material: Solid | str, power: float, speeds, spot_radius: float, **kw) -> dict:
    speeds = np.asarray(list(speeds), dtype=float)
    rows = [penetration(material, power, float(v), spot_radius, **kw) for v in speeds]
    return {"speed": speeds, "depth": np.array([r["depth"] for r in rows]), "width": np.array([r["width"] for r in rows]),
            "mode": [r["mode"] for r in rows], "heat_input_J_per_m": np.array([r["heat_input_J_per_m"] for r in rows])}


def speed_for_penetration(material: Solid | str, power: float, thickness: float, spot_radius: float, v_range=(1e-3, 1.0), **kw) -> float:
    """The fastest speed at which the keyhole depth still reaches ``thickness`` (bisection on the monotonic curve); nan when
    even the slowest speed does not."""
    lo, hi = v_range
    if penetration(material, power, lo, spot_radius, **kw)["depth"] < thickness:
        return math.nan
    if penetration(material, power, hi, spot_radius, **kw)["depth"] >= thickness:
        return hi
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if penetration(material, power, mid, spot_radius, **kw)["depth"] >= thickness:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def process_window(material: Solid | str, power: float, thicknesses, spot_radius: float, margin: float = 1.2, **kw) -> list[dict]:
    """Per thickness: the fastest full-penetration speed, the speed with ``margin`` of depth in hand, the heat input there."""
    rows = []
    for d in thicknesses:
        v_full = speed_for_penetration(material, power, d, spot_radius, **kw)
        v_margin = speed_for_penetration(material, power, margin * d, spot_radius, **kw)
        rows.append({"thickness": float(d), "v_full_penetration": v_full, "v_with_margin": v_margin,
                     "heat_input_J_per_m": power / v_margin if v_margin == v_margin and v_margin > 0 else math.nan,
                     "feasible": v_margin == v_margin})
    return rows
