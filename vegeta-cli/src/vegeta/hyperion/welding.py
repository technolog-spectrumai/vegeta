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


# ----------------------------------------------------------------------------------------------------- the thin plate
def _bessel_k0(x):
    """K₀(x) for x > 0 in numpy alone (Abramowitz & Stegun 9.8.5 for x ≤ 2 with I₀ from 9.8.1, 9.8.6 beyond): better
    than 1e-7 — scipy is optional in Hyperion."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x <= 2.0
    xs = np.where(small, x, 1.0)
    t = (xs / 3.75) ** 2
    i0 = 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    y = xs * xs / 4.0
    k_small = (-np.log(xs / 2) * i0 - 0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.0000074))))))
    xl = np.where(small, 2.0, x)
    z = 2.0 / xl
    k_large = np.exp(-xl) / np.sqrt(xl) * (1.25331414 + z * (-0.07832358 + z * (0.02189568 + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208))))))
    out = np.where(small, k_small, k_large)
    return out


def thin_plate_field(material: Solid | str, absorbed_power: float, speed: float, thickness: float, xi, y, T0: float = T_ROOM) -> np.ndarray:
    """Rosenthal's moving **line** source through a thin plate (2-D, no gradient through the thickness, no surface
    losses): T − T0 = q/(2π k d) · exp(−v ξ/2κ) · K₀(v r/2κ), ξ the distance ahead of the source along the weld (negative
    behind), y across it, r = √(ξ² + y²). The field a 1–3 mm sheet sees once the keyhole has deposited its heat — the
    regime EN 1011-2 calls the two-dimensional one. Arrays broadcast; the singular point returns the melting-point cap."""
    m = _solid(material)
    xi, y = np.broadcast_arrays(np.asarray(xi, dtype=float), np.asarray(y, dtype=float))
    kap = m.diffusivity
    r = np.hypot(xi, y)
    u = speed * np.maximum(r, 1e-12) / (2 * kap)
    dT = absorbed_power / (2 * math.pi * m.k * thickness) * np.exp(-speed * xi / (2 * kap)) * _bessel_k0(u)
    return np.minimum(T0 + dT, m.T_fail)


def _isotherm_half_width(m: Solid, q: float, v: float, d: float, T_iso: float, T0: float, n: int = 400) -> tuple[float, float, float]:
    """(half-width, length behind, length ahead) [m] of the T_iso isotherm of the thin-plate field, by bisection."""
    kap = m.diffusivity
    dT = T_iso - T0
    amp = q / (2 * math.pi * m.k * d)

    def excess(xi, y):
        r = math.hypot(xi, y)
        return amp * math.exp(-v * xi / (2 * kap)) * float(_bessel_k0(v * max(r, 1e-12) / (2 * kap))) - dT

    if excess(1e-9, 0.0) <= 0:
        return 0.0, 0.0, 0.0
    # behind: exp(u) K0(u) decays as √(π/2u): a long tail — bracket by doubling
    hi = 1e-4
    while excess(-hi, 0.0) > 0 and hi < 10.0:
        hi *= 2
    lo = 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(-mid, 0.0) > 0 else (lo, mid)
    behind = 0.5 * (lo + hi)
    hi = 1e-4
    while excess(hi, 0.0) > 0 and hi < 10.0:
        hi *= 2
    lo = 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid, 0.0) > 0 else (lo, mid)
    ahead = 0.5 * (lo + hi)
    half = 0.0
    for xi in np.linspace(-behind, ahead, n):
        lo, hi = 0.0, behind + ahead
        if excess(xi, 1e-9) <= 0:
            continue
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if excess(xi, mid) > 0 else (lo, mid)
        half = max(half, 0.5 * (lo + hi))
    return half, behind, ahead


def rosenthal_thin_plate(material: Solid | str, absorbed_power: float, speed: float, thickness: float, T0: float = T_ROOM, n_y: int = 60) -> dict:
    """The thin-plate (line-source) pool and its surroundings: the melt isotherm's half-width, length behind and ahead
    (``width``, ``length``), the solidification time of a point on the centreline (length behind / v), the peak
    temperature against the lateral distance (``y``, ``peak_T``: each point's maximum over the pass), the thermal
    gradient at the fusion boundary, and ``field(xi, y)``. Rosenthal 1946 / Rykalin; the regime of EN 1011-2 Annex D
    (two-dimensional heat flow) — valid while the plate is thinner than the pool is wide, which 1–3 mm sheets under a
    keyhole are."""
    m = _solid(material)
    half, behind, ahead = _isotherm_half_width(m, absorbed_power, speed, thickness, m.T_melt, T0)
    ys = np.linspace(0.0, max(6 * half, 2e-3), n_y)
    xis = np.linspace(-max(4 * behind, 5e-3), max(2 * ahead, 1e-3), 400)
    XI, Y = np.meshgrid(xis, ys)
    field = thin_plate_field(m, absorbed_power, speed, thickness, XI, Y, T0)
    peak = field.max(axis=1)
    i_fz = int(np.searchsorted(-peak, -m.T_melt))                      # the first lateral point below the melting point
    grad = float((peak[max(i_fz - 1, 0)] - peak[min(i_fz, n_y - 1)]) / max(ys[1] - ys[0], 1e-12)) if 0 < i_fz < n_y else math.nan
    return {"width": 2 * half, "half_width": half, "length_behind": behind, "length_ahead": ahead, "length": behind + ahead,
            "solidification_time": behind / speed if speed > 0 else math.inf, "y": ys, "peak_T": peak, "gradient_at_fusion_K_per_m": grad,
            "absorbed_power": absorbed_power, "speed": speed, "thickness": thickness, "material": m.name, "regime": "thin plate (2-D line source)",
            "field": lambda xi, y: thin_plate_field(m, absorbed_power, speed, thickness, xi, y, T0)}


def cooling_time_t85(material: Solid | str, heat_input_J_per_m: float, thickness: float, T0: float = T_ROOM) -> dict:
    """The time from 800 to 500 °C on the weld's centreline (Rykalin; EN 1011-2 Annex D with its thermal efficiency
    already in the absorbed heat input): thin plate t = (Q/d)² (1/(773 − T0)² − 1/(1073 − T0)²)/(4π k ρ c), thick plate
    t = Q (1/(773 − T0) − 1/(1073 − T0))/(2π k); the critical thickness between the regimes and the one that applies
    (the larger of the two times is the physical one near the transition). Steel's hardness and hydrogen cracking
    hang on t8/5; for aluminium the temperatures mean nothing — use ``cooling_rate_at``."""
    m = _solid(material)
    T5, T8 = 773.15, 1073.15
    Q = heat_input_J_per_m
    thin = (Q / thickness) ** 2 * (1 / (T5 - T0) ** 2 - 1 / (T8 - T0) ** 2) / (4 * math.pi * m.k * m.rho * m.c)
    thick = Q * (1 / (T5 - T0) - 1 / (T8 - T0)) / (2 * math.pi * m.k)
    d_crit = math.sqrt(Q / (2 * m.rho * m.c) * (1 / (T5 - T0) + 1 / (T8 - T0)))
    regime = "thin" if thickness < d_crit else "thick"
    return {"t85_s": thin if regime == "thin" else thick, "t85_thin_s": thin, "t85_thick_s": thick, "critical_thickness": d_crit, "regime": regime,
            "heat_input_J_per_m": Q, "thickness": thickness, "material": m.name}


def cooling_rate_at(material: Solid | str, heat_input_J_per_m: float, thickness: float, T: float, T0: float = T_ROOM) -> dict:
    """The centreline cooling rate [K/s] as it passes ``T``: thin plate R = 2π k ρ c (d/Q)² (T − T0)³, thick plate
    R = 2π k (T − T0)²/Q (Adams 1958); the thin-plate value is the sheet's, the thick-plate one the floor a thick section
    would give. Any material — the number aluminium's dendrite arm spacing and porosity are read against."""
    m = _solid(material)
    Q = heat_input_J_per_m
    thin = 2 * math.pi * m.k * m.rho * m.c * (thickness / Q) ** 2 * (T - T0) ** 3
    thick = 2 * math.pi * m.k * (T - T0) ** 2 / Q
    return {"rate_thin_K_per_s": thin, "rate_thick_K_per_s": thick, "T": T, "heat_input_J_per_m": Q, "thickness": thickness, "material": m.name}


def haz_width(material: Solid | str, absorbed_power: float, speed: float, thickness: float, T_haz: float, T0: float = T_ROOM) -> dict:
    """The heat-affected zone of the thin-plate field: the half-width of the ``T_haz`` isotherm (steel: Ac1 ≈ 996 K; an
    aluminium alloy's over-ageing or recrystallisation ≈ 570–620 K) beyond the melt's half-width, on each side."""
    m = _solid(material)
    half_melt, _, _ = _isotherm_half_width(m, absorbed_power, speed, thickness, m.T_melt, T0)
    half_haz, _, _ = _isotherm_half_width(m, absorbed_power, speed, thickness, T_haz, T0)
    return {"haz_width_each_side": max(half_haz - half_melt, 0.0), "haz_half_width": half_haz, "melt_half_width": half_melt, "T_haz": T_haz, "material": m.name}
