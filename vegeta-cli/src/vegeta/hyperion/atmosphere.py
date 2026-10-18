"""What the air takes from a 1 µm beam over tens to hundreds of metres: the Beer–Lambert transmission through four
named atmospheres — molecular (Rayleigh) scattering and water-vapour absorption are negligible at 1070 nm; what
matters is **aerosol scattering**, which the meteorological visibility sets through Kruse's empirical law
α = (3.912 / V) (λ / 550 nm)^−q with q from the visibility (Kim et al. 2001), and in fog the Mie regime where the
droplets are larger than the wavelength and the loss no longer falls with λ. The numbers are the free-space-optics
link-budget ones: an order of magnitude by atmosphere, the visibility the one input to measure on the day."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def kruse_q(visibility_m: float) -> float:
    """Kim's piecewise exponent: 1.6 above 50 km, 1.3 for 6–50 km, 0.16 V + 0.34 for 1–6 km, V − 0.5 for 0.5–1 km, 0 below."""
    v = visibility_m / 1e3
    if v > 50:
        return 1.6
    if v > 6:
        return 1.3
    if v > 1:
        return 0.16 * v + 0.34
    if v > 0.5:
        return v - 0.5
    return 0.0


def aerosol_attenuation(visibility_m: float, wavelength: float) -> float:
    """Kruse–Kim: α [1/m] = (3.912 / V) (λ / 550 nm)^−q, the 2 % contrast visibility V."""
    return 3.912 / visibility_m * (wavelength / 550e-9) ** (-kruse_q(visibility_m))


@dataclass(frozen=True)
class Atmosphere:
    """A named day: the visibility [m] (its aerosol scattering by Kruse–Kim), plus an ``extra`` absorption [1/m] for
    what the visibility does not carry (dust's absorption, a trace of water-vapour line absorption near 1 µm: ~1e-5/m)."""
    name: str
    visibility_m: float
    extra: float = 0.0
    note: str = ""

    def attenuation(self, wavelength: float) -> float:
        return aerosol_attenuation(self.visibility_m, wavelength) + self.extra

    def transmission(self, distance, wavelength: float) -> np.ndarray:
        return np.exp(-self.attenuation(wavelength) * np.asarray(distance, dtype=float))

    def loss_db_per_km(self, wavelength: float) -> float:
        return 10 * math.log10(math.e) * self.attenuation(wavelength) * 1e3


ATMOSPHERES = {
    "clear day": Atmosphere("clear day", 40e3, 1e-5, "a dry, clear day: 40 km visibility, the water-vapour continuum near 1 µm as the floor"),
    "wet day": Atmosphere("wet day", 8e3, 3e-5, "rain or high humidity: 8 km visibility (haze between showers), more vapour; rain itself adds ~1–5 dB/km at heavy rates"),
    "dusty town": Atmosphere("dusty town", 4e3, 2e-4, "urban haze and dust: 4 km visibility and absorbing particles (soot, mineral dust)"),
    "foggy autumn morning": Atmosphere("foggy autumn morning", 300.0, 0.0, "moderate fog, 300 m visibility: Mie scattering by droplets, wavelength-independent"),
}


def transmission_table(distances, wavelength: float, atmospheres: dict | None = None) -> dict:
    """{atmosphere: {distance: transmission}} and the attenuation per atmosphere."""
    atmos = atmospheres or ATMOSPHERES
    return {name: {"attenuation_per_m": a.attenuation(wavelength), "db_per_km": a.loss_db_per_km(wavelength), "visibility_m": a.visibility_m,
                   "transmission": {float(d): float(a.transmission(d, wavelength)) for d in distances}} for name, a in atmos.items()}
