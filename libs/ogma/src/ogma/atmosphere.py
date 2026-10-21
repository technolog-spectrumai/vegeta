"""og.atmosphere — the International Standard Atmosphere (troposphere), with a temperature offset for hot and cold
days. Consolidates Vegeta's three copies (``boreas.microjet.isa``, ``nisus_plus_systems.atmosphere`` /
``density_altitude``, ``velutina_flight.isa_density``) and reproduces each (tests/test_atmosphere.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R = 287.05                        # J/(kg K), the value Vegeta's models use
T0, P0, RHO0, LAPSE = 288.15, 101325.0, 1.225, 0.0065
G0 = 9.80665
GAMMA = 1.4
P_EXPONENT = 5.2559               # g / (L R), rounded as the ISA tables do
SUTHERLAND = (1.458e-6, 110.4)    # μ = C T^1.5 / (T + S)


@dataclass(frozen=True)
class Air:
    pressure: float = P0          # Pa
    temperature: float = T0       # K

    @property
    def density(self) -> float:
        return self.pressure / (R * self.temperature)

    @property
    def speed_of_sound(self) -> float:
        return math.sqrt(GAMMA * R * self.temperature)

    @property
    def viscosity(self) -> float:
        c, s = SUTHERLAND
        return c * self.temperature ** 1.5 / (self.temperature + s)


def isa(altitude_m: float = 0.0, delta_t: float = 0.0) -> Air:
    """The troposphere at ``altitude_m`` with ``delta_t`` [K] added to the temperature (the pressure stays standard)."""
    t_std = T0 - LAPSE * altitude_m
    return Air(P0 * (t_std / T0) ** P_EXPONENT, t_std + delta_t)


def density_altitude(rho: float) -> float:
    """The ISA altitude [m] whose density is ``rho``."""
    return T0 / LAPSE * (1 - (rho / RHO0) ** (1 / (P_EXPONENT - 1)))


def atmosphere(h_m: float = 0.0, dT: float = 0.0) -> dict:
    """Everything a flight model asks for at ``h_m`` [m] and ``dT`` [K]: T, p, ρ, μ, ν, a, σ = ρ/ρ0, density altitude."""
    a = isa(float(h_m), float(dT))
    T, rho, mu = a.temperature, a.density, a.viscosity
    return {"h_m": float(h_m), "dT": float(dT), "T_K": T, "T_C": T - 273.15, "p_Pa": a.pressure, "rho": rho, "mu": mu, "nu": mu / rho,
            "a": a.speed_of_sound, "sigma": rho / RHO0, "density_altitude_m": density_altitude(rho)}


def density(altitude_m):
    """ISA density [kg/m³], vectorised: ρ0 (T/T0)^(g0/(L R) − 1) (the closed form; within 0.01 % of ``isa``)."""
    t = T0 - LAPSE * np.asarray(altitude_m, dtype=float)
    return RHO0 * (t / T0) ** (G0 / (LAPSE * R) - 1.0)
