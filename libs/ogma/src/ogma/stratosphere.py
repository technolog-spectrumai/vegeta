"""og.stratosphere — the U.S. Standard Atmosphere 1976 from sea level to 86 km, and gravity that falls with altitude.

``og.atmosphere`` is the troposphere only (one linear lapse rate): above 11 km it is wrong (at 20 km its density is
~7 % high, its temperature 60 K low, and both keep diverging). This module is the full 1976 model for vehicles that fly in the stratosphere
(balloons, airships, HAPS): seven layers of linear temperature in *geopotential* altitude, hydrostatic pressure, ideal-
gas density. Constants and layer bases as published (NOAA/NASA/USAF, *U.S. Standard Atmosphere, 1976*, NOAA-S/T 76-1562,
Table 4 and eqs. 18, 23, 33a/b); the tests check the published table values at 0, 11, 20, 25 and 30 km.

Altitudes are geometric [m] unless the name says ``geopotential``. ``delta_t`` [K] shifts the temperature at constant
pressure (hot/cold days), the convention of ``og.atmosphere``.

Verified physics (reference tables). Not modelled: humidity, latitude/season variation (use ``delta_t`` or a
climatology), winds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G0 = 9.80665                      # m/s², standard gravity at sea level
R_EARTH = 6356766.0               # m, the effective Earth radius the 1976 standard uses for geopotential altitude
R_STAR = 8.31432                  # J/(mol K), the 1976 standard's universal gas constant (the CODATA value is 8.314462)
M_AIR = 0.0289644                 # kg/mol, mean molar mass of air below 86 km
R_AIR = R_STAR / M_AIR            # 287.0531 J/(kg K)
GAMMA = 1.4
BETA, SUTHERLAND_S = 1.458e-6, 110.4      # μ = β T^1.5 / (T + S)

# geopotential base [m], lapse rate [K/m] of the seven layers (Table 4)
_H = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
_L = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])
LAYERS = ("troposphere", "tropopause", "stratosphere 1", "stratosphere 2", "stratopause", "mesosphere 1", "mesosphere 2")


def _bases():
    T = [288.15]
    p = [101325.0]
    for i in range(len(_L)):
        dh = _H[i + 1] - _H[i]
        Tb, L = T[-1], _L[i]
        if L == 0.0:
            p.append(p[-1] * math.exp(-G0 * M_AIR * dh / (R_STAR * Tb)))
        else:
            p.append(p[-1] * (Tb / (Tb + L * dh)) ** (G0 * M_AIR / (R_STAR * L)))
        T.append(Tb + L * dh)
    return np.array(T), np.array(p)


_T, _P = _bases()                 # base temperature and pressure of each layer (Table 4: 216.65 K, 22632.06 Pa at 11 km …)
H_MAX = 86000.0                   # m, geometric top of the model


def geopotential(z):
    """Geopotential altitude [m'] of geometric altitude ``z`` [m]: H = r0 z / (r0 + z)."""
    z = np.asarray(z, dtype=float)
    return R_EARTH * z / (R_EARTH + z)


def geometric(h):
    """Geometric altitude [m] of geopotential altitude ``h`` [m']."""
    h = np.asarray(h, dtype=float)
    return R_EARTH * h / (R_EARTH - h)


def gravity(z):
    """Gravitational acceleration [m/s²] at geometric altitude ``z`` [m]: g0 (r0 / (r0 + z))²."""
    z = np.asarray(z, dtype=float)
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2


def _check(z):
    z = np.asarray(z, dtype=float)
    if np.any(z < -5000.0) or np.any(z > H_MAX):
        raise ValueError(f"altitude outside the 1976 model (−5 km … 86 km): {z.min() if z.size else z} … {z.max() if z.size else z} m")
    return z


def temperature(z, delta_t: float = 0.0):
    """Kinetic temperature [K] at geometric altitude ``z`` [m] (vectorised)."""
    h = geopotential(_check(z))
    i = np.clip(np.searchsorted(_H, h, side="right") - 1, 0, len(_L) - 1)
    return _T[i] + _L[i] * (h - _H[i]) + delta_t


def pressure(z):
    """Static pressure [Pa] at geometric altitude ``z`` [m] (vectorised; independent of ``delta_t``)."""
    h = geopotential(_check(z))
    i = np.clip(np.searchsorted(_H, h, side="right") - 1, 0, len(_L) - 1)
    Tb, Pb, L, dh = _T[i], _P[i], _L[i], h - _H[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = Pb * (Tb / (Tb + L * dh)) ** (G0 * M_AIR / (R_STAR * np.where(L == 0.0, 1.0, L)))
    iso = Pb * np.exp(-G0 * M_AIR * dh / (R_STAR * Tb))
    return np.where(L == 0.0, iso, grad)


def density(z, delta_t: float = 0.0):
    """Air density [kg/m³] at geometric altitude ``z`` [m]: p / (R T)."""
    return pressure(z) / (R_AIR * temperature(z, delta_t))


def layer(z) -> str:
    h = float(geopotential(_check(z)))
    return LAYERS[int(np.clip(np.searchsorted(_H, h, side="right") - 1, 0, len(_L) - 1))]


@dataclass(frozen=True)
class State:
    """The air at one geometric altitude."""
    altitude: float               # m, geometric
    temperature: float            # K
    pressure: float               # Pa
    gravity: float                # m/s²

    @property
    def density(self) -> float:
        return self.pressure / (R_AIR * self.temperature)

    @property
    def speed_of_sound(self) -> float:
        return math.sqrt(GAMMA * R_AIR * self.temperature)

    @property
    def viscosity(self) -> float:
        """Dynamic viscosity [Pa s] (Sutherland, as the 1976 standard)."""
        return BETA * self.temperature ** 1.5 / (self.temperature + SUTHERLAND_S)

    @property
    def kinematic_viscosity(self) -> float:
        return self.viscosity / self.density

    @property
    def thermal_conductivity(self) -> float:
        """W/(m K), the 1976 standard's eq. 53: 2.648151e-3 T^1.5 / (T + 245.4 · 10^(−12/T))."""
        T = self.temperature
        return 2.648151e-3 * T ** 1.5 / (T + 245.4 * 10 ** (-12.0 / T))

    @property
    def geopotential_altitude(self) -> float:
        return float(geopotential(self.altitude))

    def dynamic_pressure(self, speed: float) -> float:
        return 0.5 * self.density * speed ** 2

    def reynolds(self, speed: float, length: float) -> float:
        return speed * length / self.kinematic_viscosity

    def as_dict(self) -> dict:
        return {"z_m": self.altitude, "H_m": self.geopotential_altitude, "T_K": self.temperature, "p_Pa": self.pressure,
                "rho": self.density, "g": self.gravity, "a": self.speed_of_sound, "mu": self.viscosity,
                "nu": self.kinematic_viscosity, "k": self.thermal_conductivity, "layer": layer(self.altitude)}


def state(z: float, delta_t: float = 0.0) -> State:
    """The 1976 atmosphere at geometric altitude ``z`` [m], ``delta_t`` [K] warmer at the same pressure."""
    return State(float(z), float(temperature(z, delta_t)), float(pressure(z)), float(gravity(z)))


def altitude_of_density(rho: float, delta_t: float = 0.0) -> float:
    """Geometric altitude [m] where the density is ``rho`` (bisection; density falls monotonically with altitude)."""
    lo, hi = -5000.0, H_MAX
    if not density(hi, delta_t) <= rho <= density(lo, delta_t):
        raise ValueError(f"density {rho} kg/m³ is outside the model's range")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if density(mid, delta_t) > rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def altitude_of_pressure(p: float) -> float:
    """Geometric altitude [m] where the pressure is ``p``."""
    lo, hi = -5000.0, H_MAX
    if not pressure(hi) <= p <= pressure(lo):
        raise ValueError(f"pressure {p} Pa is outside the model's range")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pressure(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
