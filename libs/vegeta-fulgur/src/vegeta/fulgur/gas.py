"""Air as the discharge sees it: density, ion mobility, Peek's corona onset field, the breakdown field, the net
ionisation coefficient — each with its temperature, pressure and humidity dependence, each marked where it is a fit.

Everything is SI. The reference state is 20 °C, 101 325 Pa, dry (``Air()``); ``delta`` is the relative air density
Peek's law and every field scale with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EPS0 = 8.854_187_8e-12
R_GAS = 8.314_462
M_DRY = 0.028_964           # kg/mol
M_H2O = 0.018_015
T0, P0 = 293.15, 101_325.0

MU_POS_0 = 2.0e-4           # m^2/(V s): positive ions in dry air at the reference state (literature 1.8-2.2)
MU_NEG_0 = 2.2e-4           # negative ions (O2-, hydrated); 2.1-2.7 reported
E_CRIT_0 = 2.6e6            # V/m: where net ionisation is zero in dry air at 1 atm (24.4 V/(cm Torr))
E_BREAK_0 = 3.0e6           # V/m: uniform-field breakdown, 1 atm, dry (30 kV/cm; 31.4 for a 1 cm gap)
E_PEEK_0 = 3.0e6            # V/m: Peek's constant (30 kV/cm) ...
PEEK_K_CM = 0.301           # ... and its wire term 0.301/sqrt(delta r[cm]) (Peek 1929, for a wire)
E_STAB_POS_0 = 0.5e6        # V/m: positive streamers keep propagating above this mean field in dry air (Gallimberti)
ALPHA_C = 5.2e-4            # cm Torr/V^2: (alpha - eta)/p = C (E/p - E_crit/p)^2 — anchored at 26 kV/cm -> 0 and
#                             30 kV/cm -> 11 /cm (Kuffel & Zaengl's air data); a fit for 26-40 kV/cm, it overestimates
#                             above (the inception integral is dominated by the tip, where it only shortens the line)


def saturation_pressure(T: float) -> float:
    """Water vapour saturation pressure [Pa] (Magnus, over water), ``T`` in K."""
    t = T - 273.15
    return 610.94 * math.exp(17.625 * t / (t + 243.04))


@dataclass(frozen=True)
class Air:
    """Air at ``T`` [K], ``p`` [Pa], relative humidity ``rh`` (0..1)."""

    T: float = T0
    p: float = P0
    rh: float = 0.0

    @property
    def p_water(self) -> float:
        return self.rh * saturation_pressure(self.T)

    @property
    def absolute_humidity(self) -> float:
        """Water vapour [g/m^3] (the quantity the breakdown corrections use)."""
        return self.p_water * M_H2O / (R_GAS * self.T) * 1000.0

    @property
    def density(self) -> float:
        """kg/m^3, moist air (lighter than dry: water's molar mass is lower)."""
        pw = self.p_water
        return ((self.p - pw) * M_DRY + pw * M_H2O) / (R_GAS * self.T)

    @property
    def delta(self) -> float:
        """The relative air density (p/p0)(T0/T) — the scaling of every field in Peek's and Paschen's laws."""
        return (self.p / P0) * (T0 / self.T)

    def mobility(self, polarity: str = "+") -> float:
        """Ion mobility [m^2/(V s)]: the reduced mobility is constant, so mu ~ 1/delta. Humidity: positive ions
        hydrate into heavier clusters, about -10 % at 80 % RH (an estimate from the cluster-mobility data, not a law)."""
        mu0 = MU_POS_0 if polarity == "+" else MU_NEG_0
        wet = 1.0 - 0.12 * self.rh if polarity == "+" else 1.0 - 0.05 * self.rh
        return mu0 * wet / self.delta

    def peek_field(self, r: float, roughness: float = 1.0) -> float:
        """Corona onset field [V/m] at a wire (or an edge) of radius ``r`` [m]: Peek's law, ``roughness`` m in 0.6-1."""
        d = self.delta
        return E_PEEK_0 * d * roughness * (1.0 + PEEK_K_CM / math.sqrt(d * r * 100.0))

    @property
    def e_crit(self) -> float:
        return E_CRIT_0 * self.delta

    @property
    def e_break(self) -> float:
        """Uniform-field breakdown [V/m]; humidity barely matters in a uniform field (~ +0.2 % per g/m^3)."""
        return E_BREAK_0 * self.delta * (1.0 + 0.002 * (self.absolute_humidity - 11.0))

    @property
    def e_stability(self) -> float:
        """The positive-streamer stability field [V/m]; grows ~ +1.2 % per g/m^3 of water (Gallimberti 1979)."""
        return E_STAB_POS_0 * self.delta * (1.0 + 0.012 * self.absolute_humidity)

    def alpha_eff(self, E: float) -> float:
        """Net ionisation coefficient alpha - eta [1/m] at field ``E`` [V/m] (0 below ``e_crit``)."""
        p_torr = self.p / 133.322 * (T0 / self.T)           # density-corrected pressure
        x = abs(E) / 100.0 / p_torr                          # V/(cm Torr)
        x0 = E_CRIT_0 / 100.0 / (P0 / 133.322)
        if x <= x0:
            return 0.0
        return ALPHA_C * p_torr * (x - x0) ** 2 * 100.0     # /cm -> /m


DRY = Air()
INDOOR = Air(T=294.15, rh=0.30)                               # a heated room in winter
