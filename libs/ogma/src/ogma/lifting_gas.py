"""og.lifting_gas — buoyancy of a gas cell: helium or hydrogen in an envelope, sealed or vented, with an air ballonet.

Ideal gases (exact enough below ~1 bar: helium's compressibility is within 0.1 % of 1 at these pressures). The gas is
at the ambient pressure plus the envelope's superpressure ``dp`` and at the ambient temperature plus the superheat
``dT`` (the sun warms the gas by day, IR cools it at night: see a thermal model for the values).

Two ways to run the gas, chosen by ``vented``:

- **sealed / superpressure** (``vented=False``): the gas mass is fixed. Below the *pressure altitude* the gas does not
  fill the envelope (the ballonet holds air, or the envelope is slack); above it the envelope is full, the volume is
  fixed and the superpressure rises with altitude and temperature. Lift stays constant day and night while the hull
  is full — the reason long-endurance balloons and airships are superpressure designs. Limit: the envelope's strength.
- **vented / zero-pressure** (``vented=True``): above the pressure altitude a valve releases gas to keep ``dp`` at the
  valve setting. Gas lost by day is not there at night: the cold gas shrinks, the lift falls and the ship sinks unless
  it drops ballast. Limit: the gas (and ballast) budget.

Gross lift = displaced air mass; net (static) lift = displaced air − gas = V (ρ_air − ρ_gas). Everything here is
verified physics (ideal gas, Archimedes); the inputs (purity, superheat, superpressure) are the user's.
"""
from __future__ import annotations

from dataclasses import dataclass

R_UNIVERSAL = 8.314462618         # J/(mol K), CODATA 2018
M_AIR = 0.0289644                 # kg/mol (the 1976 standard's)
GASES = {"helium": 0.0040026, "hydrogen": 0.00201588, "air": M_AIR}   # kg/mol


def specific_gas_constant(gas: str = "helium", purity: float = 1.0) -> float:
    """R [J/(kg K)] of the lifting gas, with ``1 − purity`` (molar) of air mixed in (purity 0.98 is typical of a cell
    after some weeks: air diffuses in)."""
    if gas not in GASES:
        raise ValueError(f"unknown gas {gas!r} (known: {', '.join(GASES)})")
    if not 0.0 < purity <= 1.0:
        raise ValueError("purity must be in (0, 1]")
    m = purity * GASES[gas] + (1.0 - purity) * M_AIR
    return R_UNIVERSAL / m


def gas_density(p: float, T: float, gas: str = "helium", purity: float = 1.0) -> float:
    return p / (specific_gas_constant(gas, purity) * T)


def air_density(p: float, T: float) -> float:
    return p / (R_UNIVERSAL / M_AIR * T)


def net_lift_per_volume(p_air: float, T_air: float, dp: float = 0.0, dT: float = 0.0, gas: str = "helium",
                        purity: float = 1.0) -> float:
    """Static lift [kg/m³] of gas at (p_air + dp, T_air + dT) in air at (p_air, T_air)."""
    return air_density(p_air, T_air) - gas_density(p_air + dp, T_air + dT, gas, purity)


@dataclass
class GasCell:
    """A lifting-gas cell of maximum (full) volume ``volume`` [m³] holding ``gas_mass`` [kg].

    ``state(p_air, T_air, dT)`` returns the gas volume, the superpressure and the lift; ``vent(...)`` applies the
    valve (vented cells only) and returns the mass released. The cell is the gas: the envelope's mass and strength
    are the caller's."""
    volume: float
    gas_mass: float
    gas: str = "helium"
    purity: float = 1.0
    vented: bool = False
    valve_dp: float = 0.0          # Pa, the vent's opening superpressure (vented cells)

    def __post_init__(self):
        if self.volume <= 0 or self.gas_mass < 0:
            raise ValueError("a gas cell needs a positive volume and a non-negative gas mass")

    @property
    def R(self) -> float:
        return specific_gas_constant(self.gas, self.purity)

    def free_volume(self, p_air: float, T_gas: float) -> float:
        """Volume [m³] the gas would take at ambient pressure (no envelope)."""
        return self.gas_mass * self.R * T_gas / p_air

    def state(self, p_air: float, T_air: float, dT: float = 0.0) -> dict:
        T_gas = T_air + dT
        v_free = self.free_volume(p_air, T_gas)
        full = v_free >= self.volume
        v = self.volume if full else v_free
        dp = self.gas_mass * self.R * T_gas / self.volume - p_air if full else 0.0
        rho_air = air_density(p_air, T_air)
        return {"gas_volume": v, "fill": v / self.volume, "full": full, "superpressure": dp, "T_gas": T_gas,
                "gross_lift": rho_air * v, "net_lift": rho_air * v - self.gas_mass, "gas_mass": self.gas_mass}

    def vent(self, p_air: float, T_air: float, dT: float = 0.0) -> float:
        """Release gas until the superpressure is at most ``valve_dp`` (vented cells; sealed cells release nothing).
        Returns the mass released [kg]."""
        if not self.vented:
            return 0.0
        T_gas = T_air + dT
        m_max = (p_air + self.valve_dp) * self.volume / (self.R * T_gas)
        released = max(0.0, self.gas_mass - m_max)
        self.gas_mass -= released
        return released

    def pressure_altitude_pressure(self, T_gas: float) -> float:
        """The ambient pressure [Pa] at which the gas just fills the cell at gas temperature ``T_gas`` (zero
        superpressure); with og.stratosphere.altitude_of_pressure this is the pressure altitude."""
        return self.gas_mass * self.R * T_gas / self.volume


def fill_for_full(volume: float, p_air: float, T_gas: float, dp: float = 0.0, gas: str = "helium",
                  purity: float = 1.0) -> float:
    """Gas mass [kg] that fills ``volume`` at ambient ``p_air`` plus ``dp`` and gas temperature ``T_gas`` — the
    launch fill of a superpressure cell sized to stay full (``dp`` ≥ 0) at its coldest float condition."""
    return (p_air + dp) * volume / (specific_gas_constant(gas, purity) * T_gas)


def superpressure(gas_mass: float, volume: float, p_air: float, T_gas: float, gas: str = "helium",
                  purity: float = 1.0) -> float:
    """Superpressure [Pa] of a full sealed cell (negative = not full: the cell is slack)."""
    return gas_mass * specific_gas_constant(gas, purity) * T_gas / volume - p_air


def adiabatic_rate(T_gas: float, p: float, dp_dt: float, gamma: float = 5.0 / 3.0) -> float:
    """dT/dt [K/s] of a gas parcel following its pressure adiabatically (helium γ = 5/3): (γ−1)/γ · T/p · dp/dt."""
    return (gamma - 1.0) / gamma * T_gas / p * dp_dt

