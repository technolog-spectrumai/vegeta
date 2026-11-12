"""Strato — gas superheat and solar-cell temperature (Stage V8.8). **Estimate**: a two-node lumped model.

Nodes: the envelope (laminate and array as one shell at one temperature) and the lifting gas. The envelope absorbs
sunlight on every facet (``og.solar.irradiance_on``: beam, sky, albedo), less the electricity the array takes out;
exchanges infrared with the Earth below (outgoing long-wave radiation, global mean ~240 W/m²) and with the sky above
(small at 20 km); radiates from its outside; convects to the outside air (forced, turbulent flat plate) and to the gas
inside (natural convection, a fixed coefficient). The gas warms or cools only through the envelope (helium is
transparent in the infrared). The array's own temperature is a local steady balance on the mean array irradiance,
backed by the envelope node.

What is physics: the energy balances, Stefan–Boltzmann, the flat-plate correlation. What is assumed (marked here and in
STRATO.md): the single envelope temperature (a real hull is hot on top and cold below), the coefficients below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ogma import stratosphere as st

SIGMA = 5.670374419e-8            # W/(m² K⁴)
OLR = 240.0                       # W/m², Earth's outgoing long-wave radiation (global annual mean; estimate for a site)
SKY_IR = 15.0                     # W/m², down-welling IR above ~18 km (estimate)
H_INTERNAL = 0.5                  # W/(m² K), envelope ↔ gas natural convection at stratospheric density (estimate)
CP_LAMINATE = 1300.0              # J/(kg K) (estimate, polymer laminates 1100–1500)
CV_HELIUM = 3116.0                # J/(kg K), monatomic ideal gas 3/2 R
PRANDTL = 0.71


def h_forced(speed: float, length: float, air: st.State) -> float:
    """Mean forced-convection coefficient [W/(m² K)] on a body of ``length``: turbulent flat plate
    Nu = 0.037 Re^0.8 Pr^(1/3), with a floor of 0.5 W/(m² K) for still air (estimate)."""
    re = air.reynolds(max(speed, 0.0), length)
    if re < 1.0:
        return 0.5
    nu = 0.037 * re ** 0.8 * PRANDTL ** (1 / 3)
    return max(nu * air.thermal_conductivity / length, 0.5)


@dataclass
class ThermalState:
    T_envelope: float             # K
    T_gas: float                  # K


@dataclass(frozen=True)
class Hull:
    """The thermal view of a ship: facets (from ``strato_geometry.Strato.facets``) and the node capacities."""
    normals_body: np.ndarray
    area: np.ndarray
    array_fraction: np.ndarray
    absorptance: np.ndarray       # per facet: array share × α_pv + rest × α_laminate
    emissivity: np.ndarray
    length: float
    envelope_capacity: float      # J/K
    gas_mass: float               # kg


def view_factors(normals_enu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(to the sky, to the Earth) for facets with these east-north-up normals (isotropic halves)."""
    up = normals_enu[:, 2]
    return (1 + up) / 2, (1 - up) / 2


def envelope_step(hull: Hull, s: ThermalState, irradiance: np.ndarray, electric_w: float, normals_enu: np.ndarray,
                  air: st.State, speed: float, dt: float, substeps: int = 10) -> ThermalState:
    """Advance the two nodes by ``dt`` [s] (explicit, in ``substeps``)."""
    f_sky, f_earth = view_factors(normals_enu)
    absorbed = float(np.sum(hull.absorptance * irradiance * hull.area)) - electric_w
    ir_in = float(np.sum(hull.emissivity * hull.area * (SKY_IR * f_sky + OLR * f_earth)))
    eps_a = float(np.sum(hull.emissivity * hull.area))
    A = float(np.sum(hull.area))
    h = h_forced(speed, hull.length, air)
    c_gas = hull.gas_mass * CV_HELIUM
    Te, Tg = s.T_envelope, s.T_gas
    # explicit steps below half the fastest node time constant (dense air near the ground convects strongly)
    k_env = (h + H_INTERNAL) * A + 4 * eps_a * SIGMA * max(Te, air.temperature) ** 3
    tau = min(hull.envelope_capacity / k_env, c_gas / (H_INTERNAL * A)) if c_gas > 0 else hull.envelope_capacity / k_env
    substeps = max(substeps, int(math.ceil(dt / (0.5 * tau))))
    d = dt / substeps
    for _ in range(substeps):
        q_int = H_INTERNAL * A * (Te - Tg)
        q_env = absorbed + ir_in - eps_a * SIGMA * Te ** 4 - h * A * (Te - air.temperature) - q_int
        Te += d * q_env / hull.envelope_capacity
        Tg += d * q_int / c_gas
    return ThermalState(Te, Tg)


def cell_temperature(irradiance: float, efficiency: float, absorptance: float, emissivity: float, air: st.State,
                     speed: float, length: float, T_back: float) -> float:
    """Steady temperature [K] of an array facet: (α − η) E + ε (SKY_IR) = h (T − T_air) + ε σ T⁴ + H_INTERNAL (T − T_back)
    (top-facing facet; bisection)."""
    h = h_forced(speed, length, air)
    q_in = (absorptance - efficiency) * irradiance + emissivity * SKY_IR

    def residual(T):
        return q_in - h * (T - air.temperature) - emissivity * SIGMA * T ** 4 - H_INTERNAL * (T - T_back)

    lo, hi = 100.0, 450.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equilibrium_guess(air: st.State) -> ThermalState:
    """A night-time starting point: envelope and gas a few kelvin under the air (IR loss to the sky)."""
    return ThermalState(air.temperature - 5.0, air.temperature - 3.0)


def superheat(s: ThermalState, air: st.State) -> float:
    return s.T_gas - air.temperature

