"""The corona and its thrust in the one-dimensional limit: onset voltage, the current law, thrust, power, the jet.

Relations (positive DC corona in air, the unipolar drift region doing the work):

* onset: the field at the emitter reaches Peek's value ``E_peek(r)``; the field there, for a thin emitter of tip
  radius ``r`` at distance ``d`` from a wide ground, is ``E_tip = 2 V / (r ln(4 d / r))`` (the hyperboloid
  approximation), so ``V0 = E_peek r ln(4d/r) / 2``;
* current: ``I = C V (V - V0)`` (Townsend 1914, Warburg): per metre of emitter edge ``C = 2 k eps0 mu L / d^2``
  (the wire-to-plane form of Sigmond's saturation current; ``k = collection`` is the fraction of it a ring ground
  collects — the one fitted geometry factor of this model, 0.6 from the Mark IV's 50 W; the 2-D solve replaces it);
* thrust: ``T = I d / mu`` (all the ions' momentum goes to the neutrals); thrust per watt ``T / P = d / (mu V)``;
* the jet: ``T = rho A v^2`` at the exit area ``A`` (a free jet with no losses through the rings).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .gas import EPS0, Air, DRY


@dataclass(frozen=True)
class Gap:
    """One emitter-to-ground gap: ``d`` [m], emitter tip radius ``r`` [m], emitter length ``length`` [m] (the
    sum of its edges' lengths), ``collection`` the fraction of Sigmond's saturation geometry the ground collects (0..1),
    ``roughness`` Peek's m (teeth: 0.8-1; a tarnished edge 0.6)."""

    d: float
    r: float
    length: float
    collection: float = 0.6
    roughness: float = 0.9

    def onset_voltage(self, air: Air = DRY) -> float:
        return air.peek_field(self.r, self.roughness) * self.r * math.log(4.0 * self.d / self.r) / 2.0

    def current_constant(self, air: Air = DRY) -> float:
        """C in I = C V (V - V0), [A/V^2]."""
        return 2.0 * self.collection * EPS0 * air.mobility() * self.length / self.d ** 2

    def current(self, V: float, air: Air = DRY) -> float:
        V0 = self.onset_voltage(air)
        return 0.0 if V <= V0 else self.current_constant(air) * V * (V - V0)

    def thrust(self, V: float, air: Air = DRY) -> float:
        return self.current(V, air) * self.d / air.mobility()

    def power(self, V: float, air: Air = DRY) -> float:
        return self.current(V, air) * V

    def mean_field(self, V: float) -> float:
        return V / self.d


def thrust_per_watt(d: float, V: float, air: Air = DRY) -> float:
    """N/W of the drift region: d / (mu V) = 1 / (mu E)."""
    return d / (air.mobility() * V)


def jet_velocity(thrust: float, area: float, air: Air = DRY) -> float:
    """Mean exit velocity [m/s] of a free jet carrying ``thrust`` through ``area``."""
    return math.sqrt(max(thrust, 0.0) / (air.density * area))


def thrust_from_velocity(v: float, area: float, air: Air = DRY) -> float:
    return air.density * area * v * v


def voltage_for_thrust(gap: Gap, thrust: float, air: Air = DRY) -> float:
    """The voltage at which ``gap`` makes ``thrust`` (the positive root of C V (V - V0) d / mu = T)."""
    C, V0 = gap.current_constant(air), gap.onset_voltage(air)
    I = thrust * air.mobility() / gap.d
    return (V0 + math.sqrt(V0 * V0 + 4.0 * I / C)) / 2.0
