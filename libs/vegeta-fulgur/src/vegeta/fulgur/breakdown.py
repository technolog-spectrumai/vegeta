"""Where the corona ends: the spark, predicted and kept away from (``docs/plasma.md``, *Arcs*).

Four criteria, each a breakdown voltage; the **spark margin** is the smallest of them over the operating voltage:

1. ``inception``: Meek's streamer criterion — the integral of the net ionisation coefficient along a field line
   from the emitter reaches ``K`` (18; Raether's 20);
2. ``propagation``: the streamer crosses when the gap's voltage exceeds what the stability field needs over the
   gap plus what the streamer's head needs to form, ``V = E_stab d + V_s`` (0.5 MV/m and 15 kV, positive, dry: the
   positive needle-plane DC data — ~20 kV at 1 cm, ~30 kV at 3 cm where a lifter at 30 kV sits at the edge, ~65 kV
   at 10 cm); below that a corona runs above onset without sparking;
3. ``uniform``: a uniform gap (two wide electrodes) breaks down at ``E_break d`` — the Paschen limit at 1 atm;
4. ``surface``: flashover along a dielectric support of creepage length ``creepage``: the mean tangential field
   must stay under a surface limit, 2 MV/m on clean dry resin falling as ``1 - 0.7 rh^3`` (condensation sets in
   above ~70 % RH; the factor reproduces the video's 78 %-day loss and its 50 %-day gain — an estimate until the lab
   measures it).

The inception integral needs the field along the line; ``field_line_hyperboloid`` gives it for a tooth of radius
``r`` at distance ``d`` from a plane (the same approximation as the onset voltage); the 2-D solver supplies real
lines later. Space charge is left out here (it lowers the field at the emitter and raises it in the gap), which is
why the propagation criterion, not inception, is what decides for a running corona.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gas import Air, DRY

MEEK_K = 18.0
V_STREAMER_0 = 15.0e3           # V: the propagation law's offset in dry air at 1 atm (fit to needle-plane data)
E_SURFACE_DRY = 2.0e6           # V/m along a clean dry resin surface (estimate)
SURFACE_WET_K = 0.7             # the derating 1 - K rh^3 (estimate)


def field_line_hyperboloid(V: float, r: float, d: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(s, E(s)) from a tip of radius ``r`` to a plane ``d`` away, at voltage ``V``: the prolate-spheroid field
    ``E(s) = 2 V / ((r + 2 s)(1 + 2 s / r ... )`` simplified to the standard ``E(s) = 2V / (ln(4d/r) (r + 2s - 2s^2/d))``
    on the axis (Coelho & Debeau 1971)."""
    s = np.linspace(0.0, d, n)
    denom = (r + 2.0 * s - 2.0 * s * s / d)
    denom = np.maximum(denom, r)                         # the plane end: E stays finite
    return s, 2.0 * V / (math.log(4.0 * d / r) * denom)


def inception_integral(V: float, r: float, d: float, air: Air = DRY) -> float:
    s, E = field_line_hyperboloid(V, r, d)
    a = np.array([air.alpha_eff(e) for e in E])
    return float(np.trapezoid(a, s))


def _bisect(f, lo: float, hi: float, tol: float = 1.0) -> float:
    flo = f(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) == flo:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def inception_voltage(r: float, d: float, air: Air = DRY, K: float = MEEK_K) -> float:
    """The voltage at which Meek's integral along the hyperboloid line reaches ``K``."""
    return _bisect(lambda V: inception_integral(V, r, d, air) < K, 100.0, 5.0e6)


def uniform_breakdown(d: float, air: Air = DRY) -> float:
    return air.e_break * d


def propagation_voltage(d: float, air: Air = DRY) -> float:
    return air.e_stability * d + V_STREAMER_0 * air.delta


def surface_flashover(creepage: float, air: Air = DRY) -> float:
    return E_SURFACE_DRY * air.delta * (1.0 - SURFACE_WET_K * air.rh ** 3) * creepage


@dataclass(frozen=True)
class SparkMargin:
    voltage: float
    inception: float
    propagation: float
    uniform: float
    surface: float | None

    @property
    def breakdown(self) -> float:
        """The lowest predicted breakdown voltage of the *gap* (inception alone is only a corona onset in a
        non-uniform gap: it starts the corona, it does not short it)."""
        vals = [self.propagation, self.uniform] + ([self.surface] if self.surface is not None else [])
        return min(vals)

    @property
    def margin(self) -> float:
        return self.breakdown / self.voltage

    @property
    def limited_by(self) -> str:
        pairs = {"propagation": self.propagation, "uniform": self.uniform}
        if self.surface is not None:
            pairs["surface"] = self.surface
        return min(pairs, key=pairs.get)


def spark_margin(V: float, r: float, d: float, air: Air = DRY, creepage: float | None = None) -> SparkMargin:
    """Every criterion at operating voltage ``V`` for a tooth of radius ``r``, gap ``d`` and the support's creepage."""
    return SparkMargin(V, inception_voltage(r, d, air), propagation_voltage(d, air), uniform_breakdown(d, air),
                       None if creepage is None else surface_flashover(creepage, air))
