"""The high-voltage supply, simply: a source ``v_set`` behind a series resistance ``r_series`` with a current limit
``i_max`` (fold-back: the voltage drops to hold the current), an output capacitance ``c_out`` (what a spark dumps at
once), an efficiency and a mass curve. Enough to find the operating point on a corona load, the power from the
battery, and the energy a spark delivers (``docs/plasma.md``, *Arcs*, item 4).

Mass and efficiency: hobby flyback / Cockcroft-Walton modules of 20-40 kV, 20-200 W weigh 150-500 g at 60-80 %
efficiency; the curve ``mass = m0 + k P`` with m0 = 60 g and k = 2.5 g/W is an estimate from catalogue modules,
to be replaced by the chosen part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HVSupply:
    v_set: float                       # V, the open-circuit voltage
    i_max: float = 2.0e-3              # A
    r_series: float = 1.0e6            # ohm (a ballast; protects the frame: the 4B had none)
    c_out: float = 200.0e-12           # F (the multiplier's last stage plus the electrodes)
    efficiency: float = 0.70
    m0: float = 0.060                  # kg
    k_mass: float = 0.0025             # kg/W of output

    def operating_point(self, load: Callable[[float], float]) -> tuple[float, float]:
        """(V, I) where the load's ``I(V)`` meets ``V = v_set - R I``, with the current limited to ``i_max``."""
        lo, hi = 0.0, self.v_set
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.v_set - self.r_series * load(mid) > mid:
                lo = mid
            else:
                hi = mid
        V = 0.5 * (lo + hi)
        I = load(V)
        if I > self.i_max:                                     # fold back: find V where the load draws i_max
            lo, hi = 0.0, V
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if load(mid) < self.i_max:
                    lo = mid
                else:
                    hi = mid
            V, I = 0.5 * (lo + hi), self.i_max
        return V, I

    def input_power(self, V: float, I: float) -> float:
        """From the battery: the corona's power, the ballast's loss, the converter's efficiency."""
        return (V * I + self.r_series * I * I) / self.efficiency

    def mass(self, p_out: float) -> float:
        return self.m0 + self.k_mass * p_out

    def spark_energy(self, V: float, gap: float, t_trip: float = 0.01, p_short: float | None = None) -> float:
        """Joules into a spark: the output capacitance's charge, then for ``t_trip`` until the supply cuts out either
        ``i_max`` at the arc's voltage (a current-limited supply; an arc in air holds ~ 2 kV/m plus ~ 30 V at the
        electrodes) or ``p_short``, the power a supply without a limit keeps delivering into a short — a flyback or a
        ZVS driver transfers its energy per cycle whatever the load, so ``p_short`` is about its rated power."""
        v_arc = 30.0 + 2000.0 * gap
        p_arc = self.i_max * v_arc if p_short is None else p_short
        return 0.5 * self.c_out * V * V + p_arc * t_trip


def resin_melt_energy(mass_kg: float, dT: float = 150.0, cp: float = 1500.0) -> float:
    """Heat to take ``mass_kg`` of printed resin to softening (dT above room, cp of methacrylate resin)."""
    return mass_kg * cp * dT
