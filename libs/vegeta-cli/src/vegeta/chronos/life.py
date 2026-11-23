"""Scalar fatigue at a hotspot and long-term (fleet usage) simulation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spectrum import LoadSpectrum


@dataclass(frozen=True)
class SNCurve:
    """Basquin stress-life with optional endurance limit and Goodman correction (same form as
    ``talos.FatigueCurve``; kept here so a hotspot estimate needs no FEA package)."""

    name: str
    sigma_f: float
    b: float
    ultimate: float | None = None
    endurance_limit: float | None = None

    def cycles_to_failure(self, amplitude, mean=0.0):
        a = np.abs(np.asarray(amplitude, dtype=float))
        m = np.asarray(mean, dtype=float)
        if self.ultimate is not None:
            a = a / np.clip(1.0 - np.clip(m, 0.0, None) / self.ultimate, 1e-6, 1.0)
        with np.errstate(divide="ignore", over="ignore"):
            n = 0.5 * (a / self.sigma_f) ** (1.0 / self.b)
        n = np.where(a <= 0, np.inf, n)
        if self.endurance_limit is not None:
            n = np.where(a < self.endurance_limit, np.inf, n)
        return n


def hotspot_damage(spectrum: LoadSpectrum, stress_per_unit: dict, curve: SNCurve) -> dict:
    """Miner damage of one spectrum pass at a point where each pattern produces ``stress_per_unit``
    (stress per unit load, signed). Returns per-block contributions and the total."""
    out = {"total": 0.0, "blocks": []}
    for b in spectrum.blocks:
        s = stress_per_unit.get(b.pattern)
        if s is None:
            raise ValueError(f"no stress_per_unit for pattern {b.pattern!r}")
        n = float(curve.cycles_to_failure(abs(b.amplitude * s), b.mean * s))
        d = b.cycles / n if np.isfinite(n) else 0.0
        out["blocks"].append({"source": b.source, "amplitude_stress": abs(b.amplitude * s), "cycles": b.cycles, "damage": d})
        out["total"] += d
    return out


@dataclass
class LifeSimulation:
    flights: list                      # mission name per flight, in order
    damage: np.ndarray                 # cumulative damage after each flight
    hours: np.ndarray                  # cumulative flight hours
    flights_to_failure: float          # inf if damage stays below 1
    hours_to_failure: float
    usage: dict

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(self.hours, self.damage, color="#1f3b5a")
        ax.axhline(1.0, color="#c62828", ls="--", label="Miner sum = 1 (failure)")
        if np.isfinite(self.hours_to_failure):
            ax.axvline(self.hours_to_failure, color="#c62828", ls=":", label=f"{self.hours_to_failure:.0f} h")
        ax.set(xlabel="flight hours", ylabel="cumulative damage", title=f"{len(self.flights)} flights, usage {self.usage}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return ax.figure


def simulate_life(damage_per_mission: dict, hours_per_mission: dict, usage: dict, n_flights: int = 1000,
                  seed: int | None = 0, stop_at_failure: bool = False) -> LifeSimulation:
    """Draw ``n_flights`` missions from ``usage`` (name -> fraction) and accumulate Miner damage.
    ``seed=None`` gives a deterministic round-robin in proportion to the usage instead of random draws."""
    names = list(usage)
    frac = np.array([usage[n] for n in names], dtype=float)
    if frac.sum() <= 0 or np.any(frac < 0):
        raise ValueError("usage fractions must be non-negative and not all zero")
    frac = frac / frac.sum()
    missing = [n for n in names if n not in damage_per_mission or n not in hours_per_mission]
    if missing:
        raise ValueError(f"missions without damage/hours: {missing}")
    if seed is None:
        counts = np.zeros(len(names))
        order = []
        for _ in range(n_flights):
            i = int(np.argmax(frac * (len(order) + 1) - counts))
            counts[i] += 1
            order.append(names[i])
    else:
        rng = np.random.default_rng(seed)
        order = list(rng.choice(names, size=n_flights, p=frac))
    d = np.cumsum([damage_per_mission[n] for n in order])
    h = np.cumsum([hours_per_mission[n] for n in order])
    fail = np.argmax(d >= 1.0) if np.any(d >= 1.0) else None
    if fail is not None:
        # linear interpolation inside the failing flight
        d0 = d[fail - 1] if fail > 0 else 0.0
        h0 = h[fail - 1] if fail > 0 else 0.0
        frac_in = (1.0 - d0) / (d[fail] - d0)
        f2f, h2f = fail + frac_in, h0 + frac_in * (h[fail] - h0)
        if stop_at_failure:
            order, d, h = order[:fail + 1], d[:fail + 1], h[:fail + 1]
    else:
        f2f = h2f = float("inf")
    return LifeSimulation(order, d, h, float(f2f), float(h2f), dict(usage))
