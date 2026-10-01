"""Phase generators for legged gaits: a fixed schedule or per-leg oscillators with load feedback (Tegotae).

Two descriptions of the same stride:

* the **cycle fraction** ``c ∈ [0, 1)``: stance for ``c < duty``, swing for ``c ≥ duty`` (the convention of
  ``notebooks/designs/gait.py``: ``phase = (t/T + phase_leg) % 1``, stance while ``phase < duty``);
* the **oscillator phase** ``φ ∈ [0, 2π)``: swing for ``φ ∈ [0, π)``, stance for ``φ ∈ [π, 2π)`` (Owaki et al.;
  the stability protocol in docs/, §9.1). The map is piecewise linear: stance ``c ∈ [0, duty) → φ = π + π c/duty``,
  swing ``c ∈ [duty, 1) → φ = π (c − duty)/(1 − duty)``.

Each leg integrates ``dφ/dt = ω(φ) − σ N cos φ`` where ``N`` is its foot's normal ground force [N] and ``ω`` is
piecewise — ``π / ((1 − duty) T)`` in swing and ``π / (duty T)`` in stance — so that at ``σ = 0`` swing and
stance last exactly ``(1 − duty)·T`` and ``duty·T`` and the generator *is* the fixed schedule. A loaded leg in
late stance (``cos φ > 0``) slows down and lifts off later; a leg loaded early in stance is advanced.
The integration runs in cycle-fraction space (``dc/dt = f − σ N cos φ / φ'(c)``), which is the same ODE and is
exact at ``σ = 0``. Robot-agnostic: legs are names; nothing here knows the feet's geometry.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["PhaseGenerator", "cycle_to_oscillator", "oscillator_to_cycle", "in_stance"]

TWO_PI = 2.0 * math.pi


def cycle_to_oscillator(c, duty):
    """Cycle fraction(s) → oscillator phase(s) φ ∈ [0, 2π) (stance c < duty ↔ φ ∈ [π, 2π))."""
    c = np.mod(np.asarray(c, dtype=float), 1.0)
    phi = np.where(c < duty, math.pi + math.pi * c / duty, math.pi * (c - duty) / (1.0 - duty))
    return float(phi) if phi.ndim == 0 else phi


def oscillator_to_cycle(phi, duty):
    """Oscillator phase(s) → cycle fraction(s) in [0, 1)."""
    phi = np.mod(np.asarray(phi, dtype=float), TWO_PI)
    c = np.where(phi >= math.pi, duty * (phi - math.pi) / math.pi, duty + (1.0 - duty) * phi / math.pi)
    c = np.mod(c, 1.0)
    return float(c) if c.ndim == 0 else c


def in_stance(c, duty):
    """True where the cycle fraction is in stance (c < duty)."""
    return np.mod(np.asarray(c, dtype=float), 1.0) < duty


class PhaseGenerator:
    """Per-leg phases for a gait of ``duty`` (stance fraction) at ``frequency_hz`` strides per second.

    ``legs``: leg names (the order of every array in and out); ``base_phases``: leg → cycle fraction at t = 0
    (e.g. a lateral-sequence walk: 0, 0.25, 0.5, 0.75); ``sigma`` [rad/(N·s)]: the load-feedback gain (0 = the
    fixed schedule). ``step(dt, normal_forces)`` advances every leg and returns ``(phases, in_stance)``.
    ``frequency_hz`` may be changed between steps.
    """

    def __init__(self, legs, base_phases, duty, frequency_hz, sigma=0.0):
        self.legs = list(legs)
        if not 0.0 < duty < 1.0:
            raise ValueError("duty must be in (0, 1)")
        if frequency_hz < 0:
            raise ValueError("frequency_hz must be non-negative")
        missing = [leg for leg in self.legs if leg not in base_phases]
        if missing:
            raise ValueError(f"base_phases misses legs {missing}")
        self.base = np.array([float(base_phases[leg]) % 1.0 for leg in self.legs])
        self.duty = float(duty)
        self.frequency_hz = float(frequency_hz)
        self.sigma = float(sigma)
        self.t = 0.0
        self.c = self.base.copy()
        self._index = {leg: i for i, leg in enumerate(self.legs)}

    # ---- state
    def reset(self, offset=0.0, phases=None):
        """Restart at t = 0 with the base phases shifted by ``offset`` (cycle fraction, e.g. a seed's random
        start in the stride) or with explicit ``phases`` (leg → cycle fraction)."""
        self.t = 0.0
        if phases is not None:
            self.c = np.array([float(phases[leg]) % 1.0 for leg in self.legs])
        else:
            self.c = np.mod(self.base + float(offset), 1.0)
        return self.phases

    @property
    def phases(self) -> np.ndarray:
        """Cycle fractions in [0, 1), one per leg (a copy)."""
        return self.c.copy()

    @property
    def oscillator_phases(self) -> np.ndarray:
        return cycle_to_oscillator(self.c, self.duty)

    @property
    def stance(self) -> np.ndarray:
        return self.c < self.duty

    def progress(self):
        """(in_stance, s): per leg, whether it is in stance and how far through the current stance or swing it
        is (s ∈ [0, 1))."""
        st = self.c < self.duty
        s = np.where(st, self.c / self.duty, (self.c - self.duty) / (1.0 - self.duty))
        return st, s

    def phase_of(self, leg) -> float:
        return float(self.c[self._index[leg]])

    def period(self) -> float:
        return math.inf if self.frequency_hz == 0 else 1.0 / self.frequency_hz

    # ---- dynamics
    def _forces(self, normal_forces):
        if normal_forces is None:
            return np.zeros(len(self.legs))
        if isinstance(normal_forces, dict):
            return np.array([float(normal_forces.get(leg, 0.0)) for leg in self.legs])
        arr = np.asarray(normal_forces, dtype=float)
        if arr.shape != (len(self.legs),):
            raise ValueError(f"normal_forces must have {len(self.legs)} values")
        return arr

    def rate(self, normal_forces=None) -> np.ndarray:
        """dc/dt [1/s] per leg for the given normal forces [N]."""
        dcdt = np.full(len(self.legs), self.frequency_hz)
        if self.sigma != 0.0:
            n = self._forces(normal_forces)
            phi = cycle_to_oscillator(self.c, self.duty)
            dphi_dc = np.where(self.c < self.duty, math.pi / self.duty, math.pi / (1.0 - self.duty))
            dcdt = dcdt - self.sigma * n * np.cos(phi) / dphi_dc
        return dcdt

    def step(self, dt, normal_forces=None):
        """Advance by ``dt`` [s] with each foot's normal ground force (dict leg → N or an array in ``legs``
        order; ignored when sigma = 0). Returns ``(phases, in_stance)``."""
        if self.sigma == 0.0:
            self.c = np.mod(self.c + self.frequency_hz * dt, 1.0)
        else:
            self.c = np.mod(self.c + dt * self.rate(normal_forces), 1.0)
        self.t += dt
        return self.c.copy(), self.c < self.duty
