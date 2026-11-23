"""Single-degree-of-freedom dynamic amplification from a list of natural frequencies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Structure:
    """Undamped natural frequencies (Hz, e.g. from ``talos.StructuralModel.solve_modes``) and one
    damping ratio for all modes. Amplification of a harmonic force uses the nearest mode as a
    single-degree-of-freedom oscillator: a bounding, not a modal-superposition, estimate."""

    modes_hz: tuple
    damping_ratio: float = 0.03
    source: str = ""

    def __post_init__(self):
        if not self.modes_hz or min(self.modes_hz) <= 0:
            raise ValueError("modes_hz must be positive frequencies")
        if not 0 < self.damping_ratio < 1:
            raise ValueError("damping_ratio must be in (0, 1)")

    def nearest_mode(self, frequency_hz: float) -> float:
        m = np.asarray(self.modes_hz, dtype=float)
        return float(m[np.argmin(np.abs(np.log(m / frequency_hz)))])

    def amplification(self, frequency_hz) -> np.ndarray:
        """Dynamic amplification factor of a harmonic force at ``frequency_hz`` (array-safe)."""
        f = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        fn = np.array([self.nearest_mode(x) for x in f])
        r = f / fn
        z = self.damping_ratio
        return 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * z * r) ** 2)

    def margin(self, frequency_hz: float) -> float:
        """Relative distance to the nearest mode, |f - fn| / fn (0.2 or more is the usual comfort)."""
        fn = self.nearest_mode(frequency_hz)
        return abs(frequency_hz - fn) / fn

    def campbell(self, excitation_lines: dict, rpm_range, ax=None, operating_rpm: dict | None = None):
        """Campbell diagram: excitation orders vs rpm against the natural frequencies (matplotlib).
        ``excitation_lines`` maps a label to its order per revolution (e.g. {"1P": 1, "3P": 3})."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        rpm = np.asarray(rpm_range, dtype=float)
        for label, order in excitation_lines.items():
            ax.plot(rpm, order * rpm / 60, label=label)
        for i, fn in enumerate(self.modes_hz):
            ax.axhline(fn, color="#c62828", ls="--", lw=0.8)
            ax.text(rpm[0], fn, f" mode {i + 1}: {fn:.0f} Hz", va="bottom", fontsize=8, color="#c62828")
        for name, r in (operating_rpm or {}).items():
            ax.axvline(r, color="#888", ls=":")
            ax.text(r, 0, f" {name}", rotation=90, va="bottom", fontsize=8)
        ax.set(xlabel="rpm", ylabel="frequency [Hz]", title="Campbell diagram")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        return ax.figure
