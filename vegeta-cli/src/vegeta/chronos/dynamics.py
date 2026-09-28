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


# -- shock ----------------------------------------------------------------------------------------------------------
def half_sine(peak_m_s2: float, duration_s: float, dt: float | None = None, tail_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A half-sine base-acceleration pulse ``(t, a)``: the classic drop / crush pulse (``tail_s`` of zeros after it)."""
    dt = dt or duration_s / 200
    t = np.arange(0.0, duration_s + tail_s + dt, dt)
    a = np.where(t <= duration_s, peak_m_s2 * np.sin(np.pi * np.clip(t, 0, duration_s) / duration_s), 0.0)
    return t, a


def srs(t, a, freqs_hz, damping_ratio: float = 0.05) -> np.ndarray:
    """Shock response spectrum (maximax absolute acceleration) of the base pulse ``a(t)`` for single-degree-of-freedom
    oscillators at ``freqs_hz``: what a part mounted at that natural frequency feels, as a multiple of the input's
    units. Central-difference integration of the relative motion, primary and residual response included."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    f = np.atleast_1d(np.asarray(freqs_hz, float))
    w = 2 * np.pi * f
    dt = min(float(np.min(np.diff(t))), float(1.0 / (40 * f.max())))
    t_end = t[-1] + 3.0 / f.min()
    tt = np.arange(0.0, t_end, dt)
    aa = np.interp(tt, t, a, left=0.0, right=0.0)
    y = np.zeros_like(w); v = np.zeros_like(w); peak = np.zeros_like(w)
    z = damping_ratio
    for ak in aa:
        acc = -ak - 2 * z * w * v - w * w * y
        v = v + acc * dt
        y = y + v * dt
        absolute = -(2 * z * w * v + w * w * y)
        peak = np.maximum(peak, np.abs(absolute))
    return peak


def shock_at_mount(t, a, mount_hz: float, damping_ratio: float = 0.05) -> float:
    """The peak acceleration a payload on isolators of natural frequency ``mount_hz`` sees from the base pulse."""
    return float(srs(t, a, [mount_hz], damping_ratio)[0])
