"""Atmospheric turbulence for long flights: Dryden gusts, the load factor they cause, the extremes, and a frozen
2-D field for drawing wind streaks.

``Turbulence`` is the low-altitude Dryden model of MIL-F-8785C (below 1000 ft): the vertical intensity from the wind
at 20 ft, the horizontal intensities and the scale lengths from the altitude. ``gust_series`` draws the three gust
components met along a straight flight at true airspeed ``V`` (Taylor's frozen turbulence: a spatial spectrum seen
at ``Omega = 2 pi f / V``), with Gaussian Fourier coefficients on the exact Dryden spectra — seeded, the rms and the
spectrum correct for any record length. ``load_factor`` turns vertical gusts into the load factor of a wing (the
sharp-edged-gust increment with the Pratt alleviation factor); ``rice_extreme`` is the expected largest excursion of
a Gaussian load over an exposure; ``frozen_field`` lays the gusts out on an (x, z) grid for a streak picture;
``through_mount`` passes a base acceleration through an isolator. Pure numpy; SI units outside.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
FT = 0.3048
KT = 0.514444
SEVERITY_W20_KT = {"light": 15.0, "moderate": 30.0, "severe": 45.0}     # MIL-F-8785C wind at 20 ft


@dataclass(frozen=True)
class Turbulence:
    """Low-altitude Dryden turbulence (MIL-F-8785C) at ``altitude_m`` for a wind of ``wind_20ft_m_s`` at 20 ft."""

    altitude_m: float
    wind_20ft_m_s: float

    def __post_init__(self):
        if self.altitude_m <= 0 or self.wind_20ft_m_s < 0:
            raise ValueError("altitude must be > 0 and the 20 ft wind >= 0")

    @classmethod
    def from_severity(cls, severity: str, altitude_m: float) -> "Turbulence":
        """``light`` / ``moderate`` / ``severe``: the standard 20 ft winds (15 / 30 / 45 kt)."""
        if severity not in SEVERITY_W20_KT:
            raise ValueError(f"severity must be one of {sorted(SEVERITY_W20_KT)}")
        return cls(altitude_m, SEVERITY_W20_KT[severity] * KT)

    @property
    def _h_ft(self) -> float:
        return min(max(self.altitude_m / FT, 10.0), 1000.0)            # the model's range

    @property
    def sigma_w(self) -> float:
        return 0.1 * self.wind_20ft_m_s

    @property
    def sigma_u(self) -> float:
        return self.sigma_w / (0.177 + 0.000823 * self._h_ft) ** 0.4

    sigma_v = sigma_u

    @property
    def length_w(self) -> float:
        return self._h_ft * FT

    @property
    def length_u(self) -> float:
        return self._h_ft / (0.177 + 0.000823 * self._h_ft) ** 1.2 * FT

    length_v = length_u

    def sigma(self, component: str) -> float:
        return {"u": self.sigma_u, "v": self.sigma_v, "w": self.sigma_w}[component]

    def length(self, component: str) -> float:
        return {"u": self.length_u, "v": self.length_v, "w": self.length_w}[component]

    def spectrum(self, component: str, f_hz, v_m_s: float) -> np.ndarray:
        """One-sided temporal PSD [(m/s)^2/Hz] of a gust component met at airspeed ``v_m_s`` (Dryden forms)."""
        s, L = self.sigma(component), self.length(component)
        om = 2 * math.pi * np.asarray(f_hz, dtype=float) / v_m_s             # spatial frequency, rad/m
        x2 = (L * om) ** 2
        if component == "u":
            phi = s * s * 2 * L / math.pi / (1 + x2)
        else:
            phi = s * s * L / math.pi * (1 + 3 * x2) / (1 + x2) ** 2
        return phi * 2 * math.pi / v_m_s                                     # S(f) df = Phi(Omega) dOmega

    def describe(self) -> dict:
        return {"altitude_m": self.altitude_m, "wind_20ft_m_s": self.wind_20ft_m_s, "sigma_u": self.sigma_u,
                "sigma_v": self.sigma_v, "sigma_w": self.sigma_w, "L_u": self.length_u, "L_v": self.length_v, "L_w": self.length_w}


def _series(psd_fn, n: int, dt: float, rng) -> np.ndarray:
    """A real Gaussian series with the one-sided PSD ``psd_fn(f)`` (random Fourier coefficients, inverse FFT)."""
    f = np.fft.rfftfreq(n, dt)
    df = 1.0 / (n * dt)
    c = np.sqrt(psd_fn(f) * df)
    c[0] = 0.0
    if n % 2 == 0:
        c[-1] = 0.0
    X = 0.5 * n * c * (rng.standard_normal(len(f)) - 1j * rng.standard_normal(len(f)))
    return np.fft.irfft(X, n=n)


@dataclass
class GustField:
    """The gusts met along the path: time, the three components [m/s] and the model."""

    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    turbulence: Turbulence
    v_m_s: float

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def rms(self) -> dict:
        return {k: float(np.sqrt(np.mean(getattr(self, k) ** 2))) for k in "uvw"}

    def spectrum(self, component: str, segment_s: float = 60.0):
        """Welch PSD of one component (Hann windows, half overlap): ``(f, S)``."""
        return welch(getattr(self, component), self.dt, segment_s)


def gust_series(turb: Turbulence, v_m_s: float, duration_s: float, dt: float = 0.02, seed: int = 0) -> GustField:
    """The three Dryden gust components along a straight flight at true airspeed ``v_m_s`` (seeded)."""
    if v_m_s <= 0 or duration_s <= 0 or dt <= 0:
        raise ValueError("airspeed, duration and dt must be > 0")
    n = int(round(duration_s / dt))
    n += n % 2
    rng = np.random.default_rng(seed)
    comps = {k: _series(lambda f, k=k: turb.spectrum(k, f, v_m_s), n, dt, rng) for k in "uvw"}
    return GustField(np.arange(n) * dt, comps["u"], comps["v"], comps["w"], turb, v_m_s)


def welch(x, dt: float, segment_s: float = 60.0):
    """One-sided Welch PSD (Hann, 50 % overlap), numpy only: ``(f, S)``."""
    x = np.asarray(x, dtype=float)
    m = min(len(x), max(16, int(segment_s / dt)))
    m -= m % 2
    win = np.hanning(m)
    scale = 1.0 / (np.sum(win ** 2) / dt)
    step = m // 2
    acc, k = 0.0, 0
    for i in range(0, len(x) - m + 1, step):
        seg = (x[i:i + m] - x[i:i + m].mean()) * win
        acc = acc + np.abs(np.fft.rfft(seg)) ** 2
        k += 1
    S = 2 * scale * acc / max(k, 1)
    S[0] /= 2
    if m % 2 == 0:
        S[-1] /= 2
    return np.fft.rfftfreq(m, dt), S


def mass_ratio(wing_loading_n_m2: float, mean_chord_m: float, cl_alpha: float, rho: float = 1.225) -> float:
    """Airplane mass ratio ``mu = 2 (W/S) / (rho c a g)``."""
    return 2 * wing_loading_n_m2 / (rho * mean_chord_m * cl_alpha * G)


def alleviation_factor(mu: float) -> float:
    """Pratt gust alleviation factor ``Kg = 0.88 mu / (5.3 + mu)`` (the aircraft rises with the gust)."""
    return 0.88 * mu / (5.3 + mu)


def load_factor(w_gust, v_m_s: float, cl_alpha: float, wing_loading_n_m2: float, rho: float = 1.225,
                mean_chord_m: float | None = None, n_mean: float = 1.0) -> np.ndarray:
    """Load factor from vertical gusts: ``n = n_mean + Kg rho V a w / (2 W/S)``; ``Kg`` = 1 (sharp edge, no
    alleviation) unless ``mean_chord_m`` is given, then the Pratt factor of the mass ratio."""
    kg = 1.0 if mean_chord_m is None else alleviation_factor(mass_ratio(wing_loading_n_m2, mean_chord_m, cl_alpha, rho))
    return n_mean + kg * rho * v_m_s * cl_alpha * np.asarray(w_gust, dtype=float) / (2 * wing_loading_n_m2)


def upcrossing_rate(x, dt: float) -> float:
    """Mean-level up-crossings per second of a series."""
    y = np.asarray(x, dtype=float) - np.mean(x)
    return float(np.count_nonzero((y[:-1] < 0) & (y[1:] >= 0)) / (len(y) * dt))


def rice_extreme(x, dt: float, exposures: float = 1.0) -> float:
    """Expected largest excursion above the mean of a stationary Gaussian series over ``exposures`` records of its
    length: ``sigma sqrt(2 ln(nu0 T))`` (Rice / Davenport), from the series' own rms and up-crossing rate."""
    s = float(np.std(x))
    nt = upcrossing_rate(x, dt) * len(x) * dt * exposures
    return s * math.sqrt(2 * math.log(max(nt, 1.0 + 1e-9))) if nt > 1 else s


def frozen_field(turb: Turbulence, length_m: float, heights_m, dx: float = 1.0, seed: int = 0):
    """A frozen (x, z) gust field for pictures: rows of spatial Dryden series (u along x, w up) at ``heights_m``,
    correlated between rows as ``exp(-|dz| / L)``. Returns ``(x, z, U[z, x], W[z, x])``."""
    z = np.asarray(heights_m, dtype=float)
    n = int(round(length_m / dx)); n += n % 2
    rng = np.random.default_rng(seed)
    out = {}
    for k in "uw":
        rows = np.array([_series(lambda f, k=k: turb.spectrum(k, f, 1.0), n, dx, rng) for _ in z])   # V = 1: t -> x
        C = np.exp(-np.abs(z[:, None] - z[None, :]) / turb.length(k))
        out[k] = np.linalg.cholesky(C + 1e-9 * np.eye(len(z))) @ rows
    return np.arange(n) * dx, z, out["u"], out["w"]


def through_mount(t, a, mount_hz: float, damping_ratio: float = 0.05) -> np.ndarray:
    """Acceleration of a payload on an isolator (1-DOF base excitation) for a base acceleration history ``a``,
    by the complex transmissibility in the frequency domain (steady, periodic record)."""
    a = np.asarray(a, dtype=float)
    dt = float(t[1] - t[0])
    f = np.fft.rfftfreq(len(a), dt)
    r = f / mount_hz
    H = (1 + 2j * damping_ratio * r) / (1 - r * r + 2j * damping_ratio * r)
    return np.fft.irfft(np.fft.rfft(a) * H, n=len(a))
