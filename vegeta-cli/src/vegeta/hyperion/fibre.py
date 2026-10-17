"""Optical fibres and the fibre laser: a step-index fibre's V number, cutoff, LP₀₁ mode radius (Marcuse), overlap
factors, its beam, the LP₀₁ eigenvalue and Marcuse's bend loss (Bessel functions by quadrature, so numpy alone);
Yb-doped silica's cross sections (the quasi-three-level medium); the steady-state **power evolution** of pump and
signal along a double-clad active fibre (Kelson & Hardy 1998, without ASE) — an oscillator between two fibre Bragg
gratings solved by shooting on the circulating power, a single-pass amplifier co- or counter-pumped — with the
inversion and the heat along the fibre; the stimulated-Raman threshold and the core's temperature rise.

Lengths in metres, powers in watts, wavelengths in metres. Nothing is picked: the fibre, its doping (or the
cladding absorption it is specified by), the gratings, the pump and the seed are inputs; the model is the standard
two-level rate equations with overlap factors — the datasheet's cross sections and the measured absorption decide."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .beam import GaussianBeam
from .materials import FUSED_SILICA

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
DB = 10.0 * math.log10(math.e)                                   # nepers → dB (4.343)
LP01_CUTOFF = 2.404826                                           # the first zero of J₀: single-mode for V below it


# ------------------------------------------------------------------------------------------ Bessel functions, by quadrature
def bessel_j(n: int, x) -> np.ndarray:
    """Jₙ(x) = (1/π) ∫₀^π cos(nτ − x sin τ) dτ (Bessel's integral), vectorised over x."""
    x = np.asarray(x, dtype=float)[..., None]
    tau = np.linspace(0.0, math.pi, 2001)
    f = np.cos(n * tau - x * np.sin(tau))
    return _trapz(f, tau) / math.pi


def bessel_k(n: int, x) -> np.ndarray:
    """Kₙ(x) = ∫₀^∞ exp(−x cosh t) cosh(nt) dt for x > 0, vectorised over x."""
    x = np.asarray(x, dtype=float)[..., None]
    t = np.linspace(0.0, 14.0, 6001)
    f = np.exp(-x * np.cosh(t)) * np.cosh(n * t)
    return _trapz(f, t)


def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x, axis=-1)


# -------------------------------------------------------------------------------------------------------- the fibre
@dataclass(frozen=True)
class Fibre:
    """A step-index (double-clad) fibre: ``core_radius`` and core ``NA``, ``clad_radius`` (the pump guide of a
    double-clad fibre; its NA ``clad_NA``), ``coating_radius``; the background losses ``alpha_signal`` and
    ``alpha_pump`` [1/m] (10 dB/km = 2.3e-3 /m); ``core_index`` at the signal wavelength (silica's unless given)."""
    core_radius: float
    NA: float
    clad_radius: float
    coating_radius: float | None = None
    clad_NA: float = 0.46
    alpha_signal: float = 0.0
    alpha_pump: float = 0.0
    core_index: float | None = None
    name: str = ""

    def n1(self, wavelength: float) -> float:
        return self.core_index if self.core_index else FUSED_SILICA.n(wavelength)

    def V(self, wavelength: float) -> float:
        return 2 * math.pi * self.core_radius * self.NA / wavelength

    @property
    def cutoff_wavelength(self) -> float:
        """Above it the core carries LP₀₁ alone."""
        return 2 * math.pi * self.core_radius * self.NA / LP01_CUTOFF

    def single_mode(self, wavelength: float) -> bool:
        return self.V(wavelength) < LP01_CUTOFF

    def mode_radius(self, wavelength: float) -> float:
        """Marcuse's fit for the LP₀₁ Gaussian radius: w/a = 0.65 + 1.619 V^−1.5 + 2.879 V^−6 (within 1 % for
        0.8 < V < 2.5; a few-mode fibre's LP₀₁ is read from the same fit, a stated extrapolation)."""
        v = self.V(wavelength)
        return self.core_radius * (0.65 + 1.619 * v ** -1.5 + 2.879 * v ** -6)

    def mode_area(self, wavelength: float) -> float:
        return math.pi * self.mode_radius(wavelength) ** 2

    @property
    def core_area(self) -> float:
        return math.pi * self.core_radius ** 2

    @property
    def clad_area(self) -> float:
        return math.pi * self.clad_radius ** 2

    def overlap_core(self, wavelength: float) -> float:
        """Γ_s: the share of the LP₀₁ Gaussian inside the core, 1 − exp(−2a²/w²)."""
        return 1.0 - math.exp(-2.0 * (self.core_radius / self.mode_radius(wavelength)) ** 2)

    @property
    def overlap_clad(self) -> float:
        """Γ_p: the share of a cladding-filling pump that crosses the core, the area ratio (a/b)²."""
        return (self.core_radius / self.clad_radius) ** 2

    def beam(self, wavelength: float, m2: float = 1.0, name: str = "") -> GaussianBeam:
        """The beam leaving a flat cleave: the LP₀₁ Gaussian at its waist (M² 1.0–1.1 for a single-mode fibre)."""
        return GaussianBeam(wavelength, self.mode_radius(wavelength), 0.0, m2, 1.0, name or self.name)

    def lp01(self, wavelength: float) -> dict:
        """The LP₀₁ eigenvalue: u J₁(u)/J₀(u) = w K₁(w)/K₀(w) with u² + w² = V² (bisection) → u, w, the transverse
        wavenumbers κ = u/a and γ = w/a, the propagation constant β."""
        v = self.V(wavelength)
        hi = min(v, LP01_CUTOFF) - 1e-9
        lo = 1e-6

        def f(u):
            w = math.sqrt(max(v * v - u * u, 1e-18))
            j0, j1 = float(bessel_j(0, u)), float(bessel_j(1, u))
            k0, k1 = float(bessel_k(0, w)), float(bessel_k(1, w))
            return u * j1 / j0 - w * k1 / k0

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        u = 0.5 * (lo + hi)
        w = math.sqrt(v * v - u * u)
        k = 2 * math.pi / wavelength
        n1 = self.n1(wavelength)
        beta = math.sqrt(max((k * n1) ** 2 - (u / self.core_radius) ** 2, 0.0))
        return {"V": v, "u": u, "w": w, "kappa": u / self.core_radius, "gamma": w / self.core_radius, "beta": beta, "n1": n1}

    def bend_loss(self, radius, wavelength: float, stress_factor: float = 1.28) -> np.ndarray:
        """Marcuse's (1976) pure-bend loss of LP₀₁ [1/m, power] at a bend ``radius`` (a number or an array):
        α = √π κ² exp(−2γ³R_eff/(3β²)) / (2 γ^{3/2} V² √R_eff K₁(γa)²), with R_eff = ``stress_factor`` × R for the
        stress-optic correction of silica (Schermer & Cole: 1.28). An order of magnitude, as the catalogues' curves."""
        m = self.lp01(wavelength)
        kap, gam, beta, v = m["kappa"], m["gamma"], m["beta"], m["V"]
        r = np.asarray(radius, dtype=float) * stress_factor
        k1 = float(bessel_k(1, gam * self.core_radius))
        return math.sqrt(math.pi) * kap ** 2 * np.exp(-2 * gam ** 3 * r / (3 * beta ** 2)) / (2 * gam ** 1.5 * v ** 2 * np.sqrt(r) * k1 ** 2)

    def bend_loss_db_per_m(self, radius, wavelength: float, stress_factor: float = 1.28) -> np.ndarray:
        return DB * self.bend_loss(radius, wavelength, stress_factor)

    def to_dict(self) -> dict:
        return {"core_radius": self.core_radius, "NA": self.NA, "clad_radius": self.clad_radius, "coating_radius": self.coating_radius,
                "clad_NA": self.clad_NA, "alpha_signal": self.alpha_signal, "alpha_pump": self.alpha_pump, "core_index": self.core_index, "name": self.name}


# ------------------------------------------------------------------------------------------------------- the dopant
@dataclass(frozen=True)
class Dopant:
    """A rare-earth ion in silica: ``table`` of (wavelength [m], σ_a [m²], σ_e [m²]) interpolated log-linearly,
    ``tau`` the upper-level lifetime [s], ``N`` the ion density [1/m³], ``kappa`` the glass's conductivity."""
    name: str
    table: tuple
    tau: float
    N: float
    kappa: float = 1.38
    note: str = ""

    def cross_sections(self, wavelength: float) -> tuple[float, float]:
        lams = np.array([r[0] for r in self.table])
        sa = np.log(np.array([r[1] for r in self.table]))
        se = np.log(np.array([r[2] for r in self.table]))
        if not lams.min() <= wavelength <= lams.max():
            raise ValueError(f"{self.name}: no cross sections at {wavelength * 1e9:.0f} nm (table {lams.min() * 1e9:.0f}–{lams.max() * 1e9:.0f} nm)")
        return float(np.exp(np.interp(wavelength, lams, sa))), float(np.exp(np.interp(wavelength, lams, se)))

    def with_(self, **changes) -> "Dopant":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"name": self.name, "table": [list(r) for r in self.table], "tau": self.tau, "N": self.N, "kappa": self.kappa, "note": self.note}


YB_SILICA = Dopant("Yb:silica", (
    (900e-9, 1.5e-25, 1.0e-26), (915e-9, 7.5e-25, 2.5e-26), (940e-9, 4.0e-25, 6.0e-26), (960e-9, 5.0e-25, 2.0e-25),
    (976e-9, 2.5e-24, 2.5e-24), (990e-9, 3.0e-25, 6.0e-25), (1010e-9, 1.2e-25, 7.0e-25), (1030e-9, 5.0e-26, 6.0e-25),
    (1050e-9, 2.0e-26, 4.0e-25), (1064e-9, 1.2e-26, 3.3e-25), (1070e-9, 8.0e-27, 2.8e-25), (1080e-9, 4.0e-27, 2.2e-25),
    (1090e-9, 2.0e-27, 1.7e-25), (1100e-9, 1.0e-27, 1.3e-25)),
    tau=0.85e-3, N=2.4e25, kappa=1.38,
    note="Paschotta et al. 1997's germanosilicate shape, rounded; τ 0.85 ms; N for ~1.7 dB/m cladding absorption of a 10/125 at 976 nm — "
         "the fibre's datasheet (absorption at 976 or 915 nm) sets N through ions_for_absorption")


def ions_for_absorption(fibre: Fibre, db_per_m: float, wavelength: float, dopant: Dopant = YB_SILICA) -> float:
    """The ion density that gives a cladding-pumped small-signal absorption ``db_per_m`` at ``wavelength``:
    N = α / (Γ_p σ_a) with α in nepers per metre."""
    sa, _ = dopant.cross_sections(wavelength)
    return db_per_m / DB / (fibre.overlap_clad * sa)


@dataclass(frozen=True)
class ActiveFibre:
    """The gain medium and nothing else: ``length`` metres of ``fibre`` doped with ``dopant`` — the guided optical path
    that amplifies. It is not a laser: an oscillator is this medium *composed with* two reflectors and their boundary
    conditions (``FibreLaser`` / ``FibreOscillator``); an amplifier is this medium with a seed and no feedback
    (``FibreAmplifier``)."""
    fibre: Fibre
    dopant: Dopant
    length: float
    name: str = "active fibre"

    def absorption_db_per_m(self, wavelength: float, core_pumped: bool = False) -> float:
        """Small-signal pump absorption, cladding-pumped (Γ_p = (a/b)²) or core-pumped (Γ_s)."""
        sa, _ = self.dopant.cross_sections(wavelength)
        g = self.fibre.overlap_core(wavelength) if core_pumped else self.fibre.overlap_clad
        return DB * g * sa * self.dopant.N

    def absorbed_fraction(self, wavelength: float) -> float:
        """Small-signal, cladding-pumped, no bleaching: 1 − 10^(−α L / 10)."""
        return 1.0 - 10 ** (-self.absorption_db_per_m(wavelength) * self.length / 10)

    def to_dict(self) -> dict:
        return {"fibre": self.fibre.to_dict(), "dopant": self.dopant.to_dict(), "length": self.length, "name": self.name}


# ------------------------------------------------------------------------------------------- the power evolution
@dataclass
class FibreSolution:
    """Pump and signal along z: ``Pp`` the pump, ``Ps_fwd``/``Ps_bwd`` the signal in +z/−z, ``inversion`` N₂/N,
    ``heat`` W/m; the totals an engineer reads."""
    z: np.ndarray
    Pp: np.ndarray
    Ps_fwd: np.ndarray
    Ps_bwd: np.ndarray
    inversion: np.ndarray
    heat: np.ndarray
    output_W: float
    pump_W: float
    absorbed_W: float
    unabsorbed_W: float
    gain_db: float = math.nan
    note: str = ""

    @property
    def efficiency(self) -> float:
        return self.output_W / self.pump_W if self.pump_W > 0 else 0.0

    @property
    def heat_W(self) -> float:
        return float(_trapz(self.heat, self.z))

    @property
    def peak_heat_per_m(self) -> float:
        return float(self.heat.max())

    def to_dict(self) -> dict:
        return {"output_W": self.output_W, "pump_W": self.pump_W, "absorbed_W": self.absorbed_W, "unabsorbed_W": self.unabsorbed_W, "gain_db": self.gain_db,
                "efficiency": self.efficiency, "heat_W": self.heat_W, "peak_heat_per_m": self.peak_heat_per_m, "inversion_max": float(self.inversion.max()),
                "inversion_min": float(self.inversion.min()), "note": self.note}


class _Rates:
    """The local rate equations of a two-level (quasi-three-level) ion under cladding pumping."""

    def __init__(self, active: ActiveFibre, pump_wavelength: float, signal_wavelength: float):
        fb, dp = active.fibre, active.dopant
        self.N, self.tau = dp.N, dp.tau
        self.sap, self.sep = dp.cross_sections(pump_wavelength)
        self.sas, self.ses = dp.cross_sections(signal_wavelength)
        self.Gp, self.Gs = fb.overlap_clad, fb.overlap_core(signal_wavelength)
        self.A = fb.core_area
        self.hnu_p, self.hnu_s = H_PLANCK * C_LIGHT / pump_wavelength, H_PLANCK * C_LIGHT / signal_wavelength
        self.ap, self.as_ = fb.alpha_pump, fb.alpha_signal
        self.defect = 1.0 - pump_wavelength / signal_wavelength

    def inversion(self, Pp: float, Ps: float) -> float:
        rp = max(Pp, 0.0) * self.Gp / (self.hnu_p * self.A)
        rs = max(Ps, 0.0) * self.Gs / (self.hnu_s * self.A)
        return (rp * self.sap + rs * self.sas) / (rp * (self.sap + self.sep) + rs * (self.sas + self.ses) + 1.0 / self.tau)

    def gains(self, n2: float) -> tuple[float, float]:
        """(g_p, g_s) per metre: the pump's (negative: absorbed) and the signal's net of reabsorption, before the background loss."""
        gp = self.Gp * self.N * ((self.sap + self.sep) * n2 - self.sap)
        gs = self.Gs * self.N * ((self.sas + self.ses) * n2 - self.sas)
        return gp, gs

    def heat(self, Pp: float, Ps: float, gp: float) -> float:
        """W/m: the quantum defect of the pump the ions absorb, plus the background losses."""
        return max(-gp, 0.0) * Pp * self.defect + self.ap * Pp + self.as_ * Ps


def _rk4(f, y0, z):
    y = np.array(y0, dtype=float)
    out = np.empty((len(z), len(y)))
    out[0] = y
    for i in range(1, len(z)):
        h = z[i] - z[i - 1]
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.where(y < 0, 0.0, y)
        out[i] = y
    return out


@dataclass(frozen=True)
class FibreLaser:
    """An oscillator = the gain medium (``active``, the guided path) **composed with** the feedback: a high reflector
    ``R_hr`` at z = 0 (the pump's end: co-pumped) and an output grating ``R_oc`` at z = L, which are the boundary
    conditions P_s⁺(0) = R_hr P_s⁻(0) and P_s⁻(L) = R_oc P_s⁺(L). The fibre alone is not a resonator; the two gratings
    make it one. ``pump_wavelength``, ``signal_wavelength`` (the gratings'). ``solve`` shoots on the circulating power
    until both boundary conditions hold (Rigrod's)."""
    active: ActiveFibre
    pump_wavelength: float
    signal_wavelength: float
    R_hr: float = 0.99
    R_oc: float = 0.10
    name: str = ""

    def _rates(self) -> _Rates:
        return _Rates(self.active, self.pump_wavelength, self.signal_wavelength)

    def _integrate(self, pump_W: float, x: float, n: int):
        r = self._rates()
        z = np.linspace(0.0, self.active.length, n + 1)

        def f(y):
            Pp, Pf, Pb = y
            n2 = r.inversion(Pp, Pf + Pb)
            gp, gs = r.gains(n2)
            return np.array([(gp - r.ap) * Pp, (gs - r.as_) * Pf, -(gs - r.as_) * Pb])

        y = _rk4(f, [pump_W, x, x / self.R_hr if self.R_hr > 0 else 0.0], z)
        return z, y, r

    def _residual(self, pump_W: float, x: float, n: int) -> float:
        _, y, _ = self._integrate(pump_W, x, n)
        return y[-1, 2] - self.R_oc * y[-1, 1]

    def small_signal_round_trip(self, pump_W: float, n: int = 400) -> float:
        """R_hr R_oc exp(2 ∫(g_s − α_s) dz) with no signal: above 1 the laser starts."""
        z, y, r = self._integrate(pump_W, 0.0, n)
        g = np.array([r.gains(r.inversion(p, 0.0))[1] - r.as_ for p in y[:, 0]])
        return self.R_hr * self.R_oc * math.exp(2 * float(_trapz(g, z)))

    def threshold(self, pump_max: float = 1e3, n: int = 300) -> float:
        """The pump at which the small-signal round-trip gain reaches 1 (bisection)."""
        lo, hi = 0.0, pump_max
        if self.small_signal_round_trip(hi, n) < 1.0:
            return math.inf
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.small_signal_round_trip(mid, n) < 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def solve(self, pump_W: float, n: int = 400) -> FibreSolution:
        r = self._rates()
        if pump_W <= 0 or self.small_signal_round_trip(pump_W, n) <= 1.0:
            z, y, _ = self._integrate(pump_W, 0.0, n)
            inv = np.array([r.inversion(p, 0.0) for p in y[:, 0]])
            heat = np.array([r.heat(p, 0.0, r.gains(i)[0]) for p, i in zip(y[:, 0], inv)])
            return FibreSolution(z, y[:, 0], y[:, 1] * 0, y[:, 2] * 0, inv, heat, 0.0, pump_W, pump_W - y[-1, 0], y[-1, 0], note="below threshold")
        # bracket the forward power at the HR: tiny → the residual is negative (unsaturated gain), large → positive
        lo, hi = 1e-9, max(pump_W * 10.0, 1.0)
        f_lo, f_hi = self._residual(pump_W, lo, n), self._residual(pump_W, hi, n)
        if f_lo > 0:
            lo = 1e-12
        if f_hi < 0:
            hi *= 100
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if self._residual(pump_W, mid, n) < 0:
                lo = mid
            else:
                hi = mid
        z, y, _ = self._integrate(pump_W, math.sqrt(lo * hi), n)
        inv = np.array([r.inversion(p, f + b) for p, f, b in y])
        heat = np.array([r.heat(p, f + b, r.gains(i)[0]) for (p, f, b), i in zip(y, inv)])
        out = (1.0 - self.R_oc) * y[-1, 1]
        return FibreSolution(z, y[:, 0], y[:, 1], y[:, 2], inv, heat, float(out), pump_W, float(pump_W - y[-1, 0]), float(y[-1, 0]),
                             gain_db=DB * math.log(y[-1, 1] / y[0, 1]) if y[0, 1] > 0 else math.nan)

    def power_curve(self, pumps, n: int = 300) -> dict:
        pumps = np.asarray(list(pumps), dtype=float)
        sols = [self.solve(float(p), n) for p in pumps]
        out = np.array([s.output_W for s in sols])
        lit = out > 0
        slope = float(np.polyfit(pumps[lit], out[lit], 1)[0]) if lit.sum() >= 2 else math.nan
        return {"pump_W": pumps, "output_W": out, "unabsorbed_W": np.array([s.unabsorbed_W for s in sols]), "heat_W": np.array([s.heat_W for s in sols]),
                "inversion_max": np.array([float(s.inversion.max()) for s in sols]), "slope": slope, "threshold_W": self.threshold()}

    def length_sweep(self, lengths, pump_W: float, n: int = 300) -> dict:
        rows = []
        for L in lengths:
            s = replace(self, active=replace(self.active, length=float(L))).solve(pump_W, n)
            rows.append({"length": float(L), "output_W": s.output_W, "unabsorbed_W": s.unabsorbed_W, "absorbed_W": s.absorbed_W, "efficiency": s.efficiency,
                         "heat_peak_per_m": s.peak_heat_per_m, "inversion_max": float(s.inversion.max())})
        best = max(rows, key=lambda r: r["output_W"])
        return {"rows": rows, "best_length": best["length"], "best_output_W": best["output_W"]}

    def to_dict(self) -> dict:
        return {"active": self.active.to_dict(), "pump_wavelength": self.pump_wavelength, "signal_wavelength": self.signal_wavelength, "R_hr": self.R_hr,
                "R_oc": self.R_oc, "name": self.name}


FibreOscillator = FibreLaser                                     # the name that says what it is


@dataclass(frozen=True)
class FibreAmplifier:
    """A single pass of seed through the active fibre, pumped ``"co"`` (with the seed, from z = 0) or ``"counter"``
    (from the output end); no ASE (a seed of a watt or more keeps it small; a weak seed's amplifier is not this one)."""
    active: ActiveFibre
    pump_wavelength: float
    signal_wavelength: float
    pump_direction: str = "counter"
    name: str = ""

    def solve(self, seed_W: float, pump_W: float, n: int = 400) -> FibreSolution:
        r = _Rates(self.active, self.pump_wavelength, self.signal_wavelength)
        z = np.linspace(0.0, self.active.length, n + 1)
        sign = 1.0 if self.pump_direction == "co" else -1.0        # the pump's power grows with z when it travels in −z

        def f(y):
            Pp, Ps = y
            n2 = r.inversion(Pp, Ps)
            gp, gs = r.gains(n2)
            return np.array([sign * (gp - r.ap) * Pp, (gs - r.as_) * Ps])

        if self.pump_direction == "co":
            y = _rk4(f, [pump_W, seed_W], z)
        else:                                                      # shoot on the pump left at z = 0 so that Pp(L) = pump_W
            lo, hi = 0.0, pump_W
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                y = _rk4(f, [mid, seed_W], z)
                if y[-1, 0] < pump_W:
                    lo = mid
                else:
                    hi = mid
            y = _rk4(f, [0.5 * (lo + hi), seed_W], z)
        inv = np.array([r.inversion(p, s) for p, s in y])
        heat = np.array([r.heat(p, s, r.gains(i)[0]) for (p, s), i in zip(y, inv)])
        unabs = y[-1, 0] if self.pump_direction == "co" else y[0, 0]
        return FibreSolution(z, y[:, 0], y[:, 1], y[:, 1] * 0, inv, heat, float(y[-1, 1]), pump_W, float(pump_W - unabs), float(unabs),
                             gain_db=DB * math.log(y[-1, 1] / seed_W) if seed_W > 0 else math.nan)

    def to_dict(self) -> dict:
        return {"active": self.active.to_dict(), "pump_wavelength": self.pump_wavelength, "signal_wavelength": self.signal_wavelength,
                "pump_direction": self.pump_direction, "name": self.name}


# ------------------------------------------------------------------------------------------------------- the limits
def srs_threshold(fibre: Fibre, length: float, wavelength: float, g_R: float = 1.0e-13) -> dict:
    """Stimulated Raman scattering, Smith's criterion: P_th ≈ 16 A_eff / (g_R L_eff), L_eff = (1 − e^{−αL})/α with the
    signal's background loss; g_R ≈ 1e-13 m/W in silica near 1 µm (the Stokes line 13.2 THz lower)."""
    a = fibre.alpha_signal
    l_eff = (1.0 - math.exp(-a * length)) / a if a > 0 else length
    a_eff = fibre.mode_area(wavelength)
    return {"P_threshold_W": 16.0 * a_eff / (g_R * l_eff), "L_eff": l_eff, "A_eff": a_eff, "stokes_wavelength": 1.0 / (1.0 / wavelength - 13.2e12 / C_LIGHT)}


def core_temperature(fibre: Fibre, heat_per_m: float, h: float = 50.0, kappa_glass: float = 1.38, kappa_coating: float = 0.2) -> dict:
    """Steady radial conduction with the heat generated in the core: the core-to-clad-surface rise
    q (1 + 2 ln(b/a))/(4πκ), the coating's q ln(c/b)/(2π κ_c) and the surface's q/(2π c h) to the air or the heat sink
    (``h`` W/m²/K: ~10 in still air, 50 on a plate, 1000+ in a water-cooled groove)."""
    a, b = fibre.core_radius, fibre.clad_radius
    c = fibre.coating_radius or b
    q = heat_per_m
    dt_core = q * (1.0 + 2.0 * math.log(b / a)) / (4 * math.pi * kappa_glass)
    dt_coat = q * math.log(c / b) / (2 * math.pi * kappa_coating) if c > b else 0.0
    dt_surf = q / (2 * math.pi * c * h)
    return {"dT_core_to_clad": dt_core, "dT_coating": dt_coat, "dT_surface_to_sink": dt_surf, "dT_core_to_sink": dt_core + dt_coat + dt_surf,
            "heat_per_m": q}
