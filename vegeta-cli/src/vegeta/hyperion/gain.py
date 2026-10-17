"""The gain medium of a four-level, end-pumped solid-state laser in steady state: pump absorption, threshold, slope
efficiency, output power against pump power (Rigrod in the low-gain limit), the optimum output coupler, the
small-signal gain, and the thermal lens of the pumped rod (Innocenti's end-pumped form). The numbers are the
classic ones (Koechner, *Solid-State Laser Engineering*; Fan & Byer; Laporta & Brussard): a top-hat pump of radius
w_p overlapping a TEM₀₀ mode of radius w_l through the effective area π(w_p² + w_l²)/2. Nothing is picked: the
medium, the rod, the pump spot, the mode, the coupler and the loss are inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .elements import ThermalLens
from .materials import Material, YAG

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0


@dataclass(frozen=True)
class GainMedium:
    """A four-level medium: ``sigma`` the stimulated-emission cross-section [m²], ``tau`` the upper-level lifetime [s],
    ``wavelength`` the laser line [m], ``pump_wavelength`` [m], ``alpha_pump`` the pump absorption coefficient [1/m]
    at the doping chosen, ``eta_q`` the pump quantum efficiency (upper-level ions per absorbed pump photon),
    ``eta_heat`` the fraction of absorbed pump power that becomes heat, ``material`` the host's optical and thermal
    numbers (n, dn/dT, κ, α, ν)."""
    name: str
    sigma: float
    tau: float
    wavelength: float
    pump_wavelength: float
    alpha_pump: float
    eta_q: float = 1.0
    eta_heat: float = 0.3
    material: Material = YAG
    note: str = ""

    @property
    def saturation_intensity(self) -> float:
        """hν_l / (σ τ) [W/m²]."""
        return H_PLANCK * C_LIGHT / self.wavelength / (self.sigma * self.tau)

    @property
    def quantum_defect(self) -> float:
        """1 − λ_p/λ_l: the share of a pump photon's energy that is not laser light."""
        return 1.0 - self.pump_wavelength / self.wavelength

    def n(self) -> float:
        return self.material.n(self.wavelength)

    def with_(self, **changes) -> "GainMedium":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"name": self.name, "sigma": self.sigma, "tau": self.tau, "wavelength": self.wavelength, "pump_wavelength": self.pump_wavelength,
                "alpha_pump": self.alpha_pump, "eta_q": self.eta_q, "eta_heat": self.eta_heat, "material": self.material.to_dict(), "note": self.note}


ND_YAG = GainMedium("Nd:YAG 1.0 at.%", sigma=2.8e-23, tau=230e-6, wavelength=1064e-9, pump_wavelength=808e-9, alpha_pump=400.0,
                    eta_q=0.9, eta_heat=0.3, material=YAG,
                    note="σ 2.8e-19 cm², τ 230 µs (Koechner); α ≈ 4 /cm at 808 nm for 1 at.% (polarisation and diode linewidth move it); "
                         "η_heat ≈ 0.3 for diode pumping (the quantum defect 0.24 plus non-radiative paths)")

MEDIA = {"nd:yag": ND_YAG}


def medium(name: str | GainMedium) -> GainMedium:
    if isinstance(name, GainMedium):
        return name
    key = str(name).lower()
    if key in MEDIA:
        return MEDIA[key]
    raise ValueError(f"unknown gain medium {name!r}; known: {', '.join(m.name for m in MEDIA.values())}")


@dataclass(frozen=True)
class Rod:
    """A cylindrical crystal: ``radius`` and ``length`` [m] of ``medium``."""
    medium: GainMedium
    radius: float
    length: float
    name: str = "rod"

    @property
    def absorbed_fraction(self) -> float:
        """1 − exp(−α L): the single-pass pump absorption."""
        return 1.0 - math.exp(-self.medium.alpha_pump * self.length)

    @property
    def absorption_length(self) -> float:
        return 1.0 / self.medium.alpha_pump

    def to_dict(self) -> dict:
        return {"medium": self.medium.to_dict(), "radius": self.radius, "length": self.length, "name": self.name}


def photon_energy(wavelength: float) -> float:
    return H_PLANCK * C_LIGHT / wavelength


def effective_pump_radius(w_of_z, rod: Rod, n: int = 400) -> float:
    """A pump beam that changes radius along the rod: the radius whose area is the absorption-weighted mean of
    w_p(z)² over the rod (``w_of_z``: a callable on z [m] from the entrance face)."""
    z = np.linspace(0.0, rod.length, n)
    weight = np.exp(-rod.medium.alpha_pump * z)
    w2 = np.asarray(w_of_z(z), dtype=float) ** 2
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(math.sqrt(trapz(w2 * weight, z) / trapz(weight, z)))


@dataclass(frozen=True)
class Laser:
    """The steady-state laser: ``rod`` pumped end-on through a top-hat spot of radius ``w_pump`` [m], the cavity mode of
    radius ``w_mode`` [m] in the rod, an output coupler of transmission ``T`` and a round-trip loss ``loss`` from
    everything else (the other mirror, scatter, the rod's absorption at the laser line)."""
    rod: Rod
    w_pump: float
    w_mode: float
    T: float
    loss: float = 0.0
    name: str = ""

    @property
    def medium(self) -> GainMedium:
        return self.rod.medium

    @property
    def effective_area(self) -> float:
        """π (w_p² + w_l²) / 2: the overlap area of a top-hat pump and a Gaussian mode."""
        return math.pi * (self.w_pump ** 2 + self.w_mode ** 2) / 2.0

    @property
    def eta_pump(self) -> float:
        """Absorbed fraction × quantum efficiency: upper-level ions per incident pump photon."""
        return self.rod.absorbed_fraction * self.medium.eta_q

    def small_signal_gain(self, pump_W: float) -> float:
        """The round-trip small-signal gain exponent g₀ = 2 σ N₂ L."""
        m = self.medium
        n2 = self.eta_pump * pump_W * m.tau / (photon_energy(m.pump_wavelength) * self.effective_area * self.rod.length)
        return 2.0 * m.sigma * n2 * self.rod.length

    def threshold(self, T: float | None = None) -> float:
        """Pump power at threshold [W]: π hν_p (w_p² + w_l²)(T + L) / (4 σ τ η_abs η_q)."""
        T = self.T if T is None else T
        m = self.medium
        return math.pi * photon_energy(m.pump_wavelength) * (self.w_pump ** 2 + self.w_mode ** 2) * (T + self.loss) / (4 * m.sigma * m.tau * self.eta_pump)

    def slope_efficiency(self, T: float | None = None) -> float:
        """η_abs η_q (λ_p/λ_l) T / (T + L): output watts per pump watt above threshold."""
        T = self.T if T is None else T
        m = self.medium
        return self.eta_pump * (m.pump_wavelength / m.wavelength) * T / (T + self.loss)

    def output(self, pump_W, T: float | None = None):
        """Output power [W] against pump power (a number or an array): the Rigrod low-gain line
        η_s (P_p − P_th), zero below threshold."""
        T = self.T if T is None else T
        p = np.asarray(pump_W, dtype=float)
        out = self.slope_efficiency(T) * (p - self.threshold(T))
        return np.where(out > 0, out, 0.0) if out.ndim else float(max(out, 0.0))

    def intracavity(self, pump_W: float, T: float | None = None) -> float:
        """One-way circulating power [W] = output / T."""
        T = self.T if T is None else T
        return float(self.output(pump_W, T)) / T

    def optimum_coupler(self, pump_W: float, T_range=(0.002, 0.6), n: int = 600) -> dict:
        """The output coupling that maximises the output at ``pump_W`` (a scan), and Rigrod's closed form
        T_opt = √(g₀ L) − L for comparison."""
        ts = np.linspace(T_range[0], T_range[1], n)
        outs = np.array([float(self.output(pump_W, t)) for t in ts])
        i = int(np.argmax(outs))
        g0 = self.small_signal_gain(pump_W)
        return {"T_opt": float(ts[i]), "P_out": float(outs[i]), "T_opt_rigrod": math.sqrt(g0 * self.loss) - self.loss if self.loss > 0 else math.nan,
                "g0": g0, "pump_W": pump_W}

    def table(self, pump_W) -> dict:
        """The laser's numbers at one pump power (threshold, slope, output, intracavity, gain, heat) in one dict."""
        return {"pump_W": pump_W, "absorbed_W": pump_W * self.rod.absorbed_fraction, "heat_W": pump_W * self.rod.absorbed_fraction * self.medium.eta_heat,
                "threshold_W": self.threshold(), "slope_efficiency": self.slope_efficiency(), "output_W": float(self.output(pump_W)),
                "intracavity_W": self.intracavity(pump_W), "small_signal_gain": self.small_signal_gain(pump_W),
                "optical_efficiency": float(self.output(pump_W)) / pump_W if pump_W > 0 else 0.0, "effective_area": self.effective_area,
                "w_pump": self.w_pump, "w_mode": self.w_mode, "T": self.T, "loss": self.loss}

    def to_dict(self) -> dict:
        return {"rod": self.rod.to_dict(), "w_pump": self.w_pump, "w_mode": self.w_mode, "T": self.T, "loss": self.loss, "name": self.name}


def thermal_lens(rod: Rod, pump_W: float, w_pump: float, stress_and_bulge: float = 0.0) -> ThermalLens:
    """The thermal lens of an end-pumped rod (Innocenti et al. 1990): 1/f = η_h P_abs (dn/dT) / (π κ w_p²) for a
    top-hat pump of radius w_p absorbed in the rod; ``stress_and_bulge`` adds the stress-induced index change and
    the end-face bulging as a fraction of the dn/dT term (Nd:YAG: of the order of 0.2; 0 = the dn/dT term alone).
    ``f = inf`` with no pump."""
    m = rod.medium.material
    heat = rod.medium.eta_heat * rod.absorbed_fraction * pump_W
    if heat <= 0:
        return ThermalLens(math.inf, pump_W, name=f"{rod.name} thermal lens")
    power = heat * m.dn_dT * (1.0 + stress_and_bulge) / (math.pi * m.kappa * w_pump ** 2)
    return ThermalLens(1.0 / power, pump_W, name=f"{rod.name} thermal lens")


def rod_temperature_rise(rod: Rod, pump_W: float, w_pump: float) -> dict:
    """Steady radial conduction in a rod cooled at its barrel with the heat deposited inside radius w_p along the
    absorption length: the centre-to-barrel temperature rise ΔT = Q (1 + 2 ln(r₀/w_p)) / (4 π κ L_abs) where Q is the
    heat and L_abs the length it is spread over (the rod length for a rod that absorbs almost all of the pump)."""
    m = rod.medium.material
    heat = rod.medium.eta_heat * rod.absorbed_fraction * pump_W
    length = rod.length
    dt_centre = heat * (1.0 + 2.0 * math.log(rod.radius / w_pump)) / (4 * math.pi * m.kappa * length) if heat > 0 else 0.0
    dt_pump_edge = heat * 2.0 * math.log(rod.radius / w_pump) / (4 * math.pi * m.kappa * length) if heat > 0 else 0.0
    return {"heat_W": heat, "dT_centre_to_barrel": dt_centre, "dT_pump_edge_to_barrel": dt_pump_edge, "heat_per_length": heat / length}
