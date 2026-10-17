"""Svarog-fiber — the Yb-doped double-clad fibre laser designed on Hyperion (notebook 47): the fibre twin of Svarog's rod.

The device as numbers: the oscillator fibre (`FIBRE_OSC`, a 10/125 µm Yb fibre), the amplifier fibre (`FIBRE_AMP`,
20/125 µm), their doping set by the datasheet's cladding absorption (`active`), the 976 nm diode pumps through a combiner
(`Pump`), the fibre Bragg gratings (`GRATINGS`), the oscillator (`oscillator`) and its length choice (`length_choice`),
the amplifier stage (`amplifier`), the coil (`coil`: bend loss against the radius), the limits (`limits`: SRS, the
core's temperature at several coolings), the beam out and the delivery train (`beam_out`, `delivery`, through Svarog's
`delivery_train`), the comparison with the rod from notebook 46's report (`against_the_rod`), the record (`design_record`).
Everything is Hyperion's `fibre` module; nothing is run."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from vegeta import hyperion as hy
from vegeta.hyperion import fibre as fb

import svarog as sv

LAM = 1070e-9                                   # the gratings' line: Yb's gain is broad and flat here, reabsorption small
LAM_P = 976e-9                                  # the absorption peak: short fibres; a wavelength-stabilised diode holds it
LOSS_DB_PER_KM = 10.0                           # background loss of the active fibre at both wavelengths
QUANTUM_LIMIT = LAM_P / LAM                     # 0.912: the most that can come out per watt absorbed

FIBRE_OSC = hy.Fibre(core_radius=5e-6, NA=0.075, clad_radius=62.5e-6, coating_radius=125e-6, clad_NA=0.46,
                     alpha_signal=LOSS_DB_PER_KM / 1e3 / fb.DB, alpha_pump=LOSS_DB_PER_KM / 1e3 / fb.DB, name="10/125 Yb double-clad")
FIBRE_AMP = hy.Fibre(core_radius=10e-6, NA=0.06, clad_radius=62.5e-6, coating_radius=125e-6, clad_NA=0.46,
                     alpha_signal=LOSS_DB_PER_KM / 1e3 / fb.DB, alpha_pump=LOSS_DB_PER_KM / 1e3 / fb.DB, name="20/125 Yb double-clad")
ABSORPTION_DB_PER_M = {"10/125 Yb double-clad": 1.7, "20/125 Yb double-clad": 5.0}     # the datasheets' cladding absorption at 976 nm


def active(fibre: hy.Fibre, length: float) -> hy.ActiveFibre:
    """The fibre doped to its datasheet's cladding absorption at 976 nm."""
    yb = fb.YB_SILICA.with_(N=fb.ions_for_absorption(fibre, ABSORPTION_DB_PER_M[fibre.name], LAM_P))
    return hy.ActiveFibre(fibre, yb, length, name=f"{fibre.name}, {length:g} m")


@dataclass(frozen=True)
class Pump:
    """Fibre-coupled 976 nm diodes (105/125 µm, NA 0.22 pigtails, ``diode_W`` each) into a (N+1)×1 pump combiner of
    ``combiner_efficiency``; ``n_diodes`` ports used. ``launched(P_diodes)`` is what reaches the active fibre."""
    diode_W: float = 9.0
    n_diodes: int = 3
    combiner_efficiency: float = 0.95
    pigtail_core_radius: float = 52.5e-6
    pigtail_NA: float = 0.22
    name: str = "976 nm diodes, (6+1)×1 combiner"

    @property
    def power_max(self) -> float:
        return self.diode_W * self.n_diodes * self.combiner_efficiency

    def launched(self, diodes_W: float) -> float:
        return diodes_W * self.combiner_efficiency

    def fits_the_cladding(self, fibre: hy.Fibre) -> dict:
        """Brightness: the pigtails' étendue against the cladding's — (n r NA)² summed over the ports must fit (b NA_clad)²."""
        per_port = (self.pigtail_core_radius * self.pigtail_NA) ** 2
        clad = (fibre.clad_radius * fibre.clad_NA) ** 2
        return {"ports_that_fit": int(clad / per_port), "used": self.n_diodes, "fits": self.n_diodes * per_port <= clad}


PUMP_OSC = Pump(n_diodes=3)
PUMP_AMP = Pump(n_diodes=6, name="976 nm diodes, (6+1)×1 combiner, six ports")


@dataclass(frozen=True)
class Gratings:
    """The two fibre Bragg gratings and the round trip's excess loss: ``excess_loss_db`` per pass for the two splices
    and the gratings' own loss (0.3 dB is a clean build), folded into the effective HR reflectivity."""
    R_hr: float = 0.99
    R_oc: float = 0.10
    bandwidth_hr_nm: float = 0.5
    bandwidth_oc_nm: float = 0.1
    excess_loss_db: float = 0.3
    name: str = "FBG pair at 1070 nm"

    @property
    def R_hr_effective(self) -> float:
        return self.R_hr * 10 ** (-2 * self.excess_loss_db / 10)


GRATINGS = Gratings()
R_OC_CANDIDATES = (0.04, 0.07, 0.10, 0.15, 0.20, 0.30)


# ------------------------------------------------------------------------------------------------ the oscillator
def oscillator(length: float, R_oc: float = GRATINGS.R_oc, gratings: Gratings = GRATINGS) -> hy.FibreLaser:
    return hy.FibreLaser(active(FIBRE_OSC, length), LAM_P, LAM, gratings.R_hr_effective, R_oc, name=f"Svarog-fiber oscillator, {length:g} m")


def length_choice(pump_W: float, lengths=None, within: float = 0.03, n: int = 200) -> dict:
    """The length sweep at the design pump and the rule: the shortest fibre whose output is within ``within`` of the
    best (a shorter fibre is cheaper, stiffer against SRS and bends, and the unabsorbed pump it leaves is reported)."""
    lengths = np.arange(2.0, 20.01, 1.0) if lengths is None else np.asarray(lengths, dtype=float)
    sw = oscillator(lengths[0]).length_sweep(lengths, pump_W, n)
    best = sw["best_output_W"]
    chosen = next(r for r in sw["rows"] if r["output_W"] >= (1.0 - within) * best)
    return {"rows": sw["rows"], "best_length": sw["best_length"], "best_output_W": best, "chosen_length": chosen["length"],
            "chosen_output_W": chosen["output_W"], "chosen_unabsorbed_W": chosen["unabsorbed_W"], "rule": f"shortest within {within:.0%} of the best"}


def coupler_choice(length: float, pump_W: float, n: int = 200) -> dict:
    """Output against the output grating's reflectivity at the design pump, with the threshold and the circulating power."""
    rows = {}
    for r in R_OC_CANDIDATES:
        las = oscillator(length, r)
        s = las.solve(pump_W, n)
        rows[r] = {"output_W": s.output_W, "threshold_W": las.threshold(n=n // 2), "circulating_at_OC_W": s.Ps_fwd[-1], "gain_db_one_way": s.gain_db,
                   "inversion_max": float(s.inversion.max())}
    return rows


# ------------------------------------------------------------------------------------------------- the amplifier
def amplifier(length: float, direction: str = "counter") -> hy.FibreAmplifier:
    return hy.FibreAmplifier(active(FIBRE_AMP, length), LAM_P, LAM, direction, name=f"Svarog-fiber amplifier, {length:g} m, {direction}-pumped")


def amplifier_length_choice(seed_W: float, pump_W: float, lengths=None, within: float = 0.03, n: int = 200) -> dict:
    lengths = np.arange(1.0, 10.01, 0.5) if lengths is None else np.asarray(lengths, dtype=float)
    rows = []
    for L in lengths:
        s = amplifier(float(L)).solve(seed_W, pump_W, n)
        rows.append({"length": float(L), "output_W": s.output_W, "gain_db": s.gain_db, "unabsorbed_W": s.unabsorbed_W, "heat_peak_per_m": s.peak_heat_per_m,
                     "inversion_max": float(s.inversion.max())})
    best = max(rows, key=lambda r: r["output_W"])
    chosen = next(r for r in rows if r["output_W"] >= (1.0 - within) * best["output_W"])
    return {"rows": rows, "best_length": best["length"], "chosen_length": chosen["length"], "chosen_output_W": chosen["output_W"], "rule": f"shortest within {within:.0%} of the best"}


# ------------------------------------------------------------------------------------------------------ the coil
def coil(fibre: hy.Fibre, radii=None, wavelength: float = LAM, budget_db: float = 0.05) -> dict:
    """LP₀₁'s bend loss against the coil radius (dB per metre and per turn) and the smallest radius within a loss
    budget for the whole length — the spool's size. Higher-mode stripping of a few-mode fibre is not modelled here."""
    radii = np.linspace(0.015, 0.12, 106) if radii is None else np.asarray(radii, dtype=float)
    per_m = fibre.bend_loss_db_per_m(radii, wavelength)
    return {"radii": radii, "db_per_m": per_m, "db_per_turn": per_m * 2 * math.pi * radii, "wavelength": wavelength, "V": fibre.V(wavelength),
            "single_mode": fibre.single_mode(wavelength)}


def coil_radius_for(fibre: hy.Fibre, length: float, budget_db: float = 0.05, wavelength: float = LAM) -> float:
    """The smallest coil radius at which the whole length loses no more than ``budget_db``."""
    radii = np.linspace(0.01, 0.2, 381)
    total = fibre.bend_loss_db_per_m(radii, wavelength) * length
    ok = np.where(total <= budget_db)[0]
    return float(radii[ok[0]]) if len(ok) else math.inf


# ---------------------------------------------------------------------------------------------------- the limits
COOLING = {"still air": 10.0, "on a plate": 50.0, "on a cooled spool": 500.0, "in a water-cooled groove": 2000.0}


def limits(fibre: hy.Fibre, length: float, sol: fb.FibreSolution, signal_W: float) -> dict:
    """SRS against the signal power in the fibre, the core's temperature at the hottest point for each cooling."""
    srs = hy.srs_threshold(fibre, length, LAM)
    temps = {name: hy.core_temperature(fibre, sol.peak_heat_per_m, h=h) for name, h in COOLING.items()}
    return {"srs": srs, "srs_margin": srs["P_threshold_W"] / signal_W if signal_W > 0 else math.inf, "temperatures": temps,
            "peak_heat_per_m": sol.peak_heat_per_m, "heat_W": sol.heat_W}


# ------------------------------------------------------------------------------------------------- the beam out
def beam_out(fibre: hy.Fibre, m2: float = 1.05) -> hy.GaussianBeam:
    """The LP₀₁ leaving an angled cleave (or an end cap's flat face — the same waist): M² 1.05 for a single-mode fibre;
    a few-mode fibre coiled to strip LP₁₁ is taken at the same number, which is what the coil must deliver."""
    return fibre.beam(LAM, m2, name=f"{fibre.name} output")


def delivery(beam: hy.GaussianBeam, w_collimated: float = 1.0e-3, f_focus: float = 0.050, standoff: float = 0.150) -> dict:
    """Svarog's delivery train on the fibre's beam: the collimator sized for the same collimated radius, the same
    stand-off and focusing lens, so the two work spots compare."""
    return sv.delivery_train(beam, w_collimated, f_focus, standoff)


# ------------------------------------------------------------------------------------------------ the comparison
def rod_profiles(report46: dict | None, n: int = 200) -> dict | None:
    """The rod's pump, inversion, circulating signal and heat along its 10 mm from notebook 46's record, on the same
    footing as the fibre's `FibreSolution`: the pump decays as e^(−αz) (single pass, end-pumped), the heat is η_h times
    the pump absorbed per metre, the inversion is the local four-level steady state saturated by the circulating
    intensity (N₂ = R τ / (1 + I/I_sat), R the pump rate density in the overlap area), the signal is the circulating
    power — nearly flat along a rod whose single-pass gain is a few per cent. None without the report."""
    if not report46:
        return None
    las = report46["laser"]; rod = report46["rod"]; med = rod["medium"]; pump = report46["pump"]
    alpha, L = med["alpha_pump"], rod["length"]
    P = pump["design_power"]; w_p = pump["w_effective"]; w_l = las["w_rod"]
    z = np.linspace(0.0, L, n)
    Pp = P * np.exp(-alpha * z)
    heat = med["eta_heat"] * alpha * Pp                                            # W/m
    area = math.pi * (w_p ** 2 + w_l ** 2) / 2
    hnu_p = hy.photon_energy(med["pump_wavelength"]); hnu_s = hy.photon_energy(med["wavelength"])
    rate = med["eta_q"] * alpha * Pp / (hnu_p * area)                              # upper-level ions per m³ per s
    i_sat = hnu_s / (med["sigma"] * med["tau"])
    circ = las["intracavity_W"]
    n2 = rate * med["tau"] / (1.0 + 2 * circ / (math.pi * w_l ** 2) / i_sat)       # saturated by the two-way circulating intensity
    n_total = 1.38e26                                                               # Nd ions/m³ at 1.0 at.% in YAG
    return {"z": z, "Pp": Pp, "Ps": np.full_like(z, circ), "inversion": n2 / n_total, "heat": heat, "length": L, "pump_W": P, "output_W": las["output_W"]}



def against_the_rod(report46: dict | None, osc: fb.FibreSolution, osc_threshold: float, osc_slope: float, dl: dict, beam: hy.GaussianBeam, pump_W: float) -> dict:
    """The oscillator at the rod's design pump against notebook 46's record: efficiency, threshold, slope, the beam, the
    work spot, what each is sensitive to. Rows with the rod's numbers are empty when the report is missing."""
    rod = (report46 or {}).get("laser", {})
    rdl = (report46 or {}).get("delivery", {})
    rb = (report46 or {}).get("beam_out", {})
    rows = {
        "pump at the work point [W]": (pump_W, (report46 or {}).get("meta", {}).get("design_pump_W", math.nan)),
        "output [W]": (osc.output_W, rod.get("output_W", math.nan)),
        "optical efficiency": (osc.efficiency, rod.get("output_W", math.nan) / (report46 or {}).get("meta", {}).get("design_pump_W", math.nan) if rod else math.nan),
        "threshold [W]": (osc_threshold, rod.get("threshold_W", math.nan)),
        "slope": (osc_slope, rod.get("slope_efficiency", math.nan)),
        "heat in the gain medium [W]": (osc.heat_W, rod.get("heat_W", math.nan)),
        "beam waist [µm]": (beam.w0 * 1e6, rb.get("w0", math.nan) * 1e6 if rb else math.nan),
        "beam divergence [mrad]": (beam.divergence * 1e3, rb.get("divergence", math.nan) * 1e3 if rb else math.nan),
        "M²": (beam.m2, 1.0),
        "work spot radius [µm]": (dl["work_spot"] * 1e6, rdl.get("work_spot", math.nan) * 1e6 if rdl else math.nan),
        "depth of focus [mm]": (dl["depth_of_focus"] * 1e3, rdl.get("depth_of_focus", math.nan) * 1e3 if rdl else math.nan),
        "collimator f [mm]": (dl["f_collimator"] * 1e3, rdl.get("f_collimator", math.nan) * 1e3 if rdl else math.nan),
    }
    return {"rows": rows, "have_rod": bool(rod),
            "sensitivities": {"fibre": "the gratings' wavelengths and temperature, splice losses, the pump diodes' wavelength against the 976 nm peak, bend loss, SRS at high power",
                              "rod": "the thermal lens (η_heat, the pump spot), the cavity's alignment, the mode–pump overlap, the rod's fracture limit"}}


# ---------------------------------------------------------------------------------------------------- the record
def design_record(osc_length: float, R_oc: float, pump_osc_W: float, amp_length: float, pump_amp_W: float, beam: hy.GaussianBeam, dl: dict) -> dict:
    """What the next notebook (the build) reads: the oscillator and the amplifier as Hyperion objects, the pumps, the
    beam out and the delivery train."""
    osc = oscillator(osc_length, R_oc)
    amp = amplifier(amp_length)
    s_osc = osc.solve(pump_osc_W, 200)
    s_amp = amp.solve(s_osc.output_W, pump_amp_W, 200)
    return {"name": "Svarog-fiber", "wavelength": LAM, "pump_wavelength": LAM_P,
            "oscillator": {"laser": osc.to_dict(), "fibre": FIBRE_OSC.to_dict(), "pump_W": pump_osc_W, "pump": {"diodes": PUMP_OSC.n_diodes, "diode_W": PUMP_OSC.diode_W},
                           "gratings": {"R_hr": osc.R_hr, "R_oc": osc.R_oc}, **s_osc.to_dict(), "coil_radius": coil_radius_for(FIBRE_OSC, osc_length)},
            "amplifier": {"amplifier": amp.to_dict(), "fibre": FIBRE_AMP.to_dict(), "pump_W": pump_amp_W, "seed_W": s_osc.output_W,
                          "pump": {"diodes": PUMP_AMP.n_diodes, "diode_W": PUMP_AMP.diode_W}, **s_amp.to_dict(), "coil_radius": coil_radius_for(FIBRE_AMP, amp_length)},
            "beam_out": beam, "delivery": {"train": dl["train"], "f_collimator": dl["f_collimator"], "work_spot": dl["work_spot"],
                                           "depth_of_focus": dl["depth_of_focus"], "w_collimated": dl["w_collimated"]}}
