"""Svarog III — four commercial fibre lasers on Hyperion (notebook 47): the published limits of IPG's YLS-SM, Coherent's
FL2000C and FL2000CSM and nLIGHT's WFL-3000 taken as they are (``designs/data/svarog_commercial_specs.json``, unknowns
preserved), then the physics Svarog's device was designed on applied to each: the beam (BPP ⇄ M², the delivery fibre it
implies, Gaussian against multimode propagation through Svarog's 100/200 head), the energy balance (electrical in, waste
heat, cooling — bounded assumptions where the sheet is silent, a sensitivity table), the thin-sheet welds (steel and
aluminium at 1, 2, 3 mm: the keyhole threshold, the speed that penetrates with a margin, the thin-plate Rosenthal field,
cooling, HAZ, distortion indicators — a model, never a validated weld), and the shielding assessment (the raw collimated
beam and a specular reflection at 5–100 m against polycarbonate, aluminium and wood: exposure against the MPE, the
transmitted and reflected shares, a non-destructive bound on the barrier's heating, a fire-risk indicator for wood — and
a verdict PASS / FAIL / INCOMPLETE where PASS needs certified protection evidence this module cannot produce).

Nothing here optimises damage: no burn-through time is computed for a barrier (``svarog_industrial.screen`` has those
bounds for the guard's design; this module asks only whether a barrier could ever be shown safe). Every assumption is a
named entry of ``ASSUMPTIONS`` with its range; every unknown of the sheet stays ``None``."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from vegeta import hyperion as hy
from vegeta.hyperion import thermal as th
from vegeta.hyperion import welding as wd

import svarog_industrial as si

DATA = Path(__file__).resolve().parent / "data"
SPECS_PATH = DATA / "svarog_commercial_specs.json"
LAM = 1070e-9
HEAD = si.HEAD                                          # Svarog's 100/200 welding head: the one optic every laser is sent through
EXPOSURE = si.EXPOSURE                                  # IEC 60825-4's T1 10 s, T2 100 s, T3 30 000 s
STANDARDS = {"IEC 60825-1": "safety of laser products — classification, MPE tables (edition used for the formulae: 2014+A11:2021; verify the current one)",
             "IEC 60825-4": "laser guards — the exposure classes T1/T2/T3 and the guard's protective exposure limit (verify the current edition)",
             "ISO 11553-1": "laser processing machines — general safety requirements (verify the current edition)",
             "EN 12254": "screens for laser working places — protection levels AB1–AB10 per power density, wavelength and test time (verify the current edition)",
             "EN 207": "eye protectors — scale numbers LB1–LB10 (D mode: continuous wave) (cited for the OD ladder only; verify the current edition)"}


def known(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ================================================================================================== the lasers, as published
@dataclass(frozen=True)
class Spec:
    """One laser's published limits; ``None`` is Unknown and stays so."""
    key: str
    maker: str
    model: str
    power_W: float
    wavelength: float                                   # m
    mode: str                                           # "single" | "multi"
    m2_max: float | None = None
    bpp_max_mm_mrad: float | None = None
    stability_pct: float | None = None
    modulation_kHz: float | None = None
    connected_kVA: float | None = None
    electrical_max_kW: float | None = None
    cooling_kW: float | None = None
    notes: str = ""
    source: str = "published limits (user-supplied table)"

    @property
    def name(self) -> str:
        return f"{self.maker} {self.model}"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def load_specs(path: Path = SPECS_PATH) -> dict[str, Spec]:
    d = json.loads(Path(path).read_text())
    out = {}
    for key, s in d["lasers"].items():
        out[key] = Spec(key, s["maker"], s["model"], float(s["power_W"]), float(s["wavelength_nm"]) * 1e-9, s["mode"], s.get("m2_max"), s.get("bpp_max_mm_mrad"),
                        s.get("stability_pct"), s.get("modulation_kHz"), s.get("connected_kVA"), s.get("electrical_max_kW"), s.get("cooling_kW"), s.get("notes", ""),
                        source=d.get("source", "published limits"))
    return out


LASERS = load_specs()
KEYS = tuple(LASERS)


# ================================================================================================== the assumptions, named
@dataclass(frozen=True)
class Assumption:
    """A number the sheets do not give: its ``nominal`` and ``(low, high)`` range, what it is and where it comes from."""
    key: str
    nominal: float
    low: float
    high: float
    unit: str
    basis: str

    @property
    def interval(self) -> tuple[float, float]:
        return (self.low, self.high)


ASSUMPTIONS = {a.key: a for a in [
    Assumption("diode_eff", 0.50, 0.45, 0.55, "-", "976 nm pump diodes' electrical-to-optical efficiency (class of today's modules)"),
    Assumption("driver_eff", 0.90, 0.88, 0.93, "-", "the diode drivers' (switch-mode) efficiency"),
    Assumption("optical_eff", 0.75, 0.70, 0.85, "-", "pump-to-signal conversion of a Yb MOPA including combiners and splices (Svarog's 1 kW stage: 0.93 on the launched pump × its combiner 0.95 × ~0.85 of the chain)"),
    Assumption("aux_W", 300.0, 200.0, 500.0, "W", "controls, fans, pumps and the diode's thermo-electric coolers where fitted"),
    Assumption("cooling_COP", 3.0, 2.5, 3.5, "-", "a compressor chiller's coefficient of performance at 20 °C water, 35 °C air"),
    Assumption("power_factor", 0.95, 0.90, 0.98, "-", "a PFC-corrected three-phase supply"),
    Assumption("delivery_loss_dB", 0.3, 0.2, 0.5, "dB", "the process fibre and its connectors (Svarog: 0.3 dB on 20 m)"),
    Assumption("lens_absorption", 0.002, 0.001, 0.003, "-", "each of the head's two lenses (coated fused silica at 1 µm)"),
    Assumption("abs_keyhole_steel", 0.70, 0.50, 0.80, "-", "keyhole absorptivity of mild steel at 1 µm (multiple reflections; svarog_industrial.WELD_MATERIALS)"),
    Assumption("abs_keyhole_al", 0.50, 0.35, 0.60, "-", "keyhole absorptivity of aluminium at 1 µm"),
    Assumption("melting_eff", 0.40, 0.30, 0.48, "-", "the share of the absorbed energy that ends in the melt (Swift-Hook & Gick's bound 0.48)"),
    Assumption("workpiece_reflectivity", 0.9, 0.6, 0.92, "-", "a specular reflection off a flat, polished or molten workpiece at 1 µm (steel 0.6, aluminium 0.9)"),
    Assumption("seam_gap_mm", 1.0, 0.2, 3.0, "mm", "a seam or gap between barrier panels the beam can pass through"),
    Assumption("wood_piloted_flux", 12e3, 10e3, 13e3, "W/m²", "critical absorbed flux for the piloted ignition of wood (fire-safety handbooks: 10–13 kW/m²)"),
    Assumption("wood_spontaneous_flux", 28e3, 25e3, 30e3, "W/m²", "critical flux for the unpiloted ignition of wood"),
    Assumption("pc_degraded_transmission_x", 2.0, 1.0, 5.0, "-", "how much a scratched, crazed or thinned certified sheet's transmission grows (no product data: a sensitivity)"),
]}


def _a(key: str) -> Assumption:
    return ASSUMPTIONS[key]


# ================================================================================================== optics
FIBRE_CLASSES = [wd.DeliveryFibre(7e-6, 0.08, 20.0, "14/125 feeding fibre"), wd.DeliveryFibre(10e-6, 0.10, 20.0, "20/125 feeding fibre"),
                 wd.DeliveryFibre(25e-6, 0.12, 20.0, "50/125 process fibre"), wd.DeliveryFibre(50e-6, 0.12, 20.0, "100/125 process fibre"),
                 wd.DeliveryFibre(100e-6, 0.22, 20.0, "200/220 process fibre")]
GAUSSIAN_M2_LIMIT = 1.5                                 # below it the embedded Gaussian is the beam; above, the core is imaged (geometric)


def bpp_from_m2(m2: float, wavelength: float = LAM) -> float:
    """mm·mrad from M²: BPP = M² λ/π."""
    return m2 * wavelength / math.pi * 1e6


def m2_from_bpp(bpp_mm_mrad: float, wavelength: float = LAM) -> float:
    return math.pi * bpp_mm_mrad * 1e-6 / wavelength


def implied_fibre(bpp_mm_mrad: float) -> wd.DeliveryFibre:
    """The smallest listed fibre whose r·NA holds the beam (an étendue match, not a vendor's choice)."""
    for f in FIBRE_CLASSES:
        if f.bpp * 1e6 >= bpp_mm_mrad:
            return f
    return FIBRE_CLASSES[-1]


def optics(spec: Spec, head: wd.WeldHead = HEAD) -> dict:
    """What the published beam quality implies: BPP and M² (one from the other where only one is given, both flagged as
    limits), the consistency of M²/BPP with the stated mode, the delivery fibre class, the beam after Svarog's head two
    ways — the core imaged (geometric, what a multimode fibre gives) and the embedded Gaussian of that M² (what a
    single-mode beam gives) — with the spot, cone, Rayleigh range, collimated radius and the collimated beam's own
    divergence; ``regime`` says which description to trust."""
    if known(spec.m2_max) and known(spec.bpp_max_mm_mrad):
        m2, bpp = spec.m2_max, spec.bpp_max_mm_mrad
        basis = "both published"
        consistent = abs(bpp_from_m2(m2) - bpp) / bpp < 0.25
    elif known(spec.m2_max):
        m2, bpp, basis, consistent = spec.m2_max, bpp_from_m2(spec.m2_max), "M² published; BPP = M² λ/π", True
    elif known(spec.bpp_max_mm_mrad):
        bpp, m2, basis, consistent = spec.bpp_max_mm_mrad, m2_from_bpp(spec.bpp_max_mm_mrad), "BPP published; M² = π BPP/λ", True
    else:
        raise ValueError(f"{spec.key}: neither M² nor BPP is published")
    mode_ok = (m2 <= GAUSSIAN_M2_LIMIT) == (spec.mode == "single")
    fibre = implied_fibre(bpp)
    geo_spot = head.spot_radius(fibre)
    w_coll = head.collimated_radius(fibre)
    gauss_spot = m2 * spec.wavelength * head.f_focus / (math.pi * w_coll)                 # the embedded Gaussian focused from the collimated radius
    gauss = hy.GaussianBeam(spec.wavelength, gauss_spot, 0.0, m2, 1.0, f"{spec.key} focus")
    geo_cone = head.cone_half_angle(fibre)
    theta_coll_geo = fibre.core_radius / head.f_collimator                               # the core's image at infinity: the collimated beam's growth
    theta_coll_gauss = m2 * spec.wavelength / (math.pi * w_coll)
    regime = "Gaussian (embedded M² is the beam)" if m2 <= GAUSSIAN_M2_LIMIT else "multimode (the core is imaged; the Gaussian is only an envelope)"
    spot = gauss_spot if m2 <= GAUSSIAN_M2_LIMIT else geo_spot
    cone = gauss.divergence if m2 <= GAUSSIAN_M2_LIMIT else geo_cone
    d_loss, lens = _a("delivery_loss_dB"), _a("lens_absorption")
    eff = {lvl: 10 ** (-d_loss.__getattribute__(lvl) / 10) * (1 - lens.__getattribute__(lvl)) ** 2 for lvl in ("low", "nominal", "high")}
    eff_interval = (min(eff.values()), max(eff.values()))
    stab = spec.stability_pct
    return {"laser": spec.key, "mode": spec.mode, "m2": m2, "bpp_mm_mrad": bpp, "basis": basis, "m2_bpp_consistent": consistent,
            "mode_matches_m2": mode_ok, "fibre": fibre, "fibre_name": fibre.name, "fibre_bpp_mm_mrad": fibre.bpp * 1e6, "regime": regime,
            "spot_radius": spot, "spot_geometric": geo_spot, "spot_gaussian": gauss_spot, "cone_half_angle": cone, "cone_geometric": geo_cone,
            "cone_gaussian": gauss.divergence, "rayleigh_length": spot / cone, "rayleigh_gaussian": gauss.rayleigh_range, "collimated_radius": w_coll,
            "collimated_divergence": max(theta_coll_geo, theta_coll_gauss), "collimated_divergence_geometric": theta_coll_geo, "collimated_divergence_gaussian": theta_coll_gauss,
            "optical_efficiency_head": eff["nominal"], "optical_efficiency_interval": eff_interval,
            "delivered_W": spec.power_W * eff["nominal"], "delivered_interval_W": (spec.power_W * eff_interval[0], spec.power_W * eff_interval[1]),
            "power_band_W": ((1 - stab / 100) * spec.power_W, (1 + stab / 100) * spec.power_W) if known(stab) else None,
            "mean_irradiance_W_per_cm2": spec.power_W * eff["nominal"] / (math.pi * spot ** 2) / 1e4,
            "gaussian": gauss, "head": head}


def beam_radius_at(opt: dict, distance, kind: str = "collimated") -> np.ndarray:
    """The stray beam's radius at ``distance`` [m]: the raw collimated beam (W_c + θ_c·d, θ_c the larger of the geometric
    and Gaussian growth: the worst case, a beam that stays small) or the beam diverging past the focus (θ·d)."""
    d = np.asarray(distance, dtype=float)
    if kind == "collimated":
        return opt["collimated_radius"] + opt["collimated_divergence"] * d
    return np.maximum(opt["cone_half_angle"] * d, opt["spot_radius"])


def optics_table(lasers: dict[str, Spec] = LASERS) -> pd.DataFrame:
    rows = {}
    for k, s in lasers.items():
        o = optics(s)
        rows[s.name] = {"mode (published)": s.mode, "M²": o["m2"], "BPP [mm·mrad]": o["bpp_mm_mrad"], "basis": o["basis"], "M²/BPP consistent": o["m2_bpp_consistent"],
                        "mode matches M²": o["mode_matches_m2"], "fibre class": o["fibre_name"], "regime": o["regime"], "spot radius [µm]": o["spot_radius"] * 1e6,
                        "spot, core imaged [µm]": o["spot_geometric"] * 1e6, "spot, Gaussian [µm]": o["spot_gaussian"] * 1e6, "cone half-angle [mrad]": o["cone_half_angle"] * 1e3,
                        "Rayleigh length [mm]": o["rayleigh_length"] * 1e3, "collimated radius [mm]": o["collimated_radius"] * 1e3,
                        "collimated divergence [µrad]": o["collimated_divergence"] * 1e6, "optical efficiency to the work": o["optical_efficiency_head"],
                        "delivered [W]": o["delivered_W"], "delivered low–high [W]": tuple(round(x) for x in o["delivered_interval_W"]),
                        "power band (stability) [W]": o["power_band_W"], "irradiance at the spot [MW/cm²]": o["mean_irradiance_W_per_cm2"] / 1e6}
    return pd.DataFrame(rows).T


# ================================================================================================== electrical and thermal
def _triple(fn, keys: tuple[str, ...]) -> dict:
    """``fn(**{key: value})`` at the nominal, and the extreme combinations of the assumptions' ranges → (low, nominal, high)."""
    nom = fn(**{k: _a(k).nominal for k in keys})
    vals = [nom]
    for bits in range(2 ** len(keys)):
        kw = {k: (_a(k).low if bits >> i & 1 else _a(k).high) for i, k in enumerate(keys)}
        vals.append(fn(**kw))
    return {"low": min(vals), "nominal": nom, "high": max(vals)}


def energy_balance(spec: Spec) -> dict:
    """Electrical in, wall-plug efficiency, waste heat, the chiller's draw and the connected power — from the sheet where it
    speaks (the electrical maximum and the cooling capacity are limits: the balance is checked, not closed) and from the
    assumptions' ranges where it is silent. Every number a (low, nominal, high) with its basis."""
    P = spec.power_W
    keys = ("diode_eff", "driver_eff", "optical_eff", "aux_W")

    def electrical(diode_eff, driver_eff, optical_eff, aux_W):
        return P / (diode_eff * driver_eff * optical_eff) + aux_W

    est = _triple(electrical, keys)
    out = {"laser": spec.key, "power_W": P}
    if known(spec.electrical_max_kW):
        e_max = spec.electrical_max_kW * 1e3
        out["electrical_W"] = {"low": min(est["low"], e_max), "nominal": min(est["nominal"], e_max), "high": e_max,
                               "basis": f"published maximum {spec.electrical_max_kW} kW caps the estimate ({est['low']:.0f}–{est['high']:.0f} W from the assumptions)"}
    else:
        out["electrical_W"] = {**est, "basis": "no published electrical input: diode, driver and optical efficiencies and the auxiliaries (ASSUMPTIONS)"}
    e = out["electrical_W"]
    out["wall_plug"] = {"low": P / e["high"], "nominal": P / e["nominal"], "high": P / e["low"], "basis": "P_out / electrical in"}
    out["waste_heat_W"] = {"low": e["low"] - P, "nominal": e["nominal"] - P, "high": e["high"] - P, "basis": "electrical in − optical out (all of it ends in the coolant or the air)"}
    if known(spec.cooling_kW):
        c = spec.cooling_kW * 1e3
        out["cooling_published_W"] = c
        out["cooling_consistent"] = out["waste_heat_W"]["high"] <= 1.1 * c and c <= 1.5 * out["waste_heat_W"]["high"]
        out["cooling_note"] = f"published capacity {c:.0f} W against the waste heat's bound {out['waste_heat_W']['high']:.0f} W: " + ("consistent" if out["cooling_consistent"] else "NOT consistent — one of the limits is not the operating value")
    else:
        out["cooling_published_W"] = None
        out["cooling_consistent"] = None
        out["cooling_note"] = "no published cooling capacity: the waste heat's bound sizes the chiller"
    cop = _a("cooling_COP")
    heat = out["waste_heat_W"]
    out["chiller_electrical_W"] = {"low": heat["low"] / cop.high, "nominal": heat["nominal"] / cop.nominal, "high": heat["high"] / cop.low, "basis": "waste heat / COP"}
    pf = _a("power_factor")
    tot = {lvl: e[lvl] + out["chiller_electrical_W"][lvl] for lvl in ("low", "nominal", "high")}
    out["total_electrical_W"] = {**tot, "basis": "laser + chiller"}
    out["connected_kVA"] = {"low": tot["low"] / pf.high / 1e3, "nominal": tot["nominal"] / pf.nominal / 1e3, "high": tot["high"] / pf.low / 1e3, "basis": "total / power factor"}
    if known(spec.connected_kVA):
        out["connected_published_kVA"] = spec.connected_kVA
        out["connected_note"] = (f"published {spec.connected_kVA} kVA is the laser alone: it holds the electrical maximum {spec.electrical_max_kW} kW at pf ≥ {spec.electrical_max_kW / spec.connected_kVA:.2f}"
                                 if known(spec.electrical_max_kW) else f"published {spec.connected_kVA} kVA")
    else:
        out["connected_published_kVA"] = None
        out["connected_note"] = "no published connected load"
    return out


def energy_table(lasers: dict[str, Spec] = LASERS) -> pd.DataFrame:
    rows = {}
    for k, s in lasers.items():
        b = energy_balance(s)
        f = lambda q, scale=1.0: f"{b[q]['nominal'] * scale:.2f} ({b[q]['low'] * scale:.2f}–{b[q]['high'] * scale:.2f})"
        rows[s.name] = {"optical out [kW]": s.power_W / 1e3, "electrical in [kW]": f("electrical_W", 1e-3), "wall-plug": f("wall_plug"), "waste heat [kW]": f("waste_heat_W", 1e-3),
                        "published cooling [kW]": s.cooling_kW, "cooling check": b["cooling_note"], "chiller electrical [kW]": f("chiller_electrical_W", 1e-3),
                        "total with chiller [kW]": f("total_electrical_W", 1e-3), "connected, laser + chiller [kVA]": f("connected_kVA"), "published connected [kVA]": s.connected_kVA}
    return pd.DataFrame(rows).T


def energy_sensitivity(spec: Spec) -> pd.DataFrame:
    """Each assumption moved to its range's ends with the others nominal: the electrical input and the wall-plug it gives."""
    rows = []
    for key in ("diode_eff", "driver_eff", "optical_eff", "aux_W", "cooling_COP"):
        a = _a(key)
        for lvl in ("low", "high"):
            kw = {k: _a(k).nominal for k in ("diode_eff", "driver_eff", "optical_eff", "aux_W")}
            cop = _a("cooling_COP").nominal
            if key == "cooling_COP":
                cop = getattr(a, lvl)
            else:
                kw[key] = getattr(a, lvl)
            e = spec.power_W / (kw["diode_eff"] * kw["driver_eff"] * kw["optical_eff"]) + kw["aux_W"]
            if known(spec.electrical_max_kW):
                e = min(e, spec.electrical_max_kW * 1e3)
            rows.append({"assumption": key, "end": lvl, "value": getattr(a, lvl), "electrical_W": e, "wall_plug": spec.power_W / e, "total_with_chiller_W": e + (e - spec.power_W) / cop})
    return pd.DataFrame(rows)


# ================================================================================================== the welds
SHEET_MATERIALS = {
    "steel": {"solid": th.MILD_STEEL.with_(name="S235JR-class mild steel"), "absorption": "abs_keyhole_steel", "T_haz": 996.0,
              "alloy": "S235JR / DC01 (unalloyed low-carbon sheet, uncoated); a zinc coating is a separate defect driver",
              "defects": "zinc outgassing if coated; hardness in the HAZ at short t8/5 (low carbon: tolerant); humping and undercut at high speed; lack of fusion in lap joints with a gap"},
    "aluminium": {"solid": th.ALUMINIUM.with_(name="EN AW-5754-class aluminium"), "absorption": "abs_keyhole_al", "T_haz": 573.0,
                  "alloy": "EN AW-5754 (AlMg3, H22/H111); 6082 would add hot-cracking sensitivity and need 4043/5356 filler",
                  "defects": "hydrogen porosity (oxide, moisture); keyhole collapse porosity; magnesium loss and spatter; hot cracking in 6xxx (not 5754); softening of the HAZ in H-tempers"},
}
THICKNESSES = (1e-3, 2e-3, 3e-3)
MARGIN = 1.2                                            # the depth in hand over the sheet (si.welds' rule)
HUMPING_SPEED = 0.1                                     # m/s: above ~6 m/min a narrow keyhole weld humps (literature's rule of thumb; a trial decides)


def weld_case(spec: Spec, family: str, thickness: float, opt: dict | None = None) -> dict:
    """One sheet under one laser: the keyhole threshold at the laser's spot, the speed that puts ``MARGIN``× the sheet's
    thickness of depth (nan if none does), the heat input, the thin-plate field's pool and cooling (t8/5 for steel, the
    cooling rate at the melt for aluminium), the HAZ, the distortion indicator (angular distortion ∝ heat input / t²,
    Watanabe–Satoh, relative to the 2 mm steel case at Svarog's 1 kW) and the defect flags — with the speed's interval
    over the absorptivity and melting-efficiency ranges. A model: no penetration or strength here is validated."""
    m = SHEET_MATERIALS[family]
    s = m["solid"]
    opt = optics(spec) if opt is None else opt
    P = opt["delivered_W"]
    spot = opt["spot_radius"]
    a = _a(m["absorption"])
    eta = _a("melting_eff")
    kh = wd.keyhole_threshold(s, spot)
    v = wd.speed_for_penetration(s, P, MARGIN * thickness, spot, absorption=a.nominal, melting_efficiency=eta.nominal)
    v_lo = wd.speed_for_penetration(s, P, MARGIN * thickness, spot, absorption=a.low, melting_efficiency=eta.low)
    v_hi = wd.speed_for_penetration(s, P, MARGIN * thickness, spot, absorption=a.high, melting_efficiency=eta.high)
    out = {"laser": spec.key, "material": s.name, "family": family, "thickness": thickness, "spot_radius": spot, "delivered_W": P,
           "keyhole_threshold_W": kh["P_threshold_W"], "keyhole_ratio": P / kh["P_threshold_W"], "speed": v, "speed_interval": (v_lo, v_hi), "feasible": v == v,
           "absorption": a.nominal, "melting_efficiency": eta.nominal, "validated": False, "basis": "wd.penetration energy balance + thin-plate Rosenthal; a model, not a weld"}
    if v != v:
        out.update({"heat_input_J_per_m": math.nan, "note": "no speed in the range reaches the depth with the margin"})
        return out
    pen = wd.penetration(s, P, v, spot, absorption=a.nominal, melting_efficiency=eta.nominal)
    Q = P / v
    q_abs = a.nominal * P
    pool = wd.rosenthal_thin_plate(s, q_abs, v, thickness)
    haz = wd.haz_width(s, q_abs, v, thickness, m["T_haz"])
    t85 = wd.cooling_time_t85(s, a.nominal * Q, thickness)
    rate = wd.cooling_rate_at(s, a.nominal * Q, thickness, s.T_melt - 50.0)
    out.update({"depth": pen["depth"], "width_energy_balance": pen["width"], "heat_input_J_per_m": Q, "absorbed_heat_input_J_per_m": a.nominal * Q,
                "pool_width": pool["width"], "pool_length": pool["length"], "solidification_time_s": pool["solidification_time"],
                "gradient_at_fusion_K_per_mm": pool["gradient_at_fusion_K_per_m"] * 1e-3, "peak_T_y": pool["y"], "peak_T": pool["peak_T"],
                "t85_s": t85["t85_s"], "t85_regime": t85["regime"], "cooling_rate_K_per_s": rate["rate_thin_K_per_s"],
                "haz_each_side": haz["haz_width_each_side"], "distortion_index": a.nominal * Q / (s.rho * s.c * thickness ** 2),
                "humping_risk": v > HUMPING_SPEED, "keyhole_marginal": P < 3 * kh["P_threshold_W"], "defects": m["defects"], "pool": pool})
    return out


def welding_cases(spec: Spec, opt: dict | None = None) -> list[dict]:
    opt = optics(spec) if opt is None else opt
    return [weld_case(spec, fam, t, opt) for fam in SHEET_MATERIALS for t in THICKNESSES]


def welding_table(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cases:
        r = {"laser": c["laser"], "material": c["material"], "t [mm]": c["thickness"] * 1e3, "spot [µm]": c["spot_radius"] * 1e6, "P/P_keyhole": c["keyhole_ratio"],
             "v [m/min]": c["speed"] * 60 if c["feasible"] else math.nan, "v low–high [m/min]": tuple(round(x * 60, 2) for x in c["speed_interval"]) if c["feasible"] else None,
             "feasible (model)": c["feasible"]}
        if c["feasible"]:
            r.update({"heat input [J/mm]": c["heat_input_J_per_m"] * 1e-3, "pool width [mm]": c["pool_width"] * 1e3, "pool length [mm]": c["pool_length"] * 1e3,
                      "solidification [ms]": c["solidification_time_s"] * 1e3, "t8/5 [s]": c["t85_s"], "cooling rate at melt [K/s]": c["cooling_rate_K_per_s"],
                      "gradient at fusion [K/mm]": c["gradient_at_fusion_K_per_mm"], "HAZ each side [mm]": c["haz_each_side"] * 1e3,
                      "distortion index [K]": c["distortion_index"], "humping risk": c["humping_risk"], "keyhole marginal": c["keyhole_marginal"]})
        rows.append(r)
    return pd.DataFrame(rows)


def welding_uncertainty(cases: list[dict]) -> pd.DataFrame:
    """What each indicator hangs on, and how far the assumptions' ranges move it (the speed interval is the summary)."""
    rows = []
    for c in cases:
        if not c["feasible"]:
            continue
        lo, hi = c["speed_interval"]
        rows.append({"laser": c["laser"], "material": c["material"], "t [mm]": c["thickness"] * 1e3, "speed spread [%]": 100 * (hi - lo) / c["speed"] if lo == lo and hi == hi else math.nan,
                     "hangs on": "keyhole absorptivity, melting efficiency (the two numbers a trial recalibrates: svarog_todo M.3)", "field model": "thin-plate line source, no losses, constant properties",
                     "validated": "no"})
    return pd.DataFrame(rows)


# ================================================================================================== shielding
@dataclass(frozen=True)
class Shield:
    family: str
    variant: str
    solid: th.Solid
    thickness: float
    evidence: str                                       # "none" | "certified-rating-required": what a PASS would need
    note: str = ""

    @property
    def name(self) -> str:
        return f"{self.family}: {self.variant} ({self.solid.name} {self.thickness * 1e3:g} mm)"


SHIELDS = {
    "polycarbonate": [Shield("polycarbonate", "clear", th.CLEAR_PC, 5e-3, "none", "transmits at 1 µm: no barrier"),
                      Shield("polycarbonate", "opaque", th.BLACK_PC, 5e-3, "none", "a black panel: absorbs at the face, chars"),
                      Shield("polycarbonate", "certified laser-protective", th.LASER_PC, 5e-3, "certified-rating-required", "dyed for 1 µm; PASS only with its EN 12254 protection level for this irradiance and time")],
    "aluminium": [Shield("aluminium", "bare", th.ALUMINIUM, 1.5e-3, "none", "a reflective sheet: what it does not absorb it sends elsewhere"),
                  Shield("aluminium", "oxidised", th.OXIDISED_ALUMINIUM, 1.5e-3, "none"),
                  Shield("aluminium", "coated", th.COATED_ALUMINIUM, 1.5e-3, "none", "the paint takes the beam first")],
    "wood": [Shield("wood", "solid", th.WOOD, 12e-3, "none", "chars, then burns"),
             Shield("wood", "plywood", th.PLYWOOD, 12e-3, "none"),
             Shield("wood", "fire-retardant", th.FR_WOOD, 12e-3, "none", "the retardant's class (EN 13501-1) is product data; the laser heating is the untreated board's")],
}
BASE_VARIANTS = {fam: lst[0].variant for fam, lst in SHIELDS.items()}
SHIELD_DISTANCES = (5.0, 10.0, 20.0, 50.0, 100.0)
BEAM_KINDS = ("collimated", "reflected")


def mpe_eye_1070(t, verify: bool = True) -> np.ndarray:
    """The eye's MPE [W/m²] at 1050–1400 nm (C7 = 1 below 1150 nm) as an irradiance over an exposure of ``t`` seconds:
    H = 90 t^0.75 J/m² for 5 µs ≤ t < 10 s (so E = 90 t^−0.25), E = 50 W/m² for 10 s ≤ t ≤ 30 000 s — IEC 60825-1 Table
    A.1, point source. Outside that span nan. ``STANDARDS`` says which edition the constants were read from: verify."""
    t = np.asarray(t, dtype=float)
    E = np.where(t < 10.0, 90.0 * t ** -0.25, 50.0)
    return np.where((t >= 5e-6) & (t <= 3e4), E, math.nan)


def mpe_skin_1070(t) -> np.ndarray:
    """The skin's MPE [W/m²] at 400–1400 nm with C4 = 5 (1050–1400 nm): H = 11 000 t^0.25 C4 J/m² for 1e-7 ≤ t < 10 s,
    E = 2000 C4 W/m² from 10 s — IEC 60825-1 Table A.5 (verify the edition)."""
    t = np.asarray(t, dtype=float)
    E = np.where(t < 10.0, 11000.0 * 5 * t ** -0.75, 2000.0 * 5)
    return np.where((t >= 1e-7) & (t <= 3e4), E, math.nan)


def nohd(opt: dict, mpe_W_per_m2: float = 50.0, kind: str = "collimated") -> float:
    """The distance [m] at which the stray beam's mean irradiance falls to ``mpe``: for the collimated beam
    (√(P/πE) − W_c)/θ_c; for the diverging beam √(P/πE)/θ. Air left out (it only shortens it by per cents)."""
    P = opt["delivered_W"]
    r = math.sqrt(P / (math.pi * mpe_W_per_m2))
    if kind == "collimated":
        return max(r - opt["collimated_radius"], 0.0) / opt["collimated_divergence"]
    return r / opt["cone_half_angle"]


def od_required(irradiance_W_per_m2: float, mpe_W_per_m2: float) -> dict:
    """The optical density that brings ``irradiance`` under the MPE, and EN 207's D-mode (continuous) scale number: LB n is
    tested at 10^(n+3) W/m² for 315–1400 nm — the ladder a protective window or eyewear is specified on (verify the
    edition; the certified product's own marking decides)."""
    od = max(math.log10(irradiance_W_per_m2 / mpe_W_per_m2), 0.0) if irradiance_W_per_m2 > 0 else 0.0
    lb = max(math.ceil(math.log10(irradiance_W_per_m2) - 3), 1) if irradiance_W_per_m2 > 0 else 0
    return {"od_required": od, "en207_lb_scale": lb, "en207_tested_at_W_per_m2": 10 ** (lb + 3)}


def stray_beam(spec: Spec, distance: float, kind: str = "collimated", opt: dict | None = None) -> si.StrayBeam:
    """The laser's stray beam at ``distance`` in svarog_industrial's terms: the raw collimated beam (the full delivered
    power, the worst case: it escapes before the focusing optic or a reflector re-collimates it) or a specular
    reflection off the workpiece (the beam through the focus, ``workpiece_reflectivity`` of the power, diverging at the
    head's cone)."""
    opt = optics(spec) if opt is None else opt
    r = float(beam_radius_at(opt, distance, kind))
    if kind == "collimated":
        return si.StrayBeam(f"{spec.key} raw collimated beam", opt["delivered_W"], r, "tophat", "the full power in a beam that grows by its own divergence only")
    return si.StrayBeam(f"{spec.key} specular reflection", _a("workpiece_reflectivity").nominal * opt["delivered_W"], r, "tophat",
                        "the focused beam reflected by a flat workpiece, diverging at the head's cone")


def thermal_response(shield: Shield, beam: si.StrayBeam, T0: float = th.T_ROOM) -> dict:
    """The barrier's heating without destroying it: the absorbed irradiance, the semi-infinite surface rise at T1 and T2
    (an upper bound for a thick, lossless solid), the lumped plate's rise at T1 (no losses), the steady upper bound of
    ``si.plate_steady_bound`` — each against the material's damage point (melting, or charring/decomposition). The
    ratio ``bound_ratio`` = the smallest of the bounds over the damage rise: below 1 the closed forms say it cannot reach
    damage; above 1 nothing is concluded (a bound is not a prediction) and only a certified test decides."""
    s, d = shield.solid, shield.thickness
    q = s.absorptivity * beam.mean_irradiance
    damage_T = s.T_melt if s.failure == "melt" else s.T_fail
    rise_T1 = float(th.semi_infinite_surface_rise(s, q, EXPOSURE["T1"]))
    rise_T2 = float(th.semi_infinite_surface_rise(s, q, EXPOSURE["T2"]))
    lumped_T1 = q * EXPOSURE["T1"] / (s.rho * s.c * d)
    steady = si.plate_steady_bound(s, d, s.absorptivity * beam.power, beam.radius, max(8 * beam.radius, 12 * d, 0.3))
    smallest = min(rise_T1, lumped_T1, steady)
    return {"absorbed_irradiance_W_per_cm2": q / 1e4, "damage_T": damage_T, "damage_rise_K": damage_T - T0, "surface_rise_T1_bound_K": rise_T1,
            "surface_rise_T2_bound_K": rise_T2, "lumped_rise_T1_K": lumped_T1, "steady_rise_bound_K": steady, "bound_ratio": smallest / (damage_T - T0),
            "cannot_reach_damage": smallest < damage_T - T0, "transmitted_fraction": s.transmitted(d), "reflected_fraction": max(1.0 - s.absorptivity - s.transmitted(d), 0.0)}


def fire_risk(shield: Shield, beam: si.StrayBeam) -> dict:
    """Wood only: the absorbed flux against the piloted and unpiloted ignition fluxes (fire-safety handbook classes), the
    level and the note; other families: not applicable."""
    if shield.family != "wood":
        return {"fire_risk": "n/a", "fire_note": "not a combustible panel"}
    q = shield.solid.absorptivity * beam.mean_irradiance
    pil, spon = _a("wood_piloted_flux"), _a("wood_spontaneous_flux")
    if q >= spon.low:
        lvl = "high: above the unpiloted-ignition flux"
    elif q >= pil.low:
        lvl = "high: above the piloted-ignition flux (any spark or flame ignites it)"
    elif q >= 0.3 * pil.low:
        lvl = "moderate: a third of the piloted flux — chars under long exposure"
    else:
        lvl = "low: below a third of the piloted-ignition flux"
    note = "fire-retardant treatment: the ignition flux rises by the product's tested factor only — product data needed (INCOMPLETE)" if shield.variant == "fire-retardant" else ""
    return {"fire_risk": lvl, "fire_flux_ratio_piloted": q / pil.nominal, "fire_note": note}


def seam_leak(beam: si.StrayBeam, gap_m: float | None = None) -> dict:
    """A seam of width ``gap`` across the beam: the share of the power that passes (a strip through a top-hat disc)."""
    g = _a("seam_gap_mm").nominal * 1e-3 if gap_m is None else gap_m
    r = beam.radius
    share = min(1.0, g * 2 * r / (math.pi * r ** 2)) if r > 0 else 1.0
    return {"seam_gap_m": g, "seam_power_W": share * beam.power, "seam_share": share}


def verdict(case: dict) -> tuple[str, str]:
    """PASS / FAIL / INCOMPLETE for one barrier case. FAIL: the barrier transmits more than 1 %, or the beam's irradiance
    at it is above the MPE and nothing certified stands in the way of a person behind it once it is gone (a clear
    sheet), or the case is a seam. PASS: only a certified product whose protection level covers this irradiance and
    exposure time — evidence this module does not hold, so it never returns PASS by itself; a case with
    ``evidence == "certified"`` (the rating entered by hand) can. Everything else is INCOMPLETE: a simulation bound is
    not protection evidence (IEC 60825-4 asks for the test)."""
    if case.get("seam"):
        return "FAIL", "a seam passes the beam: geometry, not material"
    if case["transmitted_fraction"] > 0.01:
        return "FAIL", f"transmits {100 * case['transmitted_fraction']:.0f} % at 1 µm"
    if case["evidence"] == "certified":
        return "PASS", "certified protection level covers this irradiance and exposure (entered evidence)"
    if case["evidence"] == "certified-rating-required":
        return "INCOMPLETE", f"needs the product's EN 12254 level for {case['irradiance_W_per_cm2']:.1f} W/cm² over {case['exposure_s']:.0f} s"
    if case["cannot_reach_damage"]:
        return "INCOMPLETE", "the closed-form bounds stay under the damage point, but a bound is not a test: certify by EN 12254 / IEC 60825-4 testing"
    return "INCOMPLETE", "the bounds exceed the damage point: no safety can be claimed; a tested barrier is required"


def shield_case(spec: Spec, shield: Shield, distance: float, kind: str = "collimated", exposure: str = "T2", opt: dict | None = None) -> dict:
    opt = optics(spec) if opt is None else opt
    beam = stray_beam(spec, distance, kind, opt)
    t = EXPOSURE[exposure]
    mpe = float(mpe_eye_1070(t))
    E = beam.mean_irradiance
    resp = thermal_response(shield, beam)
    od = od_required(E, mpe)
    reflected_W = resp["reflected_fraction"] * beam.power
    case = {"laser": spec.key, "family": shield.family, "variant": shield.variant, "barrier": shield.name, "base": shield.variant == BASE_VARIANTS[shield.family],
            "distance_m": distance, "beam": kind, "power_W": beam.power, "beam_radius_mm": beam.radius * 1e3, "irradiance_W_per_cm2": E / 1e4, "exposure": exposure, "exposure_s": t,
            "mpe_eye_W_per_m2": mpe, "mpe_skin_W_per_m2": float(mpe_skin_1070(t)), "over_mpe_eye": E / mpe, "over_mpe_skin": E / float(mpe_skin_1070(t)),
            "nohd_m": nohd(opt, mpe, kind), **od, **resp, "reflected_W": reflected_W, "transmitted_W": resp["transmitted_fraction"] * beam.power,
            **fire_risk(shield, beam), **seam_leak(beam), "evidence": shield.evidence, "interlock_required": True,
            "uncertainty": "absorptivity at 1 µm (×2 either way on metals and wood), the bounds' lossless assumptions, the MPE edition"}
    case["verdict"], case["reason"] = verdict(case)
    return case


def shield_cases(lasers: dict[str, Spec] = LASERS, distances=SHIELD_DISTANCES, variants: bool = False, kinds=("collimated",), exposure: str = "T2") -> pd.DataFrame:
    """The matrix: every laser × distance × family (the base variant of each: 60 rows for four lasers at five distances)
    and, with ``variants``, every variant (180); ``kinds`` adds the reflected beam's rows."""
    rows = []
    for k, s in lasers.items():
        opt = optics(s)
        for fam, lst in SHIELDS.items():
            for sh in (lst if variants else lst[:1]):
                for d in distances:
                    for kind in kinds:
                        rows.append(shield_case(s, sh, d, kind, exposure, opt))
    return pd.DataFrame(rows)


def verdict_matrix(df: pd.DataFrame, kind: str = "collimated") -> pd.DataFrame:
    """laser × (family, variant) rows against distance columns, the verdict in each cell."""
    sub = df[df["beam"] == kind]
    return sub.pivot_table(index=["laser", "family", "variant"], columns="distance_m", values="verdict", aggfunc="first")


def degraded_sensitivity(spec: Spec, distance: float = 10.0) -> pd.DataFrame:
    """The certified sheet with its transmission grown by the degradation factor's range, the opaque sheet with its
    absorptivity at the ends of the metals' span — how the verdict's inputs move when the barrier is not as new."""
    opt = optics(spec)
    rows = []
    sh = SHIELDS["polycarbonate"][2]
    for x in (_a("pc_degraded_transmission_x").low, _a("pc_degraded_transmission_x").nominal, _a("pc_degraded_transmission_x").high):
        s2 = sh.solid.with_(absorption_depth=sh.solid.absorption_depth * x)
        c = shield_case(spec, replace(sh, solid=s2, variant=f"certified, transmission ×{x:g}"), distance, "collimated", "T2", opt)
        rows.append({"barrier": c["variant"], "transmitted [%]": 100 * c["transmitted_fraction"], "verdict": c["verdict"], "reason": c["reason"]})
    for a in (0.12, 0.25, 0.7):
        s2 = th.ALUMINIUM.with_(absorptivity=a, name=f"aluminium, absorptivity {a}")
        c = shield_case(spec, Shield("aluminium", f"absorptivity {a}", s2, 1.5e-3, "none"), distance, "collimated", "T2", opt)
        rows.append({"barrier": c["variant"], "transmitted [%]": 0.0, "steady bound [K]": c["steady_rise_bound_K"], "reflected [W]": c["reflected_W"], "verdict": c["verdict"], "reason": c["reason"]})
    return pd.DataFrame(rows)


# ================================================================================================== the record
def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if callable(o):
        return None
    return str(o)


def record(lasers: dict[str, Spec], optic: dict, energy: dict, welds: dict, shields: pd.DataFrame, fenix: dict | None = None, debug: bool = False) -> dict:
    """The notebook's report: the specs verbatim, the assumptions with their ranges, the standards cited, and every
    section's numbers with units and the open validation list."""
    def clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in ("gaussian", "head", "fibre", "pool", "peak_T_y", "peak_T")}
    return {"meta": {"notebook": "47_svarog_iii", "debug": debug, "specs_source": next(iter(lasers.values())).source, "standards": STANDARDS,
                     "assumptions": {k: {"nominal": a.nominal, "low": a.low, "high": a.high, "unit": a.unit, "basis": a.basis} for k, a in ASSUMPTIONS.items()},
                     "units": {"power": "W", "length": "m", "irradiance": "W/m² (tables: W/cm²)", "speed": "m/s (tables: m/min)", "heat input": "J/m", "temperature": "K"},
                     "rule": "PASS needs certified protection evidence; a simulation bound gives INCOMPLETE; distance alone never establishes safety"},
            "specs": {k: s.to_dict() for k, s in lasers.items()},
            "optics": {k: clean(o) for k, o in optic.items()},
            "energy": energy,
            "welding": {k: [clean(c) for c in cs] for k, cs in welds.items()},
            "shielding": {"base_count": int(shields["base"].sum()) if "base" in shields else len(shields), "count": len(shields),
                          "cases": shields.to_dict(orient="records"), "fenix": fenix or {}},
            "validation_open": ["the MPE constants against the current IEC 60825-1 edition", "every barrier: an EN 12254 / IEC 60825-4 test report for the irradiance and exposure class",
                                "the metals' and wood's absorptivity at 1 µm, measured on the actual panels (svarog_todo M.1)", "the welds: trials per the test matrix (svarog_todo M.3) before any penetration or strength is quoted",
                                "the sheets' unknowns (YLS-SM and WFL-3000 electrical numbers, FL2000C/CSM M²) from the makers' current data"]}


def write_csvs(out_dir: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p)
        paths.append(p)
    return paths


# ================================================================================================== endurance: how long a shield stands
ENDURANCE_SHEETS = {"polycarbonate": th.BLACK_PC, "steel": th.MILD_STEEL, "aluminium": th.ALUMINIUM}
ENDURANCE_THICKNESSES = (1e-3, 2e-3, 3e-3)


def endurance(spec: Spec, distances=SHIELD_DISTANCES, sheets: dict | None = None, thicknesses=ENDURANCE_THICKNESSES, run: bool = True, opt: dict | None = None,
              sheet_half: float = si.SHEET_HALF, t_max: float = si.SIMULATION_CAP_S) -> list[dict]:
    """Notebook 46 Part IV §3's question for a commercial laser: how long a sheet of ``thickness`` holds the raw collimated
    beam at each distance — ``si.screen`` (the closed forms: transparent → 0 s; a steady or energy bound → for ever; else
    the steady solve), the enthalpy/ablation transient (``th.heat_plate``, the grid ending at the sheet's edge) only where
    the screening cannot decide, ``si.verdict`` for ``holds_s`` and the IEC 60825-4 classes T1/T2/T3. This is the guard
    *designer's* number — the time a panel buys before the beam is through — and no part of the PASS/FAIL/INCOMPLETE
    verdicts: a time from a cell model certifies nothing (the standard's test on the panel does). ``run=False`` leaves
    the undecided cases at nan."""
    opt = optics(spec) if opt is None else opt
    sheets = ENDURANCE_SHEETS if sheets is None else sheets
    rows = []
    for fam, solid in sheets.items():
        for d_sheet in thicknesses:
            b = si.Barrier(solid, d_sheet)
            for d in distances:
                beam = stray_beam(spec, d, "collimated", opt)
                scr = si.screen(b, beam)
                sim = None
                if scr["decision"] == "simulate" and run:
                    sim = th.heat_plate(solid, d_sheet, beam.power, beam.radius, t_max, shape=beam.shape, h=10.0, nr=si.NR_GUARD, nz=si._nz(d_sheet), r_max=sheet_half)
                v = si.verdict(scr, sim)
                rows.append({"laser": spec.key, "family": fam, "sheet": b.name, "thickness_mm": d_sheet * 1e3, "distance_m": d, "beam_radius_mm": beam.radius * 1e3,
                             "irradiance_W_per_cm2": beam.mean_irradiance / 1e4, "absorbed_W_per_cm2": scr["absorbed_irradiance_W_per_cm2"], "decision": scr["decision"],
                             "surface_damage_s": v["surface_damage_s"], "holds_s": v["holds_s"], **{k: v[k] for k in EXPOSURE}, "source": v["source"], "simulated": sim is not None,
                             "note": "a cell-model time for the guard's design (IEC 60825-4 classes); not protection evidence"})
    return rows


def endurance_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["laser", "family", "thickness_mm"], columns="distance_m", values="holds_s", aggfunc="first", dropna=False)


ENDURANCE_PATH = DATA / "svarog_commercial_endurance.json"


def endurance_record(lasers: dict[str, Spec] = LASERS, screening: bool | None = None, **kw) -> dict:
    """Every laser's endurance rows with their provenance (the grid, the cap, the sheet) — the recorded result the notebook
    reads (``ENDURANCE_PATH``): minutes per laser, so it is made by ``python svarog_commercial.py --endurance``, never
    inside the notebook."""
    import datetime
    rows = {k: endurance(s, **kw) for k, s in lasers.items()}
    return {"meta": {"made": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "screening": si.SCREENING if screening is None else screening,
                     "grid": {"nr": si.NR_GUARD, "nz": "si._nz(thickness)", "r_max": si.SHEET_HALF, "t_max_s": si.SIMULATION_CAP_S, "h": 10.0},
                     "route": "si.screen → th.heat_plate (numpy enthalpy/ablation cell model) where undecided → si.verdict; the collimated beam of sc.stray_beam",
                     "sheets": {k: v.name for k, v in ENDURANCE_SHEETS.items()}, "thicknesses_m": list(ENDURANCE_THICKNESSES), "distances_m": list(SHIELD_DISTANCES),
                     "units": {"holds_s": "s (inf = holds for ever by a bound or a stalled front; 0 = transparent)", "irradiance": "W/cm²"},
                     "caveat": "a cell-model time for the guard's design; not protection evidence (IEC 60825-4 asks for the test)"},
            "rows": {k: [{kk: (None if isinstance(vv, float) and math.isnan(vv) else ("inf" if vv == math.inf else vv)) for kk, vv in r.items()} for r in rs] for k, rs in rows.items()}}


def load_endurance(path: Path = ENDURANCE_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    for rs in d["rows"].values():
        for r in rs:
            r["holds_s"] = math.inf if r["holds_s"] == "inf" else (math.nan if r["holds_s"] is None else r["holds_s"])
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Svarog III: make the recorded results the notebook reads")
    ap.add_argument("--endurance", action="store_true", help=f"the hold times of PC/steel/aluminium sheets at 1–3 mm against 5–100 m, every laser → {ENDURANCE_PATH} (minutes per laser)")
    ap.add_argument("--laser", choices=KEYS, nargs="*", default=list(KEYS))
    args = ap.parse_args()
    if args.endurance:
        import time as _t
        t0 = _t.monotonic()
        rec = endurance_record({k: LASERS[k] for k in args.laser})
        ENDURANCE_PATH.write_text(json.dumps(rec, indent=1))
        print(f"wrote {ENDURANCE_PATH}: {sum(len(v) for v in rec['rows'].values())} rows in {_t.monotonic() - t0:.0f} s")


# ================================================================================================== per euro: the planning allowances
def load_prices(path: Path = SPECS_PATH) -> dict:
    """The user-supplied planning allowances (EUR, new, low–high) and the listings, with their source — not quotations."""
    return json.loads(Path(path).read_text()).get("prices", {})


PRICES = load_prices()
ESCAPE_TIMES = (1.0, 3.0, 5.0)                          # s: the time a person needs to leave the beam's path once a panel is hit


def price_keur(key: str, prices: dict | None = None) -> tuple[float, float]:
    p = (PRICES if prices is None else prices)["allowance_new_EUR"][key]
    return (p[0] / 1e3, p[1] / 1e3)


def speed_per_keur(cases: list[dict], prices: dict | None = None) -> pd.DataFrame:
    """Welding speed bought per thousand euro of laser source: v [m/min] / allowance [k€], at the allowance's low and high
    ends — the source alone, the same head and no machine, so a ranking of the sources, not of welding systems."""
    rows = []
    for c in cases:
        lo, hi = price_keur(c["laser"], prices)
        v = c["speed"] * 60 if c["feasible"] else math.nan
        rows.append({"laser": c["laser"], "material": c["material"], "t [mm]": c["thickness"] * 1e3, "v [m/min]": v, "allowance [k€]": f"{lo:g}–{hi:g}",
                     "v per k€, at the low allowance [m/min/k€]": v / lo, "v per k€, at the high allowance [m/min/k€]": v / hi, "basis": "model speed (unvalidated) / planning allowance (unverified)"})
    return pd.DataFrame(rows)


def shield_for_escape(endurance_rows: list[dict], escape_times=ESCAPE_TIMES, prices: dict | None = None) -> pd.DataFrame:
    """The paradox of the dearer source: per laser, sheet family and distance, the thinnest recorded sheet (1, 2 or 3 mm)
    that holds at least ``t_escape`` seconds — the time a person needs to get out of the beam's path — and that thickness
    per thousand euro of the source's allowance (the shield burden each k€ of laser buys). ``> 3 mm`` when none of the
    recorded sheets holds that long; the hold times are the cell model's (46's route), not protection evidence."""
    df = pd.DataFrame(endurance_rows)
    rows = []
    for (laser, fam, d), g in df.groupby(["laser", "family", "distance_m"]):
        lo, hi = price_keur(laser, prices)
        for t in escape_times:
            ok = g[g["holds_s"] >= t].sort_values("thickness_mm")
            need = float(ok["thickness_mm"].iloc[0]) if len(ok) else math.nan
            rows.append({"laser": laser, "family": fam, "distance_m": d, "escape_s": t, "shield_needed_mm": need, "shield_label": f"{need:g} mm" if need == need else "> 3 mm",
                         "mm per k€ (low allowance)": need / lo, "mm per k€ (high allowance)": need / hi})
    return pd.DataFrame(rows)


def escape_per_keur(endurance_rows: list[dict], prices: dict | None = None) -> pd.DataFrame:
    """The other way round: per laser, sheet (family, thickness) and distance, the seconds the sheet holds per thousand
    euro of the source's allowance — inf where a bound says the sheet holds for ever, 0 where it is transparent."""
    df = pd.DataFrame(endurance_rows)
    out = []
    for _, r in df.iterrows():
        lo, hi = price_keur(r["laser"], prices)
        out.append({"laser": r["laser"], "family": r["family"], "thickness_mm": r["thickness_mm"], "distance_m": r["distance_m"], "holds_s": r["holds_s"],
                    "s per k€ (low allowance)": r["holds_s"] / lo, "s per k€ (high allowance)": r["holds_s"] / hi})
    return pd.DataFrame(out)


# ================================================================================================== the economics, on sourced parameters
ECON = json.loads(SPECS_PATH.read_text()).get("economics", {})


def econ(key: str) -> dict:
    """One economic parameter: {low, nominal, high, unit, basis, source} from the specs file's ``economics`` block."""
    return ECON[key]


def econ_table() -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in ECON.items() if isinstance(v, dict)}).T[["low", "nominal", "high", "unit", "basis", "source"]]


def _lvl(key: str, lvl: str) -> float:
    return float(ECON[key][lvl])


def capex(spec: Spec) -> dict:
    """The source (per kW for the 3 kW), the head and the chiller: (low, nominal, high) EUR."""
    out = {}
    for lvl in ("low", "nominal", "high"):
        src = _lvl("source_capex_EUR", lvl) if spec.power_W <= 2000 else _lvl("source_capex_per_kW_EUR", lvl) * spec.power_W / 1e3
        out[lvl] = {"source": src, "head": _lvl("head_EUR", lvl), "chiller": _lvl("chiller_EUR", lvl), "total": src + _lvl("head_EUR", lvl) + _lvl("chiller_EUR", lvl)}
    return out


def opex_per_hour(spec: Spec, energy: dict) -> dict:
    """EUR per operating hour: the electricity of laser + chiller from the energy balance, the shielding gas, maintenance
    (a share of the capex per year over the hours), the pump diodes' wear (their share of the capex over their life),
    and the operator — each (low, nominal, high) with the parameter ends paired (low electricity with low tariff, …)."""
    cx = capex(spec)
    out = {}
    for lvl in ("low", "nominal", "high"):
        kw = energy["total_electrical_W"][lvl] / 1e3
        elec = kw * _lvl("electricity_EUR_per_kWh", lvl)
        gas = _lvl("argon_flow_l_per_min", lvl) * 60 / 1e3 * _lvl("argon_EUR_per_m3", lvl)
        maint = cx[lvl]["total"] * _lvl("maintenance_pct_of_capex_per_year", lvl) / 100 / _lvl("hours_per_year", "high" if lvl == "low" else ("low" if lvl == "high" else "nominal"))
        diodes = cx[lvl]["source"] * _lvl("diode_share_of_capex", lvl) / _lvl("diode_life_h", "high" if lvl == "low" else ("low" if lvl == "high" else "nominal"))
        labour = _lvl("labour_EUR_per_h", lvl)
        out[lvl] = {"electricity": elec, "gas": gas, "maintenance": maint, "diodes": diodes, "labour": labour, "machine (no labour)": elec + gas + maint + diodes,
                    "total": elec + gas + maint + diodes + labour, "kW_drawn": kw}
    return out


def cost_per_metre(cases: list[dict], energy: dict) -> pd.DataFrame:
    """EUR per metre of weld per sheet case: the hourly cost over the model speed (the arc-on time only: no handling,
    fixturing or rework — a lower bound per metre), with and without the operator, and the capex per m/min of speed."""
    rows = []
    for c in cases:
        spec = LASERS[c["laser"]]
        ox = opex_per_hour(spec, energy)
        cx = capex(spec)
        v = c["speed"] * 60 if c["feasible"] else math.nan
        rows.append({"laser": c["laser"], "material": c["material"], "t [mm]": c["thickness"] * 1e3, "v [m/min]": v,
                     "machine €/h": ox["nominal"]["machine (no labour)"], "€/h with operator": ox["nominal"]["total"],
                     "€/m machine": ox["nominal"]["machine (no labour)"] / (v * 60), "€/m with operator": ox["nominal"]["total"] / (v * 60),
                     "€/m low–high": (ox["low"]["total"] / (v * 60), ox["high"]["total"] / (v * 60)),
                     "capex €/(m/min)": cx["nominal"]["total"] / v, "basis": "model speed (unvalidated), arc-on time only; sourced parameters (ECON)"})
    return pd.DataFrame(rows)


def panel_cost_per_m2(family: str, thickness_mm: float, lvl: str = "nominal") -> float:
    """EUR per m² of a shielding panel: steel and aluminium by mass, black polycarbonate per mm, the certified sheet per m²."""
    if family == "steel":
        return _lvl("steel_sheet_EUR_per_t", lvl) / 1e3 * th.MILD_STEEL.rho * thickness_mm * 1e-3
    if family == "aluminium":
        return _lvl("aluminium_sheet_EUR_per_kg", lvl) * th.ALUMINIUM.rho * thickness_mm * 1e-3
    if family == "polycarbonate":
        return _lvl("black_pc_EUR_per_m2_per_mm", lvl) * thickness_mm
    if family == "certified":
        return _lvl("certified_sheet_EUR_per_m2", lvl)
    raise KeyError(family)


def shield_economics(endurance_rows: list[dict], escape_times=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tables on the endurance record and the panel prices. (1) Per laser, family, distance and escape time: the
    thinnest recorded sheet that holds that long, its cost per m², and that cost per thousand euro of the source's capex
    (the shield burden the dearer source carries). (2) Per sheet and distance: the escape seconds bought per euro of panel
    (per m²) — inf where a bound holds for ever. Hold times are the cell model's (46's route): design numbers, not evidence."""
    df = pd.DataFrame(endurance_rows)
    times = ECON.get("escape_times_s", [1.0, 3.0, 5.0]) if escape_times is None else escape_times
    need, per_eur = [], []
    for (laser, fam, d), g in df.groupby(["laser", "family", "distance_m"]):
        cx = capex(LASERS[laser])["nominal"]["source"] / 1e3
        for t in times:
            ok = g[g["holds_s"] >= t].sort_values("thickness_mm")
            mm = float(ok["thickness_mm"].iloc[0]) if len(ok) else math.nan
            cost = panel_cost_per_m2(fam, mm) if mm == mm else math.nan
            need.append({"laser": laser, "family": fam, "distance_m": d, "escape_s": t, "shield_needed_mm": mm, "shield_label": f"{mm:g} mm" if mm == mm else "> 3 mm",
                         "panel €/m²": cost, "panel € per k€ of source": cost / cx, "source capex k€": cx})
    for _, r in df.iterrows():
        cost = panel_cost_per_m2(r["family"], r["thickness_mm"])
        per_eur.append({"laser": r["laser"], "family": r["family"], "thickness_mm": r["thickness_mm"], "distance_m": r["distance_m"], "holds_s": r["holds_s"], "panel €/m²": cost,
                        "escape s per € of panel": r["holds_s"] / cost, "escape s per k€ of source": r["holds_s"] / (capex(LASERS[r["laser"]])["nominal"]["source"] / 1e3)})
    return pd.DataFrame(need), pd.DataFrame(per_eur)
