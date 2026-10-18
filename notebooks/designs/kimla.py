"""KIMLA as a welding platform — notebook 48: KIMLA's fibre-laser *cutting* machines (Finecut, Flashcut, Powercut,
Extremecut: up to 6, 12, 20, 30 kW as found) taken as hypothetical welding platforms at 3, 6, 12, 20 and 30 kW — analysis
scenarios, not orderable variants — with a welding head of stated optics in place of the cutting head. Two questions on
the physics Svarog's device and notebook 47's commercial lasers were judged on: the close-range manufacturing weld at the
head's normal focus (carbon steel, stainless, aluminium at 1, 2, 3 mm; butt and fillet joints; the regime, the speed
interval, the pool, the cooling, the HAZ, the distortion index, the throughput with its handling overheads and the euro
per metre) and the stray beam at 20–200 m (the beam past the focus and the specular reflections, against the eye's and
the skin's exposure limits; the collimated fault separately; barriers INCOMPLETE until certified).

Multimode at every tier — the 2 kW single-mode optics of notebook 47 are **not** applied here: each power class carries
its own process fibre and BPP range (``kimla_specs.json``, assumptions marked). Nothing is validated; the trials decide."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vegeta.hyperion import thermal as th
from vegeta.hyperion import welding as wd

import svarog_commercial as sc
import svarog_industrial as si

DATA = Path(__file__).resolve().parent / "data"
SPECS_PATH = DATA / "kimla_specs.json"
SPECS = json.loads(SPECS_PATH.read_text())
LAM = 1070e-9
EXPOSURE = si.EXPOSURE
ASSUMPTIONS = {k: sc.Assumption(k, v["nominal"], v["low"], v["high"], v["unit"], v["basis"]) for k, v in SPECS["assumptions"].items()}


def _a(key: str) -> sc.Assumption:
    return ASSUMPTIONS[key]


def assumptions_table() -> pd.DataFrame:
    return pd.DataFrame({k: {"low": a.low, "nominal": a.nominal, "high": a.high, "unit": a.unit, "basis": a.basis} for k, a in ASSUMPTIONS.items()}).T


# ================================================================================================== the scenarios
@dataclass(frozen=True)
class Scenario:
    key: str
    power_W: float
    fibre: wd.DeliveryFibre
    bpp_low: float
    bpp_nominal: float
    bpp_high: float
    basis: str
    head: wd.WeldHead

    @property
    def bpp_mm_mrad(self) -> float:
        return self.bpp_nominal

    def to_dict(self) -> dict:
        return {"key": self.key, "power_W": self.power_W, "fibre": self.fibre.to_dict(), "bpp_mm_mrad": [self.bpp_low, self.bpp_nominal, self.bpp_high], "basis": self.basis,
                "head": self.head.to_dict(), "status": "analysis scenario on a hypothetical welding head; not an orderable KIMLA variant"}


def _scenarios() -> dict[str, Scenario]:
    h = SPECS["head"]
    head = wd.WeldHead(h["f_collimator_m"], h["f_focus_m"][1], "hypothetical welding head 100/200 on the cutting gantry")
    out = {}
    for k, s in SPECS["scenarios"].items():
        fibre = wd.DeliveryFibre(s["fibre_core_um"] * 1e-6 / 2, s["fibre_NA"], 20.0, f"{s['fibre_core_um']}/{s['fibre_core_um'] + 25} process fibre")
        lo, nom, hi = s["bpp_mm_mrad"]
        out[k] = Scenario(k, float(s["power_W"]), fibre, lo, nom, hi, s["basis"], head)
    return out


SCENARIOS = _scenarios()
KEYS = tuple(SCENARIOS)
COL = {"3 kW": "tab:green", "6 kW": "tab:blue", "12 kW": "tab:orange", "20 kW": "tab:red", "30 kW": "tab:purple"}


def optics(s: Scenario) -> dict:
    """The core imaged through the head (multimode at every tier): the spot, the cone, the Rayleigh length, the
    collimated radius and its growth (the fibre's BPP over the collimated radius), the delivered power and the
    irradiance at the focus — with the BPP range's spot interval (the spot a fibre of that BPP would image at the
    head's NA: r_spot = BPP / NA_head)."""
    head, f = s.head, s.fibre
    spot = head.spot_radius(f)
    cone = head.cone_half_angle(f)
    w_c = head.collimated_radius(f)
    eff = _a("optical_efficiency")
    P = s.power_W * eff.nominal
    theta_c = f.bpp / w_c                                              # the collimated beam's divergence: BPP over its radius
    spot_lo, spot_hi = s.bpp_low * 1e-6 / cone, s.bpp_high * 1e-6 / cone
    return {"scenario": s.key, "power_W": s.power_W, "delivered_W": P, "delivered_interval_W": (s.power_W * eff.low, s.power_W * eff.high), "fibre": f.name, "bpp_mm_mrad": f.bpp * 1e6,
            "bpp_range_mm_mrad": (s.bpp_low, s.bpp_high), "m2": f.m2(LAM), "spot_radius": spot, "spot_interval": (spot_lo, spot_hi), "cone_half_angle": cone, "rayleigh_length": spot / cone,
            "collimated_radius": w_c, "collimated_divergence": theta_c, "irradiance_MW_per_cm2": P / (math.pi * spot ** 2) / 1e10, "regime": "multimode: the core is imaged (geometric optics)"}


def optics_table() -> pd.DataFrame:
    rows = {}
    for k, s in SCENARIOS.items():
        o = optics(s)
        rows[k] = {"power [kW]": s.power_W / 1e3, "fibre": o["fibre"], "BPP [mm·mrad]": o["bpp_mm_mrad"], "BPP range": o["bpp_range_mm_mrad"], "M²": o["m2"], "spot radius [µm]": o["spot_radius"] * 1e6,
                   "spot range [µm]": tuple(round(x * 1e6) for x in o["spot_interval"]), "cone half-angle [mrad]": o["cone_half_angle"] * 1e3, "Rayleigh [mm]": o["rayleigh_length"] * 1e3,
                   "collimated radius [mm]": o["collimated_radius"] * 1e3, "collimated divergence [µrad]": o["collimated_divergence"] * 1e6, "delivered [W]": o["delivered_W"],
                   "irradiance at focus [MW/cm²]": o["irradiance_MW_per_cm2"], "basis": s.basis}
    return pd.DataFrame(rows).T


# ================================================================================================== the welds
MATERIALS = {
    "carbon steel": {"solid": th.MILD_STEEL.with_(name="S235JR-class carbon steel"), "absorption": "abs_keyhole_carbon_steel", "reflectivity": "reflectivity_carbon_steel", "T_haz": 996.0,
                     "gas": "nitrogen or argon", "defects": "zinc if coated; HAZ hardness at short t8/5; humping at speed; undercut"},
    "stainless": {"solid": th.STAINLESS.with_(name="1.4301 / 304 stainless"), "absorption": "abs_keyhole_stainless", "reflectivity": "reflectivity_stainless", "T_haz": 1123.0,
                  "gas": "argon (root shielding for colour)", "defects": "hot cracking in fully austenitic grades; sensitisation at slow cooling; oxidation without root gas"},
    "aluminium": {"solid": th.ALUMINIUM.with_(name="EN AW-5754-class aluminium"), "absorption": "abs_keyhole_aluminium", "reflectivity": "reflectivity_aluminium", "T_haz": 573.0,
                  "gas": "argon or argon–helium", "defects": "hydrogen porosity; keyhole-collapse porosity; magnesium loss; HAZ softening in H-tempers"},
}
THICKNESSES = (1e-3, 2e-3, 3e-3)
JOINTS = {"butt": {"depth_factor": 1.2, "note": "square butt, no gap: full penetration with 20 % in hand"},
          "fillet": {"depth_factor": 1.2 * math.sqrt(2), "note": "a fillet welded from the corner: the throat (t/√2 per leg) asks √2·t of depth along the bisector, with 20 % in hand; an edge-fillet lap is cheaper"}}


def weld_case(s: Scenario, material: str, thickness: float, joint: str = "butt", opt: dict | None = None) -> dict:
    """One joint under one scenario at the head's focus: the regime (conduction below the keyhole threshold, keyhole
    above; the ratio), the speed that puts the joint's depth (``wd.speed_for_penetration``, the absorptivity and
    melting-efficiency ranges giving the interval) **capped** by the gantry's feed and the humping onset (the cap
    named), the heat input, the thin-plate field (pool, solidification, gradient), t8/5 or the cooling rate, the HAZ,
    the distortion index, and the model's limitations where it leaves its range (a pool wider than the sheet is thick,
    a keyhole deeper than its width allows, conduction mode). Unvalidated."""
    m = MATERIALS[material]
    sol = m["solid"]
    opt = optics(s) if opt is None else opt
    P, spot = opt["delivered_W"], opt["spot_radius"]
    a, eta = _a(m["absorption"]), _a("melting_efficiency")
    depth_needed = JOINTS[joint]["depth_factor"] * thickness
    kh = wd.keyhole_threshold(sol, spot)
    rng = (1e-3, 5.0)
    v_nom = wd.speed_for_penetration(sol, P, depth_needed, spot, v_range=rng, absorption=a.nominal, melting_efficiency=eta.nominal)
    v_lo = wd.speed_for_penetration(sol, P, depth_needed, spot, v_range=rng, absorption=a.low, melting_efficiency=eta.low)
    v_hi = wd.speed_for_penetration(sol, P, depth_needed, spot, v_range=rng, absorption=a.high, melting_efficiency=eta.high)
    feed, hump = _a("max_feed_m_per_s"), _a("humping_m_per_s")
    cap_v, cap = min(feed.nominal, hump.nominal), ("humping onset" if hump.nominal < feed.nominal else "gantry feed")
    out = {"scenario": s.key, "power_W": s.power_W, "material": sol.name, "family": material, "thickness": thickness, "joint": joint, "depth_needed": depth_needed, "spot_radius": spot,
           "keyhole_threshold_W": kh["P_threshold_W"], "keyhole_ratio": P / kh["P_threshold_W"], "regime": "keyhole" if P >= kh["P_threshold_W"] else "conduction",
           "v_model": v_nom, "v_model_interval": (v_lo, v_hi), "feasible": v_nom == v_nom, "validated": False, "gas": m["gas"]}
    if v_nom != v_nom:
        out.update({"speed": math.nan, "speed_interval": (math.nan, math.nan), "cap": "none: no speed reaches the depth", "limitation": "the energy balance finds no speed in range"})
        return out
    v = min(v_nom, cap_v)
    v_int = (min(v_lo, cap_v) if v_lo == v_lo else math.nan, min(v_hi, cap_v) if v_hi == v_hi else math.nan)
    capped = v_nom > cap_v
    pen = wd.penetration(sol, P, v, spot, absorption=a.nominal, melting_efficiency=eta.nominal)
    Q = P / v
    q_abs = a.nominal * P
    pool = wd.rosenthal_thin_plate(sol, q_abs, v, thickness)
    haz = wd.haz_width(sol, q_abs, v, thickness, m["T_haz"])
    t85 = wd.cooling_time_t85(sol, a.nominal * Q, thickness)
    rate = wd.cooling_rate_at(sol, a.nominal * Q, thickness, sol.T_melt - 50.0)
    lim = []
    if capped:
        lim.append(f"speed capped by the {cap} ({cap_v * 60:.0f} m/min): the depth at the cap is {pen['depth'] * 1e3:.1f} mm — more than the joint needs; the surplus power is spatter and a wider pool, or a defocus the model does not cover")
    if t85["regime"] != "thin":
        lim.append("thick-plate regime by EN 1011-2: the thin-plate field over-estimates the width")
    if pool["width"] < thickness:
        lim.append("the pool is narrower than the sheet is thick: a deep keyhole, where the 2-D line source is only a lateral envelope")
    if out["regime"] == "conduction":
        lim.append("conduction mode: the keyhole energy balance does not apply; Rosenthal's pool is the estimate")
    if pen["depth"] / max(pen["width"], 1e-9) > 10:
        lim.append("keyhole aspect ratio > 10: beyond the stable-keyhole range of the energy balance")
    out.update({"speed": v, "speed_interval": v_int, "capped": capped, "cap": cap if capped else "none", "cap_speed": cap_v, "depth_at_speed": pen["depth"], "width_energy_balance": pen["width"],
                "heat_input_J_per_m": Q, "absorbed_heat_input_J_per_m": a.nominal * Q, "pool_width": pool["width"], "pool_length": pool["length"], "solidification_time_s": pool["solidification_time"],
                "gradient_at_fusion_K_per_mm": pool["gradient_at_fusion_K_per_m"] * 1e-3, "t85_s": t85["t85_s"], "t85_regime": t85["regime"], "cooling_rate_K_per_s": rate["rate_thin_K_per_s"],
                "haz_each_side": haz["haz_width_each_side"], "distortion_index": a.nominal * Q / (sol.rho * sol.c * thickness ** 2), "limitation": "; ".join(lim) if lim else "within the model's range",
                "defects": m["defects"]})
    return out


def weld_matrix(scenarios: dict[str, Scenario] = SCENARIOS, materials=tuple(MATERIALS), thicknesses=THICKNESSES, joints=tuple(JOINTS)) -> list[dict]:
    out = []
    for k, s in scenarios.items():
        opt = optics(s)
        for mat in materials:
            for t in thicknesses:
                for j in joints:
                    out.append(weld_case(s, mat, t, j, opt))
    return out


def weld_table(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cases:
        r = {"scenario": c["scenario"], "material": c["material"], "t [mm]": c["thickness"] * 1e3, "joint": c["joint"], "regime": c["regime"], "P/P_keyhole": c["keyhole_ratio"],
             "v model [m/min]": c["v_model"] * 60 if c["feasible"] else math.nan, "v [m/min]": c["speed"] * 60 if c["feasible"] else math.nan,
             "v low–high [m/min]": tuple(round(x * 60, 1) for x in c["speed_interval"]) if c["feasible"] else None, "cap": c["cap"]}
        if c["feasible"]:
            r.update({"depth at v [mm]": c["depth_at_speed"] * 1e3, "energy [J/mm]": c["heat_input_J_per_m"] * 1e-3, "pool width [mm]": c["pool_width"] * 1e3, "pool length [mm]": c["pool_length"] * 1e3,
                      "solidification [ms]": c["solidification_time_s"] * 1e3, "t8/5 [s]": c["t85_s"], "cooling rate [K/s]": c["cooling_rate_K_per_s"], "HAZ each side [mm]": c["haz_each_side"] * 1e3,
                      "distortion index [K]": c["distortion_index"], "limitation": c["limitation"]})
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================================================================== throughput and the euro
OVERHEADS = {k: v for k, v in SPECS["overheads_s"].items() if k != "basis"}
JOBS = SPECS["jobs"]


def throughput(case: dict, job: dict, overheads: dict | None = None) -> dict:
    """One job (``seams``, ``length_m``) on one weld case: the arc-on time (length / speed) and the practical time with
    the handling overheads (load, clamp, tacks per seam and per metre, repositioning per seam, inspection per part and
    per metre, unload) — parts per hour both ways, and the practical share that is arc-on."""
    o = OVERHEADS if overheads is None else overheads
    if not case["feasible"]:
        return {"arc_on_s": math.nan, "practical_s": math.nan, "parts_per_h_arc_on": math.nan, "parts_per_h_practical": math.nan, "arc_on_share": math.nan}
    arc = job["length_m"] / case["speed"]
    over = (o["load_s_per_part"] + o["clamp_s_per_part"] + o["unload_s_per_part"] + o["inspection_s_per_part"] + o["inspection_s_per_m"] * job["length_m"]
            + job["seams"] * (o["tack_s_per_seam"] + o["reposition_s_per_seam"]) + o["tack_s_per_m"] * job["length_m"])
    return {"arc_on_s": arc, "overhead_s": over, "practical_s": arc + over, "parts_per_h_arc_on": 3600 / arc, "parts_per_h_practical": 3600 / (arc + over), "arc_on_share": arc / (arc + over),
            "metres_per_h_practical": job["length_m"] * 3600 / (arc + over)}


def capex(s: Scenario) -> dict:
    src, mach = _a("source_capex_EUR_per_kW"), _a("machine_capex_EUR")
    return {lvl: {"source": getattr(src, lvl) * s.power_W / 1e3, "machine": getattr(mach, lvl), "head": sc._lvl("head_EUR", lvl), "chiller": sc._lvl("chiller_EUR", lvl) * max(1.0, s.power_W / 2000),
                  "total": getattr(src, lvl) * s.power_W / 1e3 + getattr(mach, lvl) + sc._lvl("head_EUR", lvl) + sc._lvl("chiller_EUR", lvl) * max(1.0, s.power_W / 2000)}
            for lvl in ("low", "nominal", "high")}


def opex_per_hour(s: Scenario) -> dict:
    """EUR per operating hour: electricity (the power over the wall-plug, the chiller over notebook 47's COP), the
    shielding gas, maintenance and the diodes (notebook 47's ECON shares), the operator — low/nominal/high."""
    cx = capex(s)
    out = {}
    for lvl in ("low", "nominal", "high"):
        anti = "high" if lvl == "low" else ("low" if lvl == "high" else "nominal")
        elec_kw = s.power_W / getattr(_a("wall_plug"), anti) / 1e3
        heat_kw = elec_kw - s.power_W / 1e3
        chiller_kw = heat_kw / sc._lvl("cooling_COP", anti) if "cooling_COP" in sc.ECON else heat_kw / sc._a("cooling_COP").__getattribute__(anti)
        elec = (elec_kw + chiller_kw) * sc._lvl("electricity_EUR_per_kWh", lvl)
        gas = getattr(_a("gas_flow_l_per_min"), lvl) * 60 / 1e3 * getattr(_a("gas_EUR_per_m3"), lvl)
        maint = cx[lvl]["total"] * sc._lvl("maintenance_pct_of_capex_per_year", lvl) / 100 / sc._lvl("hours_per_year", anti)
        diodes = cx[lvl]["source"] * sc._lvl("diode_share_of_capex", lvl) / sc._lvl("diode_life_h", anti)
        labour = sc._lvl("labour_EUR_per_h", lvl)
        out[lvl] = {"electricity": elec, "gas": gas, "maintenance": maint, "diodes": diodes, "labour": labour, "machine": elec + gas + maint + diodes, "total": elec + gas + maint + diodes + labour,
                    "kW_drawn": elec_kw + chiller_kw}
    return out


def economics_table(cases: list[dict], overheads: dict | None = None) -> pd.DataFrame:
    """Per weld case and job: €/metre arc-on (the hourly cost over the speed) and practical (over the practical metres
    per hour), parts per hour both ways — the nominal and the low–high ends."""
    rows = []
    ox = {k: opex_per_hour(s) for k, s in SCENARIOS.items()}
    for c in cases:
        if not c["feasible"]:
            continue
        o = ox[c["scenario"]]
        for jk, job in JOBS.items():
            tp = throughput(c, job, overheads)
            rows.append({"scenario": c["scenario"], "material": c["material"], "t [mm]": c["thickness"] * 1e3, "joint": c["joint"], "job": jk, "v [m/min]": c["speed"] * 60,
                         "energy [J/mm]": c["heat_input_J_per_m"] * 1e-3, "arc-on [s/part]": tp["arc_on_s"], "practical [s/part]": tp["practical_s"], "arc-on share": tp["arc_on_share"],
                         "parts/h arc-on": tp["parts_per_h_arc_on"], "parts/h practical": tp["parts_per_h_practical"],
                         "€/m arc-on": o["nominal"]["total"] / (c["speed"] * 3600), "€/m practical": o["nominal"]["total"] / tp["metres_per_h_practical"],
                         "€/m practical low–high": (o["low"]["total"] / tp["metres_per_h_practical"], o["high"]["total"] / tp["metres_per_h_practical"]),
                         "€/h": o["nominal"]["total"], "capped": c["capped"]})
    return pd.DataFrame(rows)


def capacity_sensitivity(material: str = "carbon steel", thickness: float = 2e-3, joint: str = "butt", job_key: str = "assembly_1m") -> pd.DataFrame:
    """Each assumption moved to its range's ends (the others nominal): the practical parts per hour of one job on each
    scenario — what moves the capacity most."""
    rows = []
    job = JOBS[job_key]
    keys = ("abs_keyhole_" + {"carbon steel": "carbon_steel", "stainless": "stainless", "aluminium": "aluminium"}[material], "melting_efficiency", "max_feed_m_per_s", "humping_m_per_s")
    base = {k: throughput(weld_case(s, material, thickness, joint), job)["parts_per_h_practical"] for k, s in SCENARIOS.items()}
    for key in keys:
        a = ASSUMPTIONS[key]
        for lvl in ("low", "high"):
            ASSUMPTIONS[key] = sc.Assumption(key, getattr(a, lvl), a.low, a.high, a.unit, a.basis)
            try:
                for k, s in SCENARIOS.items():
                    rows.append({"assumption": key, "end": lvl, "value": getattr(a, lvl), "scenario": k, "parts/h practical": throughput(weld_case(s, material, thickness, joint), job)["parts_per_h_practical"], "base": base[k]})
            finally:
                ASSUMPTIONS[key] = a
    for key, factor in (("overheads ×0.5", 0.5), ("overheads ×2", 2.0)):
        o = {k: (v * factor if isinstance(v, float) else v) for k, v in OVERHEADS.items()}
        for k, s in SCENARIOS.items():
            rows.append({"assumption": key, "end": "", "value": factor, "scenario": k, "parts/h practical": throughput(weld_case(s, material, thickness, joint), job, o)["parts_per_h_practical"], "base": base[k]})
    df = pd.DataFrame(rows)
    df["change [%]"] = 100 * (df["parts/h practical"] / df["base"] - 1)
    return df


# ================================================================================================== the stray beam
SAFETY_DISTANCES = (20.0, 50.0, 100.0, 200.0)
EXPOSURES = {"1 s escape": 1.0, "3 s escape": 3.0, "5 s escape": 5.0, "T1 (10 s)": 10.0, "T2 (100 s)": 100.0, "T3 (30 000 s)": 30000.0}


def stray_beam(s: Scenario, distance: float, kind: str, material: str = "carbon steel", opt: dict | None = None) -> si.StrayBeam:
    """The beam past the focus (the full delivered power diverging at the head's cone — the beam that misses or passes
    through the work), a specular reflection off the workpiece (``kind="reflected"``: the material's reflectivity of the
    power, the same cone), or the collimated fault (``kind="collimated"``: the beam escaping before the focusing optic,
    growing by its own divergence only — a fault of the head, kept apart)."""
    opt = optics(s) if opt is None else opt
    if kind == "collimated":
        r = opt["collimated_radius"] + opt["collimated_divergence"] * distance
        return si.StrayBeam(f"{s.key} collimated fault", opt["delivered_W"], r, "tophat", "a fault of the head: the beam before the focusing optic")
    r = max(opt["cone_half_angle"] * distance, opt["spot_radius"])
    if kind == "diverging":
        return si.StrayBeam(f"{s.key} past the focus", opt["delivered_W"], r, "tophat", "the beam through or past the work, diverging at the head's cone")
    refl = _a(MATERIALS[material]["reflectivity"])
    return si.StrayBeam(f"{s.key} reflected off {material}", refl.nominal * opt["delivered_W"], r, "tophat", f"a specular reflection off flat {material} ({refl.nominal:g} of the power), diverging at the cone")


def stray_cases(scenarios: dict[str, Scenario] = SCENARIOS, distances=SAFETY_DISTANCES, kinds=("diverging", "reflected"), materials=("carbon steel", "stainless", "aluminium"), exposures: dict | None = None) -> pd.DataFrame:
    """Per scenario, kind (and material for the reflection), distance and exposure duration: the beam's radius and
    irradiance, the eye's and the skin's exposure limits (IEC 60825-1 at 1070 nm — no aversion response is assumed:
    the 0.25 s blink does not apply in the infrared, so the stated durations are the exposures), the ratios, the
    attenuation (OD) a window or eyewear would need, the hazard distance, and the uncertainty from the reflectivity
    and BPP ranges."""
    exposures = EXPOSURES if exposures is None else exposures
    rows = []
    for k, s in scenarios.items():
        opt = optics(s)
        for kind in kinds:
            for mat in (materials if kind == "reflected" else (None,)):
                for d in distances:
                    beam = stray_beam(s, d, kind, mat or "carbon steel", opt)
                    E = beam.mean_irradiance
                    refl = _a(MATERIALS[mat]["reflectivity"]) if mat else None
                    E_lo, E_hi = (E * refl.low / refl.nominal, E * refl.high / refl.nominal) if refl else (E, E)
                    for name, t in exposures.items():
                        mpe_e, mpe_s = float(sc.mpe_eye_1070(t)), float(sc.mpe_skin_1070(t))
                        od = sc.od_required(E, mpe_e)
                        rows.append({"scenario": k, "kind": kind, "material": mat or "—", "distance_m": d, "exposure": name, "exposure_s": t, "power_W": beam.power, "beam_radius_m": beam.radius,
                                     "irradiance_W_per_m2": E, "irradiance_interval_W_per_m2": (E_lo, E_hi), "mpe_eye_W_per_m2": mpe_e, "mpe_skin_W_per_m2": mpe_s, "over_mpe_eye": E / mpe_e,
                                     "over_mpe_skin": E / mpe_s, "od_required_eye": od["od_required"], "en207_lb_scale": od["en207_lb_scale"], "od_required_skin": max(math.log10(E / mpe_s), 0.0) if E > 0 else 0.0,
                                     "nohd_m": math.sqrt(beam.power / (math.pi * mpe_e)) / opt["cone_half_angle"], "aversion_assumed": False,
                                     "uncertainty": "reflectivity ×0.7–1.2 (surface, angle); the BPP range moves the cone and the irradiance by ×0.5–2; the MPE edition (verify)"})
    return pd.DataFrame(rows)


def collimated_fault(scenarios: dict[str, Scenario] = SCENARIOS, distances=SAFETY_DISTANCES, exposure_s: float = 10.0) -> pd.DataFrame:
    """The head's fault kept apart from the accidental cases: the beam before the focusing optic at each distance."""
    rows = []
    for k, s in scenarios.items():
        opt = optics(s)
        for d in distances:
            b = stray_beam(s, d, "collimated", opt=opt)
            E = b.mean_irradiance
            rows.append({"scenario": k, "distance_m": d, "beam_radius_mm": b.radius * 1e3, "irradiance_W_per_cm2": E / 1e4, "over_mpe_eye": E / float(sc.mpe_eye_1070(exposure_s)),
                         "over_mpe_skin": E / float(sc.mpe_skin_1070(exposure_s)), "od_required_eye": sc.od_required(E, float(sc.mpe_eye_1070(exposure_s)))["od_required"],
                         "note": "a fault scenario (the head's cover or optic missing); the guard's design case, not an accidental exposure"})
    return pd.DataFrame(rows)


BARRIERS = [sc.Shield("polycarbonate", "opaque", th.BLACK_PC, 5e-3, "none"), sc.Shield("polycarbonate", "certified laser-protective", th.LASER_PC, 5e-3, "certified-rating-required"),
            sc.Shield("steel", "bare", th.MILD_STEEL, 1.5e-3, "none"), sc.Shield("aluminium", "bare", th.ALUMINIUM, 1.5e-3, "none")]


def barrier_assessment(scenarios: dict[str, Scenario] = SCENARIOS, distances=SAFETY_DISTANCES, kind: str = "diverging", exposure: str = "T2") -> pd.DataFrame:
    """The enclosure's panels against the beam past the focus (and, as a separate call, the collimated fault): notebook
    47's non-destructive bounds and its verdict rule — PASS only with certified evidence (none here), FAIL for a
    transparent panel, INCOMPLETE otherwise. A calculated attenuation is never certified protection."""
    rows = []
    for k, s in scenarios.items():
        opt = optics(s)
        for b in BARRIERS:
            for d in distances:
                beam = stray_beam(s, d, kind, opt=opt)
                t = EXPOSURE[exposure]
                mpe = float(sc.mpe_eye_1070(t))
                resp = sc.thermal_response(b, beam)
                case = {"scenario": k, "kind": kind, "barrier": b.name, "distance_m": d, "exposure": exposure, "exposure_s": t, "irradiance_W_per_cm2": beam.mean_irradiance / 1e4,
                        "over_mpe_eye": beam.mean_irradiance / mpe, **sc.od_required(beam.mean_irradiance, mpe), **resp, "evidence": b.evidence, "interlock_required": True}
                case["verdict"], case["reason"] = sc.verdict(case)
                rows.append(case)
    return pd.DataFrame(rows)


# ================================================================================================== the record
def record(optic: dict, welds: list[dict], econ: pd.DataFrame, stray: pd.DataFrame, fault: pd.DataFrame, barriers: pd.DataFrame, sens: pd.DataFrame, overheads: dict, debug: bool = False) -> dict:
    def clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in ("pool",)}
    return {"meta": {"notebook": "48_kimla", "debug": debug, "kimla_as_found": {k: v for k, v in SPECS.items() if k in ("note", "sources", "machines", "listed_supplies", "conflicts", "hypothetical_welding_head")},
                     "assumptions": {k: {"nominal": a.nominal, "low": a.low, "high": a.high, "unit": a.unit, "basis": a.basis} for k, a in ASSUMPTIONS.items()}, "overheads_s": overheads, "jobs": JOBS,
                     "standards": sc.STANDARDS, "rule": "analysis scenarios on a hypothetical welding head; every weld unvalidated; PASS needs certified evidence; no aversion response assumed in the infrared"},
            "scenarios": {k: {**s.to_dict(), "optics": {kk: vv for kk, vv in optic[k].items()}} for k, s in SCENARIOS.items()},
            "welding": [clean(c) for c in welds],
            "throughput": econ.to_dict(orient="records"),
            "economics": {"capex": {k: capex(s) for k, s in SCENARIOS.items()}, "opex_per_hour": {k: opex_per_hour(s) for k, s in SCENARIOS.items()}, "parameters": {k: v for k, v in sc.ECON.items() if isinstance(v, dict)}},
            "safety": {"accidental": stray.to_dict(orient="records"), "collimated_fault": fault.to_dict(orient="records"), "barriers": barriers.to_dict(orient="records"),
                       "note": "accidental exposure and protective controls; the fault is a separate table; a calculated attenuation is not certified protection"},
            "sensitivity": sens.to_dict(orient="records"),
            "validation_open": ["KIMLA's own documents (the three URLs) once reachable: the input table against them", "the welding head: a real configuration, its optics and gas nozzle, from a supplier",
                                "weld trials per material, thickness and joint before any speed, penetration or strength is quoted", "the MPE constants against the current IEC 60825-1 edition",
                                "the enclosure's panels: EN 12254 / IEC 60825-4 evidence for the irradiance and exposure", "the handling overheads timed on the real cell"]}


# ================================================================================================== the shields' endurance, and fire
SHIELDS = [si.Barrier(th.ALUMINIUM, 1e-3), si.Barrier(th.ALUMINIUM, 3e-3), si.Barrier(th.BLACK_PC, 3e-3), si.Barrier(th.MILD_STEEL, 3e-3)]
FIRE = {"pc_piloted_flux": sc.Assumption("pc_piloted_flux", 25e3, 20e3, 30e3, "W/m²", "critical absorbed flux for the piloted ignition of polycarbonate (fire-safety handbooks: 20–30 kW/m²)"),
        "pc_unpiloted_flux": sc.Assumption("pc_unpiloted_flux", 40e3, 35e3, 50e3, "W/m²", "critical flux for the unpiloted ignition of polycarbonate (assumed from the handbooks' class)"),
        "pc_ignition_T": sc.Assumption("pc_ignition_T", 790.0, 750.0, 850.0, "K", "polycarbonate's ignition temperature (handbooks: ~480–580 °C; its decomposition T_fail 720 K is lower)")}
ENDURANCE_PATH = DATA / "kimla_endurance.json"


def fire_risk(barrier: si.Barrier, beam: si.StrayBeam) -> dict:
    """Can the shield catch fire, and how soon: polycarbonate against its piloted and unpiloted ignition fluxes, and the
    semi-infinite time for its surface to reach the ignition temperature (a lower bound: a thin sheet heats faster than
    a semi-infinite one once the heat reaches its back, and the surface chars from T_fail on); aluminium and steel do
    not burn (a paint or oil film on them can, and is not modelled)."""
    s = barrier.solid
    q = s.absorptivity * beam.mean_irradiance
    if s.failure != "ablate":
        return {"combustible": False, "fire_risk": "none: a metal sheet does not burn (a coating or oil film would)", "ignition_flux_ratio": 0.0, "t_ignition_bound_s": math.inf}
    pil, spon, Ti = FIRE["pc_piloted_flux"], FIRE["pc_unpiloted_flux"], FIRE["pc_ignition_T"]
    t_ign = (Ti.nominal - th.T_ROOM) ** 2 * math.pi * s.rho * s.c * s.k / (2 * q) ** 2 if q > 0 else math.inf
    if q >= spon.low:
        lvl = "high: above the unpiloted-ignition flux — it ignites by itself"
    elif q >= pil.low:
        lvl = "high: above the piloted-ignition flux — a spark or the weld's own plume ignites it"
    elif q >= 0.3 * pil.low:
        lvl = "moderate: a third of the piloted flux — chars under long exposure"
    else:
        lvl = "low: below a third of the piloted-ignition flux"
    return {"combustible": True, "fire_risk": lvl, "ignition_flux_ratio": q / pil.nominal, "t_ignition_bound_s": t_ign}


def screen_bounds(barrier: si.Barrier, beam: si.StrayBeam, T0: float = th.T_ROOM) -> dict:
    """``si.screen``'s closed forms without its steady solve (seconds a case: that belongs to the recorded run): the
    transmitted share, the semi-infinite time to surface damage, the lossless lower bound on the burn-through time, the
    steady upper bound on the rise — and the decision: *fails: transparent*, *survives: steady bound*, *survives: energy
    bound*, or *undecided* (the recorded run's steady solve and transient decide)."""
    s, d = barrier.solid, barrier.thickness
    q = s.absorptivity * beam.mean_irradiance
    transmitted = s.transmitted(d)
    damage_T = s.T_melt if s.failure == "melt" else s.T_fail
    t_surface = (damage_T - T0) ** 2 * math.pi * s.rho * s.c * s.k / (2 * q) ** 2 if q > 0 else math.inf
    t_lower = (s.enthalpy_to_fail(T0) * d / q) if (s.failure == "ablate" and q > 0) else (th.thin_plate_time_to(s, d, q, s.T_melt) if q > 0 else math.inf)
    rise_bound = si.plate_steady_bound(s, d, s.absorptivity * beam.power, beam.radius, max(8 * beam.radius, 12 * d, 0.02))
    if transmitted > 0.01:
        decision = "fails: transparent"
    elif rise_bound < damage_T - T0:
        decision = "survives: steady bound"
    elif t_lower > EXPOSURE["T3"]:
        decision = "survives: energy bound"
    else:
        decision = "undecided: the recorded run's steady solve / transient decides"
    return {"barrier": barrier.name, "beam": beam.name, "absorbed_irradiance_W_per_cm2": q / 1e4, "transmitted_fraction": transmitted, "t_surface_damage_bound": t_surface,
            "t_burn_through_lower_bound": t_lower, "steady_rise_bound_K": rise_bound, "damage_rise_K": damage_T - T0, "steady_T_max": math.nan, "decision": decision}


def shield_endurance(scenarios: dict[str, Scenario] = SCENARIOS, shields=SHIELDS, distances=SAFETY_DISTANCES, kinds=("diverging", "collimated"), run: bool = False,
                     sheet_half: float = si.SHEET_HALF, t_max: float = si.SIMULATION_CAP_S) -> list[dict]:
    """How long each shield protects against the beam past the focus and the collimated fault at each distance: 46's
    route — ``si.screen`` (the closed forms: a transparent sheet 0 s; a steady or energy bound: for ever; the lower bound
    on the burn-through time and the surface-damage time either way), the enthalpy/ablation transient (``th.heat_plate``)
    only where the screening cannot decide and only with ``run=True`` (minutes: ``python designs/kimla.py --endurance``),
    ``si.verdict`` for ``holds_s`` and the T1/T2/T3 classes — and the fire indicator. Design numbers, not evidence."""
    rows = []
    for k, s in scenarios.items():
        opt = optics(s)
        for kind in kinds:
            for b in shields:
                for d in distances:
                    beam = stray_beam(s, d, kind, opt=opt)
                    scr = si.screen(b, beam) if run else screen_bounds(b, beam)          # the steady solve and the transient belong to the recorded run
                    sim = None
                    if scr["decision"] == "simulate" and run:
                        sim = th.heat_plate(b.solid, b.thickness, beam.power, beam.radius, t_max, shape=beam.shape, h=10.0, nr=si.NR_GUARD, nz=si._nz(b.thickness), r_max=sheet_half)
                    v = si.verdict(scr, sim)
                    rows.append({"scenario": k, "kind": kind, "shield": b.name, "material": b.solid.name, "thickness_mm": b.thickness * 1e3, "distance_m": d, "beam_radius_mm": beam.radius * 1e3,
                                 "irradiance_W_per_cm2": beam.mean_irradiance / 1e4, "absorbed_W_per_cm2": scr["absorbed_irradiance_W_per_cm2"], "decision": scr["decision"],
                                 "t_surface_damage_bound_s": scr["t_surface_damage_bound"], "t_burn_through_lower_bound_s": scr["t_burn_through_lower_bound"], "steady_rise_bound_K": scr["steady_rise_bound_K"],
                                 "holds_s": v["holds_s"], **{c: v[c] for c in EXPOSURE}, "source": v["source"], "simulated": sim is not None, **fire_risk(b, beam),
                                 "note": "cell-model / closed-form times for the guard's design; not protection evidence"})
    return rows


def shield_endurance_table(rows: list[dict], value: str = "holds_s") -> pd.DataFrame:
    return pd.DataFrame(rows).pivot_table(index=["kind", "scenario", "shield"], columns="distance_m", values=value, aggfunc="first", dropna=False)


def endurance_record(**kw) -> dict:
    import datetime
    rows = shield_endurance(run=True, **kw)
    return {"meta": {"made": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "screening": si.SCREENING, "grid": {"nr": si.NR_GUARD, "r_max": si.SHEET_HALF, "t_max_s": si.SIMULATION_CAP_S},
                     "route": "si.screen → th.heat_plate where undecided → si.verdict; the beam past the focus and the collimated fault", "shields": [b.name for b in SHIELDS],
                     "caveat": "a cell-model time for the guard's design; not protection evidence (IEC 60825-4 asks for the test)"},
            "rows": [{kk: (None if isinstance(vv, float) and math.isnan(vv) else ("inf" if vv == math.inf else vv)) for kk, vv in r.items()} for r in rows]}


def load_endurance(path: Path = ENDURANCE_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    for r in d["rows"]:
        for key in ("holds_s", "t_ignition_bound_s", "t_burn_through_lower_bound_s", "t_surface_damage_bound_s"):
            r[key] = math.inf if r[key] == "inf" else (math.nan if r[key] is None else r[key])
    return d


if __name__ == "__main__":
    import argparse
    import time as _t
    ap = argparse.ArgumentParser(description="notebook 48: the recorded results the notebook reads")
    ap.add_argument("--endurance", action="store_true", help=f"the shields' hold times with the transients (minutes per scenario) → {ENDURANCE_PATH}")
    args = ap.parse_args()
    if args.endurance:
        t0 = _t.monotonic()
        rec = endurance_record()
        ENDURANCE_PATH.write_text(json.dumps(rec, indent=1))
        print(f"wrote {ENDURANCE_PATH}: {len(rec['rows'])} rows in {_t.monotonic() - t0:.0f} s")


def mock_endurance_record(scenarios: dict[str, Scenario] = SCENARIOS, **kw) -> dict:
    """A **mock** of ``endurance_record`` for the notebook's debug mode and the tests: no transient runs — where the
    screening is undecided the hold time is set to 3 × the lossless lower bound on the burn-through (a stand-in of the
    cell model's usual ratio, nothing more) and the source says MOCK; the decided rows are the closed forms'. Never
    written to ``ENDURANCE_PATH``: it exercises the read-back path only."""
    rows = shield_endurance(scenarios, run=False, **kw)
    for r in rows:
        if r["decision"].startswith("undecided"):
            r["holds_s"] = 3.0 * r["t_burn_through_lower_bound_s"]
            for c, t in EXPOSURE.items():
                r[c] = bool(r["holds_s"] > t)
            r["source"] = "MOCK: 3 × the lossless lower bound (no transient was run)"
            r["simulated"] = False
    return {"meta": {"made": "MOCK", "screening": True, "grid": {"nr": 0, "r_max": si.SHEET_HALF, "t_max_s": si.SIMULATION_CAP_S}, "route": "MOCK — screen_bounds with 3 × the lower bound where undecided; no transient",
                     "shields": [b.name for b in SHIELDS], "caveat": "a MOCK record for the notebook's debug mode and the tests: not a result"}, "rows": rows, "mock": True}
