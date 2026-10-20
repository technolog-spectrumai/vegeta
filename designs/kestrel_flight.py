"""Kestrel-4Q — the biplane's aerodynamics, the cruise and the full-range polar for the transition (notebook 40).

- ``aero``: the zero-stagger biplane in PEREGRINE's vortex lattice (``peregrine_flight.vlm``: two flat lifting
  surfaces at z = ±gap/2, Trefftz-plane induced drag) against one of its wings alone and against a monoplane of the
  same span and area: the lift slope, the span efficiency, the gap's interference, the wings' neutral point.
- ``neutral_point``: the lattice's plus the pod (Raymer's fuselage term) and the four propellers' normal force ahead of
  the CG (``prop_normal_force_slope``: a blade-element estimate in De Young's form; the momentum bound N = ṁ·V·sin α of
  a disk that turns the whole cross-flow axial is ~10 × higher and is kept for comparison) — a tailsitter's propellers
  sit far ahead and are destabilising; with no tail, the CG alone sets the margin.
- ``drag_buildup``: Cd0 from the parts (friction and form on the wings, pod, fin, nacelles; the posts and diagonals as
  round tubes in cross-flow, or ``faired`` in teardrop sleeves; a 10 % interference allowance): what the bracing costs.
- ``Polar``: the airframe's coefficients over the whole range of angle of attack — attached (the lattice's slope, the
  build-up's Cd0, the lattice's k) blended through the stall into the separated flat plate (Viterna's CD90 for the
  wing's aspect ratio, the centre of pressure moving to mid-chord at 90°). It is the transition model's first
  estimate; the CFD (``kestrel_cfd``) is the check that replaces it where it has been run.
- ``cruise``: the power required, the speeds of best endurance and range, the stall speed, the pitch trim by
  differential thrust (no tail: the lower pair pushes harder to hold the nose up) and whether it closes.

Units SI. Frame: ``kestrel``'s (x aft, y right, z up); angles of attack positive nose-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kestrel
import kestrel_systems as ks

G = ks.G
RHO = ks.RHO0
NU = 1.46e-5
CL_MAX_2D = 0.90                 # NACA 0012 at Re ~ 1e5 (assumption: low-Re polars put it 0.8-1.0)
ALPHA0_DEG = 0.0                 # symmetric section
FUSELAGE_K = 0.008               # Raymer's fuselage pitching-moment factor (as peregrine_flight)
INTERFERENCE = 1.10              # the build-up's allowance for the junctions (wing-pod, strut ends, nacelles)
CD_CYLINDER = 1.10               # a carbon tube in cross-flow at Re ~ 5e3
CD_FAIRED = 0.12                 # the same tube in a 3:1 teardrop sleeve, on the tube's frontal area (sleeve's own friction included)
OSWALD_VISCOUS = 0.90            # e = 0.90 × the lattice's inviscid span efficiency
V_CRUISE = 16.0                  # m/s, the survey speed this study assumes (checked against the endurance speed)


# ================================================================================================= the lattice
def quads(p=None, *, wings=("upper", "lower")) -> list:
    """The lifting surfaces for the lattice: the wings as flat rectangles at their chord planes."""
    L = kestrel.layout(p)
    out = []
    for w, z in (("upper", L["h"]), ("lower", -L["h"])):
        if w in wings:
            b2 = L["span"] / 2
            out.append(np.array([[L["x_le"], -b2, z], [L["x_le"], b2, z], [L["x_te"], b2, z], [L["x_te"], -b2, z]]))
    return out


def _vlm(qs, alpha, S, c, x_ref, n_span=20, n_chord=4):
    import peregrine_flight as pf
    return pf.vlm(qs, alpha_deg=alpha, S_ref=S, c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)


def aero(p=None, *, n_span=20, n_chord=4) -> dict:
    """The lattice's numbers for the biplane, for one wing alone and for a monoplane of the same span and area:
    CLα per rad (on the total wing area), k = CDi / CL², e (on the span), the neutral point of the wings, and the
    biplane factors: the lift slope against two isolated wings and the induced drag against the monoplane."""
    L = kestrel.layout(p)
    x_ref = L["x_ac"]
    out = {}
    cases = {"biplane": (quads(p), L["S_ref"], L["span"]), "one wing alone": (quads(p, wings=("upper",)), L["S_wing"], L["span"])}
    b2 = L["span"] / 2
    mono_c = 2 * L["chord"]                       # same span, the two wings' area in one: AR halves
    cases["monoplane, same span and area"] = ([np.array([[L["x_le"], -b2, 0.0], [L["x_le"], b2, 0.0], [L["x_le"] + mono_c, b2, 0.0],
                                                         [L["x_le"] + mono_c, -b2, 0.0]])], L["S_ref"], L["span"])
    for name, (qs, S, b) in cases.items():
        r0 = _vlm(qs, 0.0, S, L["chord"], x_ref, n_span, n_chord)
        r1 = _vlm(qs, 4.0, S, L["chord"], x_ref, n_span, n_chord)
        da = math.radians(4.0)
        CLa = (r1["CL"] - r0["CL"]) / da
        Cma = (r1["Cm"] - r0["Cm"]) / da
        k = r1["CDi"] / r1["CL"] ** 2
        AR = b ** 2 / S
        out[name] = {"CLa": CLa, "k": k, "e": 1 / (math.pi * AR * k), "AR": AR, "S": S,
                     "x_np": x_ref - Cma / CLa * L["chord"], "upper share": None}
        if name == "biplane":
            lat, F = r1["lattice"], r1["F"]
            up = lat["quad"] == 0
            out[name]["upper share"] = float(F[up, 2].sum() / F[:, 2].sum())
    bi, one = out["biplane"], out["one wing alone"]
    out["factors"] = {"lift slope vs two isolated wings": bi["CLa"] / one["CLa"],
                      "induced drag vs the monoplane": bi["k"] / out["monoplane, same span and area"]["k"],
                      "gap / chord": L["gap_chord"], "gap / span": L["gap_span"]}
    return out


def prandtl_sigma(gap_over_span: float) -> float:
    """Prandtl's biplane interference factor σ for equal spans, fitted to his curve: σ ≈ (1 − 0.66 g/b) / (1.05 + 3.7 g/b).
    The pair's induced drag over the equal-span monoplane's carrying the same lift is (1 + σ) / 2."""
    gb = float(gap_over_span)
    return (1 - 0.66 * gb) / (1.05 + 3.7 * gb)


GAP_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)        # gap / span: Kestrel sits near 0.25–0.28


def interference(p=None, *, gaps=GAP_SWEEP, n_span=16, n_chord=3) -> dict:
    """The **measure of the wing interference** of the biplane, analytic (seconds): for the design's gap and for a
    sweep of gap / span, the lattice's lift slope of the pair against one wing alone (the same per-area slope would
    be 1: the shortfall is what the wings take from each other), its induced-drag factor against the equal-span,
    equal-area monoplane (Munk: a biplane of the same span and lift has **less** induced drag than that monoplane
    only through its higher per-wing aspect ratio; against the same span it has more, by (1 + σ) / 2), and Prandtl's
    σ beside it. Returns ``{"design": {...}, "sweep": DataFrame, "equivalents": DataFrame, "notes": [...]}``.

    Definitions (all coefficients per unit area of the planform they belong to):
    - **lift interference factor** = CLα(pair, on 2 S_wing) / CLα(one wing alone, on S_wing)  — 1.0 = no interference;
    - **induced drag vs the monoplane of the same span and area** = k(pair) / k(monoplane, chord 2c, on 2 S_wing);
    - **Prandtl σ** and (1 + σ) / 2 = the pair's induced drag over the equal-span monoplane **of the same lift** (span loading);
    - **monoplane equivalents**: the span a single wing of the same area would need for the pair's induced drag, and the
      area it would need at the same span for the pair's lift slope × area (lift at a given α and speed)."""
    import pandas as pd
    base = kestrel.resolve(kestrel.overrides(p))
    L0 = kestrel.layout(base)
    rows = []
    gbs = sorted(set(list(gaps) + [round(L0["gap"] / L0["span"], 4)]))
    design_gb = L0["gap"] / L0["span"]
    for gb in gbs:
        q = kestrel.resolve(dict(kestrel.overrides(p), gap=gb * L0["span"] * 1000.0))
        a = aero(q, n_span=n_span, n_chord=n_chord)
        bi, one, mono = a["biplane"], a["one wing alone"], a["monoplane, same span and area"]
        sig = prandtl_sigma(gb)
        rows.append({"gap / span": gb, "gap / chord": gb * L0["span"] / L0["chord"], "design": abs(gb - design_gb) < 1e-6,
                     "CLα pair [1/rad]": bi["CLa"], "CLα one wing [1/rad]": one["CLa"], "CLα monoplane 2c [1/rad]": mono["CLa"],
                     "lift interference factor": bi["CLa"] / one["CLa"], "e pair (on the span)": bi["e"], "e monoplane 2c": mono["e"],
                     "k pair": bi["k"], "k one wing": one["k"], "k monoplane 2c": mono["k"],
                     "k pair / k monoplane 2c (lattice)": bi["k"] / mono["k"], "Prandtl σ": sig, "(1+σ)/2, same span and lift": (1 + sig) / 2,
                     "pair lift / one wing's lift, same α": bi["CLa"] * bi["S"] / (one["CLa"] * one["S"]), "upper wing share": bi["upper share"]})
    sweep = pd.DataFrame(rows).set_index("gap / span")
    d = sweep[sweep["design"]].iloc[0]
    # the monoplane equivalents of the design's pair
    S2, b = L0["S_ref"], L0["span"]
    k_pair, k_one, k_mono = float(d["k pair"]), float(d["k one wing"]), float(d["k monoplane 2c"])     # on each planform's own area
    e_mono = d["e monoplane 2c"]
    b_eq_drag = math.sqrt(S2 / (math.pi * e_mono * k_pair))                      # the single wing's span for the pair's k at the same area
    cla_one = d["CLα one wing [1/rad]"]
    S_eq_lift = d["CLα pair [1/rad]"] * S2 / cla_one                              # the single wing's area (at the design's chord → span) for the pair's lift
    equivalents = pd.DataFrame({
        "the pair as built": {"span [m]": b, "area [m²]": S2, "AR (per wing)": L0["span"] / L0["chord"], "CLα [1/rad]": d["CLα pair [1/rad]"], "k = CDi/CL²": k_pair, "e (on the span)": d["e pair (on the span)"]},
        "monoplane, same span and area (chord 2c)": {"span [m]": b, "area [m²]": S2, "AR (per wing)": b ** 2 / S2, "CLα [1/rad]": d["CLα monoplane 2c [1/rad]"], "k = CDi/CL²": k_mono, "e (on the span)": e_mono},
        "monoplane, same span and chord (one wing)": {"span [m]": b, "area [m²]": S2 / 2, "AR (per wing)": L0["span"] / L0["chord"], "CLα [1/rad]": cla_one, "k = CDi/CL²": k_one, "e (on the span)": 1 / (math.pi * (b ** 2 / (S2 / 2)) * k_one)},
        "monoplane with the pair's induced drag, same area": {"span [m]": b_eq_drag, "area [m²]": S2, "AR (per wing)": b_eq_drag ** 2 / S2, "CLα [1/rad]": float("nan"), "k = CDi/CL²": k_pair, "e (on the span)": e_mono},
        "monoplane with the pair's lift at a given α, same chord": {"span [m]": S_eq_lift / L0["chord"], "area [m²]": S_eq_lift, "AR (per wing)": S_eq_lift / L0["chord"] ** 2, "CLα [1/rad]": cla_one, "k = CDi/CL²": float("nan"), "e (on the span)": float("nan")},
    }).T
    design = {"gap / span": design_gb, "gap / chord": L0["gap_chord"], "lift interference factor": float(d["lift interference factor"]),
              "lift lost to interference [%]": 100 * (1 - float(d["lift interference factor"])), "Prandtl σ": float(d["Prandtl σ"]),
              "induced drag vs equal-span monoplane of the same lift, Prandtl": float(d["(1+σ)/2, same span and lift"]),
              "induced drag vs monoplane of the same span and area, lattice": float(d["k pair / k monoplane 2c (lattice)"]),
              "span a monoplane of the same area needs for the pair's induced drag [m]": b_eq_drag,
              "area a monoplane of the same chord needs for the pair's lift [m²]": S_eq_lift, "upper wing share of the lift": float(d["upper wing share"])}
    notes = ["lattice: inviscid, attached flow, small angles (the lift interference factor is a slope ratio at 0–4°); Prandtl's σ: the classical fit for equal spans, no stagger, no decalage",
             "per-area coefficients: a factor of 1 means the two wings lift as if alone; the equal-span monoplane of the same area has half the aspect ratio, which is why the biplane beats it on induced drag while losing to it on lift slope per area",
             "the CFD chapter K12 (wing-interference) measures the same three planforms with RANS at 0, 4, 8°: its factors replace these when its run is on file"]
    return {"design": design, "sweep": sweep, "equivalents": equivalents, "notes": notes}


def prop_cn_alpha(thrust_each: float, V: float, p=None, rho: float = RHO, *, calibrated: bool = True) -> float:
    """One propeller's normal-force slope dC_N/dα [1/rad] on q·A (free-stream q, disk area A) in De Young's
    blade-element form: 4.25 σₑ / (1 + 2 σₑ) · sin(β₀.₇₅ + 8°) · f(T_c), f = 1 + ½ (√(1 + 8 T_c / π) − 1), with the
    effective solidity σₑ = (4/3) B c₀.₇₅ / (π R) and T_c = T / (q A), times prop_lab's measured factor when its
    campaign is on file (``calibrated``). The CFD's rotor disks carry the in-plane force of their blade elements
    (printed in the solver log), not on the airframe's force coefficients."""
    c = ks.COMPONENTS["propeller"]
    L = kestrel.layout(p)
    R = L["R_prop"]
    pitch = c["pitch_in"] * 0.0254
    c75 = c["chord_max_m"] + (c["chord_tip_m"] - c["chord_max_m"]) * ((0.75 - 0.45) / 0.55) ** 1.5
    sig = 4 / 3 * c["blades"] * c75 / (math.pi * R)
    beta = math.atan(pitch / (2 * math.pi * 0.75 * R))
    q = 0.5 * rho * max(V, 0.1) ** 2
    tc = max(thrust_each, 0.0) / (q * L["A_disc"])
    f = ks.prop_lab_calibration()["cn_factor"] if calibrated else 1.0          # prop_lab's measured slope over De Young's
    return f * 4.25 * sig / (1 + 2 * sig) * math.sin(beta + math.radians(8.0)) * (1 + 0.5 * (math.sqrt(1 + 8 * tc / math.pi) - 1))


def prop_normal_force_slope(V: float, thrust_each: float, p=None, rho: float = RHO, *, bound: bool = False) -> float:
    """dN/dα [N/rad] of the four propellers together at airspeed V: De Young's estimate (``prop_cn_alpha``), or with
    ``bound`` the momentum upper bound of a disk turning the whole cross-flow axial (N = ṁ V sin α, ṁ = ρ A (V + vᵢ))."""
    L = kestrel.layout(p)
    A = L["A_disc"]
    if bound:
        vi = induced_velocity(thrust_each, V, 0.0, A, rho)
        return 4 * rho * A * (V + vi) * V
    return 4 * prop_cn_alpha(thrust_each, V, p, rho) * 0.5 * rho * V ** 2 * A


def induced_velocity(thrust, V, alpha_rad, A, rho=RHO, iterations=40):
    """Glauert's induced velocity at a disk [m/s] in an inflow V at α to its axis (axial component V cos α): Ogma's."""
    from ogma import rotors
    return rotors.induced_velocity(thrust, V, alpha_rad, A, rho, iterations)


def neutral_point(p=None, *, V: float = V_CRUISE, thrust_each: float = 0.6, x_cg: float, a: dict | None = None) -> dict:
    """The neutral point [m from the nose tip] with the pod and the propellers, and the static margin of ``x_cg`` (in
    chords). The propellers' destabilising share depends on their thrust: given at cruise."""
    L = kestrel.layout(p)
    a = a or aero(p)
    bi = a["biplane"]
    q = 0.5 * RHO * V ** 2
    c, S = L["chord"], L["S_ref"]
    Cma_wing = -bi["CLa"] * (bi["x_np"] - x_cg) / c
    Wf, Lf = L["pod_width"], L["pod_length"]
    Cma_pod = FUSELAGE_K * Wf ** 2 * Lf / (c * S) * (180 / math.pi)
    dN = prop_normal_force_slope(V, thrust_each, p)
    Cma_prop = dN * (x_cg - L["prop_x"]) / (q * S * c)
    Cma = Cma_wing + Cma_pod + Cma_prop
    CLa = bi["CLa"] + dN / (q * S)
    x_np = x_cg - Cma / CLa * c
    return {"x_np_wings": bi["x_np"], "x_np": x_np, "Cma_wings": Cma_wing, "Cma_pod": Cma_pod, "Cma_props": Cma_prop,
            "Cma": Cma, "static_margin": (x_np - x_cg) / c, "x_cg": x_cg}


# ================================================================================================= drag
def _cf(Re):
    """Mixed laminar–turbulent flat-plate friction (Prandtl–Schlichting with the 1700/Re transition term)."""
    Re = max(Re, 1e4)
    return max(0.455 / math.log10(Re) ** 2.58 - 1700.0 / Re, 1.328 / math.sqrt(Re))


def drag_buildup(p=None, *, V: float = V_CRUISE, faired: bool = False) -> dict:
    """The parts' zero-lift drag areas [m²] and Cd0 on S_ref; ``bracing`` is the posts and diagonals (round tubes, or
    ``faired`` in teardrop sleeves)."""
    pr = kestrel.resolve(p)
    L = kestrel.layout(p)
    t = pr["thickness"]
    rows = {}
    ff_wing = 1 + 0.6 / 0.3 * t + 100 * t ** 4
    wet_wing = 2 * L["S_wing"] * (1 + 0.2 * t)
    rows["wings (2)"] = 2 * _cf(V * L["chord"] / NU) * ff_wing * wet_wing
    lf, df = L["pod_length"], math.sqrt(L["pod_width"] * L["gap"] * 0.6)
    fr = lf / df
    ff_pod = 1 + 60 / fr ** 3 + fr / 400
    wet_pod = math.pi * df * lf * 0.75
    rows["pod (between the wings)"] = _cf(V * lf / NU) * ff_pod * wet_pod
    fin_area = (L["fin_height"] + L["ventral_height"]) * 0.5 * (L["x_feet"] - L["fin_root_x"] + L["fin_tip_chord"])
    rows["fin plate"] = _cf(V * 0.15 / NU) * 1.05 * 2 * fin_area
    dn, ln = pr["nacelle_diameter"] / 1000, (L["x_le"] + 0.6 * L["chord"] - L["prop_x"])
    rows["nacelles (4)"] = 4 * (_cf(V * ln / NU) * 1.3 * math.pi * dn * ln + 0.1 * math.pi * dn ** 2 / 4)
    cd = CD_FAIRED if faired else CD_CYLINDER
    rows["posts (2)"] = 2 * cd * pr["post_diameter"] / 1000 * L["post_length"]
    rows["diagonals (4)"] = 4 * cd * pr["diagonal_diameter"] / 1000 * L["diagonal_length"] * (L["gap"] / L["diagonal_length"])
    rows["foot rods (4), axial"] = 4 * 0.02 * math.pi * (pr["foot_rod_diameter"] / 1000) ** 2 / 4 * 10
    rows["camera dome"] = 0.4 * math.pi * (pr["camera_diameter"] / 2000) ** 2
    total = sum(rows.values()) * INTERFERENCE
    bracing = rows["posts (2)"] + rows["diagonals (4)"]
    return {"faired": faired, "parts_m2": rows, "CdA_m2": total, "Cd0": total / L["S_ref"], "bracing_m2": bracing * INTERFERENCE,
            "bracing_share": bracing * INTERFERENCE / total, "Cd0_wings": rows["wings (2)"] / L["S_ref"]}


# ================================================================================================= the full-range polar
@dataclass(frozen=True)
class Polar:
    """One lifting surface's coefficients over the whole range of α (radians in, coefficients on its own area):
    attached flow on the lattice's slope with a parabolic polar, blended through ``alpha_stall`` (width ``width``) into
    the separated flat plate CL = CD90/2 · sin 2α, CD = cd0 + CD90 sin² α. ``cm`` is about the quarter chord: 0 attached
    (symmetric section) and −CN (x_cp − 1/4) separated, with x_cp = 1/4 + 1/4 |sin α| (mid-chord at 90°)."""
    cl_alpha: float
    cd0: float
    k: float
    cl_max: float
    cd90: float
    alpha0: float = 0.0
    width: float = math.radians(1.5)
    source: str = "model"

    @property
    def alpha_stall(self) -> float:
        return self.alpha0 + self.cl_max / self.cl_alpha

    def coefficients(self, alpha):
        a = np.asarray(alpha, dtype=float)
        a_w = np.arctan2(np.sin(a), np.cos(a))                        # wrapped to (−π, π]
        att = self.cl_alpha * np.sin(a_w - self.alpha0) * np.cos(a_w - self.alpha0)     # linear near 0, bounded in reverse flow
        sig = 1.0 / (1.0 + np.exp(-(np.abs(a_w) - self.alpha_stall) / self.width))
        cl_s = 0.5 * self.cd90 * np.sin(2 * a_w)
        cd_s = self.cd0 + self.cd90 * np.sin(a_w) ** 2
        cl = (1 - sig) * att + sig * cl_s
        cd = (1 - sig) * (self.cd0 + self.k * att ** 2) + sig * cd_s
        cn_s = cl_s * np.cos(a_w) + cd_s * np.sin(a_w)
        cm = -sig * cn_s * (0.25 * np.abs(np.sin(a_w)))
        return cl, cd, cm


def viterna_cd90(AR: float) -> float:
    """Viterna & Janetzke's CD_max of a finite wing: 1.11 + 0.018 AR (AR ≤ 50)."""
    return 1.11 + 0.018 * min(AR, 50.0)


def polar(p=None, a: dict | None = None, d: dict | None = None, *, faired: bool = False) -> Polar:
    """The airframe's wing polar for the transition model, per unit of wing area (both wings alike): the lattice's
    biplane slope and k (with the viscous span factor), the build-up's Cd0 (the whole airframe's, carried by the wing
    area), the 3-D maximum lift 0.9 × CL_MAX_2D."""
    L = kestrel.layout(p)
    a = a or aero(p)
    d = d or drag_buildup(p, faired=faired)
    bi = a["biplane"]
    return Polar(cl_alpha=bi["CLa"], cd0=d["Cd0"], k=bi["k"] / OSWALD_VISCOUS, cl_max=0.9 * CL_MAX_2D,
                 cd90=viterna_cd90(L["AR_wing"]), alpha0=math.radians(ALPHA0_DEG))


# ================================================================================================= cruise
def cruise(p=None, *, mass_kg: float, x_cg: float, z_cg: float = 0.0, speeds=np.arange(9.0, 26.0, 1.0), drive=None,
           pol: Polar | None = None, avionics_W: float = 13.0) -> "pd.DataFrame":
    """Level flight at each speed: CL, α, drag, the thrust split that trims the pitch with no tail (the aerodynamic
    moment about the CG balanced by the lower pair pushing harder than the upper), its feasibility (every rotor at
    least ``T_min``), the electrical power of the four drives (Boreas tables) plus the avionics, and the endurance on
    the usable battery."""
    import pandas as pd
    L = kestrel.layout(p)
    drive = drive or ks.drive_table()
    pol = pol or polar(p)
    S, c, h = L["S_ref"], L["chord"], L["h"]
    W = mass_kg * G
    np_ = neutral_point(p, x_cg=x_cg)
    Ewh = ks.COMPONENTS["battery"]["cells"] * 3.7 * ks.COMPONENTS["battery"]["capacity_ah"] * ks.COMPONENTS["battery"]["usable"]
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V ** 2
        CL = W / (q * S)
        if CL > pol.cl_max:
            rows.append({"V [m/s]": V, "CL": CL, "feasible": False, "note": "beyond the wing's CL max"})
            continue
        alpha = CL / pol.cl_alpha + pol.alpha0
        CD = pol.cd0 + pol.k * CL ** 2
        D = q * S * CD
        Cm_cg = -np_["static_margin"] * CL                     # symmetric section: the moment is the margin's
        M_aero = q * S * c * Cm_cg
        # thrust lines: upper pair at z = +h, lower at −h; total D, moment −M_aero to cancel (nose-up = lower pushes)
        # Σ T = D;  −(h − z_cg)·T_up + (h + z_cg)·T_lo = −M_aero
        A = np.array([[1.0, 1.0], [-(h - z_cg), (h + z_cg)]])
        T_up, T_lo = np.linalg.solve(A, [D, -M_aero])
        ok = min(T_up, T_lo) / 2 >= drive.T_min and max(T_up, T_lo) / 2 <= float(drive.t_max(V))
        P = 2 * drive.power(T_up / 2, V) + 2 * drive.power(T_lo / 2, V) + avionics_W
        rows.append({"V [m/s]": V, "CL": CL, "α [deg]": math.degrees(alpha), "L/D": CL / CD, "drag [N]": D,
                     "Cm about CG": Cm_cg, "upper rotor [N]": T_up / 2, "lower rotor [N]": T_lo / 2,
                     "trim by thrust": ok, "electrical [W]": P, "endurance [min]": Ewh / P * 60, "range [km]": Ewh / P * V * 3.6,
                     "feasible": True, "note": "" if ok else "the upper pair below T_min: thrust alone cannot trim"})
    return pd.DataFrame(rows).set_index("V [m/s]")


def stall_speed(p=None, *, mass_kg: float, pol: Polar | None = None) -> float:
    L = kestrel.layout(p)
    pol = pol or polar(p)
    return math.sqrt(2 * mass_kg * G / (RHO * L["S_ref"] * pol.cl_max))
