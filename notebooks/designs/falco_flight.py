"""FALCO flight — the aerodynamics with flaps and crow, the derivative table, trim, the envelope against altitude, the
climb and descent capability, the propeller brake and its regeneration, the launch at altitude (notebook 33).

Reduced models, honest about what they are, on top of NISUS's (``nisus_flight``, whose constants this module reuses):

- **the lattice**: PEREGRINE's vortex lattice (``peregrine_flight.vlm``) on ``falco.planform_split``'s quads — the
  flap and aileron quads take their deflection as an incidence increment ``τ K(δ) δ`` (thin-airfoil flap theory with
  the plain flap's effectiveness ``FLAP_TAU`` / ``AILERON_TAU`` and its loss at large deflection ``K(δ)``: **assumed**,
  Roskam/Raymer class charts) — so ΔCL, ΔCm and the induced drag of crow are **calculated (lattice)** in the linear
  sense; their profile drag is the plain-flap formula ``ΔCD = 1.7 (cf/c)^1.38 (S_flapped/S) sin²δ`` (Raymer's
  approximation: **assumed**, the crow CFD case of ``falco_cfd`` is the check). Nothing here predicts the separation at
  55° of flap: the numbers are a first approximation for sizing and simulation;
- **the air**: everything takes the density (``falco_systems.atmosphere``); speeds are TAS unless named EAS; the
  structure's speeds (V_NE, V_FE, V_A) are EAS (``V_TAS = V_EAS / √σ``) — the flutter margin at TAS is not computed;
- **the drive**: ``falco_systems.FalcoDrive`` — full thrust, the free windmill, the brake (``brake``) with its
  battery-side power;
- **descent**: the steady glide ``W sin γ = D − T`` with ``L = W cos γ`` solved for γ at each EAS and configuration
  (clean, crow, crow + the propeller brake), the sink rate ``V sin γ`` and the regeneration's power along it.

Units SI; angles in degrees where the name says so; the aircraft frame is ``falco``'s (x aft, y right, z up).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import falco
import falco_systems as fs
import merlin_flight as mf
import nisus_flight as nf
import peregrine_flight as pf
from merlin_flight import Airframe
from nisus_flight import AIRFOIL, TAIL_SECTION, ELEVATOR_TAU, RUDDER_TAU, AILERON_TAU, TAIL_EFFICIENCY, FUSELAGE_K, CL_MAX_FACTOR

G, RHO0 = fs.G, fs.RHO0
FLAP_TAU = 0.62                      # plain flap of 30 % chord: thin-airfoil effectiveness ~0.62 (Nelson fig. 2.21 class)
N_STRUCTURAL = 4.4                   # limit load factor: the pull-out after the fast descent and the mountain gusts
V_NE_EAS = 35.0                      # never-exceed [m/s EAS]
V_FE_EAS = 22.0                      # the most the flaps take deflected (crow) [m/s EAS]: their hinge loads and the servo
V_CRUISE_EAS = 18.0
FLAP_CL_MAX_2D = 0.9                 # Δcl_max of a 30 % plain flap at large deflection (Raymer fig. 12.12 class): assumed
CROW = dict(flap_deg=55.0, aileron_deg=-25.0)


def flap_k(delta_deg) -> float:
    """The plain flap's effectiveness against its linear value at large deflection (1 up to 12°, 0.75 at 30°, 0.55 at
    45°, 0.45 at 60°: assumed, Roskam Part VI fig. 8.53 class)."""
    return float(np.interp(abs(delta_deg), (0.0, 12.0, 30.0, 45.0, 60.0, 70.0), (1.0, 1.0, 0.75, 0.55, 0.45, 0.42)))


def flap_profile_drag(delta_deg, cf_c, area_share) -> float:
    """Raymer's plain-flap profile drag increment: 1.7 (cf/c)^1.38 (S_flapped/S) sin²δ (assumed, his eq. 12.61 class)."""
    return 1.7 * cf_c ** 1.38 * area_share * math.sin(math.radians(delta_deg)) ** 2


# ================================================================================================= the lattice
def _surfaces(pl, it_deg, de_deg=0.0, df_deg=0.0, da_deg=0.0, da_anti=0.0):
    """The wing's split quads and the tail with their incidences: the wing's camber zero-lift angle and incidence, the
    flap and aileron deflections (``da_deg`` symmetric, ``da_anti`` antisymmetric: right aileron up for a roll to the
    right), the tail's setting and elevator."""
    quads = list(pl["wing"]) + list(pl["tail"])
    inc = []
    for tag, side in zip(pl["wing_tags"], pl["wing_side"]):
        d = 0.0
        if tag == "flap":
            d = FLAP_TAU * flap_k(df_deg) * df_deg
        elif tag == "aileron":
            da = da_deg - side * da_anti
            d = AILERON_TAU * flap_k(da) * da
        inc.append(pl["wing_incidence_deg"] - AIRFOIL["alpha0_deg"] + d)
    inc += [it_deg + ELEVATOR_TAU * de_deg] * 2
    n_span = [max(2, int(round(np.ptp(q[:, 1]) / 0.04))) for q in pl["wing"]] + [6, 6]
    return quads, inc, n_span


def aero(p=None, *, x_ref=None, it_deg=None, n_chord=4) -> dict:
    """NISUS's lattice derivatives on FALCO (``nisus_flight.aero``'s recipe) plus the flaps' and the symmetric ailerons'
    CL and Cm per degree (linear: small deflections)."""
    p = falco.resolve(p)
    pl = falco.planform_split(p)
    it = p["tail_incidence_deg"] if it_deg is None else it_deg
    S, c = pl["S_ref"], pl["c_ref"]
    x_ref = pl["x_ac_wing"] if x_ref is None else x_ref
    kw = dict(S_ref=S, c_ref=c, x_ref=x_ref, n_chord=n_chord)

    def run(alpha=0.0, **d):
        q, inc, ns_ = _surfaces(pl, it, **d)
        return pf.vlm(q, alpha_deg=alpha, incidence_deg=inc, n_span=ns_, **kw)

    r0, r1 = run(0.0), run(4.0)
    re, rf, ra = run(de_deg=1.0), run(df_deg=1.0), run(da_deg=1.0)
    da = math.radians(4.0)
    CLa, Cma = (r1["CL"] - r0["CL"]) / da, (r1["Cm"] - r0["Cm"]) / da
    Wf, Lf = p["pod_width"] / 1000, p["pod_length"] / 1000
    Cma_f = FUSELAGE_K * Wf ** 2 * Lf / (c * S) * (180 / math.pi)
    x_np = x_ref - (Cma + Cma_f) / CLa * c
    lat = r0["lattice"]
    n_wing = len(pl["wing"])
    wing_panels = lat["quad"] < n_wing
    chord_strip = lat["area"][wing_panels] / np.maximum(np.abs(lat["B"][wing_panels, 1] - lat["A"][wing_panels, 1]), 1e-9)
    cm_af = AIRFOIL["cm_ac"] * float((lat["area"][wing_panels] * chord_strip).sum() / n_chord / (S * c))
    k = (r1["CDi"] - r0["CDi"]) / (r1["CL"] ** 2 - r0["CL"] ** 2)
    AR = pl["b_ref"] ** 2 / S
    tq = list(pl["tail"])
    t0 = pf.vlm(tq, alpha_deg=0.0, incidence_deg=[it, it], S_ref=S, c_ref=c, x_ref=x_ref, n_span=[6, 6], n_chord=n_chord)
    t1 = pf.vlm(tq, alpha_deg=4.0, incidence_deg=[it, it], S_ref=S, c_ref=c, x_ref=x_ref, n_span=[6, 6], n_chord=n_chord)
    CLa_t_alone = (t1["CL"] - t0["CL"]) / da
    lift_dir = lambda r, a: float((r["F"][~wing_panels] @ np.array([-math.sin(a), 0, math.cos(a)])).sum() / S)
    CLa_t_in_wake = (lift_dir(r1, da) - lift_dir(r0, 0.0)) / da
    deps = 1 - CLa_t_in_wake / CLa_t_alone if CLa_t_alone > 0 else math.nan
    return {"S_ref": S, "c_ref": c, "AR": AR, "b": pl["b_ref"], "x_ref": x_ref, "CL0": r0["CL"], "CLa": CLa, "Cm0": r0["Cm"] + cm_af,
            "Cma": Cma + Cma_f, "Cma_lattice": Cma, "Cma_pod": Cma_f, "x_np": x_np, "k": k, "e": 1 / (math.pi * AR * k),
            "CLde": re["CL"] - r0["CL"], "Cmde": re["Cm"] - r0["Cm"], "CLdf": rf["CL"] - r0["CL"], "Cmdf": rf["Cm"] - r0["Cm"],
            "CLda_sym": ra["CL"] - r0["CL"], "Cmda_sym": ra["Cm"] - r0["Cm"], "it_deg": it, "CLa_tail_alone": CLa_t_alone, "deps_dalpha": deps,
            "V_h": pl["V_h"], "V_v": pl["V_v"], "l_t": pl["l_t"], "l_v": pl["l_v"], "S_h": pl["S_h"], "S_v": pl["S_v"], "pl": pl}


def crow_increments(p=None, flap_deg=CROW["flap_deg"], aileron_deg=CROW["aileron_deg"], *, x_ref=None, a=None) -> dict:
    """The increments of a flap / aileron setting (crow: flaps down, ailerons up) against the clean wing at the same
    angle of attack: ΔCL, ΔCm (about ``x_ref``, default the MAC quarter chord), ΔCDi from the lattice with the
    effectiveness K(δ); the profile drag ΔCD_p of both surfaces (Raymer's plain-flap formula); ΔCL_max (the flaps'
    Δcl_max over the flapped span, the ailerons' loss likewise)."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    a = a or aero(p)
    pl = a["pl"]
    x_ref = pl["x_ac_wing"] if x_ref is None else x_ref
    kw = dict(S_ref=pl["S_ref"], c_ref=pl["c_ref"], x_ref=x_ref, n_chord=4)
    q0, i0, n0 = _surfaces(pl, a["it_deg"])
    q1, i1, n1 = _surfaces(pl, a["it_deg"], df_deg=flap_deg, da_deg=aileron_deg)
    out = {}
    for alpha in (0.0, 4.0):
        r0 = pf.vlm(q0, alpha_deg=alpha, incidence_deg=i0, n_span=n0, **kw)
        r1 = pf.vlm(q1, alpha_deg=alpha, incidence_deg=i1, n_span=n1, **kw)
        out[alpha] = (r0, r1)
    r0, r1 = out[0.0]
    dCL, dCm = r1["CL"] - r0["CL"], r1["Cm"] - r0["Cm"]
    # the induced drag at the same CL (the crow wing's span loading is worse): k_eff from the two angles
    (s0, s1) = out[4.0]
    k1 = (s1["CDi"] - r1["CDi"]) / max(s1["CL"] ** 2 - r1["CL"] ** 2, 1e-9)
    # profile drag
    share_f = L["flapped_area_share"]
    ya0, b2 = L["y_aileron0"], L["b2"]
    c_a0 = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * (ya0 - L["yc"]) / (b2 - L["yc"])
    share_a = 2 * 0.5 * (c_a0 + p["tip_chord"]) * (b2 - ya0) * 1e-6 / L["S_ref"]
    dCDp = flap_profile_drag(flap_deg, p["flap_chord_frac"], share_f) + flap_profile_drag(aileron_deg, p["aileron_chord_frac"], share_a)
    sgn = lambda x: (x > 0) - (x < 0)
    dCLmax = 0.9 * FLAP_CL_MAX_2D * (share_f * min(abs(flap_deg) / 40.0, 1.0) * sgn(flap_deg) + share_a * min(abs(aileron_deg) / 40.0, 1.0) * sgn(aileron_deg))
    return {"flap_deg": flap_deg, "aileron_deg": aileron_deg, "dCL": dCL, "dCm": dCm, "k_crow": k1, "k_clean": a["k"], "dCD_profile": dCDp,
            "dCL_max": dCLmax, "flapped_share": share_f, "aileron_share": share_a,
            "source": "ΔCL, ΔCm, k: calculated (lattice, K(δ) assumed); ΔCD profile: assumed (Raymer's plain-flap formula); ΔCL_max assumed"}


def crow_table(p=None, settings=((0, 0), (15, 0), (30, 0), (30, -15), (45, -20), (55, -25), (60, -30), (0, -25)), a=None) -> pd.DataFrame:
    a = a or aero(p)
    rows = {}
    for df_, da_ in settings:
        c = crow_increments(p, df_, da_, a=a)
        rows[f"flap {df_:+.0f}°, aileron {da_:+.0f}°"] = {k: c[k] for k in ("dCL", "dCm", "dCD_profile", "k_crow", "dCL_max")}
    return pd.DataFrame(rows).T


def cl_max(p=None) -> float:
    return CL_MAX_FACTOR * AIRFOIL["cl_max_2d"]


# ================================================================================================= the derivatives
def derivatives(p=None, *, x_cg_m: float, z_cg_m: float = -0.05, cd0: float | None = None, CL_ref: float = 0.45, h_m: float = 0.0, a=None,
                crow=CROW) -> pd.DataFrame:
    """NISUS's derivative table (``nisus_flight.derivatives``, the same formulas and tags) on FALCO's geometry, the Cd0
    from the build-up at the altitude ``h_m``'s viscosity, and the flap and crow rows: CLdf, Cmdf (lattice, per rad,
    linear), the crow increments at ``crow`` (CL, Cm about the CG, the profile drag, k), the symmetric aileron's CL, Cm."""
    p = falco.resolve(p)
    a = a or aero(p)
    L = falco.Falco.layout(p)
    atm = fs.atmosphere(h_m)
    cd0 = falco.drag_buildup(p, V_CRUISE_EAS / math.sqrt(atm["sigma"]), atm["nu"])["cd0"] if cd0 is None else cd0
    c, b, S = a["c_ref"], a["b"], a["S_ref"]
    shift = (x_cg_m - a["x_ref"]) / c
    CLa, CL0 = a["CLa"], a["CL0"]
    Cm0 = a["Cm0"] + CL0 * shift
    Cma = a["Cma"] + CLa * shift
    r2d = 180 / math.pi
    Cmde = (a["Cmde"] + a["CLde"] * shift) * r2d
    CLde = a["CLde"] * r2d
    CLdf, Cmdf = a["CLdf"] * r2d, (a["Cmdf"] + a["CLdf"] * shift) * r2d
    CLda_s, Cmda_s = a["CLda_sym"] * r2d, (a["Cmda_sym"] + a["CLda_sym"] * shift) * r2d
    eta = TAIL_EFFICIENCY
    CLa_t = a["CLa_tail_alone"] * S / a["S_h"]
    l_t = (L["x_ac_tail"] / 1000 - x_cg_m)
    V_h = a["S_h"] * l_t / (S * c)
    Cmq = -2 * eta * CLa_t * V_h * l_t / c
    CLq = 2 * eta * CLa_t * V_h
    Cmadot = Cmq * a["deps_dalpha"]
    h_v = (p["fin_height"] + p["fin_ventral"]) / 1000
    S_v1 = a["S_v"] / 2
    AR_v = 1.55 * h_v ** 2 / S_v1
    CLa_v = 2 * math.pi * AR_v / (2 + math.sqrt(AR_v ** 2 + 4))
    l_v = L["x_ac_fin"] / 1000 - x_cg_m
    z_v = (p["boom_z"] + 0.5 * (p["fin_height"] - p["fin_ventral"])) / 1000 - z_cg_m
    S_v = a["S_v"]
    Cyb_v = -eta * (S_v / S) * CLa_v
    Cyb_body = -0.08
    Cyb = Cyb_v + Cyb_body
    Cnb_v = -Cyb_v * l_v / b
    Cnb_body = -0.025
    Cnb = Cnb_v + Cnb_body
    gamma = math.radians(p["dihedral_deg"])
    Clb_dihedral = -CLa * gamma / 4 * (1 + 2 * L["taper"]) / (3 * (1 + L["taper"])) * 2
    Clb_wing_pos = -0.020
    Clb_fin = Cyb_v * z_v / b
    Clb = Clb_dihedral + Clb_wing_pos + Clb_fin
    Clp = -0.52
    Clr = CL_ref / 4 - 2 * Cyb_v * (l_v / b) * (z_v / b)
    Cnp = -CL_ref / 8
    Cnr = -2 * eta * (S_v / S) * (l_v / b) ** 2 * CLa_v - 0.02
    Cyp = -2 * Cyb_v * z_v / b * 0.5
    Cyr = -2 * Cyb_v * l_v / b
    y1, y2 = L["y_aileron0"] / 1000, b / 2
    c_r, c_t = p["root_chord"] / 1000, p["tip_chord"] / 1000
    yc = L["yc"] / 1000
    slope = (c_t - c_r) / (b / 2 - yc)
    integral = c_r * (y2 ** 2 - y1 ** 2) / 2 + slope * ((y2 ** 3 - y1 ** 3) / 3 - yc * (y2 ** 2 - y1 ** 2) / 2)
    Clda = 2 * CLa * AILERON_TAU / (S * b) * integral
    Cnda = -0.20 * CL_ref * Clda
    Cydr = eta * (S_v / S) * CLa_v * RUDDER_TAU
    Cndr = -Cydr * l_v / b
    Cldr = Cydr * z_v / b
    CD_de = 0.05
    cr = crow_increments(p, crow["flap_deg"], crow["aileron_deg"], x_ref=x_cg_m, a=a)
    rows = [
        ("CL0", CL0, "-", "calculated (lattice)", "lift at α = 0 from the pod axis: wing incidence and camber"),
        ("CLa", CLa, "1/rad", "calculated (lattice)", "lift slope, wing + tail"),
        ("CLq", CLq, "1/rad", "calculated (analytic)", "2 η CLα_t V_h"),
        ("CLde", CLde, "1/rad", "calculated (lattice)", f"elevator effectiveness τ = {ELEVATOR_TAU}"),
        ("CLdf", CLdf, "1/rad", "calculated (lattice)", f"both flaps, linear (τ = {FLAP_TAU}); K(δ) for large deflections"),
        ("Cmdf", Cmdf, "1/rad", "calculated (lattice)", "both flaps, about the CG, linear"),
        ("CLda_sym", CLda_s, "1/rad", "calculated (lattice)", "both ailerons the same way (crow's reflex), linear"),
        ("Cmda_sym", Cmda_s, "1/rad", "calculated (lattice)", "both ailerons the same way, about the CG"),
        ("dCL_crow", cr["dCL"], "-", "calculated (lattice, K(δ) assumed)", f"crow flaps {crow['flap_deg']:g}° / ailerons {crow['aileron_deg']:g}°"),
        ("dCm_crow", cr["dCm"], "-", "calculated (lattice, K(δ) assumed)", "crow, about the CG"),
        ("dCD_crow", cr["dCD_profile"], "-", "assumed", "Raymer's plain-flap profile drag of both surfaces (the crow CFD to check)"),
        ("k_crow", cr["k_crow"], "-", "calculated (lattice)", "induced-drag factor with crow set"),
        ("dCLmax_crow", cr["dCL_max"], "-", "assumed", f"Δcl_max {FLAP_CL_MAX_2D} over the flapped span, the ailerons' loss"),
        ("CD0", cd0, "-", "calculated (build-up)", f"flat-plate build-up at {V_CRUISE_EAS:g} m/s EAS, ν of {h_m:.0f} m (CFD to confirm)"),
        ("k", a["k"], "-", "calculated (lattice)", "CDi = k CL²"),
        ("e", a["e"], "-", "calculated (lattice)", "span efficiency (inviscid)"),
        ("CD_de", CD_de, "1/rad²", "assumed", "control deflection drag ΔCD = CD_de δ²"),
        ("Cm0", Cm0, "-", "calculated (lattice)", "about the CG; section Cm_ac included"),
        ("Cma", Cma, "1/rad", "calculated (lattice + pod term)", f"pod term {a['Cma_pod']:+.3f} (Raymer K = {FUSELAGE_K})"),
        ("Cmq", Cmq, "1/rad", "calculated (analytic)", "-2 η CLα_t V_h l_t/c"),
        ("Cmadot", Cmadot, "1/rad", "calculated (analytic)", "Cmq x dε/dα (not used by the simulation: quasi-steady)"),
        ("Cmde", Cmde, "1/rad", "calculated (lattice)", "elevator power about the CG"),
        ("Cyb", Cyb, "1/rad", "calculated (analytic) + assumed", f"fins {Cyb_v:+.3f} + pod {Cyb_body:+.3f} (assumed)"),
        ("Cyp", Cyp, "1/rad", "calculated (analytic)", "fins"),
        ("Cyr", Cyr, "1/rad", "calculated (analytic)", "fins"),
        ("Cydr", Cydr, "1/rad", "calculated (analytic)", f"rudders τ = {RUDDER_TAU}"),
        ("Clb", Clb, "1/rad", "calculated (analytic) + assumed", f"dihedral {Clb_dihedral:+.3f}, high wing {Clb_wing_pos:+.3f} (assumed), fins {Clb_fin:+.3f}"),
        ("Clp", Clp, "1/rad", "assumed", "roll damping, chart value for AR 9.6 (Roskam Part VI class)"),
        ("Clr", Clr, "1/rad", "calculated (analytic)", "CL/4 + fins (at CL_ref; rescaled with CL in the simulation)"),
        ("Clda", Clda, "1/rad", "calculated (analytic)", f"strip theory, τ = {AILERON_TAU}, ailerons from y = {y1:.3f} m"),
        ("Cldr", Cldr, "1/rad", "calculated (analytic)", "rudder roll coupling (fins above the CG)"),
        ("Cnb", Cnb, "1/rad", "calculated (analytic) + assumed", f"fins {Cnb_v:+.3f} + pod {Cnb_body:+.3f} (assumed)"),
        ("Cnp", Cnp, "1/rad", "calculated (analytic)", "-CL/8 (at CL_ref)"),
        ("Cnr", Cnr, "1/rad", "calculated (analytic) + assumed", "fins + 0.02 wing profile (assumed)"),
        ("Cnda", Cnda, "1/rad", "assumed", "adverse yaw -0.2 CL Clδa"),
        ("Cndr", Cndr, "1/rad", "calculated (analytic)", "rudder power"),
        ("CL_max", cl_max(p), "-", "assumed", f"{CL_MAX_FACTOR} x section cl_max {AIRFOIL['cl_max_2d']} (no lattice stall)"),
        ("alpha_stall_deg", math.degrees((cl_max(p) - CL0) / CLa), "deg", "assumed", "from CL_max and the linear slope"),
        ("x_np_m", a["x_np"], "m", "calculated (lattice + pod term)", "neutral point in the falco frame"),
        ("static_margin", (a["x_np"] - x_cg_m) / c, "MAC", "calculated", f"CG at x = {x_cg_m:.4f} m"),
        ("deps_dalpha", a["deps_dalpha"], "-", "calculated (lattice)", "downwash gradient at the tail"),
        ("CLa_t", CLa_t, "1/rad", "calculated (lattice)", "tail lift slope on S_h, isolated"),
        ("CLa_v", CLa_v, "1/rad", "calculated (analytic)", f"fin slope at AR_eff {AR_v:.2f} (end-plate factor 1.55 assumed)"),
    ]
    df = pd.DataFrame(rows, columns=["derivative", "value", "unit", "source", "note"]).set_index("derivative")
    df.attrs.update(S=S, c=c, b=b, x_cg_m=x_cg_m, z_cg_m=z_cg_m, l_t=l_t, l_v=l_v, V_h=V_h, z_v=z_v, cd0=cd0, h_m=h_m,
                    crow_flap_deg=crow["flap_deg"], crow_aileron_deg=crow["aileron_deg"])
    return df


coefficients = nf.coefficients
trim = nf.trim


def trim_crow(deriv: pd.DataFrame, mass_kg: float, V: float, rho=RHO0, crow_frac: float = 1.0, gamma_deg: float = 0.0) -> dict:
    """α and elevator with crow set (``crow_frac`` of the table's setting) in a glide at the path angle γ: lift W cos γ."""
    c = coefficients(deriv)
    cl = mass_kg * G * math.cos(math.radians(gamma_deg)) / (0.5 * rho * V ** 2 * c["S"])
    M = np.array([[c["CLa"], c["CLde"]], [c["Cma"], c["Cmde"]]])
    al, de = np.linalg.solve(M, [cl - c["CL0"] - crow_frac * c["dCL_crow"], -c["Cm0"] - crow_frac * c["dCm_crow"]])
    return {"V": V, "CL": cl, "alpha_deg": math.degrees(al), "elevator_deg": math.degrees(de), "stalled": cl > c["CL_max"] + crow_frac * c["dCLmax_crow"]}


def cg_range(p, mass_kg, *, sm_min: float = 0.05, de_max_deg: float = 20.0, rho=RHO0, a=None) -> dict:
    """NISUS's CG range (aft: the minimum static margin; forward: the elevator trims the stall in crow too) at density rho."""
    p = falco.resolve(p)
    a = a or aero(p)
    L = falco.Falco.layout(p)
    c = a["c_ref"]
    aft = a["x_np"] - sm_min * c
    xs = np.linspace(L["x_mac_le"] / 1000 - 0.1 * c, aft, 40)
    fwd = xs[0]
    for x in xs:
        d = derivatives(p, x_cg_m=x, a=a)
        cf = coefficients(d)
        Vs = math.sqrt(2 * mass_kg * G / (rho * cf["S"] * (cf["CL_max"] + cf["dCLmax_crow"])))
        t = trim_crow(d, mass_kg, 1.1 * Vs, rho)
        if abs(t["elevator_deg"]) <= de_max_deg:
            fwd = x
            break
    return {"x_fwd_m": float(fwd), "x_aft_m": float(aft), "fwd_frac_mac": (fwd - L["x_mac_le"] / 1000) / c, "aft_frac_mac": (aft - L["x_mac_le"] / 1000) / c,
            "x_np_frac_mac": (a["x_np"] - L["x_mac_le"] / 1000) / c, "sm_min": sm_min, "de_max_deg": de_max_deg}


# ================================================================================================= the airframe and the drive
def airframe(battery_key: str = fs.DEFAULT_PACK, p=None, *, h_m: float = 0.0, cd0_correction: float = 0.0, a=None) -> Airframe:
    """Merlin's parabolic-polar ``Airframe``: mass from the mass table, Cd0 from the build-up at ``h_m``'s viscosity
    (+ a CFD correction), Oswald e = 0.9 x the lattice's span efficiency, CL_max as assumed (clean)."""
    p = falco.resolve(p)
    a = a or aero(p)
    m = fs.cg_inertia(fs.mass_table(battery_key, p=p), p)["mass_kg"]
    atm = fs.atmosphere(h_m)
    b = falco.drag_buildup(p, V_CRUISE_EAS / math.sqrt(atm["sigma"]), atm["nu"])
    return Airframe("Falco-Zero", m, a["S_ref"], a["AR"], b["cd0"] + cd0_correction, oswald=pf.OSWALD_VISCOUS * a["e"], cl_max=cl_max(p))


def airframe_dict(af: Airframe) -> dict:
    return {"mass_kg": af.mass_kg, "cd0": af.cd0, "AR": af.aspect_ratio, "oswald": af.oswald, "S": af.wing_area_m2}


@dataclass
class DriveUnit:
    """``FalcoDrive`` at one density wearing Merlin's ``Unit`` interface (``full(V)``, ``electrical_power(V, T)``), so
    ``merlin_flight.performance`` and PEREGRINE's turn arithmetic run unchanged; ``fixed_w`` the electronics."""
    drive: fs.FalcoDrive
    rho: float = RHO0
    fixed_w: float = 0.0
    v_batt: float | None = None
    name: str = "AT4125 KV540 + APC 15x8E"

    @property
    def V(self):
        return self.drive.V

    def full(self, V):
        a = self.drive.at(V, 1.0, self.rho, self.v_batt)
        return a["thrust"], a["electrical"] + self.fixed_w, a["rpm"]

    def electrical_power(self, V, T_net):
        if T_net <= 0:
            return self.fixed_w
        P = self.drive.electrical_for_thrust(V, T_net, self.rho, self.v_batt)
        return P + self.fixed_w if np.isfinite(P) else math.inf


def envelope(af: Airframe, dr: fs.FalcoDrive, h_m: float = 0.0, *, dT: float = 0.0, battery_wh: float = 350.0, usable: float = 0.85 * 0.9,
             electronics_w: float = 0.0, n_struct=N_STRUCTURAL) -> dict:
    """NISUS's envelope at an altitude: the speed polar (``merlin_flight.performance``), endurance and range, glide, turns."""
    atm = fs.atmosphere(h_m, dT)
    rho = atm["rho"]
    unit = DriveUnit(dr, rho, electronics_w)
    V = np.linspace(max(af.stall_speed(rho) * 1.05, 6.0), V_NE_EAS / math.sqrt(atm["sigma"]), 100)
    perf = mf.performance(af, unit, rho, V=V)
    P = np.where(np.isfinite(perf["P_level"]), perf["P_level"], np.inf)
    i_e = int(np.argmin(P))
    tr = pf.turn(af, unit, V, n_struct, rho)
    ld = af.mass_kg * G / af.drag(V, rho)
    sink = af.drag(V, rho) * V / (af.mass_kg * G)
    return {**perf, **tr, "h_m": h_m, "rho": rho, "sigma": atm["sigma"], "L/D": ld, "sink": sink, "v_endurance": float(V[i_e]), "P_min": float(P[i_e]),
            "endurance_min": battery_wh * usable / P[i_e] * 60, "range_km": battery_wh * usable / perf["wh_per_km_best"],
            "L/D_max": float(ld.max()), "v_L/D_max": float(V[np.argmax(ld)]), "sink_min_m_s": float(sink.min()), "v_sink_min": float(V[np.argmin(sink)])}


def envelope_vs_altitude(af: Airframe, dr: fs.FalcoDrive, heights=(0.0, 1500.0, 3000.0, 4500.0, 6000.0), *, dT: float = 0.0,
                         electronics_w: float = 0.0, battery_wh: float = 350.0) -> pd.DataFrame:
    """The envelope at each altitude: stall (TAS, and its EAS stays put), best climb, the most the drive climbs,
    minimum power, endurance, top speed, glide, minimum sink, the never-exceed speed in TAS."""
    rows = {}
    for h in heights:
        e = envelope(af, dr, h, dT=dT, electronics_w=electronics_w, battery_wh=battery_wh)
        rows[f"{h:.0f} m"] = {"σ": e["sigma"], "stall TAS [m/s]": af.stall_speed(e["rho"]), "best-climb TAS [m/s]": e["v_climb"], "climb max [m/s]": e["roc_max"],
                              "min power [W]": e["P_min"], "at TAS [m/s]": e["v_endurance"], "endurance [min]": e["endurance_min"], "top speed TAS [m/s]": e["v_top"],
                              "L/D max": e["L/D_max"], "min sink [m/s]": e["sink_min_m_s"], "V_NE TAS [m/s]": V_NE_EAS / math.sqrt(e["sigma"])}
    return pd.DataFrame(rows).T


def ceiling(af: Airframe, dr: fs.FalcoDrive, *, dT: float = 0.0, service_roc: float = 0.5, h_max: float = 9000.0) -> dict:
    """The service (``service_roc``) and absolute (0) ceilings: bisection on the altitude of the best climb rate."""
    def roc(h):
        return envelope(af, dr, h, dT=dT)["roc_max"]

    out = {}
    for name, target in (("service", service_roc), ("absolute", 0.0)):
        lo, hi = 0.0, h_max
        if roc(hi) > target:
            out[name] = hi
            continue
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if roc(mid) > target else (lo, mid)
        out[name] = lo
    return {"service_ceiling_m": out["service"], "absolute_ceiling_m": out["absolute"], "service_roc": service_roc, "dT": dT,
            "note": "the drive scales with the density only (no Reynolds or Mach effect on the blades), the pack at its nominal voltage; a ceiling at h_max means beyond the model's edge (9000 m: the troposphere ISA, the blades' Re)"}


# ================================================================================================= climb and descent
def climb_strategy(dr: fs.FalcoDrive, af: Airframe, plan: fs.MissionPlan = fs.MissionPlan(), rates=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0),
                   electronics_w: float = 33.6) -> pd.DataFrame:
    """Is a fast climb cheap? The climb from the launch to the work altitude at each rate (capped at what the drive gives):
    its time, the propulsion energy, the electronics' energy over the climb, the total and the energy per 100 m."""
    rows = {}
    for roc in rates:
        q = fs.MissionPlan(**{**asdict(plan), "roc": roc})
        cl = fs.climb_profile(dr, airframe_dict(af), q)
        t = float(cl["time [s]"].sum())
        E_p = float(cl["energy [Wh]"].sum())
        E_e = electronics_w * t / 3600
        dh = plan.work_alt_m - plan.launch_alt_m
        rows[f"{roc:g} m/s"] = {"climb rate held (mean) [m/s]": dh / t, "capped by the drive": bool((cl["roc [m/s]"] < roc - 1e-6).any()), "time [min]": t / 60,
                                "propulsion [Wh]": E_p, "electronics [Wh]": E_e, "total [Wh]": E_p + E_e, "per 100 m [Wh]": 100 * (E_p + E_e) / dh,
                                "potential energy share": af.mass_kg * G * dh / 3600 / (E_p + E_e), "horizontal [km]": float(cl["horizontal [m]"].sum()) / 1000}
    return pd.DataFrame(rows).T


def glide_state(af: Airframe, deriv_c: dict, V: float, rho: float, *, crow: float = 0.0, thrust: float = 0.0) -> dict:
    """The steady glide at TAS V: W sin γ = D − T, L = W cos γ (fixed point in γ), with crow (0..1 of the table's
    setting: its ΔCD and k) and the propeller's thrust (negative: the brake's drag). Returns γ [deg, down +], the
    sink rate, CL, CD and whether the wing stalls."""
    W = af.mass_kg * G
    q = 0.5 * rho * V ** 2
    S = af.wing_area_m2
    k = af.k + crow * (deriv_c.get("k_crow", af.k) - deriv_c.get("k", af.k))
    cd0 = af.cd0 + crow * deriv_c.get("dCD_crow", 0.0)
    g = 0.0
    for _ in range(60):
        CL = W * math.cos(g) / (q * S)
        D = q * S * (cd0 + k * CL ** 2)
        s = max(-1.0, min(1.0, (D - thrust) / W))
        g_new = math.asin(s)
        if abs(g_new - g) < 1e-7:
            break
        g = 0.5 * (g + g_new)
    clmax = af.cl_max + crow * deriv_c.get("dCLmax_crow", 0.0)
    return {"gamma_deg": math.degrees(g), "sink": V * math.sin(g), "CL": CL, "CD": cd0 + k * CL ** 2, "drag": D, "stalls": CL > clmax}


def descent_table(af: Airframe, dr: fs.FalcoDrive, deriv: pd.DataFrame, h_m: float = 3000.0, *, dT: float = 0.0, eas=(14.0, 16.0, 18.0, 20.0, 22.0, 26.0, 30.0, 35.0),
                  pack_T_C: float = 15.0, soc: float = 0.6, pk: fs.LiIonPack | None = None) -> pd.DataFrame:
    """The descent at altitude ``h_m``, EAS by EAS, in four configurations: clean with the motor freewheeling, crow
    (only up to V_FE), the propeller brake alone (its full brake, the regeneration within the pack's charge limit), crow +
    brake. Sink rate [m/s], path angle, the battery-side power (negative: charging)."""
    c = coefficients(deriv)
    atm = fs.atmosphere(h_m, dT)
    rho = atm["rho"]
    pk = pk or fs.pack()
    i_chg = pk.charge_limit_a(pack_T_C, soc)
    rows = []
    for ve in eas:
        V = ve / math.sqrt(atm["sigma"])
        fw = dr.freewheel(V, rho)
        br = dr.brake(V, 1.0, rho, i_charge_max=i_chg)
        for name, crow, prop in (("clean, motor freewheeling", 0.0, fw), ("crow", 1.0, fw), ("propeller brake", 0.0, br), ("crow + propeller brake", 1.0, br)):
            if crow and ve > V_FE_EAS + 1e-9:
                continue
            g = glide_state(af, c, V, rho, crow=crow, thrust=prop["thrust"])
            rows.append({"EAS [m/s]": ve, "TAS [m/s]": V, "configuration": name, "sink [m/s]": g["sink"], "path angle [deg]": g["gamma_deg"],
                         "propeller thrust [N]": prop["thrust"], "battery side [W]": prop["electrical"], "rpm": prop["rpm"], "mode": prop["mode"],
                         "CL": g["CL"], "stalls": g["stalls"]})
    return pd.DataFrame(rows)


def best_descent(table: pd.DataFrame) -> pd.DataFrame:
    """The fastest sink of each configuration in a descent table and the EAS where it happens."""
    i = table.groupby("configuration")["sink [m/s]"].idxmax()
    return table.loc[i].set_index("configuration")


def regen_table(af: Airframe, dr: fs.FalcoDrive, deriv: pd.DataFrame, *, h_top: float = 4000.0, h_bottom: float = 1200.0, eas: float = 22.0,
                pk: fs.LiIonPack | None = None, pack_T_C: float = 15.0, soc: float = 0.6, brakes=(0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Does regeneration pay? The descent from ``h_top`` to ``h_bottom`` at ``eas`` with crow, the propeller braked
    by b (0: freewheeling), integrated in 100 m steps: the time, the sink, the energy returned to the pack and its
    share of the pack, against the potential energy given up; also with a cold pack (no charging: the short-circuit
    brake makes the drag, the pack gets nothing)."""
    pk = pk or fs.pack()
    c = coefficients(deriv)
    W = af.mass_kg * G
    rows = {}
    for label, T_C in (("warm pack", pack_T_C), ("cold pack (< 5 °C: no charging)", min(pack_T_C, 0.0))):
        i_chg = pk.charge_limit_a(T_C, soc)
        for b in (0.0,) + tuple(brakes):
            t = E = 0.0
            for h in np.arange(h_bottom, h_top, 100.0):
                atm = fs.atmosphere(h + 50.0)
                V = eas / math.sqrt(atm["sigma"])
                pr = dr.freewheel(V, atm["rho"]) if b == 0 else dr.brake(V, b, atm["rho"], i_charge_max=i_chg)
                g = glide_state(af, c, V, atm["rho"], crow=1.0, thrust=pr["thrust"])
                dt = 100.0 / max(g["sink"], 0.1)
                t += dt
                E += -min(pr["electrical"], 0.0) * dt / 3600
            rows[(label, f"brake {b:.2f}")] = {"descent time [s]": t, "mean sink [m/s]": (h_top - h_bottom) / t, "recovered [Wh]": E,
                                              "recovered per 1000 m [Wh]": E * 1000 / (h_top - h_bottom),
                                              "share of the pack [%]": 100 * E / pk.energy_wh,
                                              "share of the potential energy [%]": 100 * E / (W * (h_top - h_bottom) / 3600)}
    return pd.DataFrame(rows).T


def downdraft_escape(af: Airframe, dr: fs.FalcoDrive, *, heights=(1500.0, 3000.0, 4000.0, 4500.0), downdrafts=(2.0, 4.0, 6.0), dT: float = 0.0) -> pd.DataFrame:
    """Can the climb out-climb a lee downdraft? The best climb rate at each altitude against the sink of the air: the
    net climb (positive: it still gains height)."""
    rows = {}
    for h in heights:
        r = envelope(af, dr, h, dT=dT)["roc_max"]
        rows[f"{h:.0f} m"] = {"climb max [m/s]": r, **{f"net in a {u:g} m/s downdraft [m/s]": r - u for u in downdrafts}}
    return pd.DataFrame(rows).T


def launch_check(af: Airframe, dr: fs.FalcoDrive, h_m: float = 3000.0, *, v_release: float = 11.0, flap_deg: float = 15.0, pitch_deg: float = 8.0,
                 dT: float = 0.0, a=None, p=None, duration: float = 4.0) -> dict:
    """A launch at a site of altitude ``h_m``: released at ``v_release`` (a strong hand throw ~11 m/s; a light bungee
    ~18 m/s) at full power with the take-off flap, the wing held at the stall's angle at most: a point mass in the
    vertical plane; the height lost before it climbs and the time to reach 1.2 V_s (flaps set)."""
    p = falco.resolve(p)
    a = a or aero(p)
    atm = fs.atmosphere(h_m, dT)
    rho = atm["rho"]
    cr = crow_increments(p, flap_deg, 0.0, a=a)
    W, m, S = af.mass_kg * G, af.mass_kg, af.wing_area_m2
    clmax = af.cl_max + cr["dCL_max"]
    Vs = math.sqrt(2 * W / (rho * S * clmax))
    V, gam, z, x = v_release, math.radians(pitch_deg) * 0.5, 0.0, 0.0
    dt = 0.01
    z_min, t_ok = 0.0, None
    for i in range(int(duration / dt)):
        q = 0.5 * rho * V * V
        CL = min(clmax, W * math.cos(gam) / (q * S) * 1.05)
        T = dr.at(V, 1.0, rho)["thrust"]
        D = q * S * (af.cd0 + cr["dCD_profile"] + af.k * CL * CL)
        L = q * S * CL
        dV = (T - D - W * math.sin(gam)) / m
        dg = (L - W * math.cos(gam)) / (m * max(V, 3.0))
        V += dV * dt
        gam += dg * dt
        x += V * math.cos(gam) * dt
        z += V * math.sin(gam) * dt
        z_min = min(z_min, z)
        if t_ok is None and V >= 1.2 * Vs:
            t_ok = (i + 1) * dt
    return {"altitude_m": h_m, "release_speed": v_release, "flap_deg": flap_deg, "V_stall_flaps": Vs, "static_T_over_W": dr.at(0.0, 1.0, rho)["thrust"] / W,
            "height_lost_m": -z_min, "time_to_1.2Vs_s": t_ok, "height_after_s": z, "speed_after": V, "ok": bool(-z_min < 1.0 and t_ok is not None)}


def launch_table(af, dr, heights=(1200.0, 2000.0, 3000.0), releases=(9.0, 11.0, 18.0), **kw) -> pd.DataFrame:
    rows = {}
    for h in heights:
        for v in releases:
            r = launch_check(af, dr, h, v_release=v, **kw)
            rows[(f"{h:.0f} m", f"{v:g} m/s")] = {k: r[k] for k in ("V_stall_flaps", "static_T_over_W", "height_lost_m", "time_to_1.2Vs_s", "height_after_s", "ok")}
    return pd.DataFrame(rows).T


# ================================================================================================= flight checks
stall_table = nf.stall_table
gust_response = nf.gust_response
pull_up = nf.pull_up


def operating_limits(af: Airframe, dr: fs.FalcoDrive, deriv: pd.DataFrame, cg: dict, *, h_work: float = 4000.0, dT: float = 0.0) -> pd.DataFrame:
    """Provisional operating limits (EAS where the structure decides, TAS where the performance does)."""
    atm = fs.atmosphere(h_work, dT)
    Vs0 = af.stall_speed(RHO0)
    e = envelope(af, dr, h_work, dT=dT)
    g = gust_response(af, deriv, V_CRUISE_EAS / math.sqrt(atm["sigma"]), 8.0, atm["rho"])
    rows = {
        "stall speed, level [m/s EAS]": (Vs0, f"CL_max {af.cl_max:.2f} (clean, assumed); TAS at {h_work:.0f} m: {Vs0 / math.sqrt(atm['sigma']):.1f}"),
        "minimum speed in flight [m/s EAS]": (1.25 * Vs0, "1.25 Vs (the mountain air's gusts)"),
        "approach speed [m/s EAS]": (1.3 * Vs0, "1.3 Vs, crow set"),
        "cruise speed [m/s EAS]": (V_CRUISE_EAS, f"TAS at {h_work:.0f} m: {V_CRUISE_EAS / math.sqrt(atm['sigma']):.1f}"),
        "never-exceed speed [m/s EAS]": (V_NE_EAS, f"TAS at {h_work:.0f} m: {V_NE_EAS / math.sqrt(atm['sigma']):.1f}; flutter is a TAS problem: not computed (todo)"),
        "maximum flap-extended (crow) speed V_FE [m/s EAS]": (V_FE_EAS, "the flaps' hinge loads and servo (falco_systems.servo_check)"),
        "manoeuvring speed V_A [m/s EAS]": (math.sqrt(2 * N_STRUCTURAL * af.mass_kg * G / (RHO0 * af.wing_area_m2 * af.cl_max)), f"n = {N_STRUCTURAL:g} at CL_max"),
        "limit load factor": (N_STRUCTURAL, "the pull-out after a fast descent; the mountain gusts (falco_structure)"),
        "maximum bank, survey turns [deg]": (45.0, "n = 1.41"),
        f"best climb at {h_work:.0f} m [m/s]": (e["roc_max"], f"at {e['v_climb']:.1f} m/s TAS, full throttle"),
        "maximum steady wind [m/s]": (12.0, "two thirds of the cruise EAS"),
        f"gust Δn at cruise, {h_work:.0f} m, 8 m/s": (g["delta_n"], f"{'stalls' if g['stalls_in_gust'] else 'no stall'} at cruise CL"),
        "CG forward limit [% MAC]": (100 * cg["fwd_frac_mac"], f"elevator ≤ {cg['de_max_deg']:g}° trims 1.1 Vs in crow"),
        "CG aft limit [% MAC]": (100 * cg["aft_frac_mac"], f"static margin ≥ {100 * cg['sm_min']:.0f} %"),
        "maximum take-off mass [kg]": (af.mass_kg * 1.08, "8 % over the budget; re-check the launch at the highest site"),
        "battery reserve at landing": (0.20, "of the derated available energy"),
        "pack temperature for regeneration [°C]": (fs.CHARGE_MIN_C, "no charging below it: the brake is the short circuit then"),
    }
    return pd.DataFrame({k: {"value": v[0], "basis": v[1]} for k, v in rows.items()}).T


__all__ = ["FLAP_TAU", "N_STRUCTURAL", "V_NE_EAS", "V_FE_EAS", "V_CRUISE_EAS", "CROW", "flap_k", "flap_profile_drag", "aero", "crow_increments",
           "crow_table", "cl_max", "derivatives", "coefficients", "trim", "trim_crow", "cg_range", "airframe", "airframe_dict", "DriveUnit", "envelope",
           "envelope_vs_altitude", "ceiling", "climb_strategy", "glide_state", "descent_table", "best_descent", "regen_table", "downdraft_escape",
           "launch_check", "launch_table", "stall_table", "gust_response", "pull_up", "operating_limits"]
