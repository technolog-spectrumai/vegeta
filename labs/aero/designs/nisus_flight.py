"""NISUS flight — aerodynamics, stability, trim, the envelope, the flight checks and the coefficient table the
six-degree-of-freedom simulation flies with (notebook 31).

Reduced models, honest about what they are, on top of what the repository already validated:

- the longitudinal aerodynamics come from PEREGRINE's **vortex lattice** (``peregrine_flight.vlm``: horseshoe
  vortices on flat panels, Trefftz-plane induced drag) on ``nisus.planform``'s quads — lift slope, induced-drag
  factor, span efficiency, neutral point, elevator power — **calculated**;
- the lateral-directional derivatives are **analytic** estimates (tail-volume and strip-theory formulas of the
  textbooks: Nelson, Roskam, Etkin) with the pod's terms **assumed**; the rate derivatives Cmq, CLq, Cnr likewise;
- the parasite drag is ``nisus.drag_buildup`` (flat-plate build-up) until the CFD or a flight test replaces it;
- the stall is a section value (**assumed**) and a smooth post-stall blend for the simulation: the lattice knows
  nothing about separation, and nothing here is a validated stall or recovery prediction.

Every number in ``derivatives`` carries its tag (``calculated (lattice)``, ``calculated (analytic)``, ``assumed``).
Units SI; angles in degrees where the name says so; the aircraft frame is ``nisus``'s (x aft, y right, z up) for
positions, the usual body axes (x forward, y right, z down) for the coefficients the simulation uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import merlin_flight as mf
import nisus
import nisus_systems as ns
import peregrine_flight as pf
from merlin_flight import G, RHO, Airframe

# ------------------------------------------------------------------------------------------------- section values
AIRFOIL = {
    "name": "NACA 3412 (wing)", "alpha0_deg": -3.1, "cm_ac": -0.080, "cl_max_2d": 1.30,
    "source": "thin-airfoil theory for 3 % camber at 40 % chord (α0 = 1.5 x the NACA 2412 value of -2.1°, Cm_ac likewise); "
              "cl_max 1.3 at Re ~ 2e5 is an assumption for a 12 % section (XFOIL/wind-tunnel data to replace it)",
}
TAIL_SECTION = {"name": "flat plate / NACA 0005 (tail)", "cl_max_2d": 0.80, "source": "assumed: thin symmetric section at Re ~ 1e5"}
ELEVATOR_TAU, RUDDER_TAU, AILERON_TAU = 0.55, 0.55, 0.50     # flap effectiveness for 35 / 35 / 25 % chord plain flaps (Nelson fig. 2.21 class)
TAIL_EFFICIENCY = 0.90                                       # dynamic pressure ratio at the tail (clear of the pod, in the propeller's wake edge): assumed
FUSELAGE_K = 0.010                                           # Raymer's pitching-moment factor per degree for a wing at ~45 % of the body: assumed
CL_MAX_FACTOR = 0.90                                         # aircraft CL_max / section cl_max (taper, pod cut-out, tail download)
N_STRUCTURAL = 4.0                                           # the limit load factor the structure is checked for (the FEA cases)
V_NE = 28.0                                                  # never-exceed airspeed [m/s] (the FEA's dash case; 1.75 x cruise)


# ================================================================================================= the lattice
def _surfaces(pl, it_deg, de_deg=0.0):
    quads = list(pl["wing"]) + list(pl["tail"])
    inc = [pl["wing_incidence_deg"] - AIRFOIL["alpha0_deg"]] * len(pl["wing"]) + [it_deg + ELEVATOR_TAU * de_deg] * 2
    n_span = [max(2, int(round(np.ptp(q[:, 1]) / 0.03))) for q in pl["wing"]] + [5, 5]
    return quads, inc, n_span


def aero(p=None, *, x_ref=None, it_deg=None, n_chord=4) -> dict:
    """The lattice's longitudinal derivatives referred to ``S_ref`` and the MAC: CL0 and CLα (per rad, the
    incidences included), the induced-drag factor k = CDi/CL², the span efficiency e, the neutral point (lattice +
    the pod's destabilising term), Cm0 about ``x_ref`` (default the MAC quarter chord) with the section's own
    moment, the elevator's CLδe and Cmδe (per degree), the tail's share and the downwash gradient at the tail
    (from a run without the wing). Angles of attack are measured from the pod axis."""
    p = nisus.resolve(p)
    pl = nisus.planform(p)
    it = p["tail_incidence_deg"] if it_deg is None else it_deg
    S, c = pl["S_ref"], pl["c_ref"]
    x_ref = pl["x_ac_wing"] if x_ref is None else x_ref
    quads, inc, ns_ = _surfaces(pl, it)
    kw = dict(S_ref=S, c_ref=c, x_ref=x_ref, n_span=ns_, n_chord=n_chord)
    r0 = pf.vlm(quads, alpha_deg=0.0, incidence_deg=inc, **kw)
    r1 = pf.vlm(quads, alpha_deg=4.0, incidence_deg=inc, **kw)
    _, inc_e, _ = _surfaces(pl, it, de_deg=1.0)
    re = pf.vlm(quads, alpha_deg=0.0, incidence_deg=inc_e, **kw)
    da = math.radians(4.0)
    CLa, Cma = (r1["CL"] - r0["CL"]) / da, (r1["Cm"] - r0["Cm"]) / da
    # the pod: Raymer's Cm_alpha,fus = K W_f^2 L_f / (c S) per degree
    Wf, Lf = p["pod_width"] / 1000, p["pod_length"] / 1000
    Cma_f = FUSELAGE_K * Wf ** 2 * Lf / (c * S) * (180 / math.pi)
    x_np = x_ref - (Cma + Cma_f) / CLa * c
    lat = r0["lattice"]
    wing_panels = lat["quad"] < len(pl["wing"])
    chord_strip = lat["area"][wing_panels] / np.maximum(np.abs(lat["B"][wing_panels, 1] - lat["A"][wing_panels, 1]), 1e-9)
    cm_af = AIRFOIL["cm_ac"] * float((lat["area"][wing_panels] * chord_strip).sum() / n_chord / (S * c))
    k = (r1["CDi"] - r0["CDi"]) / (r1["CL"] ** 2 - r0["CL"] ** 2)
    AR = pl["b_ref"] ** 2 / S
    tail_share = float(r1["F"][~wing_panels, 2].sum() / r1["F"][:, 2].sum())
    # the tail alone (no wing) gives its isolated lift slope; the ratio to its slope in the wing's wake is 1 - dε/dα
    tq = list(pl["tail"])
    t0 = pf.vlm(tq, alpha_deg=0.0, incidence_deg=[it, it], S_ref=S, c_ref=c, x_ref=x_ref, n_span=[5, 5], n_chord=n_chord)
    t1 = pf.vlm(tq, alpha_deg=4.0, incidence_deg=[it, it], S_ref=S, c_ref=c, x_ref=x_ref, n_span=[5, 5], n_chord=n_chord)
    CLa_t_alone = (t1["CL"] - t0["CL"]) / da                                     # referred to S_ref
    lift_dir = lambda r, a: float((r["F"][~wing_panels] @ np.array([-math.sin(a), 0, math.cos(a)])).sum() / S)
    CLa_t_in_wake = (lift_dir(r1, da) - lift_dir(r0, 0.0)) / da
    deps = 1 - CLa_t_in_wake / CLa_t_alone if CLa_t_alone > 0 else math.nan
    return {"S_ref": S, "c_ref": c, "AR": AR, "b": pl["b_ref"], "x_ref": x_ref, "CL0": r0["CL"], "CLa": CLa, "Cm0": r0["Cm"] + cm_af,
            "Cma": Cma + Cma_f, "Cma_lattice": Cma, "Cma_pod": Cma_f, "x_np": x_np, "k": k, "e": 1 / (math.pi * AR * k),
            "CLde": re["CL"] - r0["CL"], "Cmde": re["Cm"] - r0["Cm"], "tail_share": tail_share, "it_deg": it,
            "CLa_tail_alone": CLa_t_alone, "deps_dalpha": deps, "V_h": pl["V_h"], "V_v": pl["V_v"], "l_t": pl["l_t"], "l_v": pl["l_v"],
            "S_h": pl["S_h"], "S_v": pl["S_v"]}


def polar(p=None, alphas_deg=np.arange(-6, 15, 2), *, cd0=None, de_deg=0.0) -> pd.DataFrame:
    """CL, CDi, CD (with the build-up's Cd0), Cm about the quarter MAC against the angle of attack from the lattice
    (attached flow: no stall; the rows above ~10° are extrapolations the CFD or the tunnel must check)."""
    p = nisus.resolve(p)
    pl = nisus.planform(p)
    cd0 = nisus.drag_buildup(p, 16.0)["cd0"] if cd0 is None else cd0
    quads, inc, ns_ = _surfaces(pl, p["tail_incidence_deg"], de_deg)
    rows = []
    for a in alphas_deg:
        r = pf.vlm(quads, alpha_deg=float(a), incidence_deg=inc, S_ref=pl["S_ref"], c_ref=pl["c_ref"], x_ref=pl["x_ac_wing"], n_span=ns_, n_chord=4)
        rows.append({"alpha [deg]": float(a), "CL": r["CL"], "CDi": r["CDi"], "CD": cd0 + r["CDi"], "Cm (quarter MAC)": r["Cm"] + AIRFOIL["cm_ac"] * 0.95,
                     "L/D": r["CL"] / (cd0 + r["CDi"]) if cd0 + r["CDi"] > 0 else np.nan})
    return pd.DataFrame(rows).set_index("alpha [deg]")


def cl_max(p=None) -> float:
    return CL_MAX_FACTOR * AIRFOIL["cl_max_2d"]


# ================================================================================================= the derivatives
def derivatives(p=None, *, x_cg_m: float, z_cg_m: float = -0.04, cd0: float | None = None, CL_ref: float = 0.4) -> pd.DataFrame:
    """The full set of non-dimensional derivatives about the centre of gravity (body axes: x forward, y right, z
    down; rates non-dimensionalised with b/2V and c/2V, deflections in radians), one row each with value, unit and
    source tag. ``CL_ref`` sets the lift-dependent lateral terms (Clr, Cnp, adverse yaw) at the reference flight
    condition; the simulation rescales them with the instantaneous CL."""
    p = nisus.resolve(p)
    a = aero(p)
    L = nisus.Nisus.layout(p)
    cd0 = nisus.drag_buildup(p, 16.0)["cd0"] if cd0 is None else cd0
    c, b, S = a["c_ref"], a["b"], a["S_ref"]
    shift = (x_cg_m - a["x_ref"]) / c
    CLa, CL0 = a["CLa"], a["CL0"]
    Cm0 = a["Cm0"] + CL0 * shift
    Cma = a["Cma"] + CLa * shift
    Cmde = (a["Cmde"] + a["CLde"] * shift) * 180 / math.pi
    CLde = a["CLde"] * 180 / math.pi
    eta = TAIL_EFFICIENCY
    # the tail's slope on S_h (the lattice gives it on S_ref)
    CLa_t = a["CLa_tail_alone"] * S / a["S_h"]
    l_t = (L["x_ac_tail"] / 1000 - x_cg_m)
    V_h = a["S_h"] * l_t / (S * c)
    Cmq = -2 * eta * CLa_t * V_h * l_t / c
    CLq = 2 * eta * CLa_t * V_h
    Cmadot = Cmq * a["deps_dalpha"]
    # the fins: each an end-plated low-aspect-ratio surface
    h_v = (p["fin_height"] + p["fin_ventral"]) / 1000
    S_v1 = a["S_v"] / 2
    AR_v = 1.55 * h_v ** 2 / S_v1                                 # end-plate effect of the stabiliser (assumed factor)
    CLa_v = 2 * math.pi * AR_v / (2 + math.sqrt(AR_v ** 2 + 4))
    l_v = L["x_ac_fin"] / 1000 - x_cg_m
    z_v = (p["boom_z"] + 0.5 * (p["fin_height"] - p["fin_ventral"])) / 1000 - z_cg_m     # fin ac above the CG
    S_v = a["S_v"]
    Cyb_v = -eta * (S_v / S) * CLa_v
    Cyb_body = -0.10                                              # the pod and booms (assumed)
    Cyb = Cyb_v + Cyb_body
    Cnb_v = -Cyb_v * l_v / b
    Cnb_body = -0.03                                              # the pod (assumed; a slender body ahead of the CG)
    Cnb = Cnb_v + Cnb_body
    gamma = math.radians(p["dihedral_deg"])
    Clb_dihedral = -CLa * gamma / 4 * (1 + 2 * L["taper"]) / (3 * (1 + L["taper"])) * 2    # Nelson: -CLα Γ/4 x (taper weighting)
    Clb_wing_pos = -0.020                                          # high wing on the pod: stabilising (assumed, Roskam class value)
    Clb_fin = Cyb_v * z_v / b
    Clb = Clb_dihedral + Clb_wing_pos + Clb_fin
    Clp = -0.48                                                    # AR 7, taper 0.74: Roskam Part VI chart class (assumed)
    Clr = CL_ref / 4 - 2 * Cyb_v * (l_v / b) * (z_v / b)          # CL/4 (classic) + the fins above the CG
    Cnp = -CL_ref / 8                                              # -CL/8 (classic)
    Cnr = -2 * eta * (S_v / S) * (l_v / b) ** 2 * CLa_v - 0.02      # fins + the wing's profile drag term (assumed 0.02)
    Cyp = -2 * Cyb_v * z_v / b * 0.5                                # small
    Cyr = -2 * Cyb_v * l_v / b
    # ailerons: strip theory (Nelson eq. 3.12) over the aileron span with the flap effectiveness
    y1, y2 = L["y_aileron0"] / 1000, b / 2
    c_r, c_t = p["root_chord"] / 1000, p["tip_chord"] / 1000
    yc = L["yc"] / 1000
    slope = (c_t - c_r) / (b / 2 - yc)
    c_at = lambda y: c_r + slope * (y - yc)
    integral = c_r * (y2 ** 2 - y1 ** 2) / 2 + slope * ((y2 ** 3 - y1 ** 3) / 3 - yc * (y2 ** 2 - y1 ** 2) / 2)
    Clda = 2 * CLa * AILERON_TAU / (S * b) * integral             # per rad, both ailerons differential
    Cnda = -0.20 * CL_ref * Clda                                    # adverse yaw (assumed factor)
    Cydr = eta * (S_v / S) * CLa_v * RUDDER_TAU
    Cndr = -Cydr * l_v / b
    Cldr = Cydr * z_v / b
    CD_de = 0.05                                                   # drag of a deflected elevator/rudder: ΔCD = CD_de δ² (assumed)
    rows = [
        ("CL0", CL0, "-", "calculated (lattice)", "lift at α = 0 from the pod axis: wing incidence and camber"),
        ("CLa", CLa, "1/rad", "calculated (lattice)", "lift slope, wing + tail"),
        ("CLq", CLq, "1/rad", "calculated (analytic)", "2 η CLα_t V_h"),
        ("CLde", CLde, "1/rad", "calculated (lattice)", f"elevator effectiveness τ = {ELEVATOR_TAU}"),
        ("CD0", cd0, "-", "calculated (build-up)", "flat-plate build-up at 16 m/s (CFD to confirm)"),
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
        ("Clp", Clp, "1/rad", "assumed", "roll damping, chart value for AR 7 (Roskam Part VI class)"),
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
        ("x_np_m", a["x_np"], "m", "calculated (lattice + pod term)", "neutral point in the nisus frame"),
        ("static_margin", (a["x_np"] - x_cg_m) / c, "MAC", "calculated", f"CG at x = {x_cg_m:.4f} m"),
        ("deps_dalpha", a["deps_dalpha"], "-", "calculated (lattice)", "downwash gradient at the tail"),
        ("CLa_t", CLa_t, "1/rad", "calculated (lattice)", "tail lift slope on S_h, isolated"),
        ("CLa_v", CLa_v, "1/rad", "calculated (analytic)", f"fin slope at AR_eff {AR_v:.2f} (end-plate factor 1.55 assumed)"),
    ]
    df = pd.DataFrame(rows, columns=["derivative", "value", "unit", "source", "note"]).set_index("derivative")
    df.attrs.update(S=S, c=c, b=b, x_cg_m=x_cg_m, z_cg_m=z_cg_m, l_t=l_t, l_v=l_v, V_h=V_h, z_v=z_v, cd0=cd0)
    return df


def coefficients(deriv: pd.DataFrame) -> dict:
    """The derivative table as a plain dict (value by name) plus the reference geometry — what the simulation reads."""
    d = {k: float(v) for k, v in deriv["value"].items()}
    d.update({k: float(v) for k, v in deriv.attrs.items()})
    return d


# ================================================================================================= stability and trim
def trim(deriv: pd.DataFrame, mass_kg: float, V: float, rho=RHO, n: float = 1.0) -> dict:
    """Angle of attack (from the pod axis) and elevator for level flight at V and load factor n, from the linear
    model: CL0 + CLα α + CLδ δ = n W / (q S); Cm0 + Cmα α + Cmδ δ = 0."""
    c = coefficients(deriv)
    cl = n * mass_kg * G / (0.5 * rho * V ** 2 * c["S"])
    M = np.array([[c["CLa"], c["CLde"]], [c["Cma"], c["Cmde"]]])
    al, de = np.linalg.solve(M, [cl - c["CL0"], -c["Cm0"]])
    return {"V": V, "CL": cl, "alpha_deg": math.degrees(al), "elevator_deg": math.degrees(de), "stalled": cl > c["CL_max"]}


def cg_range(p, mass_kg, *, sm_min: float = 0.05, de_max_deg: float = 20.0, V_app_factor: float = 1.0) -> dict:
    """The allowed CG range [fraction of MAC]: the aft limit from the minimum static margin; the forward limit where
    the elevator (``de_max_deg`` of its 25°, 5° left for control) still trims ``V_app_factor`` x the stall speed —
    1.0: the aircraft can reach CL_max in the flare."""
    p = nisus.resolve(p)
    a = aero(p)
    L = nisus.Nisus.layout(p)
    c = a["c_ref"]
    aft = (a["x_np"] - sm_min * c)
    xs = np.linspace(L["x_mac_le"] / 1000 - 0.1 * c, aft, 60)
    fwd = xs[0]
    for x in xs:
        d = derivatives(p, x_cg_m=x)
        cf = coefficients(d)
        Vs = math.sqrt(2 * mass_kg * G / (RHO * cf["S"] * cf["CL_max"]))
        t = trim(d, mass_kg, V_app_factor * Vs)
        if abs(t["elevator_deg"]) <= de_max_deg:
            fwd = x
            break
    return {"x_fwd_m": float(fwd), "x_aft_m": float(aft), "fwd_frac_mac": (fwd - L["x_mac_le"] / 1000) / c, "aft_frac_mac": (aft - L["x_mac_le"] / 1000) / c,
            "x_np_frac_mac": (a["x_np"] - L["x_mac_le"] / 1000) / c, "sm_min": sm_min, "de_max_deg": de_max_deg}


# ================================================================================================= the drive as a Unit
@dataclass
class MapUnit:
    """``nisus_systems.PropulsionMap`` wearing Merlin's ``Unit`` interface (``full(V)``, ``electrical_power(V, T)``)
    so ``merlin_flight.performance`` and PEREGRINE's envelope arithmetic run unchanged; ``fixed_w`` is the
    electronics' battery-side power added to every level-flight point."""
    pm: ns.PropulsionMap
    fixed_w: float = 0.0
    name: str = "X2216 KV1250 + APC 9x6E"

    @property
    def V(self):
        return self.pm.V

    def full(self, V):
        a = self.pm.at(V, 1.0)
        return a["thrust"], a["electrical"] + self.fixed_w, a["rpm"]

    def electrical_power(self, V, T_net):
        if T_net <= 0:
            return self.fixed_w
        P = self.pm.electrical_for_thrust(V, T_net)
        return P + self.fixed_w if np.isfinite(P) else math.inf


def airframe(variant: str, battery_key: str = "gens-ace-3s-2200", p=None, *, cd0_correction: float = 0.0) -> Airframe:
    """Merlin's parabolic-polar ``Airframe`` for a variant on a pack: mass from the mass table, Cd0 from the build-up
    (+ a CFD correction when known), Oswald e = 0.9 x the lattice's inviscid span efficiency, CL_max as assumed."""
    p = nisus.resolve(p)
    a = aero(p)
    m = ns.cg_inertia(ns.mass_table(variant, battery_key, p=p))["mass_kg"]
    b = nisus.drag_buildup(p, 16.0)
    return Airframe(f"Nisus-{variant}", m, a["S_ref"], a["AR"], b["cd0"] + cd0_correction, oswald=pf.OSWALD_VISCOUS * a["e"], cl_max=cl_max(p))


def envelope(af: Airframe, unit: MapUnit, *, battery_wh=24.42, usable=0.80 * 0.9, rho=RHO, n_struct=N_STRUCTURAL) -> dict:
    """The speed polar (``merlin_flight.performance``), endurance and range, glide, the turns (PEREGRINE's ``turn``)."""
    perf = mf.performance(af, unit, rho, V=np.linspace(max(af.stall_speed(rho) * 1.05, 6.0), 30.0, 100))
    V = perf["V"]
    P = np.where(np.isfinite(perf["P_level"]), perf["P_level"], np.inf)
    i_e = int(np.argmin(P))
    tr = pf.turn(af, unit, V, n_struct, rho)
    ld = af.mass_kg * G / af.drag(V, rho)
    return {**perf, **tr, "L/D": ld, "v_endurance": float(V[i_e]), "P_min": float(P[i_e]), "endurance_min": battery_wh * usable / P[i_e] * 60,
            "range_km": battery_wh * usable / perf["wh_per_km_best"], "L/D_max": float(ld.max()), "v_L/D_max": float(V[np.argmax(ld)]),
            "sink_min_m_s": float((af.drag(V, rho) * V / (af.mass_kg * G)).min()),
            "r_min_inst": float(np.min(tr["radius_inst"])), "v_r_min": float(V[np.argmin(tr["radius_inst"])]),
            "rate_max_deg_s": math.degrees(float(np.max(tr["rate_inst"]))), "r_min_sus": float(np.min(tr["radius_sus"])),
            "rate_max_sus_deg_s": math.degrees(float(np.max(tr["rate_sus"])))}


# ================================================================================================= flight checks
def stall_table(af: Airframe, banks_deg=(0, 15, 30, 45, 60), rho=RHO) -> pd.DataFrame:
    """Stall speed and the turn at each bank angle: V_s(φ) = V_s1 / √cos φ, n = 1/cos φ, radius V²/(g tan φ) at
    1.3 V_s(φ) (the survey turns), rate."""
    rows = {}
    Vs1 = af.stall_speed(rho)
    for phi in banks_deg:
        n = 1 / math.cos(math.radians(phi))
        Vs = Vs1 * math.sqrt(n)
        Vt = 1.3 * Vs
        r = Vt ** 2 / (G * math.tan(math.radians(phi))) if phi > 0 else math.inf
        rows[f"bank {phi}°"] = {"load factor": n, "stall speed [m/s]": Vs, "turn speed 1.3 Vs [m/s]": Vt, "turn radius [m]": r,
                                "turn rate [deg/s]": math.degrees(Vt / r) if phi > 0 else 0.0, "level power at 1.3 Vs needs n": n}
    return pd.DataFrame(rows).T


def gust_response(af: Airframe, deriv: pd.DataFrame, V: float, U_de: float = 5.0, rho=RHO) -> dict:
    """Pratt's discrete gust: Δn = K_g ρ V U_de CLα S / (2 W) with the alleviation K_g = 0.88 μ/(5.3 + μ),
    μ = 2 W/(ρ c CLα g S); also the sharp-edged gust (K_g = 1) and the stall check on the gusted CL."""
    c = coefficients(deriv)
    W = af.mass_kg * G
    mu = 2 * W / (rho * c["c"] * c["CLa"] * G * c["S"])
    Kg = 0.88 * mu / (5.3 + mu)
    dn = Kg * rho * V * U_de * c["CLa"] * c["S"] / (2 * W)
    CL1 = W / (0.5 * rho * V ** 2 * c["S"])
    return {"V": V, "U_de": U_de, "mass_ratio": mu, "K_g": Kg, "delta_n": dn, "n_gust": 1 + dn, "n_sharp_edged": 1 + dn / Kg,
            "CL_gusted": CL1 * (1 + dn), "stalls_in_gust": CL1 * (1 + dn) > c["CL_max"], "alpha_gust_deg": math.degrees(math.atan(U_de / V))}


def pull_up(af: Airframe, V: float, n_struct=N_STRUCTURAL, rho=RHO) -> dict:
    """A bounded pull-up at V: the load factor is the lesser of the lift limit (CL_max) and the structural limit;
    the radius V²/(g (n − 1)); the height lost recovering from a 30° dive at that radius (geometric: R (1 − cos 30°))
    plus 1 s of reaction; the speed gained in the dive is not credited (conservative)."""
    q = 0.5 * rho * V ** 2
    n_lift = q * af.wing_area_m2 * af.cl_max / (af.mass_kg * G)
    n = min(n_lift, n_struct)
    if n <= 1.0:
        return {"V": V, "n": n, "limited_by": "lift", "radius_m": math.inf, "height_lost_30deg_dive_m": math.inf}
    R = V ** 2 / (G * (n - 1))
    h = R * (1 - math.cos(math.radians(30))) + V * math.sin(math.radians(30)) * 1.0
    return {"V": V, "n": n, "limited_by": "lift" if n_lift < n_struct else "structure", "radius_m": R, "height_lost_30deg_dive_m": h,
            "V_A_corner_m_s": math.sqrt(2 * n_struct * af.mass_kg * G / (rho * af.wing_area_m2 * af.cl_max))}


def approach(af: Airframe, unit: MapUnit, rho=RHO) -> dict:
    """The approach: 1.3 Vs, the idle glide angle and sink rate (L/D at that speed), a 6° powered approach's thrust
    and power, the flare height for a 0.5 m/s touchdown sink and the ground-contact speed."""
    Vs = af.stall_speed(rho)
    Va = 1.3 * Vs
    D = float(af.drag(Va, rho))
    ld = af.mass_kg * G / D
    gamma_glide = math.degrees(math.atan(1 / ld))
    gamma = 6.0
    T = D - af.mass_kg * G * math.sin(math.radians(gamma))
    P = unit.electrical_power(Va, max(T, 0.0)) if T > 0 else unit.fixed_w
    sink = Va * math.sin(math.radians(gamma))
    return {"V_stall": Vs, "V_approach": Va, "L/D": ld, "glide_angle_idle_deg": gamma_glide, "sink_idle_m_s": Va * math.sin(math.radians(gamma_glide)),
            "approach_angle_deg": gamma, "approach_thrust_N": T, "approach_power_W": P, "approach_sink_m_s": sink,
            "flare_height_m": Va ** 2 / (2 * G * 0.3) * (1 - math.cos(math.radians(gamma))), "touchdown_speed_m_s": 1.1 * Vs}


def operating_limits(variant: str, af: Airframe, unit: MapUnit, deriv: pd.DataFrame, cg: dict, rho=RHO) -> pd.DataFrame:
    """Provisional operating limits for one variant (first flights: to be tightened or opened by test): speeds,
    load factors, bank, wind and gust, CG, mass, battery."""
    Vs = af.stall_speed(rho)
    env = envelope(af, unit)
    g = gust_response(af, deriv, 16.0)
    rows = {
        "stall speed, level [m/s]": (Vs, "CL_max assumed 1.17"),
        "minimum speed in flight [m/s]": (1.2 * Vs, "1.2 Vs"),
        "approach speed [m/s]": (1.3 * Vs, "1.3 Vs"),
        "cruise speed [m/s]": (16.0, "mission target 14-18"),
        "best endurance speed [m/s]": (env["v_endurance"], "minimum level power"),
        "never-exceed speed [m/s]": (V_NE, "the FEA's dash case; flutter and control-surface limits untested"),
        "manoeuvring speed V_A [m/s]": (math.sqrt(2 * N_STRUCTURAL * af.mass_kg * G / (rho * af.wing_area_m2 * af.cl_max)), f"n = {N_STRUCTURAL:g} at CL_max"),
        "limit load factor": (N_STRUCTURAL, "the structure's check case (FEA)"),
        "maximum bank, survey turns [deg]": (45.0, "n = 1.41; stall speed x 1.19"),
        "maximum steady wind [m/s]": (8.0, "half the cruise speed: the return's ground speed stays ≥ 8 m/s"),
        "maximum gust for first flights [m/s]": (5.0, f"Δn = {g['delta_n']:+.2f} at 16 m/s; {'stalls' if g['stalls_in_gust'] else 'no stall'} at cruise CL"),
        "CG forward limit [% MAC]": (100 * cg["fwd_frac_mac"], f"elevator ≤ {cg['de_max_deg']:g}° trims the approach"),
        "CG aft limit [% MAC]": (100 * cg["aft_frac_mac"], f"static margin ≥ {100 * cg['sm_min']:.0f} %"),
        "maximum take-off mass [kg]": (af.mass_kg * 1.1, "10 % over the budget mass (payload growth); re-check the climb"),
        "minimum climb rate at MTOM [m/s]": (float(np.interp(13.0, env["V"], env["roc"])), "full throttle at 13 m/s"),
        "battery reserve at landing": (0.20, "of the derated available energy"),
    }
    return pd.DataFrame({k: {"value": v[0], "basis": v[1]} for k, v in rows.items()}).T


__all__ = ["AIRFOIL", "TAIL_SECTION", "ELEVATOR_TAU", "RUDDER_TAU", "AILERON_TAU", "TAIL_EFFICIENCY", "N_STRUCTURAL", "V_NE", "aero", "polar",
           "cl_max", "derivatives", "coefficients", "trim", "cg_range", "MapUnit", "airframe", "envelope", "stall_table", "gust_response",
           "pull_up", "approach", "operating_limits"]
