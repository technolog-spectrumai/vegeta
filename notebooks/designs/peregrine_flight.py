"""PEREGRINE — the aerodynamics of a folding wing, the flight, the farm, the flock, the roof and the verdict (notebook 28).

Reduced models, honest about what they are, on top of what the other notebooks validated:

- ``vlm``: a vortex lattice (horseshoe vortices on flat panels, Kutta–Joukowski forces, Trefftz-plane induced drag)
  on ``peregrine.planform``'s quadrilaterals — lift slope, induced drag factor, span efficiency, neutral point and
  pitching moment of the wing with its tail at any fold angle. (New: nothing in the repository did swept wings.)
- ``stability``: centre of gravity from the mass table, neutral point (lattice + a fuselage term), static margin and
  the trim (angle of attack, elevator) against the fold.
- ``Airframe`` and ``Unit`` are MERLIN's (``merlin_flight``): a parabolic polar per variant and fold angle; the
  propeller from notebook 25's library (``propulsor_maps``), one 300 W tractor drive for all three wings.
- ``envelope``: power required, endurance and range speeds, glide, turn radius and rate, roll rate, climb.
- ``BIRDS``: the pests and the peregrine, for the agility chart.
- ``stoop``: the falcon's dive in the vertical plane, with the tuck, the servo's unfolding time and the pull-out.
- ``gap_passage``: the narrowest gap between trees or houses each wing passes (bank through it, or fold).
- ``prop_hang``, ``hand_throw``, ``roof_drop``, ``perched_wind``: starting from and resting on a roof.
- ``hinge_loads``: the hinge's bearing forces and the fold servo's torque.
- ``mass_table``, ``bom``, ``ENGINEERING_HOURS``: the bill of materials and the engineering cost.
- ``Flock`` and ``herd``: a boids flock with a fear radius pushed off the crop by the drone; ``Farm``, ``mission`` and
  ``render_movie``: the whole sortie from the roof and back, as an MP4.
- ``decision_table`` and ``weight_sensitivity``: does folding pay.

The sound is the job (the propeller's noise deters the animals), so there is no acoustics here. Units SI; the
aircraft frame is ``peregrine``'s (x aft, y right, z up); the world frame x east, y north, z up, heights above
ground.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import merlin_flight as mf
import peregrine as pg
from merlin_flight import G, RHO, Airframe, Unit

AIRFOIL_ALPHA0_DEG = -2.1          # NACA 2410 zero-lift angle (thin-airfoil theory for 2 % camber at 40 %)
AIRFOIL_CM_AC = -0.053             # its moment about the quarter chord (thin-airfoil theory)
CL_MAX_2D = 1.20                   # section maximum lift at Re ~ 2e5 (assumption, NACA 24xx class at low Re)
ELEVATOR_TAU = 0.50                # elevator effectiveness: a deflection acts as tau x the same change of tail incidence
FUSELAGE_K = 0.008                 # Raymer's fuselage pitching-moment factor (per degree) at a wing ~ 0.3 L back
OSWALD_VISCOUS = 0.90              # e = 0.90 x the lattice's inviscid span efficiency (the profile drag's lift share)


# ================================================================================================= the vortex lattice
def _cut(q, y0):
    """Where the line y = y0 crosses the polygon q (x, y, z): its most forward and most aft points (leading and
    trailing edge of the streamwise strip), or None."""
    pts = []
    for i in range(len(q)):
        a, b = q[i - 1], q[i]
        if (a[1] - y0) * (b[1] - y0) <= 0 and a[1] != b[1]:
            t = (y0 - a[1]) / (b[1] - a[1])
            pts.append(a + t * (b - a))
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    return pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])]


def _lattice(quads, n_span, n_chord, hidden=None, cosine=True):
    """Panels on flat quadrilaterals, meshed in streamwise strips (the chordwise panel edges along the free stream, so
    the trailing legs run along them even when a folded panel's own chords are not streamwise): spanwise stations
    across the quad's y-range (cosine spaced), at each station the polygon's leading and trailing edge, ``n_chord``
    panels between them; the bound vortex at the panel's quarter chord, the control point at its three-quarter chord,
    the normal upwards; the bound segment runs from smaller to larger y (positive circulation = lift). ``hidden[i]``:
    the quad is only meshed outside |y| = hidden[i] (a folded panel's part over the inner wing is not a lifting
    surface of its own)."""
    A, B, CP, N, AREA, SURF = [], [], [], [], [], []
    for qi, q in enumerate(quads):
        ns, nc = (n_span[qi] if np.ndim(n_span) else n_span), n_chord
        h = 0.0 if hidden is None else hidden[qi]
        ylo, yhi = float(q[:, 1].min()), float(q[:, 1].max())
        if h > 0:
            if yhi <= 0:
                yhi = min(yhi, -h)
            else:
                ylo = max(ylo, h)
        eps = 1e-6 * max(yhi - ylo, 1e-9)
        if yhi - ylo < 1e-4:
            continue
        s = 0.5 * (1 - np.cos(np.linspace(0, math.pi, ns + 1))) if cosine else np.linspace(0, 1, ns + 1)
        ys = ylo + eps + (yhi - ylo - 2 * eps) * s
        edges = [_cut(q, y) for y in ys]
        u = np.linspace(0, 1, nc + 1)
        for j in range(ns):
            if edges[j] is None or edges[j + 1] is None:
                continue
            (le0, te0), (le1, te1) = edges[j], edges[j + 1]
            if min(te0[0] - le0[0], te1[0] - le1[0]) < 1e-3:
                continue
            for k in range(nc):
                p00 = le0 + (te0 - le0) * u[k]; p01 = le0 + (te0 - le0) * u[k + 1]
                p10 = le1 + (te1 - le1) * u[k]; p11 = le1 + (te1 - le1) * u[k + 1]
                a = p00 + 0.25 * (p01 - p00)
                b = p10 + 0.25 * (p11 - p10)
                cp = 0.5 * ((p00 + 0.75 * (p01 - p00)) + (p10 + 0.75 * (p11 - p10)))
                n = np.cross(p11 - p00, p01 - p10)
                n = n / np.linalg.norm(n)
                if n[2] < 0:
                    n = -n
                if a[1] > b[1]:
                    a, b = b, a
                A.append(a); B.append(b); CP.append(cp); N.append(n)
                AREA.append(0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p10))); SURF.append(qi)
    return {"A": np.array(A), "B": np.array(B), "cp": np.array(CP), "n": np.array(N), "area": np.array(AREA),
            "quad": np.array(SURF)}


def _segment(P, a, b):
    """Velocity at points P [M,3] induced by unit-strength straight vortex segments a→b [N,3] (Biot–Savart)."""
    r1 = P[:, None, :] - a[None]
    r2 = P[:, None, :] - b[None]
    r0 = (b - a)[None]
    c = np.cross(r1, r2)
    c2 = np.einsum("mnk,mnk->mn", c, c)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    dot = np.einsum("mnk,mnk->mn", r0, r1 / np.maximum(n1, 1e-12)[..., None] - r2 / np.maximum(n2, 1e-12)[..., None])
    small = c2 < (1e-6 * np.einsum("mnk,mnk->mn", r0, r0) + 1e-14)
    k = np.where(small, 0.0, dot / (4 * math.pi * np.where(small, 1.0, c2)))
    return c * k[..., None]


def _horseshoes(P, lat, far):
    A, B = lat["A"], lat["B"]
    x = np.array([far, 0.0, 0.0])
    return _segment(P, A + x, A) + _segment(P, A, B) + _segment(P, B, B + x)


def vlm(surfaces, *, alpha_deg=0.0, incidence_deg=None, S_ref, c_ref, x_ref=0.0, n_span=12, n_chord=4, hidden=None):
    """Solve the lattice on ``surfaces`` (a list of quads) at ``alpha_deg`` with per-quad ``incidence_deg`` (camber's
    zero-lift angle, the tail setting, the elevator) for a unit free stream; return the circulations and the
    coefficients: CL, CDi (Trefftz plane), Cm about ``x_ref`` (nose-up positive), the panel forces and the lattice."""
    lat = _lattice(surfaces, n_span, n_chord, hidden)
    a = math.radians(alpha_deg)
    vinf = np.array([math.cos(a), 0.0, math.sin(a)])
    span = max(np.ptp(np.concatenate([q[:, 1] for q in surfaces])), 1e-3)
    AIC = np.einsum("mnk,mk->mn", _horseshoes(lat["cp"], lat, 50 * span), lat["n"])
    inc = np.zeros(len(lat["cp"])) if incidence_deg is None else np.radians(np.asarray(incidence_deg, float))[lat["quad"]]
    rhs = -(lat["n"] @ vinf) - inc * lat["n"][:, 2]
    gam = np.linalg.solve(AIC, rhs)
    l = lat["B"] - lat["A"]
    F = 2 * gam[:, None] * np.cross(vinf[None], l)                 # per panel, divided by q (rho V^2 / 2 · 2 Γ)
    lift_dir = np.array([-math.sin(a), 0.0, math.cos(a)])
    CL = float((F @ lift_dir).sum() / S_ref)
    mid = 0.5 * (lat["A"] + lat["B"])
    M = np.cross(mid - np.array([x_ref, 0.0, 0.0]), F).sum(axis=0)
    Cm = float(M[1] / (S_ref * c_ref))
    # Trefftz plane: the trailing legs as 2-D vortices (a leg at B along +x with Γ, at A with -Γ)
    yz_v = np.vstack([lat["B"][:, 1:], lat["A"][:, 1:]])
    g_v = np.r_[gam, -gam]
    yz_m = mid[:, 1:]
    d = yz_m[:, None, :] - yz_v[None]
    r2 = np.maximum((d ** 2).sum(-1), 1e-10)
    vy = (-g_v[None] * d[..., 1] / (2 * math.pi * r2)).sum(1)
    vz = (g_v[None] * d[..., 0] / (2 * math.pi * r2)).sum(1)
    lyz = l[:, 1:]
    seg = np.linalg.norm(lyz, axis=1)
    n2 = np.c_[-lyz[:, 1], lyz[:, 0]] / np.maximum(seg, 1e-12)[:, None]      # normal of the bound segment in the plane
    n2 = np.where((n2[:, 1] < 0)[:, None], -n2, n2)
    w = -(vy * n2[:, 0] + vz * n2[:, 1])                                     # downwash, positive down
    CDi = float((gam * w * seg).sum() / S_ref)
    return {"CL": CL, "CDi": CDi, "Cm": Cm, "gamma": gam, "F": F, "lattice": lat, "mid": mid}


def _surfaces(pl, it_deg=0.0, de_deg=0.0, camber=True):
    quads = list(pl["wing"]) + list(pl["tail"])
    hidden = list(pl["hidden"]) + [0.0, 0.0]
    inc = [(-AIRFOIL_ALPHA0_DEG if camber else 0.0)] * len(pl["wing"]) + [it_deg + ELEVATOR_TAU * de_deg] * 2
    n_span = []
    for q in pl["wing"]:
        n_span.append(max(2, int(round(np.ptp(q[:, 1]) / 0.03))))
    n_span += [5, 5]
    return quads, hidden, inc, n_span


def aero(p, fold_deg=None, *, x_ref=None, it_deg=-1.0, n_chord=4) -> dict:
    """The lattice's aerodynamic derivatives of PEREGRINE (wing with tail) at one fold angle, referred to the spread
    planform: CL0 and CLα (per rad), the induced drag factor k = CDi/CL^2, the span efficiency e (to the folded
    span), the neutral point (with the fuselage term), Cm0 about ``x_ref`` (default the wing's quarter MAC, with the
    airfoil's own moment), the elevator's CLδ and Cmδ (per degree), the wing's share of the lift."""
    pl = pg.planform(p, fold_deg)
    S, c = pl["S_ref"], pl["c_ref"]
    x_ref = 0.25 * c if x_ref is None else x_ref
    quads, hidden, inc, ns = _surfaces(pl, it_deg)
    kw = dict(S_ref=S, c_ref=c, x_ref=x_ref, n_span=ns, n_chord=n_chord, hidden=hidden)
    r0 = vlm(quads, alpha_deg=0.0, incidence_deg=inc, **kw)
    r1 = vlm(quads, alpha_deg=4.0, incidence_deg=inc, **kw)
    quads_e, _, inc_e, _ = _surfaces(pl, it_deg, de_deg=1.0)
    re = vlm(quads_e, alpha_deg=0.0, incidence_deg=inc_e, **kw)
    da = math.radians(4.0)
    CLa = (r1["CL"] - r0["CL"]) / da
    Cma = (r1["Cm"] - r0["Cm"]) / da
    # the fuselage's destabilising moment (Raymer: Cm_alpha,fus = K W_f^2 L_f / (c S) per degree)
    q = pg.resolve(p)
    Wf, Lf = q["fuselage_diameter"] / 1000, (q["nose_length"] + q["aft_length"] + q["spinner_length"]) / 1000
    Cma_f = FUSELAGE_K * Wf ** 2 * Lf / (c * S) * (180 / math.pi)
    x_np = x_ref - (Cma + Cma_f) / CLa * c
    # the airfoil's own moment on the wing's visible panels (sum c^2 dy / (S c))
    lat = r0["lattice"]
    wing_panels = lat["quad"] < len(pl["wing"])
    chord_strip = lat["area"][wing_panels] / np.maximum(np.abs(lat["B"][wing_panels, 1] - lat["A"][wing_panels, 1]), 1e-9)
    cm_af = AIRFOIL_CM_AC * float((lat["area"][wing_panels] * chord_strip).sum() / n_chord / (S * c)) * n_chord / n_chord
    # induced drag factor from two lift levels (CDi = k CL^2 is exact for an untwisted planar lattice only at CL0 = 0)
    k = (r1["CDi"] - r0["CDi"]) / (r1["CL"] ** 2 - r0["CL"] ** 2) if abs(r1["CL"] ** 2 - r0["CL"] ** 2) > 1e-9 else math.nan
    AR = pl["b"] ** 2 / S
    wing_share = float(r1["F"][wing_panels, 2].sum() / r1["F"][:, 2].sum())
    return {"fold_deg": pl["fold_deg"], "span": pl["b"], "AR": AR, "S_ref": S, "c_ref": c, "CL0": r0["CL"], "CLa": CLa,
            "Cm0": r0["Cm"] + cm_af, "Cma": Cma + Cma_f, "x_ref": x_ref, "x_np": x_np, "k": k, "e": 1 / (math.pi * AR * k),
            "CLde": (re["CL"] - r0["CL"]), "Cmde": (re["Cm"] - r0["Cm"]), "wing_share": wing_share, "it_deg": it_deg}


def cl_max(p, fold_deg=None) -> float:
    """The wing's maximum lift coefficient on the spread planform: 0.9 x the section's on the inner wing, and on the
    folded outer panel's visible part times cos(fold) (simple sweep theory), less what hides over the inner wing."""
    p = pg.resolve(p)
    fold = p["fold_deg"] if fold_deg is None else fold_deg
    pl = pg.planform(p, fold)
    if p["wing_hinge"] == "none":
        return 0.9 * CL_MAX_2D
    ex = pg.exposed_wing_area(p, fold)
    outer = ex["outer_panel"]
    inner = pl["S_ref"] - 2 * outer
    outer_vis = 2 * (outer - ex["hidden_per_side"])
    return 0.9 * CL_MAX_2D * (inner + outer_vis * math.cos(math.radians(fold))) / pl["S_ref"]


def fold_polar(p, folds, *, V=20.0, nu=1.5e-5, it_deg=-1.0):
    """One row per fold angle: span, aspect ratio, CLα, k, e, neutral point, CL_max, the build-up's Cd0, hidden area."""
    import pandas as pd
    rows = []
    for f in folds:
        a = aero(p, f, it_deg=it_deg)
        b = pg.drag_buildup(p, V, nu, fold_deg=f)
        rows.append({"fold [deg]": f, "span [m]": a["span"], "AR (folded span)": a["AR"], "CLα [1/rad]": a["CLa"],
                     "k = CDi/CL²": a["k"], "e (inviscid)": a["e"], "x_np [m]": a["x_np"], "CL_max": cl_max(p, f),
                     "Cd0": b["cd0"], "Cd0·S [cm²]": b["cd_area_m2"] * 1e4,
                     "hidden wing [cm²]": (pg.planform(p, 0.0)["S_ref"] - pg.exposed_wing_area(p, f)["exposed"]) * 1e4
                     - (pg.planform(p, 0.0)["S_ref"] - pg.exposed_wing_area(p, 0.0)["exposed"]) * 1e4})
    return pd.DataFrame(rows).set_index("fold [deg]")


# ================================================================================================= mass, CG and cost
# Hobby-market items: mass [g], price [EUR] (typical 2026 online retail, single units; estimates — edit them),
# position x [m] in the aircraft frame (None: placed by the code). 'variants': which wings carry the item.
BOM_ITEMS = [
    # item, g, EUR, x, variants, note
    ("motor 2212 1000 kV outrunner", 56.0, 18.0, "motor", "ABC", "300 W class"),
    ("ESC 30 A (BLHeli)", 25.0, 15.0, -0.07, "ABC", ""),
    ("propeller 9x5 (and a spare)", 10.0, 3.0, "prop", "ABC", "notebook 25's library"),
    ("battery 3S 2200 mAh 30C", 180.0, 25.0, "battery", "ABC", "24.4 Wh"),
    ("flight controller (F405 wing class, baro)", 15.0, 35.0, 0.05, "ABC", "autopilot (fixed-wing firmware)"),
    ("GPS + compass", 10.0, 15.0, 0.12, "ABC", ""),
    ("receiver + telemetry (2.4 GHz)", 5.0, 15.0, 0.15, "ABC", ""),
    ("control servos 4 x 9 g (2 ailerons, elevator, rudder)", 36.0, 16.0, "servos", "ABC", ""),
    ("carbon spar tube 8 mm", 30.0, 9.0, "spar", "ABC", "B: shorter"),
    ("wiring, connectors, leads", 20.0, 8.0, 0.02, "ABC", ""),
    ("roof skids + perch pad (TPU)", 12.0, 2.0, 0.10, "ABC", "printed"),
    ("glue, horns, pushrods, screws, tape", 15.0, 6.0, 0.10, "ABC", ""),
    ("fold servos 2 x (selected)", None, None, "fold_servo", "C", "from the hinge loads"),
    ("hinge pins + 4 flanged bearings MF63ZZ", 8.0, 6.0, "hinge", "C", "steel pin 3 mm"),
    ("hinge lugs 4 x (PETG, printed)", None, None, "hinge", "C", "from the slicer"),
    ("fold linkage (horns, ball links, pushrods)", 10.0, 5.0, "hinge", "C", ""),
    ("spar split sleeves (aluminium) + extra wiring", 14.0, 5.0, "hinge", "C", ""),
]
SERVO_PRICE_EUR = {"sub-micro 9 g": 4.0, "micro 13 g": 6.0, "micro-metal 20 g": 9.0, "micro-metal 22 g worm": 14.0,
                   "mini 32 g": 12.0, "mini 32 g worm": 18.0, "standard 55 g": 15.0, "standard 60 g worm": 24.0}
AIRFRAME_AREAL_DENSITY = 0.70      # kg/m^2 of wetted skin: ~1 mm LW-PLA shells + ribs + glue (MERLIN's assumption)
LW_PLA_EUR_PER_KG = 40.0           # LW-PLA spool price
ENGINEERING_HOURS = {              # lab hours to design, build, program and fly a first working aircraft (estimates)
    "airframe design + CAD": {"A": 30, "B": 30, "C": 30},
    "build + first flights": {"A": 30, "B": 30, "C": 30},
    "autopilot setup + tuning": {"A": 15, "B": 15, "C": 15},
    "hinge + lug design, FEA, bench tests": {"A": 0, "B": 0, "C": 35},
    "fold actuation, mixing, CG/trim schedule (software)": {"A": 0, "B": 0, "C": 30},
    "fold-in-flight test campaign (incl. a crash rebuild)": {"A": 0, "B": 0, "C": 40},
}
ENGINEERING_RATE_EUR_H = 40.0      # a lab engineer's / student assistant's cost per hour (edit)


def mass_table(p, *, fold_deg=0.0, fold_servo=None, printed_lug_g=4.0, battery_x=None, areal_density=AIRFRAME_AREAL_DENSITY):
    """The mass table [kg, m]: the BOM items placed in the aircraft frame plus the printed airframe (wing, fuselage,
    tail skins at ``areal_density``); the outer panels' mass (skin, the outer spar, the aileron servo) moves with the
    fold. ``battery_x`` None: the battery is placed for a static margin of 10 % with the wing spread (as builders do
    it — the CG is set by the battery)."""
    import pandas as pd
    import actuators as act
    p = pg.resolve(p)
    variant = {"none": "A", "root": "C"}[p["wing_hinge"]] if p["span"] >= 1000 else "B"
    L = pg.Merlin.layout(p)
    s = pg.Peregrine.stations(p)
    w = pg.wetted_areas(p, 0.0)
    pl0, plf = pg.planform(p, 0.0), pg.planform(p, fold_deg)
    n = len(pl0["wing"]) // 2

    def centroid(quads):
        a = np.array([pg._quad_area(q[:, :2]) for q in quads])
        c = np.array([q[:, :2].mean(axis=0) for q in quads])
        return float((a[:, None] * c).sum(0)[0] / a.sum()), float(a.sum())

    xo, ao = centroid([plf["wing"][n - 1], plf["wing"][2 * n - 1]])          # the two outer panels as folded
    xi, ai = centroid(pl0["wing"][:n - 1] + pl0["wing"][n:2 * n - 1])
    rows = []
    wing_kg = areal_density * w["wing"]
    rows.append(("airframe: wing skins (outer panels)", wing_kg * ao / (ao + ai), xo, 0.0))
    rows.append(("airframe: wing skins (centre + inner)", wing_kg * ai / (ao + ai), xi, 0.0))
    prof = pg.Merlin.fuselage_profile(p) / 1000
    xf = float(np.trapezoid(prof[:, 0] * prof[:, 1], prof[:, 0]) / np.trapezoid(prof[:, 1], prof[:, 0]))
    rows.append(("airframe: fuselage shell", areal_density * w["fuselage"], xf, 0.0))
    rows.append(("airframe: tail plates", areal_density * 0.6 * w["tail"], (L["tail_le"] + p["tail_chord"] / 2) / 1000, 0.0))
    pos = {"motor": (L["prop_x"] + 25) / 1000, "prop": L["prop_x"] / 1000, "servos": 0.15,
           "spar": float(xo * 0 + 0.25 * s["c0"] / 1000), "hinge": s["xh"] / 1000 if p["wing_hinge"] == "root" else 0.0}
    for name, g, eur, x, variants, note in BOM_ITEMS:
        if variant not in variants:
            continue
        if x == "fold_servo":
            a = act.get(fold_servo or "micro-metal 20 g")
            g, eur, x = 2 * a.mass_g, 2 * SERVO_PRICE_EUR.get(a.key, 10.0), s["xh"] / 1000
            name = f"fold servos 2 x {a.key}"
        if name.startswith("hinge lugs"):
            g, eur = 4 * printed_lug_g, 4 * printed_lug_g / 1000 * 30.0
        if name.startswith("carbon spar") and variant == "B":
            g, eur = g * 0.75, eur * 0.8
        if x == "battery":
            continue
        xv = pos[x] if isinstance(x, str) else x
        rows.append((name, g / 1000, xv, eur))
    df = pd.DataFrame(rows, columns=["item", "mass [kg]", "x [m]", "EUR"]).set_index("item")
    bat = next(b for b in BOM_ITEMS if b[0].startswith("battery"))
    if battery_x is None:
        a0 = aero(p, 0.0)
        x_cg_target = a0["x_np"] - 0.10 * a0["c_ref"]
        m_rest = df["mass [kg]"].sum()
        mx_rest = (df["mass [kg]"] * df["x [m]"]).sum()
        battery_x = (x_cg_target * (m_rest + bat[1] / 1000) - mx_rest) / (bat[1] / 1000)
    df.loc[bat[0]] = [bat[1] / 1000, battery_x, bat[2]]
    return df


def cg(table) -> float:
    return float((table["mass [kg]"] * table["x [m]"]).sum() / table["mass [kg]"].sum())


def bom(p, *, fold_servo=None, printed_lug_g=4.0, areal_density=AIRFRAME_AREAL_DENSITY):
    """The bill of materials [g, EUR]: the mass table's items, the printed airframe priced at ``LW_PLA_EUR_PER_KG``
    (filament only), the totals."""
    t = mass_table(p, fold_servo=fold_servo, printed_lug_g=printed_lug_g, areal_density=areal_density)
    t = t.copy()
    air = t.index.str.startswith("airframe")
    t.loc[air, "EUR"] = t.loc[air, "mass [kg]"] * LW_PLA_EUR_PER_KG
    out = t[["mass [kg]", "EUR"]].copy()
    out["mass [g]"] = out.pop("mass [kg]") * 1000
    return out[["mass [g]", "EUR"]]


def engineering_cost(rate=ENGINEERING_RATE_EUR_H, hours=ENGINEERING_HOURS):
    import pandas as pd
    df = pd.DataFrame(hours).T
    df.loc["total hours"] = df.sum()
    df.loc["total EUR"] = df.loc["total hours"] * rate
    return df


# ================================================================================================= stability and trim
def stability(p, fold_deg, table, *, it_deg=-1.0, cl=None) -> dict:
    """Centre of gravity (the mass table at this fold), neutral point, static margin, and the trim at lift
    coefficient ``cl`` (default the 1 g cruise at 15 m/s): angle of attack and elevator."""
    a = aero(p, fold_deg, it_deg=it_deg)
    x_cg = cg(table)
    m = table["mass [kg]"].sum()
    if cl is None:
        cl = m * G / (0.5 * RHO * 15.0 ** 2 * a["S_ref"])
    # moments about the CG: Cm = Cm0 + Cma a + Cmde de - CL (x_cg - x_ref)/c ... referred via the force's arm
    shift = (x_cg - a["x_ref"]) / a["c_ref"]
    Cm0_cg = a["Cm0"] + a["CL0"] * shift
    Cma_cg = a["Cma"] + a["CLa"] * shift
    Cmde_cg = a["Cmde"] + a["CLde"] * shift
    # two equations: CL0 + CLa al + CLde de = cl ; Cm0_cg + Cma_cg al + Cmde_cg de = 0
    M = np.array([[a["CLa"], a["CLde"]], [Cma_cg, Cmde_cg]])
    al, de = np.linalg.solve(M, [cl - a["CL0"], -Cm0_cg])
    return {"fold [deg]": a["fold_deg"], "x_cg [m]": x_cg, "x_np [m]": a["x_np"], "static margin [%MAC]": 100 * (a["x_np"] - x_cg) / a["c_ref"],
            "CL trim": cl, "alpha trim [deg]": math.degrees(al), "elevator trim [deg]": float(de), "mass [kg]": m}


# ================================================================================================= propulsion
FIXED_POWER_W = 8.0                # drawn whatever the thrust: the motor's no-load current (~0.5 A at 11 V) + avionics


@dataclass
class Drive(Unit):
    """MERLIN's ``Unit`` with the power a small aircraft draws whatever the thrust (``fixed_w``: the motor's no-load
    losses, the flight controller, GPS, receiver, servos) — at a 10 W cruise it is not negligible."""
    fixed_w: float = FIXED_POWER_W

    def full(self, V):
        T, P = self._row(V)
        p_max = (self.max_electrical_w - self.fixed_w) * self.drive_efficiency
        if P[-1] <= p_max:
            return float(T[-1]), float(P[-1] / self.drive_efficiency + self.fixed_w), float(self.rpm[-1])
        n = float(np.interp(p_max, P, self.rpm))
        return float(np.interp(n, self.rpm, T)), self.max_electrical_w, n

    def electrical_power(self, V, T_net):
        if T_net <= 0:
            return self.fixed_w
        return Unit.electrical_power(self, V, T_net) + self.fixed_w


def drive(prop_id="prop-9x5-b2", max_electrical_w=300.0, *, w=0.02, t=0.04, fixed_w=FIXED_POWER_W, library=None) -> Drive:
    """One tractor drive from notebook 25's propeller library at ``max_electrical_w`` (the ESC and the pack's limit),
    with the installation's wake fraction ``w`` and thrust deduction ``t`` (MERLIN's tractor values, a fuselage of
    similar proportions) and the fixed power ``fixed_w``."""
    import copy
    import dataclasses
    import propulsor_maps as pm
    lib = library or pm.load()
    e = copy.deepcopy(lib["by_id"][prop_id])
    e["installation"] = {"tractor": {"w": w, "t": t}, "pusher": {"w": w, "t": t}}
    u = pm.unit(e, max_electrical_w, layout="tractor")
    d = Drive(**{f.name: getattr(u, f.name) for f in dataclasses.fields(Unit)}, fixed_w=fixed_w)
    d.name = f"{prop_id} @ {max_electrical_w:.0f} W"
    return d


# ================================================================================================= airframes
@dataclass
class Aircraft:
    """One PEREGRINE: its name, mass, an Airframe per fold angle (A and B: only 0), the structural load factor,
    the fold servo's rate and the roll capability (p b / 2V at full aileron)."""
    name: str
    key: str
    mass_kg: float
    airframes: dict
    n_struct: float = 5.0
    fold_rate_deg_s: float = 150.0
    roll_pb2v: float = 0.09
    span_by_fold: dict = field(default_factory=dict)

    @property
    def folds(self):
        return sorted(self.airframes)

    def af(self, fold=0.0) -> Airframe:
        f = min(self.folds, key=lambda x: abs(x - fold))
        return self.airframes[f]

    def span(self, fold=0.0):
        f = min(self.folds, key=lambda x: abs(x - fold))
        return self.span_by_fold[f]

    def n_max(self, V, fold=0.0, rho=RHO):
        af = self.af(fold)
        q = 0.5 * rho * V ** 2
        return min(self.n_struct, q * af.wing_area_m2 * af.cl_max / (af.mass_kg * G))

    def roll_rate(self, V, fold=0.0):
        """Steady roll rate [rad/s] at full aileron: p = (p b / 2V) 2V / b, the ailerons' effectiveness falling as
        cos(fold) on a folded panel."""
        b = self.span(fold)
        return self.roll_pb2v * math.cos(math.radians(fold)) * 2 * V / b


def make_aircraft(key, p, mass_kg, *, folds=(0.0,), V=20.0, n_struct=5.0, fold_rate_deg_s=150.0, cd0_correction=0.0) -> Aircraft:
    afs, spans = {}, {}
    for f in folds:
        a = aero(p, f)
        b = pg.drag_buildup(p, V, fold_deg=f)
        afs[f] = Airframe(f"{key} fold {f:.0f}", mass_kg, a["S_ref"], a["AR"], b["cd0"] + cd0_correction,
                          oswald=OSWALD_VISCOUS * a["e"], cl_max=cl_max(p, f))
        spans[f] = a["span"]
    return Aircraft(pg.VARIANT_NAME[key], key, mass_kg, afs, n_struct, fold_rate_deg_s, span_by_fold=spans)


# ================================================================================================= the envelope
def turn(af: Airframe, unit: Unit, V, n_struct=5.0, rho=RHO) -> dict:
    """At airspeed V: the instantaneous load factor (lift- or structure-limited), the sustained one (full thrust =
    drag), their turn radii and rates."""
    V = np.asarray(V, float)
    q = 0.5 * rho * V ** 2
    S, W = af.wing_area_m2, af.mass_kg * G
    n_i = np.minimum(q * S * af.cl_max / W, n_struct)
    T = np.array([unit.full(v)[0] for v in np.atleast_1d(V)])
    k = af.k
    n_s = np.sqrt(np.maximum((T - q * S * af.cd0) * q * S / k, 0.0)) / W
    n_s = np.minimum(n_s, n_i)
    out = {}
    for tag, n in (("inst", n_i), ("sus", n_s)):
        s = np.sqrt(np.maximum(n ** 2 - 1, 1e-9))
        out[f"n_{tag}"] = n
        out[f"radius_{tag}"] = np.where(n > 1.0, V ** 2 / (G * s), np.inf)
        out[f"rate_{tag}"] = np.where(n > 1.0, G * s / V, 0.0)
    return out


def envelope(ac: Aircraft, unit: Unit, *, fold=0.0, battery_wh=24.4, usable=0.80, rho=RHO) -> dict:
    """The aircraft at one fold angle: MERLIN's speed polar (``merlin_flight.performance``) plus endurance, glide,
    turns and the roll rate."""
    af = ac.af(fold)
    perf = mf.performance(af, unit, rho)
    V = perf["V"]
    P = np.where(np.isfinite(perf["P_level"]), perf["P_level"], np.inf)
    i_e = int(np.argmin(P))
    tr = turn(af, unit, V, ac.n_struct, rho)
    ld = (af.mass_kg * G) / af.drag(V, rho)
    i_r = int(np.argmin(tr["radius_inst"]))
    i_w = int(np.argmax(tr["rate_inst"]))
    i_rs = int(np.argmin(tr["radius_sus"]))
    return {**perf, **tr, "L/D": ld, "v_endurance": float(V[i_e]), "P_min": float(P[i_e]),
            "endurance_min": battery_wh * usable / P[i_e] * 60, "range_km": battery_wh * usable / perf["wh_per_km_best"],
            "L/D_max": float(ld.max()), "v_L/D_max": float(V[np.argmax(ld)]),
            "r_min_inst": float(tr["radius_inst"][i_r]), "v_r_min": float(V[i_r]),
            "rate_max_deg_s": math.degrees(float(tr["rate_inst"][i_w])), "v_rate_max": float(V[i_w]),
            "r_min_sus": float(tr["radius_sus"][i_rs]), "rate_max_sus_deg_s": math.degrees(float(tr["rate_sus"].max())),
            "roll_deg_s_15": math.degrees(ac.roll_rate(15.0, fold)), "fold": fold}


def envelope_table(acs, unit, folds_for=None, **kw):
    import pandas as pd
    rows = {}
    for ac in acs:
        for f in (folds_for or {}).get(ac.key, [0.0]):
            e = envelope(ac, unit, fold=f, **kw)
            rows[f"{ac.key} fold {f:.0f}°"] = {
                "mass [kg]": ac.mass_kg, "span [m]": ac.span(f), "stall [m/s]": e["v_stall"], "top speed [m/s]": e["v_top"],
                "max climb [m/s]": e["roc_max"], "best endurance speed [m/s]": e["v_endurance"],
                "level power there [W]": e["P_min"], "endurance [min]": e["endurance_min"],
                "best range speed [m/s]": e["v_range"], "range [km]": e["range_km"], "L/D max": e["L/D_max"],
                "min turn radius [m]": e["r_min_inst"], "max turn rate [deg/s]": e["rate_max_deg_s"],
                "min sustained radius [m]": e["r_min_sus"], "max sustained rate [deg/s]": e["rate_max_sus_deg_s"],
                "roll rate at 15 m/s [deg/s]": e["roll_deg_s_15"]}
    return pd.DataFrame(rows)


# ================================================================================================= the birds
# Order-of-magnitude literature values (assumptions, to place the drones on a chart): typical flight speed, the speed
# of escape or pursuit, the load factor of their hard turns, mass. Starling: murmuration speeds 8-20 m/s (Attanasi et
# al. 2014; Pomeroy & Heppner 1992); pigeon: 15-20 m/s, banked turns at 2-3 g (Ros et al. 2011, Williams & Biewener
# 2015); rook/crow 10-15 m/s; greylag goose 15-20 m/s (Pennycuick); peregrine: level 10-25 m/s, pursuit 25-30 m/s,
# stoop 50-90 m/s, pull-outs to ~3 g in measurements (Tucker 1998, Ponitz et al. 2014, Mills et al. 2018).
BIRDS = {
    # name: (mass kg, cruise m/s, escape/pursuit m/s, hard-turn load factor, stoop m/s or None)
    "starling": (0.08, 15.0, 20.0, 2.5, None),
    "feral pigeon": (0.35, 17.0, 22.0, 2.5, None),
    "rook / crow": (0.45, 12.0, 16.0, 2.0, None),
    "greylag goose": (3.3, 16.0, 20.0, 1.5, None),
    "peregrine falcon": (0.80, 15.0, 28.0, 3.0, 70.0),
}


def bird_table():
    import pandas as pd
    rows = {}
    for name, (m, vc, vx, n, vs) in BIRDS.items():
        s = math.sqrt(n ** 2 - 1)
        rows[name] = {"mass [kg]": m, "cruise [m/s]": vc, "escape/pursuit [m/s]": vx, "hard-turn n": n,
                      "turn radius at cruise [m]": vc ** 2 / (G * s), "turn rate at cruise [deg/s]": math.degrees(G * s / vc),
                      "stoop [m/s]": vs}
    return pd.DataFrame(rows).T


# ================================================================================================= the stoop
def stoop(ac: Aircraft, unit: Unit, *, h0=60.0, x_target=120.0, h_pass=3.0, V0=15.0, tuck=0.0, throttle=0.0,
          n_pull=None, dt=0.005, h_unfold=None):
    """The falcon's dive in the vertical plane: from level flight at ``h0`` and ``V0``, a push-over (n = 0) onto the
    line through the target (``x_target`` ahead, ``h_pass`` above it), the dive at that angle with the wings tucked
    to ``tuck`` degrees (C) and the propeller at ``throttle`` (0: stopped, the falcon's way), then the unfolding (at the
    servo's rate) and the pull-out at ``n_pull`` (default: the structure's), limited by the lift at each moment, to
    level flight at ``h_pass``. The pull-out height is found by bisection so the bottom of the curve is ``h_pass``.
    Returns the time history and the numbers: time to the target, top speed, speed over the target, height used."""
    n_pull = ac.n_struct if n_pull is None else n_pull
    W, m = ac.mass_kg * G, ac.mass_kg
    gam_d = -math.atan2(h0 - h_pass, x_target)
    t_unfold = tuck / ac.fold_rate_deg_s if tuck else 0.0

    def run(h_pull, record=False):
        x, h, V, gam, t = 0.0, h0, V0, 0.0, 0.0
        phase, fold, t0u = "push-over", tuck, None
        rec = []
        while t < 60.0:
            if phase in ("push-over", "dive") and h <= h_pull:
                phase, t0u = "pull-out", t
            if phase == "pull-out" and tuck:
                fold = max(tuck - (t - t0u) * ac.fold_rate_deg_s, 0.0)
            af = ac.af(fold)
            q = 0.5 * RHO * V ** 2
            n_avail = min(q * af.wing_area_m2 * af.cl_max / W, n_pull)
            if phase == "push-over":
                n = 0.0
                if gam <= gam_d:
                    phase, gam = "dive", gam_d
            if phase == "dive":
                n = math.cos(gam)
            if phase == "pull-out":
                n = n_avail
            T = unit.full(V)[0] * throttle
            D = float(af.drag(V, RHO, max(n, 1e-3)))
            V += (T - D - W * math.sin(gam)) / m * dt
            gam += G * (n - math.cos(gam)) / max(V, 1.0) * dt
            x += V * math.cos(gam) * dt
            h += V * math.sin(gam) * dt
            t += dt
            if record:
                rec.append((t, x, h, V, gam, n, fold, phase))
            if phase == "pull-out" and gam >= 0.0:
                break
            if h < -5:
                break
        return h, t, rec

    lo, hi = h_pass, h0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        h_end, _, _ = run(mid)
        if h_end < h_pass:
            lo = mid
        else:
            hi = mid
    h_end, t_end, rec = run(hi, record=True)
    a = np.array([r[:7] for r in rec])
    phases = [r[7] for r in rec]
    return {"t": a[:, 0], "x": a[:, 1], "h": a[:, 2], "V": a[:, 3], "gamma_deg": np.degrees(a[:, 4]), "n": a[:, 5],
            "fold": a[:, 6], "phase": phases, "time_s": t_end, "V_max": float(a[:, 3].max()), "V_bottom": float(a[-1, 3]),
            "h_pull": hi, "x_bottom": float(a[-1, 1]), "dive_angle_deg": math.degrees(gam_d), "h_bottom": float(h_end),
            "n_peak": float(a[:, 5].max()), "t_unfold": t_unfold}


def terminal_speed(af: Airframe, gamma_deg=-90.0, rho=RHO):
    """Steady dive speed at zero lift (vertical) or along a dive angle with lift = W cos(gamma)."""
    W, S = af.mass_kg * G, af.wing_area_m2
    g = math.radians(gamma_deg)
    # W sin|g| = q S (cd0 + k (W cos g / q S)^2) -> solve for q
    a_, b_, c_ = S * af.cd0, -W * math.sin(abs(g)), af.k * (W * math.cos(g)) ** 2 / S
    q = (-b_ + math.sqrt(max(b_ ** 2 - 4 * a_ * c_, 0.0))) / (2 * a_)
    return math.sqrt(2 * q / rho)


# ================================================================================================= gaps between trees and houses
def gap_passage(ac: Aircraft, *, depth=4.0, V=None, dh_max=1.0, margin=0.15, body_height=0.17, folds=None):
    """The narrowest gap the aircraft passes, ``depth`` deep (a hedge, a tree line, the yard between two houses),
    losing at most ``dh_max`` of height. Bank through it: roll to phi at full aileron, hold, roll back; the lift held at
    W cos(phi) so the path stays straight, so the aircraft sinks at g sin^2(phi); the width it takes is
    ``b cos(phi) + body_height sin(phi)`` plus ``margin`` each side (the tracking error). Folding (C): the folded span,
    at a speed at least 1.25 x its stall speed, and the bank on top. Returns the best way and its width."""
    best = {"width_m": math.inf}
    for f in (folds or ac.folds):
        af = ac.af(f)
        v = max(V or 0.0, 1.25 * af.stall_speed())
        b = ac.span(f)
        p = ac.roll_rate(v, f)
        for phi in np.radians(np.arange(0.0, 90.1, 2.5)):
            t_roll = phi / p
            t_tot = 2 * t_roll + depth / v
            dh = 0.5 * G * math.sin(phi) ** 2 * (depth / v + t_roll) ** 2           # the roll-in and the hold
            if dh > dh_max:
                continue
            wdt = b * math.cos(phi) + body_height * math.sin(phi) + 2 * margin
            if wdt < best["width_m"]:
                best = {"width_m": wdt, "fold_deg": f, "bank_deg": math.degrees(phi), "speed": v, "height_lost_m": dh,
                        "time_s": t_tot}
    return best


# ================================================================================================= roof: start, landing, rest
def prop_hang(ac: Aircraft, unit: Unit, *, hang_s=4.0, climb=2.0) -> dict:
    """Take-off by hanging on the propeller (nose up, the thrust carrying the weight, as 3D aerobatic models do):
    static thrust / weight, the electrical power to hover and to climb at ``climb`` m/s, the energy of ``hang_s``
    seconds, and the transition (MERLIN's ``_accelerate``: full power from 3 m/s to 1.3 x stall)."""
    W = ac.mass_kg * G
    T0 = unit.full(0.0)[0]
    P_hover = unit.electrical_power(0.0, W)
    P_climb = unit.electrical_power(climb, W * 1.03)
    af = ac.af(0.0)
    t_tr, x_tr, e_tr = mf._accelerate(af, unit, 3.0, 1.3 * af.stall_speed(), RHO)
    return {"T/W": T0 / W, "hover power [W]": P_hover, "climb power [W]": P_climb,
            "hang energy [Wh]": P_hover * hang_s / 3600 if np.isfinite(P_hover) else math.inf,
            "transition time [s]": t_tr, "transition distance [m]": x_tr, "transition energy [Wh]": e_tr,
            "possible": T0 > 1.1 * W}


def hand_throw(ac: Aircraft, unit: Unit, *, V0=9.0, h0=1.8, pitch_deg=10.0, fold=0.0, dt=0.005, t_max=8.0) -> dict:
    """A hand throw at ``V0`` from ``h0`` at ``pitch_deg`` with full power: the autopilot holds the lift at
    min(0.9 CL_max, the 1 g lift) — it flies away when it gets to 1.3 x stall without touching the ground."""
    af = ac.af(fold)
    W, m, S = ac.mass_kg * G, ac.mass_kg, af.wing_area_m2
    x, h, V, gam, t = 0.0, h0, V0, math.radians(pitch_deg), 0.0
    hs = []
    v_ok = 1.3 * af.stall_speed()
    while t < t_max:
        q = 0.5 * RHO * V ** 2
        L = min(0.9 * q * S * af.cl_max, W * 1.0 if V < v_ok else W * 1.15)
        cl = L / (q * S)
        D = q * S * (af.cd0 + af.k * cl ** 2)
        T = unit.full(V)[0]
        V += (T - D - W * math.sin(gam)) / m * dt
        gam += (L - W * math.cos(gam)) / (m * max(V, 1.0)) * dt
        h += V * math.sin(gam) * dt
        x += V * math.cos(gam) * dt
        t += dt
        hs.append(h)
        if h <= 0:
            return {"flies": False, "min height [m]": 0.0, "time to 1.3 stall [s]": math.nan, "V0 [m/s]": V0}
        if V >= v_ok and gam > 0:
            break
    return {"flies": V >= v_ok, "min height [m]": float(min(hs)), "time to 1.3 stall [s]": t, "V0 [m/s]": V0,
            "stall [m/s]": af.stall_speed()}


def min_throw_speed(ac, unit, **kw):
    for v in np.arange(3.0, 15.01, 0.25):
        r = hand_throw(ac, unit, V0=v, **kw)
        if r["flies"] and r["min height [m]"] > 0.3:
            return float(v)
    return math.nan


def roof_drop(ac: Aircraft, unit: Unit, *, throttle=1.0, fold=0.0, V0=1.0, dt=0.005) -> dict:
    """The falcon's start: step off the roof edge (``V0``), dive under full power with the lift at 0.9 CL_max at most,
    and level off at 1.2 x stall: the height it needs. A folded wing (C) unfolds as it levels."""
    af0 = ac.af(0.0)
    W, m = ac.mass_kg * G, ac.mass_kg
    h, V, gam, t, f = 0.0, V0, -math.pi / 2 * 0.9, 0.0, fold
    while t < 10.0:
        af = ac.af(f)
        q = 0.5 * RHO * V ** 2
        L = min(0.9 * q * af.wing_area_m2 * af.cl_max, 3.0 * W)
        cl = L / max(q * af.wing_area_m2, 1e-6)
        D = q * af.wing_area_m2 * (af.cd0 + af.k * cl ** 2)
        V += (unit.full(V)[0] * throttle - D - W * math.sin(gam)) / m * dt
        gam += (L - W * math.cos(gam)) / (m * max(V, 0.5)) * dt
        h += V * math.sin(gam) * dt
        t += dt
        if f > 0 and V > 1.3 * af0.stall_speed():
            f = max(f - ac.fold_rate_deg_s * dt, 0.0)
        if gam >= 0.0 and V >= 1.2 * af0.stall_speed() and f == 0:
            break
    return {"height needed [m]": -h, "time [s]": t, "speed [m/s]": V}


def perched_wind(p, fold, mass_kg, *, alpha_deg=5.0, friction=0.6, speed_up=1.3, gust=1.4, cd_body=0.9):
    """Wind on the drone resting on a roof ridge or a flat roof edge, facing into the wind (it lands into the wind),
    sitting at ``alpha_deg`` on its skids. The ridge speeds the wind up (``speed_up``), gusts add ``gust``. The lattice
    gives the lift at that attitude and fold; it lifts off when L > W, slides when the drag exceeds friction x (W - L)
    (the drag: the polar's plus the body's ``cd_body`` on the side area of a standing fuselage, small). Returns the
    mean wind speeds [m/s] at which it lifts off and slides."""
    a = aero(p, fold)
    S = a["S_ref"]
    CL = a["CL0"] + a["CLa"] * math.radians(alpha_deg)
    cd0 = pg.drag_buildup(p, 8.0, fold_deg=fold)["cd0"]
    CD = cd0 + a["k"] * CL ** 2 + 0.02
    W = mass_kg * G
    k_wind = speed_up * gust

    def lift(U):
        return 0.5 * RHO * (k_wind * U) ** 2 * S * CL

    U_lift = math.sqrt(2 * W / (RHO * S * max(CL, 1e-6))) / k_wind if CL > 0 else math.inf
    # slide: 0.5 rho U^2 S (CD + mu CL) = mu W
    U_slide = math.sqrt(2 * friction * W / (RHO * S * (CD + friction * max(CL, 0.0)))) / k_wind
    return {"fold [deg]": fold, "alpha [deg]": alpha_deg, "CL": CL, "CD": CD, "lift-off wind [m/s]": U_lift,
            "slide wind [m/s]": U_slide, "holds to [m/s]": min(U_lift, U_slide)}


# ================================================================================================= the hinge
def hinge_loads(p, fold, mass_kg, *, n=4.0, V=25.0, mu=0.25, fold_time_s=0.4, panel_kg=None) -> dict:
    """What the hinge and the fold servo see. The outer panel's lift (from the lattice at the load factor ``n``) acts at
    its centre of pressure: the bending moment about the hinge is carried by the two bearings of the vertical pin,
    spaced by the local wing depth, as a couple; the pin's shear and the lug's bearing stress follow. The servo turns
    the panel about the vertical pin against the panel's drag times its arm, the bearings' friction under that couple,
    and the panel's inertia for a bang-bang fold in ``fold_time_s``."""
    q_ = pg.resolve(p)
    s = pg.Peregrine.stations(q_)
    pl = pg.planform(q_, fold)
    quads, hidden, inc, ns = _surfaces(pl)
    W = mass_kg * G
    qd = 0.5 * RHO * V ** 2
    CL_req = n * W / (qd * pl["S_ref"])
    a = aero(q_, fold)
    alpha = math.degrees((CL_req - a["CL0"]) / a["CLa"])
    r = vlm(quads, alpha_deg=alpha, incidence_deg=inc, S_ref=pl["S_ref"], c_ref=pl["c_ref"], n_span=ns, n_chord=4, hidden=hidden)
    lat = r["lattice"]
    n_w = len(pl["wing"]) // 2
    left_outer = lat["quad"] == (n_w - 1)
    Fz = r["F"][left_outer, 2] * qd
    mid = r["mid"][left_outer]
    hx, hy = s["xh"] / 1000, -s["yh"] / 1000
    L_panel = float(Fz.sum())
    cp_xy = (mid[:, :2] * Fz[:, None]).sum(0) / L_panel
    arm = float(np.hypot(cp_xy[0] - hx, cp_xy[1] - hy))
    M_bend = L_panel * arm
    t_loc, _ = pg._thickness_at(q_, q_["hinge_x_frac"])
    depth = 0.75 * t_loc * s["ch"] / 1000                      # bearing spacing: 75 % of the local wing depth
    F_bear = M_bend / depth
    d = q_["hinge_pin_diameter"] / 1000
    tau_pin = (F_bear + L_panel / 2) / (math.pi * d ** 2 / 4)
    sigma_br = (F_bear + L_panel / 2) / (d * q_["lug_thickness"] / 1000)
    S_panel = pg._quad_area(pl["wing"][n_w - 1][:, :2])
    D_panel = qd * S_panel * 0.012 + qd * pl["S_ref"] * a["k"] * CL_req ** 2 * L_panel / max(W * n, 1e-9) * 0.5
    Q_drag = D_panel * abs(cp_xy[1] - hy)
    Q_fric = 2 * mu * F_bear * d / 2
    panel_kg = panel_kg if panel_kg is not None else pg.wetted_areas(q_, 0.0)["wing"] * pg.exposed_wing_area(q_, 0.0)["outer_panel"] \
        / pg.exposed_wing_area(q_, 0.0)["exposed"] * AIRFRAME_AREAL_DENSITY + 0.012 + 0.009
    span_out = s["b2"] / 1000 - s["yh"] / 1000
    I_z = panel_kg * span_out ** 2 / 3
    theta = math.radians(max(fold, pg.FOLD_TUCK))
    Q_inertia = I_z * 4 * theta / fold_time_s ** 2
    Q = Q_drag + Q_fric + Q_inertia
    return {"fold [deg]": fold, "n": n, "V [m/s]": V, "panel lift [N]": L_panel, "arm [m]": arm, "bending at hinge [N m]": M_bend,
            "bearing spacing [mm]": depth * 1000, "bearing force [N]": F_bear, "pin shear [MPa]": tau_pin / 1e6,
            "lug bearing stress [MPa]": sigma_br / 1e6, "drag moment [N m]": Q_drag, "friction moment [N m]": Q_fric,
            "inertia moment [N m]": Q_inertia, "servo torque [N m]": Q, "panel mass [kg]": panel_kg}


# ================================================================================================= the farm and the flock
@dataclass
class Farm:
    """An abstract farm (metres, x east, y north): the crop field (a rectangle), the farmhouse and the barn (boxes
    with pitched roofs), tree lines with gaps (rows of crowns), the roof perch on the farmhouse's ridge, the safe area
    the birds are pushed to (a wood, away from the crop)."""
    field_xy: tuple = (40.0, -110.0, 260.0, 110.0)                       # xmin, ymin, xmax, ymax
    buildings: tuple = ((-30.0, -20.0, 14.0, 10.0, 5.0, 8.5),            # x, y (centre), length (x), width (y), eave, ridge
                        (-30.0, 22.0, 22.0, 12.0, 6.0, 9.5))
    tree_lines: tuple = (((20.0, -120.0), (20.0, 120.0), 9.0, 12.0, ((0.0, 2.6), (60.0, 3.2))),   # start, end, spacing, height, gaps (y, width)
                         ((40.0, 130.0), (270.0, 130.0), 10.0, 14.0, ((150.0, 4.0),)))
    perch: tuple = (-30.0, -20.0, 8.5)
    safe_xy: tuple = (180.0, 420.0)
    safe_r: float = 80.0

    @property
    def crop_centre(self):
        x0, y0, x1, y1 = self.field_xy
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def trees(self):
        out = []
        for (a, b, sp, h, gaps) in self.tree_lines:
            a, b = np.array(a), np.array(b)
            L = np.linalg.norm(b - a)
            u = (b - a) / L
            for s in np.arange(0, L + 1e-6, sp):
                p = a + u * s
                along = s if abs(u[0]) > 0.5 else p[1]
                if any(abs(along - g) < sp * 0.6 + w / 2 for g, w in gaps):
                    continue
                out.append((p[0], p[1], h, sp * 0.55))
        return np.array(out)

    def in_field(self, xy):
        x0, y0, x1, y1 = self.field_xy
        xy = np.atleast_2d(xy)
        return (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)


@dataclass
class Flock:
    """A boids flock of one species: cohesion, alignment and separation among neighbours, the pull of the crop (the
    birds want to feed there and circle it), and fear of the drone within ``fear_r`` (they flee at up to ``v_max``,
    after Paranjape et al. 2018, robotic herding of bird flocks with a UAV — the noise and the shape together)."""
    n: int = 40
    species: str = "starling"
    v_cruise: float = 15.0
    v_max: float = 20.0
    a_max: float = 2.5 * G
    fear_r: float = 45.0
    neigh_r: float = 12.0
    sep_r: float = 2.0
    w_coh: float = 0.6
    w_ali: float = 1.0
    w_sep: float = 4.0
    w_crop: float = 0.8
    w_fear: float = 14.0
    altitude: float = 12.0
    seed: int = 0

    @classmethod
    def of(cls, species, **kw):
        m, vc, vx, n, _ = BIRDS[species]
        return cls(species=species, v_cruise=vc, v_max=vx, a_max=n * G, **kw)

    def start(self, centre):
        rng = np.random.default_rng(self.seed)
        pos = np.c_[centre[0] + rng.normal(0, 8, self.n), centre[1] + rng.normal(0, 8, self.n), self.altitude + rng.normal(0, 2, self.n)]
        hd = rng.uniform(0, 2 * math.pi)
        vel = np.c_[np.full(self.n, math.cos(hd)), np.full(self.n, math.sin(hd)), np.zeros(self.n)] * self.v_cruise
        return pos, vel

    def step(self, pos, vel, drone, crop_centre, dt):
        d = pos[:, None, :] - pos[None]
        r = np.linalg.norm(d, axis=-1) + np.eye(self.n) * 1e6
        nb = r < self.neigh_r
        cnt = np.maximum(nb.sum(1), 1)[:, None]
        coh = (np.where(nb[..., None], pos[None], 0).sum(1) / cnt - pos) * nb.any(1)[:, None]
        ali = (np.where(nb[..., None], vel[None], 0).sum(1) / cnt - vel) * nb.any(1)[:, None]
        close = r < self.sep_r
        sep = np.where(close[..., None], d / np.maximum(r, 1e-3)[..., None] ** 2, 0).sum(1)
        to_c = np.c_[crop_centre - pos[:, :2], self.altitude - pos[:, 2]]
        dist_c = np.linalg.norm(to_c[:, :2], axis=1)[:, None]
        tang = np.c_[-to_c[:, 1], to_c[:, 0], np.zeros(self.n)] / np.maximum(dist_c, 1.0)
        crop = to_c / np.maximum(dist_c, 1.0) * np.clip(dist_c / 40.0, 0, 3) + 0.6 * tang * self.v_cruise / 10
        crop[:, 2] = 0.3 * to_c[:, 2]
        acc = self.w_coh * coh * 0.2 + self.w_ali * ali * 0.5 + self.w_sep * sep * 3.0 + self.w_crop * crop
        afraid = np.zeros(self.n, bool)
        if drone is not None:
            dd = pos - drone
            rd = np.linalg.norm(dd, axis=1)
            afraid = rd < self.fear_r
            f = np.where(afraid, (self.fear_r - rd) / self.fear_r, 0.0)
            acc = acc * (1 - 0.8 * afraid)[:, None] + self.w_fear * (f / np.maximum(rd, 1.0))[:, None] * dd
        a = np.linalg.norm(acc, axis=1)[:, None]
        acc = np.where(a > self.a_max, acc / a * self.a_max, acc)
        vel = vel + acc * dt
        v = np.linalg.norm(vel, axis=1)[:, None]
        vt = np.where(afraid, self.v_max, self.v_cruise)[:, None]
        vel = vel / np.maximum(v, 1e-6) * np.clip(v, 0.6 * self.v_cruise, vt)
        pos = pos + vel * dt
        pos[:, 2] = np.maximum(pos[:, 2], 2.0)
        return pos, vel, afraid


def _pm_step(ac, unit, st, target, v_cmd, dt, *, fold_cmd=0.0, vz_cmd=None, throttle_max=1.0):
    """One step of the point-mass aircraft (MERLIN's ``fly`` logic in still air): heading towards ``target`` (x, y, z)
    within the turn rate the lift and the structure allow at this fold, speed and height from the excess thrust; the
    fold follows ``fold_cmd`` at the servo's rate."""
    fold = st["fold"]
    df = fold_cmd - fold
    fold += float(np.clip(df, -ac.fold_rate_deg_s * dt, ac.fold_rate_deg_s * dt))
    af = ac.af(fold)
    V, psi, pos = st["V"], st["psi"], st["pos"]
    W, m = ac.mass_kg * G, ac.mass_kg
    dxy = target[:2] - pos[:2]
    chi = math.atan2(dxy[1], dxy[0])
    dpsi = (chi - psi + math.pi) % (2 * math.pi) - math.pi
    n_lim = ac.n_max(V, fold)
    rate_max = G * math.sqrt(max(n_lim ** 2 - 1, 0.0)) / max(V, 1.0)
    rate = float(np.clip(dpsi / 1.0, -rate_max, rate_max))
    # roll: the bank cannot change faster than the roll rate
    bank_cmd = math.atan(rate * V / G)
    p_roll = ac.roll_rate(V, fold)
    bank = st["bank"] + float(np.clip(bank_cmd - st["bank"], -p_roll * dt, p_roll * dt))
    rate = G * math.tan(bank) / max(V, 1.0)
    psi += rate * dt
    nload = 1.0 / max(math.cos(bank), 0.05)
    h_err = target[2] - pos[2]
    vz_c = float(np.clip(0.4 * h_err, -6.0, 6.0)) if vz_cmd is None else vz_cmd
    Tmax = unit.full(V)[0] * throttle_max
    D = float(af.drag(V, RHO, nload))
    a_cmd = float(np.clip((v_cmd - V) / 2.0, -3.0, 5.0))
    T_need = D + W * vz_c / max(V, 1.0) + m * a_cmd
    if T_need > Tmax:
        T = Tmax
        vz = vz_c if vz_c <= 0 else min(vz_c, max(T - D, 0.0) * V / W)
        a = (T - D - W * vz / max(V, 1.0)) / m
    else:
        T, vz, a = max(T_need, 0.0), vz_c, a_cmd
        if T_need < 0:
            a = (-D - W * vz / max(V, 1.0)) / m
    P = unit.electrical_power(V, T) if T > 0 else getattr(unit, "fixed_w", 0.0)
    P = min(P, unit.max_electrical_w) if np.isfinite(P) else unit.max_electrical_w
    V = max(V + a * dt, 1.02 * af.stall_speed())
    pos = pos + np.array([V * math.cos(psi), V * math.sin(psi), vz]) * dt
    return {"pos": pos, "V": V, "psi": psi, "bank": bank, "fold": fold, "T": T, "P": P, "n": nload,
            "E": st["E"] + P * dt / 3600}


def herd(ac: Aircraft, unit: Unit, farm: Farm, flock: Flock, *, t_max=240.0, dt=0.05, standoff=0.7, v_herd=None,
         fold_fast=None, start=None):
    """Herd the flock off the crop towards the safe area: the drone aims at the point ``standoff`` x fear radius behind
    the flock's centre (on the far side from the safe area: Paranjape's push), at its herding speed (default its top
    speed, capped at 1.4 x the birds' escape speed; never below 1.3 x its stall speed): a fixed wing cannot hover
    behind the flock, so it circles the aim point fast, turning within its limits; a folding wing tucks to ``fold_fast``
    when the aim is straight ahead and spreads in turns. Done when 90 % of the birds are 60 m outside the field and
    moving away. Returns the time histories and the numbers."""
    pos_b, vel_b = flock.start(farm.crop_centre)
    safe = np.array(farm.safe_xy)
    e = envelope(ac, unit)
    v_h = v_herd or min(e["v_top"] * 0.95, 1.4 * flock.v_max)
    st0 = start or {"pos": np.array([farm.perch[0] + 60.0, farm.perch[1], 20.0]), "V": e["v_range"], "psi": 0.0}
    st = {"pos": np.array(st0["pos"], float), "V": st0["V"], "psi": st0["psi"], "bank": 0.0, "fold": 0.0, "E": 0.0, "T": 0.0, "P": 0.0, "n": 1.0}
    rec = {k: [] for k in ("t", "pos", "V", "bank", "fold", "P", "E", "birds", "afraid", "dist")}
    t, done = 0.0, None
    x0, y0, x1, y1 = farm.field_xy
    while t < t_max:
        c = pos_b.mean(axis=0)
        away = c[:2] - safe
        away /= max(np.linalg.norm(away), 1e-6)
        aim = np.r_[c[:2] + away * standoff * flock.fear_r, c[2] + 3.0]
        dpsi = abs((math.atan2(aim[1] - st["pos"][1], aim[0] - st["pos"][0]) - st["psi"] + math.pi) % (2 * math.pi) - math.pi)
        d_aim = float(np.linalg.norm(aim[:2] - st["pos"][:2]))
        fold_cmd = (fold_fast if (fold_fast and dpsi < math.radians(15) and d_aim > 60) else 0.0)
        st = _pm_step(ac, unit, st, aim, max(v_h, 1.3 * ac.af(st["fold"]).stall_speed()), dt, fold_cmd=fold_cmd)
        pos_b, vel_b, afraid = flock.step(pos_b, vel_b, st["pos"], farm.crop_centre, dt)
        t += dt
        outside = ~farm.in_field(pos_b[:, :2])
        far = outside & ((pos_b[:, 0] < x0 - 60) | (pos_b[:, 0] > x1 + 60) | (pos_b[:, 1] < y0 - 60) | (pos_b[:, 1] > y1 + 60))
        rec["t"].append(t); rec["pos"].append(st["pos"].copy()); rec["V"].append(st["V"]); rec["bank"].append(st["bank"])
        rec["fold"].append(st["fold"]); rec["P"].append(st["P"]); rec["E"].append(st["E"]); rec["birds"].append(pos_b.copy())
        rec["afraid"].append(afraid.mean()); rec["dist"].append(float(np.linalg.norm(c - st["pos"])))
        if far.mean() >= 0.9:
            done = t
            break
    out = {k: np.array(v) for k, v in rec.items()}
    out.update({"cleared": done is not None, "time_s": done if done is not None else math.nan, "energy_wh": float(st["E"]),
                "v_herd": v_h, "mean_distance_m": float(np.mean(rec["dist"])), "aircraft": ac.name})
    return out


# ================================================================================================= the sortie and the movie
@dataclass
class Episode:
    t: np.ndarray
    pos: np.ndarray
    V: np.ndarray
    psi: np.ndarray
    bank: np.ndarray
    fold: np.ndarray
    power: np.ndarray
    energy_wh: np.ndarray
    phase: list
    birds: list
    deer: np.ndarray
    events: list = field(default_factory=list)
    aircraft: str = ""
    key: str = ""

    def phase_table(self):
        import pandas as pd
        rows, start = [], 0
        for i in range(1, len(self.t) + 1):
            if i == len(self.t) or self.phase[i] != self.phase[start]:
                sl = slice(start, i)
                rows.append({"phase": self.phase[start], "start [s]": self.t[start], "duration [s]": self.t[i - 1] - self.t[start],
                             "mean airspeed [m/s]": float(np.mean(self.V[sl])), "max airspeed [m/s]": float(np.max(self.V[sl])),
                             "energy [Wh]": self.energy_wh[i - 1] - self.energy_wh[start]})
                start = i
        return pd.DataFrame(rows).set_index("phase")

    def summary(self):
        return {"aircraft": self.aircraft, "flight_time_s": float(self.t[-1]), "energy_wh": float(self.energy_wh[-1]),
                "max_airspeed": float(self.V.max()), "events": list(self.events)}


def mission(ac: Aircraft, unit: Unit, farm: Farm, flock: Flock, *, patrol_agl=25.0, stoop_h=55.0, deer_xy=(200.0, -60.0),
            tuck=0.0, fold_fast=None, dt=0.05, battery_wh=24.4):
    """One sortie from the farmhouse roof and back: prop-hang take-off and transition, a patrol lap of the field at the
    best-endurance speed (the sound on the crop), a climb and a stoop at the deer (a low pass at full power over it),
    the flock arriving over the crop and herded to the safe area, the flight home, the prop-hang landing on the ridge."""
    e = envelope(ac, unit)
    ph = prop_hang(ac, unit)
    perch = np.array(farm.perch, float)
    st = {"pos": perch + [0, 0, 0.3], "V": 0.0, "psi": 0.0, "bank": 0.0, "fold": 0.0, "E": 0.0, "T": 0.0, "P": 0.0, "n": 1.0}
    rec = {k: [] for k in ("t", "pos", "V", "psi", "bank", "fold", "P", "E", "phase", "birds")}
    events, t = [], 0.0
    deer = np.array([*deer_xy, 0.0])
    W = ac.mass_kg * G

    def log(phase, birds=None):
        rec["t"].append(t); rec["pos"].append(st["pos"].copy()); rec["V"].append(st["V"]); rec["psi"].append(st["psi"])
        rec["bank"].append(st["bank"]); rec["fold"].append(st["fold"]); rec["P"].append(st["P"]); rec["E"].append(st["E"])
        rec["phase"].append(phase); rec["birds"].append(None if birds is None else birds.copy())

    # 1. hang on the propeller: climb 4 m, then the transition
    P_h = ph["climb power [W]"]
    while st["pos"][2] < perch[2] + 4.0:
        st["pos"] = st["pos"] + [0, 0, 2.0 * dt]; st["P"] = P_h; st["E"] += P_h * dt / 3600; t += dt; log("prop-hang take-off")
    v_cr = e["v_endurance"]
    n_tr = max(1, int(ph["transition time [s]"] / dt))
    for i in range(n_tr):
        st["V"] = 3.0 + (1.3 * ac.af(0).stall_speed() - 3.0) * (i + 1) / n_tr
        st["pos"] = st["pos"] + [st["V"] * dt, 0, 0]; st["P"] = unit.max_electrical_w; st["E"] += st["P"] * dt / 3600; t += dt
        log("transition")
    events.append(f"t = {t:.0f} s: wing-borne")
    # 2. patrol lap
    x0, y0, x1, y1 = farm.field_xy
    m_ = 15.0
    lap = [(x0 + m_, y0 + m_), (x1 - m_, y0 + m_), (x1 - m_, y1 - m_), (x0 + m_, y1 - m_), (x0 + m_, y0 + m_)]
    for wx, wy in lap:
        tgt = np.array([wx, wy, patrol_agl])
        while np.linalg.norm(tgt[:2] - st["pos"][:2]) > 12.0 and t < 2000:
            st = _pm_step(ac, unit, st, tgt, v_cr, dt); t += dt; log("patrol lap")
    # 3. climb and stoop at the deer
    tgt = np.r_[deer[:2] + [-260.0, 0.0], stoop_h]
    while (np.linalg.norm(tgt[:2] - st["pos"][:2]) > 15.0 or st["pos"][2] < stoop_h - 3) and t < 3000:
        st = _pm_step(ac, unit, st, tgt, e["v_climb"], dt); t += dt; log("climb for the stoop")
    gam_d = math.atan2(stoop_h - 3.0, np.linalg.norm(deer[:2] - st["pos"][:2]) - 30.0)
    while st["pos"][2] > 3.0 + st["V"] ** 2 / (G * max(ac.n_max(st["V"], 0.0) - 1, 0.5)) * (1 - math.cos(gam_d)) and t < 3000:
        vz = -st["V"] * math.sin(gam_d)
        st = _pm_step(ac, unit, st, deer + [0, 0, 3.0], 99.0, dt, fold_cmd=tuck, vz_cmd=vz, throttle_max=0.0); t += dt; log("stoop")
    events.append(f"t = {t:.0f} s: stoop at {st['V']:.0f} m/s")
    k = 0
    while st["pos"][0] < deer[0] + 60.0 and k < 4000:
        st = _pm_step(ac, unit, st, deer + [200.0, 0, 3.0], 99.0, dt, fold_cmd=0.0); t += dt; k += 1; log("low pass over the deer, full power")
    # 4. herd the flock that arrived over the crop
    events.append(f"t = {t:.0f} s: a {flock.species} flock over the crop")
    h = herd(ac, unit, farm, flock, start={"pos": st["pos"].copy(), "V": st["V"], "psi": st["psi"]}, fold_fast=fold_fast, dt=dt)
    for i in range(len(h["t"])):
        st["pos"], st["V"], st["bank"], st["fold"] = h["pos"][i], h["V"][i], h["bank"][i], h["fold"][i]
        st["P"] = h["P"][i]; st["E"] = rec["E"][-1] + (h["E"][i] - (h["E"][i - 1] if i else 0.0)) if rec["E"] else h["E"][i]
        if i:
            d = h["pos"][i] - h["pos"][i - 1]
            st["psi"] = math.atan2(d[1], d[0])
        t += dt; log(f"herding the {flock.species}s", h["birds"][i])
    events.append(f"t = {t:.0f} s: flock {'cleared' if h['cleared'] else 'NOT cleared'}"
                  + (f" in {h['time_s']:.0f} s" if h["cleared"] else ""))
    birds_last = h["birds"][-1]
    # 5. home, the transition to the hang over the ridge and the prop-hang landing
    tgt = np.r_[perch[:2] + [45.0, 0.0], perch[2] + 6.0]
    pos_b = birds_last.copy()
    drift = np.r_[pos_b[:, :2].mean(axis=0) - farm.crop_centre, 0.0]
    drift = drift / max(np.linalg.norm(drift), 1e-6) * 12.0
    k = 0
    while np.linalg.norm(tgt[:2] - st["pos"][:2]) > 10.0 and k < 20000:
        st = _pm_step(ac, unit, st, tgt, e["v_range"], dt); t += dt; k += 1
        pos_b = pos_b + drift * dt
        log("home", pos_b)
    P_hov = unit.electrical_power(0.0, W)
    P_hov = P_hov if np.isfinite(P_hov) else unit.max_electrical_w
    p0, v0, goal = st["pos"].copy(), st["V"], perch + np.r_[0.0, 0.0, 3.0]
    st["psi"] = math.atan2(goal[1] - p0[1], goal[0] - p0[0])
    n_tr = max(2, int(np.linalg.norm(goal - p0) / (0.5 * (v0 + 1.0)) / dt))
    for i in range(n_tr):
        f = (i + 1) / n_tr
        st["pos"] = p0 + (goal - p0) * (1 - (1 - f) ** 2)
        st["V"] = v0 + (1.0 - v0) * f
        st["bank"] = 0.0
        st["P"] = P_hov * f + st["P"] * (1 - f)
        st["E"] += st["P"] * dt / 3600; t += dt
        pos_b = pos_b + drift * dt
        log("transition to the hang", pos_b)
    while st["pos"][2] > perch[2] + 0.25:
        st["pos"] = st["pos"] + np.r_[0.0, 0.0, -1.0 * dt]; st["V"] = 0.0
        st["P"] = P_hov; st["E"] += P_hov * dt / 3600; t += dt
        pos_b = pos_b + drift * dt
        log("prop-hang landing on the ridge", pos_b)
    events.append(f"t = {t:.0f} s: on the ridge, {st['E']:.1f} Wh used ({st['E'] / (battery_wh * 0.8) * 100:.0f} % of the usable pack)")
    return Episode(np.array(rec["t"]), np.array(rec["pos"]), np.array(rec["V"]), np.array(rec["psi"]), np.array(rec["bank"]),
                   np.array(rec["fold"]), np.array(rec["P"]), np.array(rec["E"]), rec["phase"], rec["birds"], deer,
                   events, ac.name, ac.key)


def profile_figure(eps, colors=None):
    """Height, airspeed, fold angle and energy against time for one or more episodes (``colors``: by ``Episode.key``)."""
    import matplotlib.pyplot as plt
    eps = eps if isinstance(eps, (list, tuple)) else [eps]
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    for ep in eps:
        kw = {"label": ep.aircraft, "color": (colors or {}).get(ep.key)}
        ax[0, 0].plot(ep.t, ep.pos[:, 2], **kw)
        ax[0, 1].plot(ep.t, ep.V, **kw)
        ax[1, 0].plot(ep.t, ep.fold, **kw)
        ax[1, 1].plot(ep.t, ep.energy_wh, **kw)
    for a, (yl, ti) in zip(ax.flat, (("m AGL", "height"), ("m/s", "airspeed"), ("deg", "fold"), ("Wh", "energy used"))):
        a.set(xlabel="time [s]", ylabel=yl, title=ti); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _box(ax, x, y, L, Wd, eave, ridge, color="#b0745a"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    x0, x1, y0, y1 = x - L / 2, x + L / 2, y - Wd / 2, y + Wd / 2
    walls = [[(x0, y0, 0), (x1, y0, 0), (x1, y0, eave), (x0, y0, eave)], [(x0, y1, 0), (x1, y1, 0), (x1, y1, eave), (x0, y1, eave)],
             [(x0, y0, 0), (x0, y1, 0), (x0, y1, eave), (x0, y0, eave)], [(x1, y0, 0), (x1, y1, 0), (x1, y1, eave), (x1, y0, eave)]]
    roof = [[(x0, y0, eave), (x1, y0, eave), (x1, y, ridge), (x0, y, ridge)], [(x0, y1, eave), (x1, y1, eave), (x1, y, ridge), (x0, y, ridge)],
            [(x0, y0, eave), (x0, y1, eave), (x0, y, ridge)], [(x1, y0, eave), (x1, y1, eave), (x1, y, ridge)]]
    ax.add_collection3d(Poly3DCollection(walls, facecolor="#e8dcc8", edgecolor="#7a6a55", lw=0.4))
    ax.add_collection3d(Poly3DCollection(roof, facecolor=color, edgecolor="#5a3a2a", lw=0.4))


def render_movie(ep: Episode, farm: Farm, p, path, *, fps=20, seconds=24.0, size=(1280, 720), glyph_scale=8.0,
                 title="PEREGRINE", progress=False):
    """The sortie as an MP4 (MERLIN's pipeline: matplotlib 3D on an off-screen Agg canvas → RGB frames → the Vegeta watermark → OpenCV's
    mp4v writer): the farm (crop, tree lines, farmhouse and barn), the drone's planform drawn ``glyph_scale`` times
    life size with its current fold, its trail, the flock as dots (red when afraid of the drone), a time-lapse
    clock. Returns (path, frames)."""
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from vegeta.aeromant._watermark import watermark
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(2, int(fps * seconds))
    idx = np.linspace(0, len(ep.t) - 1, n_frames).astype(int)
    w, h = size
    fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#dfeaf5")       # off-screen: the notebook's backend stays
    FigureCanvasAgg(fig)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trees = farm.trees()
    tops = {}
    fx0, fy0, fx1, fy1 = farm.field_xy
    for k, i in enumerate(idx):
        fig.clf()
        ax = fig.add_axes([-0.22, -0.30, 1.44, 1.55], projection="3d", computed_zorder=False)
        ax.set_facecolor("#dfeaf5")
        c = ep.pos[i]
        r = 75.0
        xa, xb, ya, yb = c[0] - r, c[0] + r, c[1] - r, c[1] + r
        ax.add_collection3d(Poly3DCollection([[(xa, ya, 0), (xb, ya, 0), (xb, yb, 0), (xa, yb, 0)]], facecolor="#a9c48a",
                                             edgecolor="none", zorder=0))
        cx0, cy0, cx1, cy1 = max(fx0, xa), max(fy0, ya), min(fx1, xb), min(fy1, yb)
        if cx1 > cx0 and cy1 > cy0:
            ax.add_collection3d(Poly3DCollection([[(cx0, cy0, 0.05), (cx1, cy0, 0.05), (cx1, cy1, 0.05), (cx0, cy1, 0.05)]],
                                                 facecolor="#e2c35a", edgecolor="#9a7d22", lw=0.6, zorder=1))
        sx, sy = farm.safe_xy
        th = np.linspace(0, 2 * np.pi, 60)
        ring = np.c_[sx + farm.safe_r * np.cos(th), sy + farm.safe_r * np.sin(th)]
        inside = (ring[:, 0] > xa) & (ring[:, 0] < xb) & (ring[:, 1] > ya) & (ring[:, 1] < yb)
        if inside.any():
            ax.plot(np.where(inside, ring[:, 0], np.nan), np.where(inside, ring[:, 1], np.nan), 0.1, color="#2e6b2e", lw=1.5, zorder=1)
        for bld in farm.buildings:
            if xa - 15 < bld[0] < xb + 15 and ya - 15 < bld[1] < yb + 15:
                _box(ax, *bld)
        tv = trees[(trees[:, 0] > xa) & (trees[:, 0] < xb) & (trees[:, 1] > ya) & (trees[:, 1] < yb)]
        for tx, ty, th_, cr in tv:
            ax.plot([tx, tx], [ty, ty], [0, th_ * 0.55], color="#6b4a2b", lw=1.5, zorder=3)
        if len(tv):
            ax.scatter(tv[:, 0], tv[:, 1], tv[:, 2] * 0.75, s=(tv[:, 3] * 9) ** 2 / 4, c="#2f6b33", alpha=0.95, depthshade=False, zorder=3)
        if xa < ep.deer[0] < xb and ya < ep.deer[1] < yb:
            ax.scatter([ep.deer[0]], [ep.deer[1]], [0.8], s=60, c="#8b5a2b", marker="^", depthshade=False, zorder=4)
        bds = ep.birds[i]
        if bds is not None:
            ax.scatter(bds[:, 0], bds[:, 1], bds[:, 2], s=8, c="k", depthshade=False, zorder=5)
        j0 = max(0, i - int(30 / max(ep.t[1] - ep.t[0], 1e-3)))
        ax.plot(ep.pos[j0:i + 1, 0], ep.pos[j0:i + 1, 1], ep.pos[j0:i + 1, 2], color="#c62828", lw=1.4, zorder=6)
        ax.plot([c[0], c[0]], [c[1], c[1]], [0, c[2]], color="#c62828", lw=0.6, ls=":", zorder=6)
        f = float(ep.fold[i])
        key = f"{f:.0f}"
        if key not in tops:
            tops[key] = pg.outline(p, fold_deg=f if pg.resolve(p)["wing_hinge"] == "root" else 0.0)["top"]
        cps, sps = math.cos(ep.psi[i]), math.sin(ep.psi[i])
        cb = math.cos(ep.bank[i])
        polys = []
        for poly in tops[key]:
            # aircraft frame (x aft, y right) → world: forward = -x along the heading psi; the bank tilts the span
            xa_, ya_ = -poly[:, 0] * glyph_scale, poly[:, 1] * glyph_scale
            za_ = -ya_ * math.sin(ep.bank[i])
            ya_ = ya_ * cb
            X = c[0] + xa_ * cps + ya_ * sps
            Y = c[1] + xa_ * sps - ya_ * cps
            polys.append(list(zip(X, Y, c[2] + za_)))
        ax.add_collection3d(Poly3DCollection(polys, facecolor="#37474f", edgecolor="k", lw=0.5, alpha=0.95, zorder=7))
        ax.set_xlim(xa, xb); ax.set_ylim(ya, yb); ax.set_zlim(0, 2 * r * 0.4)
        ax.view_init(elev=22, azim=math.degrees(ep.psi[i]) - 160)
        ax.set_box_aspect((1, 1, 0.4))
        ax.set_axis_off()
        fig.text(0.02, 0.95, f"{title} — {ep.aircraft}", fontsize=14, weight="bold")
        fig.text(0.02, 0.91, f"t = {ep.t[i]:5.0f} s   {ep.phase[i]}", fontsize=12)
        fig.text(0.02, 0.87, f"airspeed {ep.V[i]:4.1f} m/s   height {ep.pos[i, 2]:4.0f} m   fold {f:3.0f}°   "
                             f"power {ep.power[i]:4.0f} W   energy {ep.energy_wh[i]:4.1f} Wh", fontsize=11)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(watermark(frame))
        if progress and k % 50 == 0:
            print(f"frame {k}/{n_frames}")
    writer.release()
    return path, n_frames


# ================================================================================================= the verdict
def decision_table(metrics, better, weights):
    """Normalise each metric over the variants (the best = 1, the worst = 0; ``better`` says whether high or low is
    good), weight them and sum: one score per variant."""
    import pandas as pd
    m = metrics.astype(float)
    norm = pd.DataFrame(index=m.index, columns=m.columns, dtype=float)
    for k in m.index:
        v = m.loc[k]
        lo, hi = v.min(), v.max()
        s = (v - lo) / (hi - lo) if hi > lo else v * 0 + 1.0
        norm.loc[k] = s if better[k] == "high" else 1 - s
    w = pd.Series(weights, dtype=float).reindex(m.index).fillna(0.0)
    w = w / w.sum()
    score = (norm.mul(w, axis=0)).sum()
    return norm, score


def weight_sensitivity(metrics, better, *, n=20000, seed=0, groups=None):
    """Random weightings (uniform on the simplex): how often each variant wins. ``groups`` ({name: [metrics]}) draws
    the weights per group and splits them evenly inside it."""
    rng = np.random.default_rng(seed)
    norm, _ = decision_table(metrics, better, {k: 1.0 for k in metrics.index})
    keys = list(groups) if groups else list(metrics.index)
    wins = {c: 0 for c in metrics.columns}
    for _ in range(n):
        wg = rng.dirichlet(np.ones(len(keys)))
        if groups:
            w = {m_: wg[i] / len(groups[g]) for i, g in enumerate(keys) for m_ in groups[g]}
        else:
            w = dict(zip(keys, wg))
        ws = np.array([w.get(k, 0.0) for k in norm.index])
        sc = (norm.values * ws[:, None]).sum(0)
        wins[metrics.columns[int(np.argmax(sc))]] += 1
    return {k: v / n for k, v in wins.items()}


__all__ = ["vlm", "aero", "cl_max", "fold_polar", "BOM_ITEMS", "SERVO_PRICE_EUR", "ENGINEERING_HOURS", "mass_table", "cg",
           "bom", "engineering_cost", "stability", "drive", "Aircraft", "make_aircraft", "turn", "envelope",
           "envelope_table", "BIRDS", "bird_table", "stoop", "terminal_speed", "gap_passage", "prop_hang", "hand_throw",
           "min_throw_speed", "roof_drop", "perched_wind", "hinge_loads", "Farm", "Flock", "herd", "Episode", "mission",
           "profile_figure", "render_movie", "decision_table", "weight_sensitivity"]
