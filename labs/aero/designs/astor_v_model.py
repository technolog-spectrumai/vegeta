"""ASTOR-V's flight model, first step: the longitudinal aerodynamics over the wing's sweep, from the vortex lattice
(``ogma.aero.vlm``) on the rotated panels themselves — lift slope, neutral point, the balance about a fixed CG, the
tail fan's power to trim, and the roll that sweeping one panel more than the other gives.

The lattice sees each wing panel as it lies at its sweep (cut into streamwise strips of equal width, the part over
the other side of the centre plane left out) and the two tail panels at the fan's state; the tail fan's incidence is
the elevator. Inviscid, attached flow: slopes and balance, not the stall and not the drag. References fixed for every
sweep: S_REF (the spread wing's exposed area), C_REF, so coefficients compare directly.
"""
from __future__ import annotations

import math

import numpy as np

import astor_v as V

S_REF, C_REF = 0.198, 0.200
N_STRIP = 12


def _outline(root_c, tip_c, span, base_sweep, pivot_frac, rot_deg, px, py, side):
    """A panel's four corners [mm] (LE root, LE tip, TE tip, TE root) after its rotation about the pivot."""
    dx = span * math.tan(math.radians(base_sweep))
    P = np.array([[0.0, 0.0], [dx, span], [dx + tip_c, span], [root_c, 0.0]]) - np.array([pivot_frac * root_c, 0.0])
    a = math.radians(rot_deg)
    R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])          # +y toward +x: aft sweep
    P = P @ R.T
    P[:, 1] *= side
    return P + np.array([px, side * py])


def _strips(P, side, z, n=N_STRIP):
    """Streamwise strips of the outline on its own side of the centre plane: quads [m] for the lattice."""
    ys = np.abs(P[:, 1])
    y0, y1 = max(0.0, float(ys.min())) if (P[:, 1] * side).min() > 0 else 0.0, float(ys.max())
    edges = [(P[i], P[(i + 1) % 4]) for i in range(4)]

    def cut(y):
        xs = []
        for a, b in edges:
            ya, yb = a[1] * side, b[1] * side
            if (ya - y) * (yb - y) <= 0 and abs(yb - ya) > 1e-9:
                xs.append(a[0] + (b[0] - a[0]) * (y - ya) / (yb - ya))
        return (min(xs), max(xs)) if len(xs) >= 2 else None
    yy = np.linspace(y0 + 1e-3, y1 - 1e-3, n + 1)
    quads = []
    for k in range(n):
        c0, c1 = cut(yy[k]), cut(yy[k + 1])
        if c0 is None or c1 is None or c0[1] - c0[0] < 2.0 or c1[1] - c1[0] < 2.0:
            continue
        q = np.array([[c0[0], side * yy[k], z], [c1[0], side * yy[k + 1], z], [c1[1], side * yy[k + 1], z], [c0[1], side * yy[k], z]]) / 1000.0
        quads.append(q if side > 0 else q[[1, 0, 3, 2]])
    return quads


def lattice(sweep_r, sweep_l=None, tail_fan=1.0, alpha=0.0, tail_inc=0.0, p=None) -> dict:
    from ogma.aero import vlm
    p = p or V.resolve()
    sweep_l = sweep_r if sweep_l is None else sweep_l
    quads, inc = [], []
    for side, sw in ((1, sweep_r), (-1, sweep_l)):
        P = _outline(p["panel_root_chord"], p["panel_tip_chord"], p["panel_span"], V.SPREAD_DEG, p["panel_pivot_chord"],
                     sw - V.SPREAD_DEG, V.pivot_x(p, 0.5 * (sweep_r + sweep_l)), p["pivot_y"], side)
        q = _strips(P, side, p["wing_plane_z"])
        quads += q
        inc += [0.0] * len(q)
    ts = 85.0 - 50.0 * tail_fan
    for side in (1, -1):
        P = _outline(p["tail_panel_root"], p["tail_panel_tip"], p["tail_panel_span"], 35.0, 0.25, ts - 35.0, p["tail_pivot_x"], p["tail_pivot_y"], side)
        q = _strips(P, side, 6.0, n=5)
        quads += q
        inc += [tail_inc] * len(q)
    r = vlm(quads, alpha_deg=alpha, incidence_deg=inc, S_ref=S_REF, c_ref=C_REF, x_ref=0.0, n_span=2)
    return r


def longitudinal(sweep, tail_fan=1.0, x_cg_mm=None) -> dict:
    r0, r4 = lattice(sweep, tail_fan=tail_fan, alpha=0.0), lattice(sweep, tail_fan=tail_fan, alpha=4.0)
    re = lattice(sweep, tail_fan=tail_fan, alpha=0.0, tail_inc=1.0)
    CLa = (r4["CL"] - r0["CL"]) / math.radians(4.0)
    x_np = -(r4["Cm"] - r0["Cm"]) / (r4["CL"] - r0["CL"]) * C_REF * 1000.0
    out = {"sweep_deg": sweep, "tail_fan": tail_fan, "CL_alpha_per_rad": round(CLa, 2), "neutral_point_mm": round(x_np, 0),
           "CL_per_deg_tail": round(re["CL"] - r0["CL"], 4), "Cm_nose_per_deg_tail": round(re["Cm"] - r0["Cm"], 4)}
    if x_cg_mm is not None:
        xc = x_cg_mm / 1000.0 / C_REF
        out["static_margin"] = round((x_np - x_cg_mm) / (C_REF * 1000.0), 3)
        out["Cm_cg_per_deg_tail"] = round((re["Cm"] + re["CL"] * xc) - (r0["Cm"] + r0["CL"] * xc), 4)
    return out


def trim(sweep, V_m_s, mass_kg, x_cg_mm, tail_fan=1.0) -> dict:
    """Level flight: the angle of attack and the tail fan's incidence that give lift = weight and no moment about the CG."""
    CL_req = mass_kg * 9.80665 / (0.5 * 1.225 * V_m_s ** 2 * S_REF)
    xc = x_cg_mm / 1000.0 / C_REF
    f = lambda r: (r["CL"], r["Cm"] + r["CL"] * xc)
    c0 = f(lattice(sweep, tail_fan=tail_fan))
    ca = f(lattice(sweep, tail_fan=tail_fan, alpha=1.0))
    ce = f(lattice(sweep, tail_fan=tail_fan, tail_inc=1.0))
    A = np.array([[ca[0] - c0[0], ce[0] - c0[0]], [ca[1] - c0[1], ce[1] - c0[1]]])
    al, de = np.linalg.solve(A, np.array([CL_req - c0[0], -c0[1]]))
    return {"sweep_deg": sweep, "V_m_s": V_m_s, "CL": round(CL_req, 3), "alpha_deg": round(float(al), 1), "tail_incidence_deg": round(float(de), 1),
            "tail_lift_share": round(float(de * (ce[0] - c0[0]) / CL_req), 2)}


def roll_from_sweep(sweep, d_deg=8.0, alpha=4.0) -> dict:
    """The right panel swept ``d_deg`` more than the left at an angle of attack: the rolling moment the lattice gives."""
    r = lattice(sweep + d_deg / 2, sweep - d_deg / 2, alpha=alpha)
    k = next((x for x in ("Cl", "Cl_roll", "Croll", "CR") if x in r), None)
    return {"keys": sorted(r.keys()), "roll": None if k is None else float(r[k])}
