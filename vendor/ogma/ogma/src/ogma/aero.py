"""Aerodynamics of lifting bodies: ``og.aero``.

- **Skin friction:** ``cf_turbulent`` (Prandtl–Schlichting, with Raymer's Mach factor) — the formula about ten Vegeta
  drag build-ups repeated — and ``cf_mixed`` (with the 1700/Re transition term, Kestrel's).
- **Polars:** ``Airframe`` (the parabolic polar of a whole aircraft: drag at a load factor, stall speed; Merlin's);
  ``Polar`` (one surface over the whole range of α, attached → flat plate; Kestrel's) and ``viterna_cd90``.
- **Stability derivatives:** ``stability_coefficients`` — the linear derivatives with a post-stall blend and the
  optional crow (spoilerons) increments: Nisus's and Nisus+'s ``Aero.coefficients``, one function.
- **Forces:** ``air_angles`` (V, α, β from the air velocity in the body frame) and ``aero_wrench`` (coefficients →
  force and moment in the body frame).
- **Vortex lattice:** ``vlm`` on flat quadrilaterals (Peregrine's, freed from its CAD import).

Frames: Chiron's body frame is x forward, y left, z up; the aero body axes are x forward, y right, z down. Angles in
radians unless a name says ``_deg``. Characterisation: ``tests/test_aero.py`` reproduces the Vegeta code each piece
came from (``tests/golden/aero.json``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import G_VEGETA

RHO = 1.225


# ------------------------------------------------------------------------------------------------------ skin friction
def cf_turbulent(re: float, mach: float = 0.0, re_min: float = 0.0) -> float:
    """Turbulent flat-plate skin friction ``0.455 / (log10 Re)^2.58``, divided by Raymer's ``(1 + 0.144 M²)^0.65``
    (exactly 1 at M = 0); ``re_min`` clamps Re from below (the build-ups use 1e4)."""
    re = max(re, re_min)
    return 0.455 / math.log10(re) ** 2.58 / (1 + 0.144 * mach ** 2) ** 0.65


def cf_mixed(Re: float) -> float:
    """Mixed laminar–turbulent flat-plate friction (Prandtl–Schlichting with the 1700/Re transition term)."""
    Re = max(Re, 1e4)
    return max(0.455 / math.log10(Re) ** 2.58 - 1700.0 / Re, 1.328 / math.sqrt(Re))


# ------------------------------------------------------------------------------------------------------------ polars
@dataclass
class Airframe:
    """A whole aircraft's parabolic polar CD = cd0 + k CL², k = 1/(π AR e): drag at a load factor, stall speed.
    ``g`` defaults to the 9.81 Vegeta's machines use (until the constants are unified)."""
    name: str
    mass_kg: float
    wing_area_m2: float
    aspect_ratio: float
    cd0: float
    oswald: float = 0.8
    cl_max: float = 1.2
    g: float = G_VEGETA

    @property
    def k(self):
        return 1.0 / (math.pi * self.aspect_ratio * self.oswald)

    def drag(self, V, rho=RHO, n=1.0):
        """Drag [N] at airspeed V in flight at load factor n (lift = n W)."""
        V = np.maximum(np.asarray(V, float), 1e-3)
        q = 0.5 * rho * V ** 2
        cl = n * self.mass_kg * self.g / (q * self.wing_area_m2)
        return q * self.wing_area_m2 * (self.cd0 + self.k * cl ** 2)

    def stall_speed(self, rho=RHO, n=1.0):
        return math.sqrt(2 * n * self.mass_kg * self.g / (rho * self.wing_area_m2 * self.cl_max))


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


# ------------------------------------------------------------------------------------------- stability derivatives
def stability_coefficients(c: dict, alpha, beta, p_, q_, r_, delta, V, *, post_stall: bool = True, crow: float = 0.0):
    """The six coefficients (CL, CD, CY, Cl, Cm, Cn) and whether the wing is stalled, from linear derivatives ``c``
    (keys ``S``, ``b``, ``c``, ``CL0``, ``CLa``, ``CLq``, ``CLde``, ``CD0``, ``k``, ``CD_de``, ``Cm0``, ``Cma``,
    ``Cmq``, ``Cmde``, ``alpha_stall_deg``, the lateral ``Cyb``… ``Cndr``; with ``crow`` > 0 also ``dCL_crow``,
    ``k_crow``, ``dCD_crow``, ``dCm_crow``, ``dCLmax_crow``). Rates in the aero axes [rad/s], ``delta`` = (aileron,
    elevator, rudder) [rad]. ``post_stall`` blends to a flat plate beyond the stall angle (assumption: CN = 2 sin α,
    a nose-down Cm of −0.15). Nisus's model; Nisus+'s with crow (identical at crow = 0)."""
    b, cbar = c["b"], c["c"]
    da, de, dr = delta
    ph, qh, rh = p_ * b / (2 * V), q_ * cbar / (2 * V), r_ * b / (2 * V)
    if crow:
        dCL, dCD, dCm = crow * c["dCL_crow"], crow * c["dCD_crow"], crow * c["dCm_crow"]
        k = c["k"] + crow * (c["k_crow"] - c["k"])
        a_s = math.radians(c["alpha_stall_deg"]) + crow * c["dCLmax_crow"] / c["CLa"]
    else:
        dCL = dCD = dCm = 0.0
        k = c["k"]
        a_s = math.radians(c["alpha_stall_deg"])
    CL_lin = c["CL0"] + c["CLa"] * alpha + c["CLq"] * qh + c["CLde"] * de + dCL
    CD_lin = c["CD0"] + dCD + k * (c["CL0"] + c["CLa"] * alpha + dCL) ** 2 + c["CD_de"] * (de * de + dr * dr + 0.5 * da * da)
    Cm_lin = c["Cm0"] + c["Cma"] * alpha + c["Cmq"] * qh + c["Cmde"] * de + dCm
    stalled = False
    if post_stall:
        s = 1 / (1 + math.exp(-(abs(alpha) - a_s) / math.radians(1.5)))
        stalled = s > 0.5
        CN = 2.0 * math.sin(alpha)
        CL_fp = CN * math.cos(alpha) * 0.9 + c["CLde"] * de * 0.5
        CD_fp = c["CD0"] + dCD + abs(CN * math.sin(alpha)) + 0.1
        Cm_fp = -0.15 * math.copysign(1.0, alpha) + c["Cmde"] * de * 0.5 + c["Cmq"] * qh
        CL, CD, Cm = (1 - s) * CL_lin + s * CL_fp, (1 - s) * CD_lin + s * CD_fp, (1 - s) * Cm_lin + s * Cm_fp
    else:
        CL, CD, Cm = CL_lin, CD_lin, Cm_lin
    CLn = max(min(CL, 1.5), -1.0)                                  # the lift-dependent lateral terms follow the actual lift
    CY = c["Cyb"] * beta + c["Cyp"] * ph + c["Cyr"] * rh + c["Cydr"] * dr
    Cl = c["Clb"] * beta + c["Clp"] * ph + (CLn / 4 + (c["Clr"] - 0.1)) * rh + c["Clda"] * da + c["Cldr"] * dr
    Cn = c["Cnb"] * beta + (-CLn / 8) * ph + c["Cnr"] * rh + (-0.2 * CLn * c["Clda"]) * da + c["Cndr"] * dr
    return CL, CD, CY, Cl, Cm, Cn, stalled


# ------------------------------------------------------------------------------------------------------------ forces
def air_angles(v_air_b) -> tuple:
    """From the air velocity in the body frame (x fwd, y left, z up): V, α, β and the aero-axis components
    (u, v right, w down). α and β are 0 below 1 m/s."""
    u, v_right, w_down = float(v_air_b[0]), -float(v_air_b[1]), -float(v_air_b[2])
    V = float(np.linalg.norm(v_air_b))
    if V > 1.0:
        alpha = math.atan2(w_down, u)
        beta = math.asin(max(-1.0, min(1.0, v_right / V)))
    else:
        alpha = beta = 0.0
    return V, alpha, beta, (u, v_right, w_down)


def aero_wrench(coeffs, V: float, alpha: float, uvw, rho: float, S: float, b: float, cbar: float):
    """The aerodynamic force and moment [N, N m] in the body frame (x fwd, y left, z up) from (CL, CD, CY, Cl, Cm, Cn):
    lift perpendicular to the wind in the symmetry plane, drag along it, side force along +y (aero), the moments on
    b, c̄, b."""
    CL, CD, CY, Cl, Cm, Cn = coeffs[:6]
    u, v_right, w_down = uvw
    qS = 0.5 * rho * V * V * S
    lift_dir = np.array([math.sin(alpha), 0.0, -math.cos(alpha)])
    drag_dir = -np.array([u, v_right, w_down]) / V
    F_a = qS * (CL * lift_dir + CD * drag_dir + CY * np.array([0.0, 1.0, 0.0]))
    M_a = qS * np.array([Cl * b, Cm * cbar, Cn * b])
    return np.array([F_a[0], -F_a[1], -F_a[2]]), np.array([M_a[0], -M_a[1], -M_a[2]])


# ------------------------------------------------------------------------------------------------------ vortex lattice
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


__all__ = ["cf_turbulent", "cf_mixed", "Airframe", "Polar", "viterna_cd90", "stability_coefficients", "air_angles",
           "aero_wrench", "vlm", "RHO"]
