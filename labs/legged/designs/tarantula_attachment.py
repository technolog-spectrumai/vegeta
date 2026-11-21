"""Does TARANTULA need adhesives? Its statics on a surface at any inclination, foot by foot, for four ways to hold on
(docs/spiders.md §10):

* ``friction`` — claws and a rubber sole, nothing that pulls: a foot may push and rub, never pull (μ per surface;
  checked for the whole robot — a foot that slips hands its shear to the others — and no foot may need to pull);
* ``pad`` — a dry fibrillar adhesive pad (ARACHNE's allowables, ``arachne_specs.ADHESIVE``; the peel moment of the
  shear acting above the pad, ``arachne_statics.foot_capacity``'s model, on TARANTULA's 40 × 30 mm pad);
* ``spine`` — a microspine array (``arachne_specs.SPINES``: a spine's hold is a property of its tip, so a heavier robot
  needs more of them, not bigger ones; they hold only along their direction, up the slope);
* ``magnet`` — a switchable permanent-magnet foot (steel only; holds normal, shear by friction under the clamp);
* ``hybrid`` — the pad with a microspine comb along its toe and the claws (ARACHNE's hybrid, scaled): on each
  surface the better of the two holds the foot (conservative: no sum).

The load sharing is ARACHNE's (``arachne_statics._share_linear``: compliant feet, the body's small rigid displacement
balancing the weight at the centre of mass off the surface); the gait leaves 6 feet attached (the wave) or 4 (the
alternating tetrapod), the worst group lifted. Margin = 1 / the worst foot's utilisation; the rule is margin ≥ 2.
"""
from __future__ import annotations

import math

import arachne_specs as asp
import arachne_statics as st
import numpy as np
import spider_specs as ss

__all__ = ["GAITS", "HYBRID_SPINES", "MASS_KG", "MODULES", "POSTURES", "SURFACES", "feet", "forces", "margin",
           "max_inclination", "modules_mass_g", "monte_carlo", "pad_needed", "spines_needed", "study"]

MASS_KG = 1.5                                                             # with the 0.3 kg payload (§5)
#: the centre of mass [mm, body frame, x forward, z off the surface]: x — the battery and the payload in the abdomen
#: pull it back (assumed); z — the body centre's height: standing (the CAD) or crouched for climbing (the belly 5 mm
#: off the surface)
COM_X_MM = -35.0
POSTURES = {"standing": ss.body_z(ss.TARANTULA), "crouched": ss.TARANTULA["prosoma"][2] / 2 + 5.0}
#: the gaits: groups of feet lifted together (the worst group is checked)
GAITS = {"wave": [("L1", "R3"), ("R2", "L4"), ("R1", "L3"), ("L2", "R4")],
         "tetrapod": [("L1", "R2", "L3", "R4"), ("R1", "L2", "R3", "L4")]}
#: surfaces: (dry-adhesive allowables shear, normal [kPa] — ARACHNE's, conservative; per-spine hold [N]; sole
#: friction μ (rubber); steel?) — the first three from ``arachne_specs``; brick and concrete have no adhesive contact;
#: bare steel takes a pad like glass, painted steel like a painted wall (assumed)
SURFACES = {
    "glass": (10.0, 3.0, 0.0, 0.6, False),
    "acrylic": (8.0, 2.5, 0.0, 0.6, False),
    "varnished wood": (5.0, 1.5, 0.01, 0.6, False),
    "painted wall": (2.0, 0.6, 0.03, 0.7, False),
    "sanded wood": (1.5, 0.4, 0.08, 0.7, False),
    "bark": (0.0, 0.0, 0.15, 0.7, False),
    "brick": (0.0, 0.0, 0.2, 0.8, False),
    "concrete": (0.0, 0.0, 0.2, 0.8, False),
    "bare steel": (8.0, 2.5, 0.0, 0.5, True),
    "painted steel": (2.0, 0.6, 0.03, 0.5, True),
}
for _s in ("glass", "acrylic", "varnished wood", "painted wall", "sanded wood", "bark"):
    assert SURFACES[_s][0] == asp.ADHESIVE["shear_kpa"][_s] and SURFACES[_s][1] == asp.ADHESIVE["normal_kpa"][_s]
for _s in ("bark", "brick", "painted wall", "sanded wood", "varnished wood"):
    assert SURFACES[_s][2] == asp.SPINES["per_spine_n"][_s]

#: the foot modules: mass per foot [g] (assumed class estimates) and their sizes
MODULES = {
    "friction": {"g": 6.0, "note": "two claws and a 40 × 30 mm rubber sole"},
    "pad": {"g": 14.0, "pad_mm": (40.0, 30.0), "pivot_mm": 1.5,
            "note": "40 × 30 mm fibrillar pad on a low-pivot ankle (ARACHNE's, scaled)"},
    "spine": {"g": 20.0, "spines": 160, "note": "a 160-microspine comb (5 rows × 32 at 1 mm pitch, each spine on its own "
                                               "flexure) + two claws"},
    "magnet": {"g": 30.0, "pull_n": 60.0, "pull_painted_n": 30.0, "mu": 0.5,
               "note": "switchable permanent magnet (electropermanent class): 60 N on bare steel, ~30 N through paint "
                       "(assumed: coating gaps cut the pull fast)"},
    "hybrid": {"g": 28.0, "note": "the 40 × 30 mm pad + the 160-spine comb along its toe + two claws"},
}
HYBRID_SPINES = 160
K_T, K_N = 4000.0, 8000.0          # N/m per foot: the ratio is ARACHNE's; the scale only sets the displacement


def feet(posture="standing"):
    """{foot: contact point [m]}, body frame (origin: the body centre, z off the surface): the CAD's stance."""
    s = ss.TARANTULA
    c = ss.leg_chain(s)
    z = -POSTURES[posture]
    out = {}
    for leg in ss_legs():
        (hx, hy), splay, sy = ss.hip_points(s, leg)
        a = math.radians(splay)
        u = (math.sin(a), sy * math.cos(a))
        r = c["reach"] - s["pad"][0] / 2                                  # the pad's centre
        out[leg] = np.array([hx + r * u[0], hy + r * u[1], z]) * 1e-3
    return out


def ss_legs():
    return [f"{sd}{i}" for sd in "LR" for i in (1, 2, 3, 4)]


def forces(attached, posture="standing", inclination_deg=90.0, heading_deg=0.0, mass_kg=MASS_KG, k=None):
    """The attached feet's forces on the robot [N] {foot: f} (k: {foot: stiffness factor}, for the Monte Carlo)."""
    pts = feet(posture)
    pts = {f: pts[f] for f in attached}
    W = st.gravity_body(mass_kg, inclination_deg, heading_deg)
    kt = {f: K_T * (k or {}).get(f, 1.0) for f in attached}
    kn = {f: K_N * (k or {}).get(f, 1.0) for f in attached}
    return st._share_linear(pts, W, np.array([COM_X_MM * 1e-3, 0.0, 0.0]), kt, kn)


def utilisation(force, surface, module):
    """The foot's utilisation (1 = at its allowable) carrying ``force`` (the surface's force on the robot)."""
    tau, sig, per_spine, mu, steel = SURFACES[surface]
    S = float(np.hypot(force[0], force[1]))
    N = float(force[2])                                                   # > 0 pushes the robot off: compression
    T = max(0.0, -N)                                                      # pulled towards the surface: tension
    if module == "friction":                                              # per foot: only whether it must pull
        return math.inf if T > 1e-9 else 0.0
    if module == "pad":
        m = MODULES["pad"]
        a, b = m["pad_mm"]
        A, L = a * b * 1e-6, max(a, b) * 1e-3
        if tau == 0 or sig == 0:
            return math.inf
        edge = T / A + 6 * S * m["pivot_mm"] * 1e-3 / (A * L)
        return math.hypot(S / (tau * 1e3 * A), edge / (sig * 1e3))
    if module == "hybrid":
        m = MODULES["spine"]
        n0, m["spines"] = m["spines"], HYBRID_SPINES
        try:
            return min(utilisation(force, surface, "pad"), utilisation(force, surface, "spine"))
        finally:
            m["spines"] = n0
    if module == "spine":
        n = MODULES["spine"]["spines"] * asp.SPINES["engaged_fraction"]
        c = max(0.0, float(force[0]) / S) if S > 1e-12 else 1.0             # the spines face up the slope (+x)
        cap = per_spine * n * c
        cap_t = cap * asp.SPINES["pulloff_over_shear"]
        if (cap == 0 and S > 0) or (cap_t == 0 and T > 0):
            return math.inf
        # pushing on the surface helps: rubbing under the normal load carries shear too
        s_left = max(0.0, S - mu * max(N, 0.0))
        return math.hypot(s_left / cap if cap else 0.0, T / cap_t if cap_t else 0.0)
    if module == "magnet":
        if not steel:
            return math.inf
        m = MODULES["magnet"]
        P = m["pull_n"] if surface == "bare steel" else m["pull_painted_n"]
        clamp = P - T                                                     # what is left pressing the foot down
        if clamp <= 0:
            return math.inf
        return max(T / P, S / (m["mu"] * (clamp + max(N, 0.0))))
    raise ValueError(module)


def margin(surface, module, gait="wave", posture="standing", inclination_deg=90.0, heading_deg=0.0,
           mass_kg=MASS_KG, k=None):
    """The worst margin over the gait's stances (each group lifted in turn)."""
    worst = math.inf
    for group in GAITS[gait]:
        att = [f for f in ss_legs() if f not in group]
        F = forces(att, posture, inclination_deg, heading_deg, mass_kg, k)
        u = max(utilisation(F[f], surface, module) for f in att)
        if module == "friction" and u == 0.0:                             # nobody pulls: the robot slides as one
            S = math.hypot(sum(F[f][0] for f in att), sum(F[f][1] for f in att))
            N = sum(F[f][2] for f in att)
            u = S / (SURFACES[surface][3] * N) if N > 0 else math.inf
        worst = min(worst, 1.0 / u if u > 0 else math.inf)
    return worst


def max_inclination(surface, module, gait="wave", posture="standing", need=2.0, step=5.0, **kw):
    """The steepest inclination [deg] (0 floor, 90 wall, 180 ceiling) with margin ≥ ``need``, climbing head up."""
    best = None
    a = 0.0
    while a <= 180.0 + 1e-9:
        if margin(surface, module, gait, posture, a, **kw) >= need:
            best = a
        else:
            break
        a += step
    return best


def pad_needed(surface="glass", gait="wave", posture="crouched", need=2.0):
    """The pad area [mm², 4:3] that gives margin ``need`` on a vertical ``surface`` (the pivot scaled with it)."""
    m = MODULES["pad"]
    a0, b0 = m["pad_mm"]
    lo, hi = 0.1, 20.0
    for _ in range(60):
        s = (lo + hi) / 2
        m["pad_mm"], piv = (a0 * s, b0 * s), m["pivot_mm"]
        m["pivot_mm"] = piv * s
        ok = margin(surface, "pad", gait, posture) >= need
        m["pad_mm"], m["pivot_mm"] = (a0, b0), piv
        lo, hi = (lo, s) if ok else (s, hi)
    return a0 * hi * b0 * hi


def spines_needed(surface="brick", gait="wave", posture="crouched", need=2.0):
    """The spines per foot that give margin ``need`` on a vertical ``surface``."""
    m = MODULES["spine"]
    n0 = m["spines"]
    try:
        for n in range(4, 2001, 4):
            m["spines"] = n
            if margin(surface, "spine", gait, posture) >= need:
                return n
        return None
    finally:
        m["spines"] = n0


def monte_carlo(surface, module, gait="wave", posture="crouched", inclination_deg=90.0, n=200, seed=0,
                stiffness_spread=0.3, com_mm=15.0, mass_kg=MASS_KG):
    """Margins with each foot's stiffness scattered ±``stiffness_spread`` (uniform) and the centre of mass moved
    ±``com_mm`` along the body: {"p5", "p50", "frac_ok"} (frac_ok: margin ≥ 2)."""
    global COM_X_MM
    rng = np.random.default_rng(seed)
    x0 = COM_X_MM
    out = []
    try:
        for _ in range(n):
            k = {f: 1.0 + stiffness_spread * (2 * rng.random() - 1) for f in ss_legs()}
            COM_X_MM = x0 + com_mm * (2 * rng.random() - 1)
            out.append(margin(surface, module, gait, posture, inclination_deg, mass_kg=mass_kg, k=k))
    finally:
        COM_X_MM = x0
    a = np.array([m if math.isfinite(m) else 1e9 for m in out])
    return {"p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)), "frac_ok": float(np.mean(a >= 2.0))}


def modules_mass_g(module):
    return 8 * MODULES[module]["g"]


def study(gait="wave", posture="crouched"):
    """{surface: {module: max inclination [deg] with margin ≥ 2 (None: not even level ground), "best": module}}."""
    out = {}
    for s in SURFACES:
        row = {m: max_inclination(s, m, gait, posture) for m in MODULES}
        cand = [m for m in MODULES if row[m] is not None]
        row["best"] = max(cand, key=lambda m: (row[m], -MODULES[m]["g"])) if cand else None
        out[s] = row
    return out
