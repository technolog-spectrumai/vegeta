"""ARACHNE's statics on a surface: how the attached feet share the load, what each foot's attachment must carry and
how far it is from letting go — gravity at any surface inclination and heading, the centre of mass off the wall,
uneven load sharing from unequal foot stiffness and placement, and the peel moment that shear applied above the pad
puts on its edge. Also the joint torques that hold the stance (J^T f) and the servo currents and power they cost.

**Load sharing.** Each attached foot is a spring (shear ``k_t``, normal ``k_n``: pad, ankle, link, servo backlash).
The body moves by a small rigid displacement (t, θ) until the foot forces f_i = −K_i (t + θ × r_i) balance the
weight W at the centre of mass: a 6 × 6 linear system. Stiffer or better-placed feet take more.

**Attachment.** For an adhesive pad of area A and length L along the shear: the edge stress is the pull-off T/A plus
the peel moment's 6 M / (A L), M = shear × the height of the ankle pivot above the interface (why the pivot sits
low, on the backing plate). Utilisation u = √((S / τA)² + (σ_edge / σ)²) with the surface's allowables τ, σ
(``arachne_specs.ADHESIVE``). For microspines: shear capacity n_engaged × f_spine × max(0, cos φ), φ between the
support the foot must give and the direction the spines face (they face +x, up the wall when climbing — so on bark
ARACHNE descends tail first); pull-off capacity a quarter of that. Margin = 1 / u; the design rule is margin ≥ 2 on
the allowables, which are themselves conservative.
"""
from __future__ import annotations

import math

import numpy as np

import arachne_specs as sp

__all__ = ["gravity_body", "foot_points", "share", "foot_capacity", "check", "monte_carlo", "joint_torques",
           "servo_currents", "K_T", "K_N", "PIVOT_HEIGHT_MM"]

K_T, K_N = 400.0, 800.0                 # N/m per attached foot (ASSUMPTIONS['foot stiffness'])
PIVOT_HEIGHT_MM = sp.FOOT_PAD["pivot_height"]   # the crossed-flexure ankle's virtual pivot above the interface


def gravity_body(mass_kg, inclination_deg=90.0, heading_deg=0.0):
    """The weight [N] in the body frame. ``inclination_deg``: 0 floor, 90 vertical wall, 180 ceiling;
    ``heading_deg``: the body's x from straight up the slope (0 climbing, 180 descending head first, ±90 across)."""
    b, h = math.radians(inclination_deg), math.radians(heading_deg)
    return mass_kg * sp.G * np.array([-math.sin(b) * math.cos(h), math.sin(b) * math.sin(h), -math.cos(b)])


def foot_points(kin, q=None, feet=None):
    """{foot: contact point [m], body frame with the origin at the body centre} at joint angles ``q``."""
    kin.set(kin.q0 if q is None else q)
    o = kin.data.xpos[kin.body("body")]
    out = {}
    for f in (feet or sp.LEGS):
        g = kin.model.geom(f).id
        p = kin.data.geom_xpos[g] - o
        out[f] = np.array([p[0], p[1], p[2] - kin.model.geom_size[g][0]])   # the pad's face, on the wall
    return out


def com_body(kin, q=None):
    kin.set(kin.q0 if q is None else q)
    return kin.data.subtree_com[kin.body("body")] - kin.data.xpos[kin.body("body")]


def _skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def _share_linear(points, weight, com, k_t=None, k_n=None):
    """Foot forces on the robot [N] {foot: f} for the attached ``points`` carrying ``weight`` at ``com`` (body frame),
    each foot a spring diag(k_t, k_t, k_n) (dicts per foot, or the defaults)."""
    feet = list(points)
    A = np.zeros((6, 6))
    for f in feet:
        K = np.diag([(k_t or {}).get(f, K_T)] * 2 + [(k_n or {}).get(f, K_N)])
        S = _skew(points[f])
        A[:3, :3] += -K
        A[:3, 3:] += K @ S
        A[3:, :3] += -S @ K
        A[3:, 3:] += S @ K @ S
    rhs = -np.concatenate([weight, np.cross(com, weight)])
    x = np.linalg.solve(A, rhs)
    t, th = x[:3], x[3:]
    out = {}
    for f in feet:
        K = np.diag([(k_t or {}).get(f, K_T)] * 2 + [(k_n or {}).get(f, K_N)])
        out[f] = -K @ (t + np.cross(th, points[f]))
    return out


def foot_capacity(force, surface, module="pad", pivot_mm=None):
    """(utilisation, detail) of one foot carrying ``force`` (the wall's force on the robot, body frame) on ``surface``
    with a ``module`` ('pad', 'spine' or 'hybrid')."""
    S = float(np.hypot(force[0], force[1]))
    T = max(0.0, -float(force[2]))                                        # pulled towards the wall: tension
    if module in ("pad", "hybrid"):
        a, b = sp.FOOT_PAD["pad"] if module == "pad" else sp.FOOT_HYBRID["pad"]
        A = a * b * 1e-6
        tau = sp.ADHESIVE["shear_kpa"][surface] * 1e3
        sig = sp.ADHESIVE["normal_kpa"][surface] * 1e3
        L = max(a, b) * 1e-3
        M = S * (PIVOT_HEIGHT_MM if pivot_mm is None else pivot_mm) * 1e-3
        edge = T / A + 6 * M / (A * L)
        u_pad = math.inf if (tau == 0 or sig == 0) and (S > 0 or T > 0) else \
            math.hypot(S / (tau * A) if tau else 0.0, edge / sig if sig else 0.0)
        if module == "pad":
            return u_pad, {"shear_n": S, "tension_n": T, "edge_kpa": edge / 1e3}
    n = sp.FOOT_SPINE["spines"] if module == "spine" else sp.FOOT_HYBRID["spines"]
    f1 = sp.SPINES["per_spine_n"][surface] * n * sp.SPINES["engaged_fraction"]
    if S > 1e-12:
        c = max(0.0, float(force[0]) / S)                                 # spines face +x (body)
    else:
        c = 1.0
    cap_s, cap_t = f1 * c, f1 * c * sp.SPINES["pulloff_over_shear"]
    u_sp = math.inf if (cap_s == 0 and S > 0) or (cap_t == 0 and T > 0) else \
        math.hypot(S / cap_s if cap_s else 0.0, T / cap_t if cap_t else 0.0)
    if module == "spine":
        return u_sp, {"shear_n": S, "tension_n": T, "spine_capacity_n": cap_s}
    # hybrid: the pad and the toe spines in parallel — the better of the two governs a foot (conservative: no sum)
    return min(u_pad, u_sp), {"shear_n": S, "tension_n": T}


def check(kin, attached, *, q=None, surface="glass", module="pad", inclination_deg=90.0, heading_deg=0.0,
          mass_kg=None, k_t=None, k_n=None, com_offset=(0.0, 0.0, 0.0), place_error=None, tail=False,
          pivot_mm=None):
    """The attached feet's forces and utilisations at ``q``: {"forces", "util", "margin", "worst_foot"}."""
    pts = foot_points(kin, q, attached)
    if place_error:
        pts = {f: p + np.asarray(place_error.get(f, (0, 0, 0))) for f, p in pts.items()}
    m = kin.robot.total_mass() if mass_kg is None else mass_kg
    W = gravity_body(m, inclination_deg, heading_deg)
    F = share(pts, W, com_body(kin, q) + np.asarray(com_offset), k_t, k_n,
              tail=sp.TAIL["x"] * 1e-3 if tail else None)
    util = {f: foot_capacity(F[f], surface, module, pivot_mm)[0] for f in F if f != "tail"}
    worst = max(util, key=util.get)
    return {"forces": F, "util": util, "margin": (1.0 / util[worst]) if util[worst] > 0 else math.inf,
            "worst_foot": worst}


def share(points, weight, com, k_t=None, k_n=None, tail=None):
    """Foot forces as ``_share_linear``; ``tail`` (x [m] on the body axis) adds the bark kit's tail: a contact on
    the surface that only pushes (normal stiffness, no shear). It lengthens the lever that resists the tipping moment
    of the off-wall centre of mass; if it would pull (heading down), it lifts off and the feet carry everything."""
    if tail is None:
        return _share_linear(points, weight, com, k_t, k_n)
    pts = dict(points)
    pts["tail"] = np.array([tail, 0.0, 0.0])
    kt, kn = dict(k_t or {}), dict(k_n or {})
    kt["tail"], kn["tail"] = 1e-6, sp.TAIL["stiffness_n_m"]
    F = _share_linear(pts, weight, com, kt, kn)
    if F["tail"][2] < 0:                                                   # it would pull: it lifts off
        return _share_linear(points, weight, com, k_t, k_n)
    return F


def monte_carlo(kin, attached, n=300, seed=0, *, stiffness_spread=0.3, place_mm=2.0, com_mm=5.0, **kw):
    """Margins over ``n`` draws: foot stiffness ±``stiffness_spread`` (uniform), foot placement ±``place_mm``,
    payload/CoM offset ±``com_mm`` in the wall plane. Returns the sorted margins."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kt = {f: K_T * rng.uniform(1 - stiffness_spread, 1 + stiffness_spread) for f in attached}
        kn = {f: K_N * rng.uniform(1 - stiffness_spread, 1 + stiffness_spread) for f in attached}
        pe = {f: (rng.uniform(-place_mm, place_mm) * 1e-3, rng.uniform(-place_mm, place_mm) * 1e-3, 0.0)
              for f in attached}
        co = (rng.uniform(-com_mm, com_mm) * 1e-3, rng.uniform(-com_mm, com_mm) * 1e-3, 0.0)
        out.append(check(kin, attached, k_t=kt, k_n=kn, place_error=pe, com_offset=co, **kw)["margin"])
    return np.sort(np.array(out))


def joint_torques(kin, forces, q=None):
    """{joint: τ [N·m]} the servos hold against the foot forces (τ = −Jᵀ f for each attached foot)."""
    import mujoco
    kin.set(kin.q0 if q is None else q)
    tau = np.zeros(kin.model.nv)
    jp = np.zeros((3, kin.model.nv))
    for f, F in forces.items():
        if f == "tail":                                                   # the tail is passive: no servo holds it
            continue
        g = kin.model.geom(f).id
        mujoco.mj_jacGeom(kin.model, kin.data, jp, None, g)
        tau -= jp.T @ np.asarray(F)
    return {j: float(tau[kin.dadr[k]]) for k, j in enumerate(kin.names)}


def servo_currents(torques, moving=(), move_current_a=0.06):
    """{joint: A}: idle + stall current × |τ| / τ_stall, plus ``move_current_a`` for joints in ``moving``
    (ASSUMPTIONS['servo current under load'])."""
    s = sp.SERVO
    st = sp.servo_stall_nm()
    return {j: s["idle_current_a"] + s["stall_current_a"] * min(1.0, abs(t) / st) + (move_current_a if j in moving else 0)
            for j, t in torques.items()}
