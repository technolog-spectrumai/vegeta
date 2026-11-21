"""TARANTULA's gaits: keyframed joint trajectories from foot paths by inverse kinematics on its own model
(``tarantula_robot``, ``motion_kit.Kin``), with their static stability checked at every keyframe. The dynamic runs
are ``sims/tarantula_sim.py`` (``./tarantula.sh``).

**A step**: the foot lifts, swings forward on an arc ``step_height`` high and sets down, a ``swing_s`` window; the
other feet push the body on, in stance. Feet move in groups:

* ``wave8`` — one leg at a time (L4, R2, L3, R1, R4, L2, R3, L1): seven feet always down; slow, the most stable;
* ``wave4`` — two diagonal legs at a time ((L1, R3), (R2, L4), (R1, L3), (L2, R4)): six feet down — the default;
* ``tetrapod`` — the alternating tetrapod ((L1, R2, L3, R4), (R1, L2, R3, L4)): four feet down, twice as fast;
* ``turn`` — ``wave4`` stepping round the body centre, ``stride`` [deg] per cycle;
* ``crab`` — ``wave4`` sideways (to the left).

**Posture**: ``body_drop_mm`` lowers the body between its feet (the feet's targets rise in the body frame):
0 = the CAD's standing pose (body centre 63 mm up), 21 = crouched (41.6 mm: the abdomen's belly 5 mm off the
surface — the climbing posture of ``tarantula_attachment``).

**Static stability** (``stability``): at each keyframe, the support polygon of the stance feet and the centre of mass
projected along gravity; the margin is the CoM's distance inside the polygon's nearest edge (negative: outside —
statically unstable). The tip-over angle in a direction = atan(edge distance that way / CoM height): the steepest
slope the stance stands on without tipping (friction aside: ``tan β ≤ μ``).
"""
from __future__ import annotations

import math

import motion_kit as mk
import numpy as np
import tarantula_robot as tr

__all__ = ["DEFAULTS", "GAITS", "POSTURES", "edge_distance", "gait", "hull", "stability"]

#: name -> (groups lifted together, twist direction (vx, vy, wz))
GAITS = {
    "wave8": ([("L4",), ("R2",), ("L3",), ("R1",), ("R4",), ("L2",), ("R3",), ("L1",)], (1, 0, 0)),
    "wave4": ([("L1", "R3"), ("R2", "L4"), ("R1", "L3"), ("L2", "R4")], (1, 0, 0)),
    "tetrapod": ([("L1", "R2", "L3", "R4"), ("R1", "L2", "R3", "L4")], (1, 0, 0)),
    "turn": ([("L1", "R3"), ("R2", "L4"), ("R1", "L3"), ("L2", "R4")], (0, 0, 1)),
    "crab": ([("L1", "R3"), ("R2", "L4"), ("R1", "L3"), ("L2", "R4")], (0, 1, 0)),
}
#: per gait: stride [mm] (turn: deg per cycle), swing window [s], step height [mm]
DEFAULTS = {"wave8": (80.0, 0.5, 40.0), "wave4": (80.0, 0.6, 40.0), "tetrapod": (80.0, 0.6, 40.0),
            "turn": (12.0, 0.6, 40.0), "crab": (60.0, 0.6, 40.0)}
#: body drop [mm]: crouched = as low as the abdomen allows — its belly 5 mm off the surface (body centre 41.6 mm)
POSTURES = {"standing": 0.0, "crouched": tr.S["belly"] - 5.0}


def _offsets(name, p0, t, period, stride, step_h):
    """{foot: (offset from its standing point [m], in stance?, swing fraction u or None)} at time t."""
    groups, (vx, vy, wz) = GAITS[name]
    n = len(groups)
    win = period / n
    sw = 1.0 / n
    out = {}
    for g, grp in enumerate(groups):
        for f in grp:
            p = p0[f]
            s = ((t - g * win) % period) / period
            if wz:
                a = math.radians(stride)
                D = np.array([-a * p[1], a * p[0], 0.0])
            else:
                D = np.array([vx, vy, 0.0]) * stride * 1e-3
            if s < sw:
                u = s / sw
                w = mk.smooth(u)
                off = -D / 2 + D * w
                off = off + np.array([0.0, 0.0, step_h * 1e-3 * math.sin(math.pi * u)])
                out[f] = (off, False, u)
            else:
                u = (s - sw) / (1 - sw)
                out[f] = (D / 2 - D * u, True, None)
    return out


def hull(points):
    """The convex hull (counter-clockwise) of 2-D points."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, float), 9))))
    if len(pts) < 3:
        return [np.array(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [np.array(p) for p in lower[:-1] + upper[:-1]]


def edge_distance(poly, c, direction=None):
    """Signed distance [m] from c to the polygon's boundary (positive inside); with ``direction`` (a 2-D unit
    vector), the distance from c to the boundary along it (0 if outside)."""
    c = np.asarray(c, float)
    if len(poly) < 3:
        return -math.inf if direction is None else 0.0
    if direction is None:
        d = math.inf
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)             # outward normal (ccw polygon)
            d = min(d, float(np.dot(a - c, nrm)))
        return d
    if edge_distance(poly, c) < 0:
        return 0.0
    u = np.asarray(direction, float)
    best = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        M = np.array([[u[0], -e[0]], [u[1], -e[1]]])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        s, r = np.linalg.solve(M, a - c)
        if s >= 0 and -1e-9 <= r <= 1 + 1e-9:
            best = min(best, s)
    return 0.0 if best == math.inf else best


def stability(feet_xyz, com_xyz, ground_z=0.0):
    """{"margin_m", "tip_deg": {"up", "down", "left", "right"}} of a stance: feet (stance only) and the CoM in a frame
    with z up from flat ground. "up" = the slope the stance stands on facing uphill (it tips backwards, −x)."""
    poly = hull([p[:2] for p in feet_xyz])
    c = np.asarray(com_xyz)[:2]
    h = max(float(com_xyz[2] - ground_z), 1e-6)
    out = {"margin_m": edge_distance(poly, c), "tip_deg": {}}
    for k, d in (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        out["tip_deg"][k] = math.degrees(math.atan2(edge_distance(poly, c, d), h))
    return out


def gait(name="wave4", *, robot=None, kin=None, stride=None, swing_s=None, step_height=None, body_drop_mm=0.0,
         frames=48, ik_reg=1e-7):
    """One gait cycle as a ``motion_kit.Motion`` (the 24 joints), with per keyframe the stance feet and the static
    stability (``motion.stance``, ``motion.stability``), and the gait's numbers in ``params`` / ``check``."""
    stride0, swing0, h0 = DEFAULTS[name]
    stride = stride0 if stride is None else stride
    swing_s = swing0 if swing_s is None else swing_s
    step_height = h0 if step_height is None else step_height
    kin = kin or mk.Kin(robot or tr.tarantula())
    groups, twist = GAITS[name]
    n = len(groups)
    period = swing_s * n
    kin.set(kin.q0)
    o = kin.data.xpos[kin.body("body")].copy()
    p0 = {f: kin.data.geom_xpos[kin.model.geom(f).id] - o for f in tr.LEGS}
    pts = {f: kin.geom_point(f) for f in tr.LEGS}
    lift = np.array([0.0, 0.0, body_drop_mm * 1e-3])
    ts = [period * i / frames for i in range(frames)]
    Q, q, worst, stances, stab = [], kin.q0.copy(), 0.0, [], []
    for t in ts:
        offs = _offsets(name, p0, t, period, stride, step_height)
        targets = [(pts[f][0], pts[f][1], o + p0[f] + lift + offs[f][0]) for f in tr.LEGS]
        q, err = kin.ik(targets, q_init=q, reg=ik_reg)
        worst = max(worst, err)
        Q.append(q.copy())
        kin.set(q)
        st = [f for f in tr.LEGS if offs[f][1]]
        stances.append(st)
        feet = [kin.data.geom_xpos[kin.model.geom(f).id] - np.array([0, 0, kin.model.geom_size[kin.model.geom(f).id][0]])
                for f in st]
        com = kin.data.subtree_com[kin.body("body")].copy()
        ground = o[2] - tr.STAND_Z + body_drop_mm * 1e-3              # the ground under the lowered body
        s = stability([f - np.array([0, 0, 0]) for f in feet], com, ground_z=ground)
        stab.append(s)
    Qa = np.asarray(Q)
    stance_t = period * (1 - 1 / n)
    if twist[2]:
        speed = {"yaw_rate_deg_s": stride / stance_t}
        base = {"velocity_m_s": [0.0, 0.0, math.radians(stride) / stance_t], "note": "[vx, vy, yaw rate]"}
    else:
        v = stride * 1e-3 / stance_t
        speed = {"speed_mm_s": v * 1000}
        base = {"velocity_m_s": [v * twist[0], v * twist[1], 0.0]}
    margins = [s["margin_m"] for s in stab]
    tips = {k: min(s["tip_deg"][k] for s in stab) for k in ("up", "down", "left", "right")}
    m = mk.Motion(name, "gait", list(kin.names), ts, Qa, period=period, base=base,
                  source=f"tarantula_gaits.gait('{name}')",
                  params=dict(speed, stride=stride, swing_s=swing_s, step_height_mm=step_height, period_s=period,
                              body_drop_mm=body_drop_mm, groups=[list(g) for g in groups],
                              duty=1 - 1 / n, stance_min=min(len(s) for s in stances)),
                  check={"ik_error_max_m": worst, "limit_margin_min_rad": kin.margin(Qa),
                         "static_margin_min_m": min(margins), "static_margin_mean_m": float(np.mean(margins)),
                         "tip_deg_min": tips})
    m.stance = stances
    m.stability = stab
    m.kin = kin
    return m
