"""Kerchak2 as a legionary on the march: the plain white scutum on its grip bar in the left fist, the baseball bat
in the right resting on the shoulder, walking (``space_kerchak_scenes --scene legio --machine kerchak2``).

The arms are held in a carry pose while the legs walk: the right arm's pose solved on the robot's own joints
(``bat_carry``) so the bat's handle runs through the fist's hollow and its barrel lies on the shoulder; the left
forearm forward with the scutum upright ahead of it. The scutum and the bat are free bodies set rigidly on the
hands every frame (``worn``): what the fist holds moves with it. Metres, x forward, y left, z up."""
from __future__ import annotations

import math

import numpy as np
from vegeta.chiron import Geom, Link, Prop

import kerchak_scenarios as gs
import space_unit_robots as sr
import space_unit_specs as su
import space_units as sun

mm = 1e-3
P = su.KERCHAK2
SCUTUM = {"height": 1.06, "width": 0.66, "radius": 0.52, "thickness": 0.014, "slats": 9}
BAT = {"length": 0.864, "barrel_d": 0.066}
LEGIO = {"walk_m": 5.0}
WHITE, STEEL, WOOD = (0.95, 0.94, 0.91, 1.0), (0.33, 0.35, 0.37, 1.0), (0.78, 0.63, 0.40, 1.0)
#: the left arm's carry [rad]: the upper arm down and a little forward, the forearm forward and level-ish
LEFT_CARRY = {"L_shoulder_pitch": -0.25, "L_shoulder_roll": 0.15, "L_shoulder_yaw": 0.0, "L_elbow": -1.35,
              "L_wrist_roll": 0.0, "L_wrist_pitch": 0.0, "L_wrist_yaw": 0.0}
CURL = 1.6                                                                #: the fists closed round what they hold


def hole():
    """(centre [m], radius [m]) of the fist's hollow in the hand's frame (the gripper's: ``space_units.fist_hole``)."""
    c, _, r = sun.fist_hole(dict(P), CURL)
    return np.array(c) * mm, r * mm


def scutum_prop():
    """The scutum in its own frame: the grip bar along y at the origin, the board ahead (+x) curved round a
    vertical axis behind it, upright; slats of the arc, two posts from the bar to the board."""
    S = SCUTUM
    _, rh = hole()
    lay = su.hand_layout(dict(P), 1.0)
    reach = max(max(q[0] for q in su.finger_chain(f, CURL)) for f in lay["fingers"]) * mm + 0.03
    back = reach + 0.02                                                   # the board's inner face past the knuckles
    R, n = S["radius"], S["slats"]
    half = S["width"] / 2 / R
    ax = back + S["thickness"] / 2 - R                                    # the arc's axis (x), behind the board
    g = []
    for i in range(n):
        a = -half + (i + 0.5) * 2 * half / n
        c = (ax + R * math.cos(a), R * math.sin(a), 0.0)
        g.append(Geom(f"scutum_slat_{i}", "box", (S["thickness"] / 2, R * 2 * half / n / 2 + 0.002, S["height"] / 2), pos=c,
                      quat=(math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)), mass=0.9, role="visual", rgba=WHITE))
    hw = lay["palm"][1] * mm / 2 + 0.03                                   # the bar's half length: past the fist
    g.append(Geom("scutum_bar", "capsule", (rh, hw), pos=(0.0, 0.0, 0.0), quat=(0.70710678, -0.70710678, 0.0, 0.0), mass=0.3, role="visual", rgba=STEEL))
    for sy in (-1, 1):
        L = back
        g.append(Geom(f"scutum_post_{'l' if sy > 0 else 'r'}", "capsule", (rh * 0.5, L / 2), pos=(L / 2, sy * hw, 0.0),
                      quat=(0.70710678, 0.0, 0.70710678, 0.0), mass=0.1, role="visual", rgba=STEEL))
    return Prop(Link("scutum", pos=(-5.0, 2.0, 1.0), geoms=g), free=True)


def bat_prop():
    """The bat in its own frame: along +z, the grip (the fist's hollow) at the origin, the knob below it just past
    the little finger, the barrel up."""
    B = BAT
    _, rh = hole()
    lay = su.hand_layout(dict(P), 1.0)
    z0 = -(lay["palm"][1] * mm / 2 + 0.024)                                 # the knob
    L = B["length"]
    hL = 0.36 * L
    seg = lambda name, r, a, b: Geom(name, "capsule", (r, (b - a) / 2), pos=(0.0, 0.0, (a + b) / 2), mass=0.2, role="visual", rgba=WOOD)
    g = [Geom("bat_knob", "cylinder", (rh + 0.008, 0.007), pos=(0.0, 0.0, z0), mass=0.05, role="visual", rgba=WOOD),
         seg("bat_handle", rh, z0, z0 + hL),
         seg("bat_taper", (rh + B["barrel_d"] / 2) / 2, z0 + hL, z0 + 0.62 * L),
         seg("bat_barrel", B["barrel_d"] / 2, z0 + 0.62 * L, z0 + L - B["barrel_d"] / 2)]
    return Prop(Link("bat", pos=(-5.0, -2.0, 1.0), geoms=g), free=True)


def legio_props(p):
    ground = Prop(Link("parade", pos=(0.0, 0.0, 0.0), geoms=[
        Geom("parade_ground", "box", (8.0, 3.0, 0.002), pos=(2.5, 0.0, -0.0015), mass=1.0, role="visual", rgba=(0.72, 0.66, 0.55, 1.0))]), free=False)
    return (ground, scutum_prop(), bat_prop())


def _rotm(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _seg_dist(a0, a1, b0, b1, n=31):
    best = (1e9, 0.0)
    v = b1 - b0
    for s in np.linspace(0, 1, n):
        pa = a0 + s * (a1 - a0)
        t = float(np.clip(np.dot(pa - b0, v) / max(np.dot(v, v), 1e-12), 0, 1))
        dd = float(np.linalg.norm(pa - (b0 + t * v)))
        if dd < best[0]:
            best = (dd, t)
    return best


def bat_carry(kin, q0):
    """The right arm and wrist on the robot's joints (in their ranges: a human's wrist on Kerchak2) for the bat on
    the shoulder: the bat along the fist's hollow, its barrel touching the shoulder's joint module or the upper arm,
    clear of the forearm, the head and the ribs, the fist ahead of the chest. A global search, then a polish."""
    from scipy.optimize import differential_evolution, minimize
    names = ["R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"]
    idx = [kin.index[n] for n in names]
    bounds = [(kin.lo[i], kin.hi[i]) for i in idx]
    c_h, _ = hole()
    pp = sun.SpaceMarine().resolve() | dict(P)
    d = pp["arm_diameter"] * mm
    up_, fo_, ha_, th_ = (kin.body(n) for n in ("R_upper_arm", "R_forearm", "R_hand", "thorax"))
    lay = su.hand_layout(dict(P), 1.0)
    pw = lay["palm"][1] * mm
    L = BAT["length"]
    rb = BAT["barrel_d"] / 2
    kin.set(q0)
    top = max(kin.data.xpos[th_][2] + 0.35, 0.0)

    def line(q):
        kin.set(q)
        D = kin.data
        Rh = D.xmat[ha_].reshape(3, 3)
        g = D.xpos[ha_] + Rh @ c_h
        a = Rh[:, 1].copy()
        return g, (a if a[2] >= 0 else -a), D

    def cost(x):
        q = q0.copy()
        q[idx] = x
        g, a, D = line(q)
        S, E, W = D.xpos[up_].copy(), D.xpos[fo_].copy(), D.xpos[ha_].copy()
        z0 = -(pw / 2 + 0.024)
        b0, b1 = g + a * (z0 + 0.45 * L), g + a * (z0 + L - rb)
        dd, t = _seg_dist(b0, b1, S, E)
        r_arm = d * (0.65 if t < 0.2 else 0.5)
        c = ((dd - (r_arm + rb + 0.003)) / 0.01) ** 2                       # the barrel on the shoulder or the upper arm
        c += (max(0.0, t - 0.66) * 20) ** 2
        fd, _ = _seg_dist(g + a * (pw / 2 + 0.01), b1, E, W)
        c += (max(0.0, d * 0.42 + rb + 0.005 - fd) / 0.01) ** 2               # clear of its own forearm
        th = D.xpos[th_]
        nd, _ = _seg_dist(g - a * 0.03, b1, th, th + np.array([0.0, 0.0, 0.45]))
        c += (max(0.0, 0.16 - nd) / 0.01) ** 2                               # clear of the ribs, the neck and the head
        c += (max(0.0, th[0] + 0.12 - g[0]) / 0.02) ** 2                     # the fist ahead of the chest
        return c
    r0 = differential_evolution(cost, bounds, seed=2, maxiter=80, popsize=14, tol=1e-8, polish=False)
    r1 = minimize(cost, r0.x, method="Powell", bounds=bounds, options={"maxiter": 3000})
    x = r1.x if r1.fun <= r0.fun else r0.x
    q = q0.copy()
    q[idx] = x
    g, a, D = line(q)
    dd, t = _seg_dist(g + a * (-(pw / 2 + 0.024) + 0.45 * L), g + a * (-(pw / 2 + 0.024) + L - rb), D.xpos[up_], D.xpos[fo_])
    return dict(zip(names, (float(v) for v in x))), float(min(r0.fun, r1.fun)), dd, t


def worn(kin, q):
    """{prop: (hand body, R_off, p_off)}: the scutum and the bat set on their hands in the carry pose ``q`` - the
    scutum's bar along the left fist's hollow, upright, its board forward; the bat along the right fist's hollow,
    the barrel up."""
    kin.set(q)
    D = kin.data
    c_h, _ = hole()
    out = {}
    for prop, hand in (("scutum", "L_hand"), ("bat", "R_hand")):
        b = kin.body(hand)
        Rh, ph = D.xmat[b].reshape(3, 3).copy(), D.xpos[b].copy()
        y = Rh[:, 1].copy()
        if prop == "scutum":
            z = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], y) * y
            z /= np.linalg.norm(z)
            x = np.cross(y, z)
            if x[0] < 0:
                y, x = -y, -x
            Rw = np.column_stack([x, y, z])
        else:
            zb = y if y[2] >= 0 else -y
            xb = np.cross([0.0, 1.0, 0.0], zb)
            xb = xb / np.linalg.norm(xb) if np.linalg.norm(xb) > 1e-6 else np.array([1.0, 0.0, 0.0])
            Rw = np.column_stack([xb, np.cross(zb, xb), zb])
        out[prop] = (hand, Rh.T @ Rw, c_h.copy())
    return out


def scenario_legio(kin, q_stand, p):
    """For now: Kerchak2 marches with the scutum and the bat."""
    return [("gait", p["walk_m"] / 0.6, 0.0)]


gs.DEFAULTS["legio"] = LEGIO
gs.SCENARIOS["legio"] = scenario_legio
gs.COMPOSITE["legio"] = legio_props
