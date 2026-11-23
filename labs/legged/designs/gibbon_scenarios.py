"""The gibbon's scenarios (``space_unit_robots.gibbon``): brachiation - hand over hand along the rungs of an overhead
ladder, the body a pendulum under the holding hand.

The scene: the gibbon walks upright under the ladder (its long arms held up), jumps and catches the first two rungs,
swings along ``swings`` rungs, then lets go and drops to the ground beyond. Each swing turns about the rung one hand
holds: the shoulders ride on a circle of the arm's reach about that rung (``REACH``), from midway behind it to midway
ahead, so at either end both hands reach a rung at the same stretch (the catch); the trunk swings with the arm, a
little less (``TRUNK_SWING``), the legs hang tucked. The free hand lets go as the swing starts, passes under the
holding hand's rung (the rungs are solid: it dips below them) and closes on the next one at the end. The time of a
swing is half the pendulum's period for the body's centre of mass below the rung (``half_period``).

The holds are ``kerchak_scenarios`` bars (an overhead rung: the wrist below the grip, along the arm), the motions
``kerchak_tasks.contact_motion`` (with the solid-limb check when the player sets it)."""
from __future__ import annotations

import math

import numpy as np
from vegeta.chiron import Geom, Link, Prop

import kerchak_motions as gm
import kerchak_scenarios as gs
import kerchak_tasks as gt
import motion_kit as mk
import space_unit_robots as sr

__all__ = ["LADDER", "LADDER_UP", "COURSE", "holds", "ladder_prop", "scenario_brachiate", "half_period", "hang_ref"]

LADDER = {"x": 1.2, "height": 1.55, "spacing": 0.8, "rungs": 8, "rung_radius": 0.02, "rail_y": 0.38, "swings": 5}
REACH = 0.58                     #: the shoulders' distance from the holding rung's grip, the arm a little bent [m]
GRIP_Y = 0.12                    #: the hands' grips either side of the ladder's middle [m]
TRUNK_SWING = 0.25               #: the trunk's pitch against the arm's angle from straight down
SHOULDER_Z = 0.272               #: the shoulders above the pelvis, standing (``gibbon``) [m]
G = 9.81
GREY_STEEL = (0.5, 0.52, 0.56, 1.0)


def _z(p, x):
    """The ladder's height at ``x``: it rises at ``rise_deg`` from its first rung."""
    return p["height"] + (x - p["x"]) * math.tan(math.radians(p.get("rise_deg", 0.0)))


def ladder_prop(p):
    """The overhead ladder: rungs across the gibbon's way between two side rails, on four posts; level, or rising."""
    x0, s, n, rr, ry = p["x"], p["spacing"], int(p["rungs"]), p["rung_radius"], p["rail_y"]
    x1 = x0 + (n - 1) * s
    geoms = [gs._cap(f"rung_{k}", rr, (x0 + k * s, -ry, _z(p, x0 + k * s)), (x0 + k * s, ry, _z(p, x0 + k * s)), rgba=GREY_STEEL, mass=2.0) for k in range(n)]
    xa, xb = x0 - 0.3, x1 + 0.3
    for sy in (-1, 1):
        geoms.append(gs._cap(f"rail_{'l' if sy > 0 else 'r'}", 0.025, (xa, sy * ry, _z(p, xa)), (xb, sy * ry, _z(p, xb)), rgba=GREY_STEEL, mass=10.0))
        for xp in (xa, xb):
            geoms.append(gs._cap(f"post_{xp:.1f}_{sy}", 0.03, (xp, sy * ry, 0.0), (xp, sy * ry, _z(p, xp) + 0.03), rgba=GREY_STEEL, mass=10.0))
    return Prop(Link("ladder", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def half_period(kin, q_hang):
    """Half the pendulum's period [s] for the hanging body: its centre of mass this far below the rung."""
    m = kin.model
    kin.set(q_hang)
    com = (m.body_mass[:, None] * kin.data.xipos).sum(axis=0) / m.body_mass.sum()
    L = REACH + SHOULDER_Z - float(com[2])                                 # the rung's grip to the centre of mass
    return math.pi * math.sqrt(max(L, 0.3) / G)


def hang_ref(kin, q):
    """The pose a hang is regularised to: the hips and knees a little flexed (the legs tucked under), the arms overhead."""
    q = np.asarray(q, dtype=float).copy()
    for side in ("L", "R"):
        q[kin.index[f"{side}_hip_pitch"]], q[kin.index[f"{side}_knee"]], q[kin.index[f"{side}_ankle_pitch"]] = -0.6, 0.9, -0.2
        q[kin.index[f"{side}_shoulder_pitch"]], q[kin.index[f"{side}_elbow"]] = -2.9, -0.3
    return q


def holds(p):
    """[(centre, axis)] of the holds in the order they are taken: the ladder's rungs (across the way), then, with a
    ``rope``, points along the horizontal rope from the last rung (along the way) every ``rope_step``."""
    out = []
    for k in range(int(p["rungs"])):
        x = p["x"] + k * p["spacing"]
        out.append((np.array([x, 0.0, _z(p, x)]), np.array([0.0, 1.0, 0.0])))
    if p.get("rope_end") is not None:
        c_last = out[-1][0]
        x = c_last[0] + p["rope_step"]
        while x <= p["rope_end"] + 1e-6:
            out.append((np.array([x, 0.0, c_last[2]]), np.array([1.0, 0.0, 0.0])))
            x += p["rope_step"]
    return out


def _rung(p, k):
    return holds(p)[k][0]


def _below_both(a, b):
    """The point a reach from both rungs ``a`` and ``b``, below them: where the shoulders are when both hands hold."""
    m, d = (a + b) / 2, b - a
    half = np.linalg.norm(d) / 2
    down = np.array([d[2], 0.0, -d[0]]) / (2 * half)                        # perpendicular to the rungs' line, downwards
    return m + down * math.sqrt(max(REACH ** 2 - half ** 2, 1e-6))


def _hold(p, k, side, toward):
    """The overhead rung ``k`` as a bar hold for ``side``'s hand, the wrist towards ``toward`` (the shoulder)."""
    c, axis = holds(p)[k]
    rope = abs(axis[0]) > 0.5
    c = c + (np.zeros(3) if rope else np.array([0.0, GRIP_Y if side == "L" else -GRIP_Y, 0.0]))   # on the rope both hands on its line
    wd = np.asarray(toward, dtype=float) - c
    wd[1] = 0.0 if not rope else wd[1]
    return gs.bar(c, axis=tuple(axis), n=(0.0, 0.0, -1.0), r=p["rung_radius"], wrist_dir=gs._unit(wd))


def _swing_base(p, k, u):
    """(pelvis xyz, pitch, shoulders' midpoint) at ``u`` of the swing about rung ``k``: the shoulders on the reach's
    circle about the rung from midway behind it to midway ahead, at rest at both ends (a pendulum's turn)."""
    piv = _rung(p, k)
    a0 = _below_both(_rung(p, k - 1), piv) - piv
    a1 = _below_both(piv, _rung(p, k + 1)) - piv
    t0, t1 = math.atan2(a0[0], -a0[2]), math.atan2(a1[0], -a1[2])           # the arm's angle from straight down at either end
    theta = t0 + (t1 - t0) * (1 - math.cos(math.pi * u)) / 2
    pitch = -TRUNK_SWING * theta
    sh = piv + REACH * np.array([math.sin(theta), 0.0, -math.cos(theta)])
    pelvis = sh - gm.base_rot(pitch) @ np.array([0.0, 0.0, SHOULDER_Z])
    return pelvis, pitch, sh


def _sides(i):
    """(holding, free) hand for swing ``i``: they alternate."""
    return ("R", "L") if i % 2 == 0 else ("L", "R")


def tail_tip(kin):
    """(body, local) of the prehensile tail's grip pad, or None: a frame without a tail (the gibbon)."""
    m = kin.model
    try:
        g = m.geom("tail_tip")
    except KeyError:
        return None
    return int(g.bodyid[0]), np.array(m.geom_pos[g.id])


def _tail_on(p, k, rad=0.012):
    """Where the tail's pad goes on hold ``k``: on the bar's top, wrapped over it."""
    return holds(p)[k][0] + np.array([0.0, 0.0, p["rung_radius"] + rad])


def tail_targets(kin, p, k, u, first=False, rest=None):
    """The tail as a fifth limb through a swing about hold ``k``: wrapped over the hold behind (``k - 1``) while the
    hands change over, then up and over to the hold the body is passing (``k``), wrapped on it by the swing's end
    (a spider monkey's semibrachiation: the tail takes the body's weight as the hands change). ``first``: the tail
    comes from free (the first swing: ``rest``, the tip's place in the pelvis's frame as it hangs), it only takes ``k``."""
    tip = tail_tip(kin)
    if tip is None:
        return []
    a, b = _tail_on(p, k - 1), _tail_on(p, k)
    U0, U1 = 0.25, 0.85                                                     # let go of the hold behind; wrapped on the next

    def body(u_):
        pelvis, pitch, _ = _swing_base(p, k, u_)
        return gm.base_rot(pitch), pelvis

    if u <= U0 and not first:
        return [(tip[0], tip[1], a)]
    if u >= U1:
        return [(tip[0], tip[1], b)]
    # in between: carried with the body - the tip's place in the pelvis's frame eased from where it let go to where
    # it will wrap, so it stays in the tail's reach as the body swings through the bottom
    u0 = 0.0 if first else U0
    R0, p0 = body(u0)
    R1, p1 = body(U1)
    l0, l1 = (np.asarray(rest) if first else R0.T @ (a - p0)), R1.T @ (b - p1)
    f = mk.smooth((u - u0) / (U1 - u0))
    R, pc = body(u)
    lift = np.array([0.0, 0.0, 0.08 * math.sin(math.pi * f)])
    return [(tip[0], tip[1], R @ ((1 - f) * l0 + f * l1) + pc + lift)]


def brachiate_swing(kin, q_from, p, k, i, duration):
    """One swing about rung ``k`` (swing ``i``): the holding hand on ``k``, the free hand from rung ``k - 1`` under
    rung ``k`` to rung ``k + 1``."""
    hold_side, free_side = _sides(i)
    tip = tail_tip(kin)
    rest = None
    if tip is not None:
        kin.set(q_from)
        rest = kin.world(*tip)                                             # the tail's tip in the pelvis's frame as the swing begins

    def targets(u):
        pelvis, pitch, sh = _swing_base(p, k, u)
        out = gs.hand_targets(kin, hold_side, _hold(p, k, hold_side, sh))
        f0 = _hold(p, k - 1, free_side, sh)
        f1 = _hold(p, k + 1, free_side, sh)
        if u <= 0.1:
            out += gs.hand_targets(kin, free_side, f0)
        elif u >= 0.9:
            out += gs.hand_targets(kin, free_side, f1)
        else:                                                              # under the holding hand's rung, well below it; along the rope, past the holding hand beside it
            f = mk.smooth((u - 0.1) / 0.8)
            on_rope = abs(f0["axis"][0]) > 0.5 and abs(f1["axis"][0]) > 0.5
            sy = 1.0 if free_side == "L" else -1.0
            dip = np.array([0.0, sy * 0.1, 0.07]) if on_rope else np.array([0.0, 0.0, 0.22])
            c = (1 - f) * f0["c"] + f * f1["c"] + np.array([1.0, 1.0, -1.0]) * dip * math.sin(math.pi * f)
            out += gs._hand_at(kin, free_side, c, gs._slerp(gs.hand_frame_on(f0, free_side), gs.hand_frame_on(f1, free_side), f))
        return out + tail_targets(kin, p, k, u, first=(i == 0), rest=rest)

    def base(u):
        pelvis, pitch, _ = _swing_base(p, k, u)
        return pelvis[0], pelvis[1], pelvis[2], pitch

    def fingers(u, q):
        sr.set_curl(q, kin, hold_side, gs.CURL_CLOSED)
        for sd in ("L", "R"):                                              # a grasping foot hangs curled, a loose fist
            sr.set_toe_curl(q, kin, sd, 0.9)
        return sr.set_curl(q, kin, free_side, gs._curl(u, 0.05, 0.95))

    return gt.contact_motion(kin, f"swing_{i + 1}", duration, targets, base, q_init=q_from, extra=fingers, reg=gs.IK_REG,
                             iters=gs.IK_ITERS, q_ref=hang_ref(kin, q_from),
                             source="gibbon_scenarios.brachiate_swing: the body a pendulum under one hand, the other hand to the next rung")


def jump_to_ladder(kin, q_stand, p, stop_x, duration=1.2):
    """From standing between rungs 0 and 1: a crouch, a jump, both hands up onto the two rungs (the left on 0, the
    right on 1), the legs tucking as the feet leave the ground."""
    pts = gm._contact_points(kin)
    b0 = np.array([stop_x, 0.0, gm.STANCE["h"]])
    start = gs._contacts(kin, q_stand, b0, gm.STANCE["pitch"])
    pelvis1, pitch1, sh1 = _swing_base(p, 1, 0.0)
    hands = {}
    for side, k in (("L", 0), ("R", 1)):
        end = gs.hand_targets(kin, side, _hold(p, k, side, sh1))
        kin.set(q_stand)
        hands[side] = (kin.world(end[0][0], end[0][1]), end)                # the grip point in the body's frame, standing
    crouch = b0 - np.array([0.0, 0.0, 0.12])

    def carried(side, u):                                                   # the hand where the standing pose holds it, the body wherever it is
        x, y, z, pt = base(u)
        return gm.base_rot(pt) @ hands[side][0] + np.array([x, y, z])

    def base(u):
        if u < 0.3:
            f = mk.smooth(u / 0.3)
            b, pt = b0 + f * (crouch - b0), gm.STANCE["pitch"] + f * 0.15
        else:
            f = mk.smooth((u - 0.3) / 0.7)
            b = (1 - f) * crouch + f * pelvis1 + np.array([0.0, 0.0, 0.08 * math.sin(math.pi * f)])
            pt = 0.15 + f * (pitch1 - 0.15)
        return b[0], b[1], b[2], pt

    def targets(u):
        out = []
        for side in ("L", "R"):
            _, end = hands[side]
            if u < 0.3:                                                    # the crouch: the arms ride with the body
                out.append((end[0][0], end[0][1], carried(side, u)))
            elif u < 0.85:
                f = mk.smooth((u - 0.3) / 0.55)
                out.append((end[0][0], end[0][1], (1 - f) * carried(side, 0.3) + f * end[0][2]))
            else:
                out += end
            if u < 0.35:                                                   # the feet push off
                out += [(bd, l, sk) for (bd, l), sk in zip(pts[side], start[side])]
            elif u < 0.7:                                                  # then rise with the body, clear of the ground, the toes last
                rise = base(u)[2] - crouch[2] + 0.03 * mk.smooth((u - 0.35) / 0.35)
                out += [(bd, l, sk + np.array([0.0, 0.0, rise])) for (bd, l), sk in zip(pts[side], start[side])]
        return out

    def fingers(u, q):
        for side in ("L", "R"):
            sr.set_curl(q, kin, side, gs._curl(u, 0.0, 1.0, start=1.0))
            sr.set_toe_curl(q, kin, side, 0.9 * mk.smooth(max(0.0, (u - 0.4) / 0.6)))   # off the ground, the toes curl
        return q

    return gt.contact_motion(kin, "jump_to_the_ladder", duration, targets, base, q_init=q_stand, extra=fingers, reg=gs.IK_REG,
                             iters=gs.IK_ITERS, source="gibbon_scenarios.jump_to_ladder: up onto the first two rungs")


def drop_off(kin, q_hang, q_stand, p, k, duration=1.0):
    """From the two-hand hang between rungs ``k`` and ``k + 1``: the hands open, the body falls, the legs reach for
    the ground and it lands standing (joint keyposes on a base track: nothing held in the air)."""
    pelvis0, pitch0, _ = _swing_base(p, k, 1.0)
    release = q_hang.copy()
    for side in ("L", "R"):
        sr.set_curl(release, kin, side, gs.CURL_OPEN)
    land = q_stand.copy()
    for side in ("L", "R"):
        land[kin.index[f"{side}_hip_pitch"]], land[kin.index[f"{side}_knee"]], land[kin.index[f"{side}_ankle_pitch"]] = -0.7, 1.3, -0.6
    x1 = pelvis0[0] + 0.25
    import humanoid_motions as hum
    h_land = hum.base_height(kin, land)                                    # the soles on the ground with the knees bent

    def base(u):
        f = mk.smooth(u)
        fall = min(1.0, u / 0.7) ** 2                                      # free fall, then the knees take the landing
        z = pelvis0[2] + (h_land - pelvis0[2]) * fall if u < 0.7 else h_land + (gm.STANCE["h"] - h_land) * mk.smooth((u - 0.7) / 0.3)
        return pelvis0[0] + f * (x1 - pelvis0[0]), 0.0, z, pitch0 * (1 - f) + gm.STANCE["pitch"] * f

    keys = [(0.0, q_hang), (0.12, release), (0.7, land), (1.0, q_stand)]
    return gs._joint_track(kin, "drop_to_the_ground", duration, keys, base, source="gibbon_scenarios.drop_off: let go, fall, land standing")


def scenario_brachiate(kin, q_stand, p):
    """The walk under the ladder, the jump onto it, ``swings`` swings hand over hand, the drop beyond."""
    stop_x = _swing_base(p, 1, 0.0)[0][0]                                  # under the first two rungs
    v = 0.6
    segs = [("gait", stop_x / v, 0.0)]
    j = jump_to_ladder(kin, q_stand, p, stop_x)
    segs.append(("motion", j))
    q = gt._last(kin, j, q_stand)
    T = half_period(kin, hang_ref(kin, q))
    n = min(int(p["swings"]), int(p["rungs"]) - 2)
    for i in range(n):
        m = brachiate_swing(kin, q, p, i + 1, i, T)
        m.params = {**(m.params or {}), "half_period_s": round(T, 3)}
        segs.append(("motion", m))
        q = gt._last(kin, m, q)
    if p.get("rise_deg", 0.0) > 0.0:                                       # up at the top: it hangs there, the ground well below
        pelvis, pitch, _ = _swing_base(p, n, 1.0)
        segs.append(("motion", gs._joint_track(kin, "hang_at_the_top", 1.5, [(0.0, q), (1.0, q)],
                                               lambda u: (pelvis[0], 0.0, pelvis[2], pitch), source="gibbon_scenarios: the hang at the top")))
        return segs
    segs.append(("motion", drop_off(kin, q, q_stand, p, n)))
    segs.append(("gait", 1.5, 0.0))
    return segs


LADDER_UP = {**LADDER, "rise_deg": 22.0, "rungs": 8, "swings": 6}     #: the ladder rising: each rung 0.32 m above the last
def _rope_hold(c, side, toward, r=0.02):
    """A hold on the horizontal rope at ``c``, the wrist towards ``toward``."""
    c = np.asarray(c, dtype=float)
    return gs.bar(c, axis=(1.0, 0.0, 0.0), n=(0.0, 0.0, -1.0), r=r, wrist_dir=gs._unit(np.asarray(toward, dtype=float) - c))


def _window(p):
    return {**gs.WINDOW, "x": p["window_x"], "table_x": p["table_x"]}


def rope_prop(p):
    """The horizontal rope: from the ladder's last rung through the open window to a bracket under the lintel inside."""
    w = _window(p)
    c0 = holds({**p, "rope_end": None})[-1][0]
    x_in = w["x"] + w["wall_thickness"] / 2 + 0.04
    top = w["sill"] + w["height"]
    geoms = [gs._cap("rope", 0.02, (c0[0], 0.0, c0[2]), (x_in, 0.0, c0[2]), rgba=gs.ROPE_C, mass=2.0),
             gs._box("rope_bracket", (0.02, 0.03, (top - c0[2]) / 2 + 0.01), (x_in, 0.0, (top + c0[2]) / 2), rgba=GREY_STEEL, mass=0.5)]
    return Prop(Link("rope_line", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def course_props(p):
    gt.DOOR["x"] = p["door_x"]
    door = gt.door_prop()
    for lk, _ in door.root.walk() if hasattr(door, "root") else door.link.walk():   # the door's walls named apart from the window's
        for g in lk.geoms:
            if g.name.startswith("wall_"):
                g.name = "door_" + g.name
    return (ladder_prop(p), rope_prop(p), gs.window_prop(_window(p)), door)


def window_from_rope(kin, q_from, q_stand, p, k, i_next, duration=4.0):
    """Hanging from the rope's last two holds outside the window (``k - 1``, ``k``): the feet swing up and over the
    sill's edge onto it one after the other, the back hand goes along the rope (under the front one) to above the
    sill as the body moves in to a crouch on it, then the front hand on into the opening: the rope runs through the
    window, so the hands are on it all the way."""
    w = _window(p)
    x, t, top = w["x"], w["wall_thickness"], w["sill"] + 0.06
    back_side, front_side = _sides(i_next)                                 # the hand on ``k - 1`` and the one on ``k``
    z_r = holds(p)[k][0][2]
    pelvis0, pitch0, sh0 = _swing_base(p, k - 1, 1.0)
    crouch, crouch_pitch = np.array([x - 0.12, 0.0, top + 0.3]), 0.35
    feet = {sd: gs.point(np.array([x - t / 2 + 0.02, 0.1 if sd == "L" else -0.1, top]), pitch=0.0, over=True) for sd in ("L", "R")}   # on the sill's near half
    hang = gs._contacts(kin, q_from, pelvis0, pitch0)
    pts = gm._contact_points(kin)
    win = {"L": (0.0, 0.25), "R": (0.15, 0.4)}
    back1, front1 = np.array([x - 0.05, 0.0, z_r]), np.array([x + 0.18, 0.0, z_r])

    def base(u):
        f = mk.smooth(max(0.0, min(1.0, (u - 0.35) / 0.4)))
        g = mk.smooth(min(1.0, u / 0.25))                                  # drawn in towards the wall as the feet come up
        lift = 0.06 * math.sin(math.pi * min(1.0, u / 0.4))
        b = (1 - f) * (pelvis0 + np.array([0.12 * g, 0.0, 0.0])) + f * crouch + np.array([0.0, 0.0, lift])
        return b[0], b[1], b[2], (1 - f) * (pitch0 - 0.15 * math.sin(math.pi * min(1.0, u / 0.4))) + f * crouch_pitch

    def shoulders(u):
        x_, y_, z_, pt = base(u)
        return gm.base_rot(pt) @ np.array([0.0, 0.0, SHOULDER_Z]) + np.array([x_, y_, z_])

    def hand_path(side, c0, c1, a, b, u):
        sh = shoulders(u)
        h0, h1 = (_hold(p, c0, side, sh) if isinstance(c0, int) else _rope_hold(c0, side, sh)), _rope_hold(c1, side, sh)
        if u <= a:
            return gs.hand_targets(kin, side, h0)
        if u >= b:
            return gs.hand_targets(kin, side, h1)
        f = mk.smooth((u - a) / (b - a))
        c = (1 - f) * h0["c"] + f * h1["c"] - np.array([0.0, 0.0, 0.15 * math.sin(math.pi * f)])
        return gs._hand_at(kin, side, c, gs._slerp(gs.hand_frame_on(h0, side), gs.hand_frame_on(h1, side), f))

    def targets(u):
        out = hand_path(back_side, k - 1, back1, 0.4, 0.7, u) + hand_path(front_side, k, front1, 0.75, 0.95, u)
        for sd in ("L", "R"):
            a, b = win[sd]
            if u <= a:
                out += [(bd, l, wpt) for (bd, l), wpt in zip(pts[sd], hang[sd])]
            elif u >= b:
                out += gs.foot_targets(kin, sd, feet[sd])
            else:
                ff = (u - a) / (b - a)
                c = gs.foot_swing(gs.point(np.mean(hang[sd], axis=0), pitch=0.0), feet[sd], ff, lift=0.12, clear=0.08)
                out += gs.foot_targets(kin, sd, gs.point(feet[sd]["c"], pitch=0.3 * (1 - mk.smooth(ff)), over=True), c)
        return out

    def fingers(u, q):
        sr.set_curl(q, kin, back_side, gs._curl(u, 0.4, 0.7))
        return sr.set_curl(q, kin, front_side, gs._curl(u, 0.75, 0.95))

    m = gt.contact_motion(kin, "onto_the_sill", duration, targets, base, q_init=q_from, extra=fingers, reg=gs.IK_REG, iters=gs.IK_ITERS,
                          q_ref=hang_ref(kin, q_from), source="gibbon_scenarios.window_from_rope: the feet onto the sill, the hands along the rope into the opening")
    q1 = gt._last(kin, m, q_from)
    # the hop down inside: the hands open, the legs stay folded until the pelvis is past the wall, then reach for the floor
    release = q1.copy()
    for sd in ("L", "R"):
        sr.set_curl(release, kin, sd, gs.CURL_OPEN)
    tuck = release.copy()
    for sd in ("L", "R"):
        tuck[kin.index[f"{sd}_knee"]] = min(q1[kin.index[f"{sd}_knee"]] + 0.25, kin.hi[kin.index[f"{sd}_knee"]])
        tuck[kin.index[f"{sd}_hip_pitch"]] = max(q1[kin.index[f"{sd}_hip_pitch"]] - 0.25, kin.lo[kin.index[f"{sd}_hip_pitch"]])
    x_land, x_clear = x + t / 2 + 0.45, x + t / 2 + 0.15

    def hop_base(u):
        xb = crouch[0] + mk.smooth(u) * (x_land - crouch[0])
        fz = mk.smooth(max(0.0, min(1.0, (xb - x_clear) / (x_land - x_clear))))
        return xb, 0.0, (1 - fz) * crouch[2] + fz * gm.STANCE["h"] + 0.04 * math.sin(math.pi * fz), (1 - fz) * crouch_pitch + fz * gm.STANCE["pitch"]

    hop = gs._joint_track(kin, "hop_down_inside", 1.3, [(0.0, q1), (0.12, release), (0.35, tuck), (0.65, tuck), (1.0, q_stand)], hop_base,
                          source="gibbon_scenarios.window_from_rope: the hop down off the sill into the room")
    return m, hop, x_land


def scenario_course(kin, q_stand, p):
    """Up the rising ladder hand over hand, along the horizontal rope to the house, in through the open window (the
    rope runs through it), across the room, the door opened, through it to the banana."""
    gt.DOOR["x"] = p["door_x"]
    stop_x = _swing_base(p, 1, 0.0)[0][0]
    v = 0.6
    segs = [("gait", stop_x / v, 0.0)]
    j = jump_to_ladder(kin, q_stand, p, stop_x)
    segs.append(("motion", j))
    q = gt._last(kin, j, q_stand)
    T = half_period(kin, hang_ref(kin, q))
    k_last = len(holds(p)) - 1
    n = k_last - 1
    for i in range(n):
        m = brachiate_swing(kin, q, p, i + 1, i, T if i < int(p["rungs"]) - 2 else T * 0.8)   # along the rope the steps are shorter
        segs.append(("motion", m))
        q = gt._last(kin, m, q)
    sill, hop, x_land = window_from_rope(kin, q, q_stand, p, k_last, n)
    segs += [("motion", sill), ("motion", hop)]
    segs.append(("gait", max(gt.stop_x() - x_land, 0.05) / v, 0.0))
    segs += [("motion", m) for m in gt.door_sequence(kin, q_stand)]
    table_front = p["window_x"] + p["table_x"] - 0.4                       # the banana's table: up to its edge
    segs.append(("gait", (table_front - 0.3 - gt.stop_x()) / v, 0.0))
    return segs


BRICK, MORTAR = (0.55, 0.27, 0.2, 1.0), (0.6, 0.58, 0.55, 1.0)
CHIMNEY = {**gs.WALL, "x": 2.2, "width": 1.0, "depth": 0.8, "height": 6.5, "rung_spacing": 0.3, "first_rung": 0.25,
           "rung_out": 0.12, "rung_radius": 0.016, "cycles": 17, "step": 1}


def chimney_prop(p):
    """A brick chimney standing on the ground, steel staple rungs up its face (the climbing side, -x), a stone cap
    and a flue pot on top."""
    x, w, dp, h, sp, z0, out, rr = p["x"], p["width"], p["depth"], p["height"], p["rung_spacing"], p["first_rung"], p["rung_out"], p["rung_radius"]
    geoms = [gs._box("chimney", (dp / 2, w / 2, h / 2), (x + dp / 2, 0.0, h / 2), rgba=BRICK, mass=8000.0)]
    for j in range(int(h / 0.45)):                                         # mortar courses round the brickwork
        geoms.append(gs._box(f"course_{j}", (dp / 2 + 0.004, w / 2 + 0.004, 0.006), (x + dp / 2, 0.0, 0.3 + j * 0.45), rgba=MORTAR, mass=1.0))
    z, k = z0, 0
    while z < h - 0.15:
        ya, yb = -w / 2 + 0.08, w / 2 - 0.08
        geoms.append(gs._cap(f"rung_{k}", rr, (x - out, ya, z), (x - out, yb, z), rgba=gs.STEEL, mass=1.0))
        for yk, tag in ((ya, "a"), (yb, "b")):                             # the staple's legs into the brick
            geoms.append(gs._cap(f"rung_{k}_{tag}", rr, (x - out, yk, z), (x + 0.02, yk, z), rgba=gs.STEEL, mass=0.3))
        z += sp
        k += 1
    geoms.append(gs._box("chimney_cap", (dp / 2 + 0.08, w / 2 + 0.08, 0.06), (x + dp / 2, 0.0, h + 0.06), rgba=MORTAR, mass=200.0))
    geoms.append(gs._cap("flue_pot", 0.12, (x + dp / 2, 0.0, h + 0.12), (x + dp / 2, 0.0, h + 0.45), rgba=BRICK, mass=30.0))
    return Prop(Link("chimney_stack", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


gs.DEFAULTS["chimney"] = CHIMNEY
gs.BUILDERS["chimney"] = chimney_prop
def scenario_chimney(kin, q_stand, p):
    """Up the chimney's staples: the feet on one, the hands two above (the gibbon's long arms), the pelvis well off the
    brick and low over the feet (a static search against the solid rungs and chimney), each limb up ``step`` rungs a
    cycle (two would outreach the short legs before the pelvis rises)."""
    xb, sp, z0 = p["x"] - p["rung_out"], p["rung_spacing"], p["first_rung"]
    step = int(p.get("step", 2))
    hold = lambda k: {"L_hand": gs.bar((xb, 0.2, z0 + (k * step + 2) * sp), r=p["rung_radius"]),
                      "R_hand": gs.bar((xb, -0.2, z0 + (k * step + 2) * sp), r=p["rung_radius"]),
                      "L": gs.bar((xb, 0.12, z0 + k * step * sp), r=p["rung_radius"], tip=0.3),
                      "R": gs.bar((xb, -0.12, z0 + k * step * sp), r=p["rung_radius"], tip=0.3)}
    base = lambda k: np.array([xb - 0.3, 0.0, z0 + k * step * sp + 0.35])
    n = min(int(p["cycles"]), int((p["height"] - 0.9 - z0) / (step * sp)))
    return gs._climb_sequence(kin, q_stand, hold, base, n, stop_x=xb - 0.25)


gs.SCENARIOS["chimney"] = scenario_chimney

LEAF = (0.25, 0.45, 0.2, 1.0)
SPIDER_TREE = {"x": 2.2, "trunk_radius": 0.09, "height": 5.2, "first_branch": 0.25, "spacing": 0.3, "branch_length": 0.75,
               "branch_radius": 0.028, "branch_rise": 0.06, "cycles": 9}


def spider_tree_prop(p):
    """A tree for a spider monkey: a trunk with a pair of branches at every level, out either side and rising a little,
    a crown of leaves on top. The branches are solid: the hands close on them, the feet stand on them, the tail
    wraps round them."""
    x, r, h, sp, bl, br, rise = p["x"], p["trunk_radius"], p["height"], p["spacing"], p["branch_length"], p["branch_radius"], p["branch_rise"]
    geoms = [gs._cap("trunk", r, (x, 0.0, 0.0), (x, 0.0, h), rgba=gs.BARK, mass=400.0)]
    z, k = p["first_branch"], 0
    while z < h - 0.4:
        for sy, tag in ((1.0, "l"), (-1.0, "r")):
            geoms.append(gs._cap(f"branch_{k}{tag}", br, (x, sy * r * 0.5, z), (x, sy * (r + bl), z + rise), rgba=gs.BARK, mass=8.0))
        z += sp
        k += 1
    for j, (dx, dy, dz, rr) in enumerate(((0.0, 0.0, 0.3, 0.6), (0.3, 0.35, 0.1, 0.45), (-0.25, -0.35, 0.15, 0.45), (0.1, -0.1, 0.55, 0.4))):
        geoms.append(Geom(f"crown_{j}", "sphere", (rr,), pos=(x + dx, dy, h + dz), mass=5.0, role="visual", rgba=LEAF))
    return Prop(Link("spider_tree", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def _branch_z(p, level, y):
    """The height of level ``level``'s branch axis at ``y`` (a branch rises as it goes out)."""
    r, bl = p["trunk_radius"], p["branch_length"]
    z0 = p["first_branch"] + level * p["spacing"]
    return z0 + p["branch_rise"] * max(0.0, abs(y) - r * 0.5) / (r * 0.5 + bl)


def _branch_bar(p, level, y, tip=None, side_n=(-1.0, 0.0, 0.0)):
    sy = 1.0 if y > 0 else -1.0
    a = gs._unit(np.array([0.0, sy * (p["trunk_radius"] * 0.5 + p["branch_length"]), p["branch_rise"]]))
    a = a if a[1] > 0 else -a                                              # the axis along +y either side: a foot's sides map onto it unmirrored
    return gs.bar((p["x"], y, _branch_z(p, level, y)), axis=a, n=side_n, r=p["branch_radius"], **({} if tip is None else {"tip": tip}))


TAIL_WRAP = 4                    #: the tail's last links wrapped round the trunk: past half a turn
TRUNK_BELOW = 0.12               #: the tail wraps the trunk this far below the feet's branches (clear of their roots) [m]


def _tail_links(kin):
    m = kin.model
    n = sum(1 for i in range(m.nbody) if m.body(i).name.startswith("tail_"))
    seg = float(-m.body_pos[m.body(f"tail_{n - 1}").id][0])
    return n, seg


def tail_wrap_trunk(kin, x, z, r_trunk, wind=1.0, off=0.0):
    """The tail's last ``TAIL_WRAP`` links' ends round the trunk at height ``z``: the first on the trunk's near side
    (the tail comes forward between the legs), the next round the left flank, the far side, on round - each a
    little lower (a helix down the bark, the tail not lying on itself). ``wind`` 0..1: how far round it is wound
    (0: all of them hanging on the near side, where no branch grows); ``off``: out from the bark [m].
    [(body, local, world)]."""
    n, seg = _tail_links(kin)
    R = r_trunk + 0.03 + off                                                 # the links' ends out by the tail's thickness and the chords' sag between them
    step = seg / (r_trunk + 0.03)                                           # each link's turn round the trunk, wound
    out = []
    for j in range(TAIL_WRAP + 1):
        th = j * step * wind
        drop = 0.02 * j * wind + 0.07 * j * (1.0 - wind)                    # unwound, the links hang one below the other
        pt = np.array([x - R * math.cos(th) - (1.0 - wind) * 0.02 * j, R * math.sin(th), z - drop])
        out.append((kin.body(f"tail_{n - 1 - TAIL_WRAP + j}"), np.array([-seg, 0.0, 0.0]), pt))
    return out


def tail_for_tree(kin, p, level_of):
    """``tail(q_from, k)`` for ``_climb_sequence``: the tail passed forward between the legs and wrapped round the
    trunk just below the feet's branches of level ``level_of(k)`` - a fifth hold. In a cycle, once the limbs have
    moved, it unwinds round the bark to the trunk's near side (no branch grows there: it never crosses one),
    slides up that side a level, and winds round again; on the mount it comes from where it is carried, onto the
    near side, and winds."""
    X, r = p["x"], p["trunk_radius"]
    z_of = lambda level: _branch_z(p, level, 0.0) - TRUNK_BELOW

    def make(q_from, k):
        if k is None:                                                      # the mount: the tail carried (low down it would reach the ground)
            return lambda u, base: []
        if k == 0:                                                         # the first cycle: from the carried tail to the first wrap, a level up
            z1 = z_of(level_of(1))
            first = tail_wrap_trunk(kin, X, z1, r, wind=0.0, off=0.03)
            kin.set(q_from)
            rest = [kin.world(b_, l_) for b_, l_, _ in first]                 # every wrapping link's end, as the tail is carried

            def fn(u, base):
                x, y, z, pt = base
                R, pc = gm.base_rot(pt), np.array([x, y, z])
                if u < 0.76:
                    return []
                if u < 0.88:                                               # the whole end of the tail steered to the near side, clear of the bark
                    f = mk.smooth((u - 0.76) / 0.12)
                    return [(b_, l_, (1 - f) * (R @ rs + pc) + f * w) for (b_, l_, w), rs in zip(first, rest)]
                g = mk.smooth((u - 0.88) / 0.12)
                return tail_wrap_trunk(kin, X, z1, r, wind=g, off=0.03 * (1 - g))
            return fn
        za, zb = z_of(level_of(k)), z_of(level_of(k + 1))

        def fn(u, base):
            if u <= 0.76:                                                  # held until both feet are down on their branches
                return tail_wrap_trunk(kin, X, za, r)
            f = (u - 0.76) / 0.24
            if f < 0.3:                                                    # unwinds round the bark
                g = mk.smooth(f / 0.3)
                return tail_wrap_trunk(kin, X, za, r, wind=1.0 - g, off=0.03 * g)
            if f < 0.6:                                                    # up the trunk's near side
                g = mk.smooth((f - 0.3) / 0.3)
                return tail_wrap_trunk(kin, X, (1 - g) * za + g * zb, r, wind=0.0, off=0.03)
            g = mk.smooth((f - 0.6) / 0.4)                                 # winds round again
            return tail_wrap_trunk(kin, X, zb, r, wind=g, off=0.03 * (1 - g))
        return fn
    return make


def scenario_spider_tree(kin, q_stand, p):
    """Up the tree: the hands on the branches two levels above the feet, the feet on the branches by the trunk, the
    tail wrapped round a branch beside the body (a fifth hold); each cycle the hands, then the feet, move up a
    level, then the tail lets go and wraps round the next branch up. Without a tail (the gibbon) the same climb."""
    xb = p["x"] - p["trunk_radius"] - 0.02
    hold = lambda k: {"L_hand": _branch_bar(p, k + 2, 0.25), "R_hand": _branch_bar(p, k + 2, -0.25),
                      "L": _branch_bar(p, k, 0.2, tip=0.2), "R": _branch_bar(p, k, -0.2, tip=0.2)}   # out from the trunk: the toes and the big toe clear of the bark where the branch grows
    base = lambda k: np.array([xb - 0.3, 0.0, _branch_z(p, k, 0.2) + 0.35])
    n = min(int(p["cycles"]), int((p["height"] - 1.2 - p["first_branch"]) / p["spacing"]))
    tail = tail_for_tree(kin, p, lambda k: k) if tail_tip(kin) is not None else None
    return gs._climb_sequence(kin, q_stand, hold, base, n, stop_x=xb - 0.25, tail=tail, duration=3.8 if tail else None)


gs.DEFAULTS["tree_spider"] = SPIDER_TREE
gs.BUILDERS["tree_spider"] = spider_tree_prop
gs.SCENARIOS["tree_spider"] = scenario_spider_tree

# ---- the heist: up the tree beside the house, the pane broken with the hammer from the backpack, in at the window, the banana
HEIST = {**SPIDER_TREE, "x": 2.2, "height": 3.2, "cycles": 2, "left_levels": 3, "wall_gap": 0.22, "window_y": 0.7, "table_x": 2.6,
         "table_y": 0.0}
BACKPACK = {"half": (0.04, 0.09, 0.11), "on_thorax": (-0.1, 0.0, 0.12)}     #: the pack's half sizes and its place on the thorax's back [m]
HAMMER_IN_PACK = (np.array([-0.1, 0.05, 0.18]), 0.35)                      #: the hammer's handle root in the thorax's frame, its lean back from upright [rad]
HEIST_JUMP = {**HEIST, "entry": "jump"}                                  #: the same heist, the window reached by a jump
HEIST["window_y"] = 1.0                                                   #: the crossing: the window one step further along, clear of the trunk


STUB = 0.16                      #: the cut-back left branches stand out this far past the bark [m]


def heist_tree_prop(p):
    """The spider monkey's tree beside the house: the left branches (on the window's side) cut back above the sill."""
    pr = spider_tree_prop(p)
    keep = []
    for g in pr.root.geoms:
        if g.name.startswith("branch_") and g.name.endswith("l") and int(g.name[7:-1]) >= p["left_levels"]:
            k = int(g.name[7:-1])                                          # cut back to a stub: a handhold beside the trunk, clear of the window
            r, bl, z0 = p["trunk_radius"], p["branch_length"], p["first_branch"] + k * p["spacing"]
            y1 = r + STUB
            g = gs._cap(g.name, p["branch_radius"], (p["x"], r * 0.5, z0), (p["x"], y1, _branch_z(p, k, y1)), rgba=gs.BARK, mass=3.0)
        keep.append(g)
    pr.root.geoms = keep
    return pr


def _heist_window(p):
    return {**gs.WINDOW, "banana": 0 if p.get("fire") else 1, "x": p["x"] + p["wall_gap"], "y": p["window_y"], "glass": 1, "sash": 1, "table_x": p["table_x"], "table_y": p["table_y"]}


SASH_SHOVE = 0.16                #: how far the hammer shoves the sash open [rad]; it swings on to ``SASH_OPEN``
SASH_OPEN = 1.5


def sash_point(w, phi, z_frac=0.5, y_in=0.0):
    """A point on the sash's free stile's outer face with the sash open ``phi`` (it swings about the left jamb, into the room)."""
    t, ws = w["wall_thickness"], w["width"] - 0.04
    hinge = np.array([w["x"] - t / 2 + 0.03, w["y"] + w["width"] / 2 - 0.02, w["sill"] + 0.02])
    c, s_ = math.cos(phi), math.sin(phi)
    local = np.array([-0.016, -ws + 0.02 + y_in, (w["height"] - 0.04) * z_frac])
    return hinge + np.array([c * local[0] - s_ * local[1], s_ * local[0] + c * local[1], local[2]])


def backpack_prop(p):
    """The backpack: a free body the player keeps on the thorax's back (``ScenePlayer``), the hammer's handle in it."""
    hx, hy, hz = BACKPACK["half"]
    link = Link("backpack", pos=(0.0, 0.0, 3.0), geoms=[gs._box("backpack_body", (hx, hy, hz), (0.0, 0.0, 0.0), rgba=(0.3, 0.38, 0.22, 1.0), mass=0.4)])
    return Prop(link, free=True)


def heist_props(p):
    w = _heist_window(p)
    return (heist_tree_prop(p), gs.window_prop(w), gs.hammer_prop({"hammer_x": 0.0, "hammer_y": 0.0}), backpack_prop(p))


def _trunk_hold(p, level, side="L", ang=math.radians(40.0)):
    """The trunk's flank as a vertical hold for a hand, at a level's height."""
    sy = 1.0 if side == "L" else -1.0
    n = np.array([-math.cos(ang), sy * math.sin(ang), 0.0])
    z = _branch_z(p, level, 0.0) + 0.05
    return gs.bar(np.array([p["x"], 0.0, z]) + (p["trunk_radius"] + gs.BAR_R + 0.005) * n, axis=(-n[1], n[0], 0.0), n=n, r=gs.BAR_R)   # across the bark: the fingers lie up the trunk, not round into it   # the palm on the bark, not in it


def scenario_heist(kin, q_stand, p):
    """The walk to the tree beside the house and the climb (the right hand on the branches, the left round the trunk
    - the branches on the window's side are cut back - the feet's toes round the branches, the tail round the trunk);
    at the window's height, held by the tail, both feet and the right hand, the left hand takes the hammer from the
    backpack over the shoulder, winds up and smashes the pane, lets the hammer fall; the crossing from the tree onto the
    sill, a crouch in the opening, a hop down into the room, the walk to the banana."""
    w = _heist_window(p)
    xb = p["x"] - p["trunk_radius"] - 0.02
    hold = lambda k: {"L_hand": _branch_bar(p, k + 2, p["trunk_radius"] + STUB * 0.6), "R_hand": _branch_bar(p, k + 2, -0.25),
                      "L": _branch_bar(p, k, 0.2, tip=0.2), "R": _branch_bar(p, k, -0.2, tip=0.2)}
    base = lambda k: np.array([xb - 0.3, 0.0, _branch_z(p, k, 0.2) + 0.35])
    n = int(p["cycles"])
    tail = tail_for_tree(kin, p, lambda k: k) if tail_tip(kin) is not None else None
    segs = gs._climb_sequence(kin, q_stand, hold, base, n, stop_x=xb - 0.25, tail=tail, duration=3.8, trials=gs.VERTICAL_TRIALS)
    q = q_stand.copy()
    for sg in segs:
        if sg[0] == "motion":
            q = gt._last(kin, sg[1], q)
    b = base(n)
    pitch = gs.CLIMB_PITCH
    R_b = gm.base_rot(pitch)
    h_end = hold(n)
    # the hand's grip at the handle's middle, the hammer's frame in the hand and the world as it sits in the pack
    kin.set(q)
    th = kin.body("thorax")
    R_th, p_th = R_b @ kin.data.xmat[th].reshape(3, 3), R_b @ kin.data.xpos[th] + b
    lean = HAMMER_IN_PACK[1]
    R_pack = np.array([[math.cos(lean), 0.0, -math.sin(lean)], [0.0, 1.0, 0.0], [math.sin(lean), 0.0, math.cos(lean)]])   # handle up, leant back
    H = gs.HAMMER
    root_w = R_th @ HAMMER_IN_PACK[0] + p_th
    R_ham_w = R_th @ R_pack
    handle_axis = R_ham_w[:, 2]
    grip_c = root_w + handle_axis * H["handle"] * 0.4
    pick_hold = gs.bar(grip_c, axis=handle_axis, n=gs._unit(np.cross(handle_axis, np.array([0.0, 1.0, 0.0]))), r=H["handle_r"], wrist_dir=-gs._unit(R_th[:, 0] * 0.0 + np.array([0.3, -0.2, 0.9])))
    # the blow: the glass's middle-left, the head through it
    glass_point = np.array([w["x"] - w["wall_thickness"] / 2 * 0.0, w["y"] - 0.35, w["sill"] + w["height"] * 0.4])
    strike_axis = gs._unit(np.array([0.7, 0.55, -0.45]))
    strike = (glass_point - 0.31 * strike_axis, gs.hand_frame_on(gs.bar((0, 0, 0), axis=strike_axis, n=(-1.0, 0.0, 0.0), r=H["handle_r"], wrist_dir=(-0.5, -0.2, 0.84)), "L"))
    wind = (p_th + R_th @ np.array([-0.05, 0.25, 0.3]), gs.hand_frame_on(gs.bar((0, 0, 0), axis=(-0.3, 0.0, 0.95), n=(0.0, -1.0, 0.0)), "L"))
    planted = (gs.foot_targets(kin, "L", h_end["L"]) + gs.foot_targets(kin, "R", h_end["R"]) + gs.hand_targets(kin, "R", h_end["R_hand"])
               + (tail_wrap_trunk(kin, p["x"], _branch_z(p, n, 0.0) - TRUNK_BELOW, p["trunk_radius"]) if tail else []))
    kin.set(q)
    l_start = [(bd, l, R_b @ kin.world(bd, l) + b) for bd, l, _ in gs.hand_targets(kin, "L", h_end["L_hand"])]
    pick_end = gs.hand_targets(kin, "L", pick_hold)
    holdbase = lambda u: (b[0], b[1], b[2], pitch)

    def keep(u, q_):                                                       # the right fist, the toes round their branches, the spine as on the climb
        sr.set_curl(q_, kin, "R", gs.CURL_CLOSED)
        for sd in ("L", "R"):
            gs.toe_wrap(kin, q_, sd, h_end[sd])
        return gs._spine_extra(kin, gs.CLIMB_SPINE)(1.0, q_)

    def t_pick(u):                                                         # off the trunk, over the shoulder, onto the handle
        if u < 0.7:
            f = mk.smooth(u / 0.7)
            c = (1 - f) * l_start[0][2] + f * pick_end[0][2] + np.array([-0.15, 0.1, 0.2]) * math.sin(math.pi * f)
            return planted + [(pick_end[0][0], pick_end[0][1], c)]
        return planted + pick_end

    def x_pick(u, q_):
        q_ = keep(u, q_)
        c = gs.CURL_CLOSED + (gs.CURL_OPEN - gs.CURL_CLOSED) * mk.smooth(min(1.0, u / 0.3)) if u < 0.75 else gs.CURL_OPEN + (kt_lever() - gs.CURL_OPEN) * mk.smooth((u - 0.75) / 0.25)
        return sr.set_curl(q_, kin, "L", c)

    pick = gt.contact_motion(kin, "pick_up_the_hammer", 2.2, t_pick, holdbase, q_init=q, extra=x_pick, reg=gs.IK_REG, iters=gs.IK_ITERS,
                             q_ref=gs.climb_ref(kin, q), source="gibbon_scenarios.heist: the hammer from the backpack over the shoulder")
    q1 = gt._last(kin, pick, q)
    Rh, ph = gs._hand_pose(kin, q1, b, pitch, "L")
    R_h2m = Rh.T @ R_ham_w
    p_h2m = Rh.T @ (root_w - ph)
    kin.set(q)
    pick.params = {**(pick.params or {}), "hammer_T": (R_h2m.tolist(), p_h2m.tolist()),
                   "hammer_carry": ("thorax", R_pack.tolist(), HAMMER_IN_PACK[0].tolist()), "hand": "L_hand"}
    grip_frame = (pick_end[0][2], gs.hand_frame_on(pick_hold, "L"))

    def hold_at(u):
        if u < 0.4:
            f = mk.smooth(u / 0.4)
            return (1 - f) * grip_frame[0] + f * wind[0], gs._slerp(grip_frame[1], wind[1], f)
        if u < 0.65:
            f = mk.smooth((u - 0.4) / 0.25)
            return (1 - f) * wind[0] + f * strike[0], gs._slerp(wind[1], strike[1], f)
        if u < 0.75:
            return strike
        f = mk.smooth((u - 0.75) / 0.25)
        return (1 - f) * strike[0] + f * wind[0], gs._slerp(strike[1], wind[1], f)

    LEAN = np.array(p.get("lean", (0.14, 0.08, 0.06)))                      # the pelvis swung toward the pane, hung on the tail and feet

    def leanbase(u0, u1):
        def f(u):
            s_ = mk.smooth(min(1.0, u / u0)) if u < u0 else (1.0 if u < u1 else 1.0 - mk.smooth((u - u1) / (1.0 - u1)))
            c = b + s_ * LEAN
            return (c[0], c[1], c[2], pitch)
        return f

    x_fist = lambda u, q_: sr.set_curl(keep(u, q_), kin, "L", kt_lever())
    limbs_ = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]

    def reach_err(head, axis, wd):
        """How near the left hand gets the hammer's head to ``head`` (the handle along ``axis``) with the body held."""
        fr = gs.hand_frame_on(gs.bar((0, 0, 0), axis=axis, n=(-1.0, 0.0, 0.0), r=H["handle_r"], wrist_dir=wd), "L")
        tg = [(bd, l, R_b.T @ (np.asarray(wp) - b - LEAN)) for bd, l, wp in planted + gs._hand_at(kin, "L", head - 0.31 * axis, fr)]
        qs, err = kin.ik(tg, q_init=q1, q_ref=gs.climb_ref(kin, q1), joints=limbs_, reg=3e-4, iters=300)
        if gt.SOLID is not None:                                           # a blow whose arm would pass through the trunk is no blow
            err += gt.SOLID.worst(qs, b + LEAN, pitch)[0]
        return err, (head - 0.31 * axis, fr)

    def best_blow(heads):
        """The blow the arm reaches best: over candidate points, the handle's directions and the wrist's."""
        best = None
        for head in heads:
            for ax in ((0.7, 0.55, -0.45), (0.85, 0.3, -0.43), (0.9, 0.0, -0.43), (0.75, 0.2, -0.63), (0.95, 0.3, 0.0), (0.6, 0.6, -0.5)):
                for wd in ((-0.5, -0.2, 0.84), (-0.7, 0.0, 0.7), (-0.3, -0.4, 0.86), (-0.8, -0.3, 0.5)):
                    err, frame = reach_err(np.asarray(head), gs._unit(np.array(ax)), gs._unit(np.array(wd)))
                    if best is None or err < best[0]:
                        best = (err, frame, np.asarray(head), gs._unit(np.array(ax)))
        return best

    x_pane = w["x"] - w["wall_thickness"] / 2 + 0.03                          # the pane in the sash, near the wall's outer face
    heads = [np.array([x_pane, y, z]) for y in (w["y"] - 0.5, w["y"] - 0.4, w["y"] - 0.3) for z in (w["sill"] + 0.35, w["sill"] + 0.55, w["sill"] + 0.75)]
    _err, strike, glass_point, strike_axis = best_blow(heads)
    print(f"heist: blow reach {_err:.3f} m at {glass_point.round(2)}")
    strike = (strike[0], strike[1])
    smash = gt.contact_motion(kin, "smash_the_glass", 1.8, lambda u: planted + gs._hand_at(kin, "L", *hold_at(u)), leanbase(0.6, 0.8), q_init=q1,
                              extra=x_fist, reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=gs.climb_ref(kin, q1),
                              source="gibbon_scenarios.heist: a wind-up and the blow through the pane")
    smash.params = {**(smash.params or {}), "strike_u": 0.65, "impact": glass_point.tolist()}
    q2 = gt._last(kin, smash, q1)

    # the shove: the hammer's head against the sash's free stile, pushed into the room; the sash swings on from there
    spots = [(zf, yi) for zf in (0.03, 0.25, 0.4) for yi in (0.0, 0.15, 0.3)]   # the free stile, or the rails near it
    shove_best = best_blow([sash_point(w, 0.0, *sp) for sp in spots])
    shove_axis, shove_frame = shove_best[3], shove_best[1][1]
    shove_spot = [sp for sp in spots if np.allclose(sash_point(w, 0.0, *sp), shove_best[2])][0]
    print(f"heist: shove reach {shove_best[0]:.3f} m at {shove_spot}")

    def head_at(phi):
        tgt = sash_point(w, phi, *shove_spot)
        return (tgt - 0.31 * shove_axis, shove_frame)

    def push_at(u):
        if u < 0.35:
            f = mk.smooth(u / 0.35)
            c0, R0 = wind
            c1, R1 = head_at(0.0)
            return (1 - f) * c0 + f * c1, gs._slerp(R0, R1, f)
        if u < 0.75:
            return head_at(SASH_SHOVE * mk.smooth((u - 0.35) / 0.4))
        f = mk.smooth((u - 0.75) / 0.25)
        c0, R0 = head_at(SASH_SHOVE)
        return (1 - f) * c0 + f * wind[0], gs._slerp(R0, wind[1], f)

    if gt.SOLID is not None:
        gt.SOLID.scene = lambda u: {"sash_hinge": SASH_SHOVE * mk.smooth(min(1.0, max(0.0, (u - 0.35) / 0.4)))}
    try:
        push = gt.contact_motion(kin, "push_the_window", 1.6, lambda u: planted + gs._hand_at(kin, "L", *push_at(u)), leanbase(0.3, 0.78), q_init=q2,
                                 extra=x_fist, reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=gs.climb_ref(kin, q2),
                                 source="gibbon_scenarios.heist: the hammer's head shoves the sash open into the room")
    finally:
        if gt.SOLID is not None:
            gt.SOLID.scene = None
    push.params = {**(push.params or {}), "shove_u": (0.35, 0.75), "shove_rad": SASH_SHOVE, "open_rad": SASH_OPEN}
    q2 = gt._last(kin, push, q2)

    def x_drop(u, q_):
        q_ = keep(u, q_)
        return sr.set_curl(q_, kin, "L", kt_lever() + (gs.CURL_OPEN - kt_lever()) * mk.smooth(min(1.0, u / 0.3)))

    drop = gt.contact_motion(kin, "drop_the_hammer", 1.0, lambda u: planted + gs._hand_at(kin, "L", *wind), holdbase, q_init=q2, extra=x_drop,
                             reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=gs.climb_ref(kin, q2), source="gibbon_scenarios.heist: the fist opens, the hammer falls")
    drop.params = {**(drop.params or {}), "release_u": 0.3}
    q3 = gt._last(kin, drop, q2)
    if p.get("entry") == "jump":                                          # the jump: a ballistic leap onto the sill
        # the leap: from the tree onto the sill, a crouch in the opening (joint keyposes on a base track)
        top = w["sill"] + 0.06
        crouch = np.array([w["x"] - 0.1, w["y"], top + 0.3])
        feet_sill = {sd: gs.point(np.array([w["x"] - 0.02, w["y"] + (0.1 if sd == "L" else -0.1), top]), pitch=0.0) for sd in ("L", "R")}
        limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
        Rc = gm.base_rot(0.35)
        T = [(bd, l, Rc.T @ (np.asarray(wp) - crouch)) for bd, l, wp in gs.foot_targets(kin, "L", feet_sill["L"]) + gs.foot_targets(kin, "R", feet_sill["R"])]
        q_crouch, _ = kin.ik(T, q_init=hang_ref(kin, q_stand), q_ref=hang_ref(kin, q_stand), joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
        for sd in ("L", "R"):
            sr.set_curl(q_crouch, kin, sd, gs.CURL_OPEN)
            sr.set_toe_curl(q_crouch, kin, sd, 0.0)
        tuck = q3.copy()
        for sd in ("L", "R"):
            tuck[kin.index[f"{sd}_knee"]] = min(tuck[kin.index[f"{sd}_knee"]] + 0.5, kin.hi[kin.index[f"{sd}_knee"]])
            tuck[kin.index[f"{sd}_hip_pitch"]] = max(tuck[kin.index[f"{sd}_hip_pitch"]] - 0.5, kin.lo[kin.index[f"{sd}_hip_pitch"]])
            sr.set_curl(tuck, kin, sd, gs.CURL_OPEN)
            sr.set_toe_curl(tuck, kin, sd, 0.0)
        for j in kin.actuated:                                                 # the arms come in off the branches first: none swept through the wall
            if any(j.startswith(f"{sd}_{k}") for sd in ("L", "R") for k in ("shoulder", "elbow", "wrist")):
                tuck[kin.index[j]] = q_crouch[kin.index[j]] = q_stand[kin.index[j]] if "elbow" in j else min(max(0.0, kin.lo[kin.index[j]]), kin.hi[kin.index[j]])   # upper arms down, forearms folded: under the lintel, over the sill

        def leap_base(u):
            f = mk.smooth(u)
            pc = (1 - f) * b + f * crouch + np.array([0.0, 0.0, 0.3 * math.sin(math.pi * u)])
            return pc[0], pc[1], pc[2], (1 - f) * pitch + f * 0.35

        arms_in = q3.copy()                                                    # first the arms off the branch and in, the body still on the tree
        for j in kin.actuated:
            if any(j.startswith(f"{sd}_{k}") for sd in ("L", "R") for k in ("shoulder", "elbow", "wrist", "finger", "thumb")):
                arms_in[kin.index[j]] = tuck[kin.index[j]]

        # the leap as a jump: arms in and a crouch off the trunk's side, a push-off along the launch line, a ballistic
        # flight (gravity only: the pelvis on a parabola, the horizontal speed constant), the landing taken in the legs
        G = 9.81
        T_IN, T_PUSH, T_LAND, DIP, CLEAR = 0.6, 0.22, 0.3, 0.07, 0.12
        p0 = np.array([b[0] - 0.22, b[1] + 0.3, b[2] - 0.06])                  # back off the tree and out to the window's side, sunk into the push
        p1 = crouch.copy()
        launch = p0.copy()
        for _ in range(6):                                                     # the launch point: where the push (uniform acceleration from rest) ends
            apex = max(launch[2], p1[2]) + CLEAR
            t_up = math.sqrt(2.0 * (apex - launch[2]) / G)
            t_dn = math.sqrt(2.0 * (apex - p1[2]) / G)
            T_FLY = t_up + t_dn
            v = np.array([(p1[0] - launch[0]) / T_FLY, (p1[1] - launch[1]) / T_FLY, G * t_up])
            launch = p0 + v * T_PUSH / 2.0
        D = T_IN + T_PUSH + T_FLY + T_LAND
        print(f"heist: leap {np.linalg.norm(p1[:2] - launch[:2]):.2f} m, take-off {np.linalg.norm(v):.2f} m/s, flight {T_FLY:.2f} s, apex {apex - launch[2]:.2f} m over take-off")
        push_ext = arms_in.copy()                                              # the legs straightened at take-off
        crouch_in = arms_in.copy()
        for sd in ("L", "R"):
            for jn, dq in (("knee", 0.5), ("hip_pitch", -0.4), ("ankle_pitch", 0.3)):
                j = f"{sd}_{jn}"
                if j in kin.index:
                    i = kin.index[j]
                    crouch_in[i] = min(max(arms_in[i] + dq, kin.lo[i]), kin.hi[i])
                    push_ext[i] = min(max(arms_in[i] - 0.6 * dq, kin.lo[i]), kin.hi[i])
        absorb = q_crouch.copy()
        for sd in ("L", "R"):
            for jn, dq in (("knee", 0.35), ("hip_pitch", -0.3)):
                j = f"{sd}_{jn}"
                if j in kin.index:
                    i = kin.index[j]
                    absorb[i] = min(max(q_crouch[i] + dq, kin.lo[i]), kin.hi[i])

        def leap_base(u):
            t = u * D
            if t < T_IN:                                                       # off the tree, round to the trunk's side, sinking
                fx_, fy_ = mk.smooth(min(1.0, t / (0.5 * T_IN))), mk.smooth(max(0.0, (t - 0.5 * T_IN) / (0.5 * T_IN)))   # back off the trunk, then round it
                pc, pt = np.array([b[0] + fx_ * (p0[0] - b[0]), b[1] + fy_ * (p0[1] - b[1]), b[2] + fx_ * (p0[2] - b[2])]), pitch
            elif t < T_IN + T_PUSH:                                            # the push: from rest to take-off speed
                tt = t - T_IN
                pc, pt = p0 + v * tt * tt / (2.0 * T_PUSH), pitch
            elif t < T_IN + T_PUSH + T_FLY:                                    # the flight: a parabola
                tt = t - T_IN - T_PUSH
                pc = launch + np.array([v[0] * tt, v[1] * tt, v[2] * tt - 0.5 * G * tt * tt])
                pt = pitch + (0.35 - pitch) * mk.smooth(tt / T_FLY)
            else:                                                              # the landing: the legs give and come back
                tt = (t - T_IN - T_PUSH - T_FLY) / T_LAND
                pc, pt = p1 - np.array([0.0, 0.0, DIP * math.sin(math.pi * min(1.0, tt))]), 0.35
            return pc[0], pc[1], pc[2], pt

        for qk in (q_crouch, absorb):                                          # landing: the arms forward and up for balance, the hands clear of the sill
            for sd in ("L", "R"):
                i = kin.index[f"{sd}_shoulder_pitch"]
                qk[i] = min(max(-0.6, kin.lo[i]), kin.hi[i])
        for qk in (push_ext, tuck, q_crouch, absorb):                          # the tail off the trunk at the push, raised and curled over the back in the air
            for j in kin.actuated:
                if j.startswith("tail"):
                    qk[kin.index[j]] = q_stand[kin.index[j]]
        k1, k2, k3 = T_IN / D, (T_IN + T_PUSH) / D, (T_IN + T_PUSH + T_FLY) / D
        leap = gs._joint_track(kin, "leap_to_the_sill", D, [(0.0, q3), (k1, crouch_in), (k2, push_ext), ((k2 + k3) / 2, tuck), (k3, q_crouch),
                                                            (k3 + 0.4 * (1 - k3), absorb), (1.0, q_crouch)], leap_base, frames=int(D * 40),
                               source="gibbon_scenarios.heist: the jump from the tree onto the sill - push-off, ballistic flight, landing")
        steps, q_sill = [leap], q_crouch
    else:                                                                  # the crossing: limb by limb, the tail round the trunk
        # the crossing: from the tree onto the sill one limb at a time, never fewer than three holds; the tail stays round
        # the trunk until both hands and both feet are on the sill, the body shifted slowly (quasi-static) between them
        top = w["sill"] + 0.06
        y_c = w["y"] - w["width"] / 2 + 0.17                                   # the crouch at the sill's tree-side end
        crouch = np.array([w["x"] - 0.1, y_c, top + 0.3])
        feet_sill = {sd: gs.point(np.array([w["x"] - 0.02, y_c + (0.1 if sd == "L" else -0.1), top]), pitch=0.0) for sd in ("L", "R")}
        xf, y_jamb = w["x"] - w["wall_thickness"] / 2, w["y"] - w["width"] / 2   # the wall's face, the window's tree-side jamb
        hands_sill = {sd: gs.bar(np.array([xf - 0.02, y_jamb + 0.03, top + (0.45 if sd == "L" else 0.2)]), axis=(0.0, 0.0, 1.0), n=(-1.0, 0.0, 0.0), r=0.03)
                      for sd in ("L", "R")}                                    # the hands round the jamb's edge, one above the other, beside the tree
        limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
        Rc = gm.base_rot(0.35)
        T = [(bd, l, Rc.T @ (np.asarray(wp) - crouch)) for bd, l, wp in gs.foot_targets(kin, "L", feet_sill["L"]) + gs.foot_targets(kin, "R", feet_sill["R"])]
        q_crouch, _ = kin.ik(T, q_init=hang_ref(kin, q_stand), q_ref=hang_ref(kin, q_stand), joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
        for sd in ("L", "R"):
            sr.set_curl(q_crouch, kin, sd, gs.CURL_OPEN)
            sr.set_toe_curl(q_crouch, kin, sd, 0.0)

        T_feet = {sd: gs.foot_targets(kin, sd, h_end[sd]) for sd in ("L", "R")}
        T_tail = tail_wrap_trunk(kin, p["x"], _branch_z(p, n, 0.0) - TRUNK_BELOW, p["trunk_radius"]) if tail else []
        T_rh = gs.hand_targets(kin, "R", h_end["R_hand"])
        S_feet = {sd: gs.foot_targets(kin, sd, feet_sill[sd]) for sd in ("L", "R")}
        S_hand = {sd: gs.hand_targets(kin, sd, hands_sill[sd]) for sd in ("L", "R")}

        def world_now(q_, base_, pt_, tg):
            kin.set(q_)
            R_ = gm.base_rot(pt_)
            return [(bd, l, R_ @ kin.world(bd, l) + base_) for bd, l, _ in tg]

        def mover(start, end, lift=0.08, out=np.zeros(3)):
            """A limb lifted off its hold, carried on an arc (up, and ``out`` from the wall) and set on the next."""
            def f(u):
                g_ = mk.smooth(min(1.0, max(0.0, (u - 0.1) / 0.75)))
                arc = math.sin(math.pi * g_) * (np.array([0.0, 0.0, lift]) + out)
                return [(e[0], e[1], (1 - g_) * np.asarray(s_[2]) + g_ * np.asarray(e[2]) + arc) for s_, e in zip(start, end)]
            return f

        def shift(c0, c1, p0_, p1_):
            return lambda u: tuple((1 - mk.smooth(u)) * c0 + mk.smooth(u) * c1) + ((1 - mk.smooth(u)) * p0_ + mk.smooth(u) * p1_,)

        def x_hold(feet_on_branch, tail_on, toes_open=()):
            def f(u, q_):
                sr.set_curl(q_, kin, "R", gs.CURL_CLOSED)
                sr.set_curl(q_, kin, "L", gs.CURL_CLOSED)
                for sd in feet_on_branch:
                    gs.toe_wrap(kin, q_, sd, h_end[sd])
                for sd in toes_open:
                    sr.set_toe_curl(q_, kin, sd, 0.0)
                if not tail_on:                                                # the tail unwound: raised and curled over the back, slowly
                    for j in kin.actuated:
                        if j.startswith("tail"):
                            i = kin.index[j]
                            q_[i] = q_[i] + (q_stand[i] - q_[i]) * mk.smooth(u)
                return gs._spine_extra(kin, gs.CLIMB_SPINE)(1.0, q_)
            return f

        steps, qc, bc, pc_ = [], q3, np.array(b, dtype=float), pitch
        if gt.SOLID is not None:
            gt.SOLID.ignore = {"glass"}                                        # the pane is broken: only its frame and the sash are left
        # round the trunk first (the right hand to a left stub, the right foot to a lower left branch: all of the body on
        # the window's side), then across to the window: left hand on the jamb, left foot, right hand, right foot on the sill
        stub_r = _branch_bar(p, n + 1, p["trunk_radius"] + STUB * 0.6)
        branch_rf = _branch_bar(p, n - 1, 0.2, tip=0.2)
        state = {"L_hand": None, "R_hand": h_end["R_hand"], "L": h_end["L"], "R": h_end["R"]}   # the left hand is free (the hammer dropped)
        plan = [  # name, duration, limb, its new hold, pitch at the end
            ("left_hand_to_the_jamb", 1.6, "L_hand", hands_sill["L"], pitch),
            ("right_hand_round_the_trunk", 1.8, "R_hand", stub_r, pitch),
            ("right_foot_round_the_trunk", 1.8, "R", branch_rf, pitch),
            ("left_foot_to_the_sill", 1.8, "L", feet_sill["L"], pitch),
            ("right_hand_to_the_jamb", 1.6, "R_hand", hands_sill["R"], pitch),
            ("right_foot_to_the_sill", 1.8, "R", feet_sill["R"], 0.5 * (pitch + 0.35)),
        ]

        def targets_of(limb, hold_):
            return gs.hand_targets(kin, limb[0], hold_) if limb.endswith("hand") else gs.foot_targets(kin, limb, hold_)

        def x_state(st, tail_on, toes_open=()):
            def f(u, q_):
                sr.set_curl(q_, kin, "R", gs.CURL_CLOSED)
                sr.set_curl(q_, kin, "L", gs.CURL_CLOSED)
                for sd in ("L", "R"):
                    if st[sd] is not None and st[sd]["kind"] == "bar" and sd not in toes_open:
                        gs.toe_wrap(kin, q_, sd, st[sd])
                    else:
                        sr.set_toe_curl(q_, kin, sd, 0.0)
                if not tail_on:
                    for j in kin.actuated:
                        if j.startswith("tail"):
                            i = kin.index[j]
                            q_[i] = q_[i] + (q_stand[i] - q_[i]) * mk.smooth(u)
                return gs._spine_extra(kin, gs.CLIMB_SPINE)(1.0, q_)
            return f

        def best_base(tg, q0, around, pt):
            """The pelvis where every hold of ``tg`` is reached and nothing is inside anything: a search round ``around``."""
            Rb = gm.base_rot(pt)
            best = None
            for dx in (-0.08, -0.04, 0.0, 0.04, 0.08, 0.12):
                for dy in (-0.08, -0.04, 0.0, 0.04, 0.08, 0.12):
                    for dz in (-0.06, 0.0, 0.06):
                        c = np.asarray(around) + np.array([dx, dy, dz])
                        tl = [(bd, l, Rb.T @ (np.asarray(wp) - c)) for bd, l, wp in tg]
                        qs, err = kin.ik(tl, q_init=q0, q_ref=gs.climb_ref(kin, q0), joints=limbs, reg=3e-4, iters=250)
                        pen = gt.SOLID.worst(qs, c, pt)[0] if gt.SOLID is not None else 0.0
                        score = err + 2.0 * max(0.0, pen - 0.01) + 0.05 * math.hypot(dx, dy)
                        if best is None or score < best[0]:
                            best = (score, c, err, pen)
            print(f"heist: base search {best[1].round(3)} reach {best[2]:.3f} pen {best[3]:.3f}")
            return best[1]

        for name, dur, limb, new_hold, p_to in plan:
            kept = [t_ for k_, h_ in state.items() if k_ != limb and h_ is not None for t_ in targets_of(k_, h_)] + T_tail
            end = targets_of(limb, new_hold)
            b_to = best_base(kept + end, qc, bc + np.array([0.03, 0.08, 0.0]), p_to)
            start = world_now(qc, bc, pc_, end)                                # the same points of the limb, where they are now
            mv = mover(start, end, lift=0.06 if limb.endswith("hand") else 0.1, out=np.array([-0.22 if "round" in name else -0.05, 0.0, 0.0]))   # round the trunk, not through it
            st_now = {**state, limb: None}
            m_ = gt.contact_motion(kin, name, dur, (lambda u, kept=kept, mv=mv: kept + mv(u)), shift(bc, b_to, pc_, p_to), q_init=qc,
                                   extra=x_state(st_now, True), reg=gs.IK_REG, iters=gs.IK_ITERS,
                                   q_ref=gs.climb_ref(kin, qc), source=f"gibbon_scenarios.heist: {name.replace('_', ' ')} - three holds kept, the tail round the trunk")
            steps.append(m_)
            state[limb] = new_hold
            qc, bc, pc_ = gt._last(kin, m_, qc), np.array(b_to, dtype=float), p_to
        T_settle = S_feet["L"] + S_feet["R"]                                  # the feet on the sill; the hands let go and fold in, the tail unwinds
        arm_j = [j for j in kin.actuated if any(j.startswith(f"{sd}_{k}") for sd in ("L", "R") for k in ("shoulder", "elbow", "wrist"))]
        q_fold = {j: (q_stand[kin.index[j]] if "elbow" in j else min(max(0.0, kin.lo[kin.index[j]]), kin.hi[kin.index[j]])) for j in arm_j}
        q_arm0 = {j: qc[kin.index[j]] for j in arm_j}
        x_settle = x_state(state, False, toes_open=("L", "R"))

        def x_tail_off(u, q_):
            q_ = x_settle(u, q_)
            f = mk.smooth(min(1.0, u / 0.6))                                   # off the jamb (out from the wall first: the base moves in), folded in
            for j in arm_j:
                q_[kin.index[j]] = (1 - f) * q_arm0[j] + f * q_fold[j]
            return q_
        mv = lambda u: []
        m_ = gt.contact_motion(kin, "tail_off_the_trunk", 1.6, lambda u: T_settle + mv(u), shift(bc, crouch + np.array([0.0, 0.15, 0.0]), pc_, 0.35),
                               q_init=qc, extra=x_tail_off, reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_crouch,
                               source="gibbon_scenarios.heist: on the sill on four, the tail unwound last and the body settled in the opening")
        steps.append(m_)
        q_sill = gt._last(kin, m_, qc)
        crouch = crouch + np.array([0.0, 0.15, 0.0])                       # the body settled over the sill, away from the jamb
        if gt.SOLID is not None:
            gt.SOLID.ignore = set()
    x_land = w["x"] + w["wall_thickness"] / 2 + 0.45
    x_clear = w["x"] + w["wall_thickness"] / 2 + 0.15
    fold = q_crouch.copy()
    for j in kin.actuated:                                                 # the arms as they are on the sill, folded in
        if any(j.startswith(f"{sd}_{k}") for sd in ("L", "R") for k in ("shoulder", "elbow", "wrist")):
            fold[kin.index[j]] = q_sill[kin.index[j]]
    for sd in ("L", "R"):
        fold[kin.index[f"{sd}_knee"]] = min(q_crouch[kin.index[f"{sd}_knee"]] + 0.25, kin.hi[kin.index[f"{sd}_knee"]])
        fold[kin.index[f"{sd}_hip_pitch"]] = max(q_crouch[kin.index[f"{sd}_hip_pitch"]] - 0.25, kin.lo[kin.index[f"{sd}_hip_pitch"]])

    def hop_base(u):
        xb_ = crouch[0] + mk.smooth(u) * (x_land - crouch[0])
        fz = mk.smooth(max(0.0, min(1.0, (xb_ - x_clear) / (x_land - x_clear))))
        return xb_, crouch[1] + mk.smooth(min(1.0, u / 0.35)) * (w["y"] - crouch[1]), (1 - fz) * crouch[2] + fz * gm.STANCE["h"] + 0.04 * math.sin(math.pi * fz), (1 - fz) * 0.35 + fz * gm.STANCE["pitch"]

    hop = gs._joint_track(kin, "hop_down_inside", 1.3, [(0.0, q_sill), (0.3, fold), (0.65, fold), (1.0, q_stand)], hop_base,
                          source="gibbon_scenarios.heist: the hop down into the room")
    table_front = w["x"] + p["table_x"] - 0.4
    segs += [("motion", pick), ("motion", smash), ("motion", push), ("motion", drop)] + [("motion", m_) for m_ in steps] + [("motion", hop),
             ("gait", max(table_front - 0.3 - x_land, 0.2) / 0.6, w["y"])]
    return segs


def kt_lever():
    import kerchak_tasks as kt
    return kt.LEVER_CURL


gs.DEFAULTS["heist"] = HEIST
gs.DEFAULTS["heist_jump"] = HEIST_JUMP
gs.SCENARIOS["heist_jump"] = scenario_heist
gs.COMPOSITE["heist_jump"] = heist_props
gs.SCENARIOS["heist"] = scenario_heist
gs.COMPOSITE["heist"] = heist_props

COURSE = {**LADDER, "rungs": 5, "rise_deg": math.degrees(math.atan((2.1 - LADDER["height"]) / (4 * LADDER["spacing"]))),
          "rope_step": 0.4, "window_x": 7.05, "rope_end": 7.05 - gs.WINDOW["wall_thickness"] / 2 - 0.12, "table_x": 4.6, "door_x": 9.05}
gs.DEFAULTS["course"] = COURSE
gs.SCENARIOS["course"] = scenario_course
gs.COMPOSITE["course"] = course_props
gs.DEFAULTS["brachiate"] = LADDER
gs.DEFAULTS["brachiate_up"] = LADDER_UP
gs.BUILDERS["brachiate_up"] = ladder_prop
gs.SCENARIOS["brachiate_up"] = scenario_brachiate
gs.BUILDERS["brachiate"] = ladder_prop
gs.SCENARIOS["brachiate"] = scenario_brachiate


# ---- the firefighter: the heist by the jump, the table on fire instead of the banana; a helmet on the head, a hose
# from the backpack with its nozzle in a holster on the pack's right side; in the room the monkey stops short of the
# table, draws the nozzle over its right shoulder, braces the hose with the left hand and sprays the fire out

FIRE = {**HEIST_JUMP, "fire": 1, "spray_range": 1.1}                       #: how far short of the fire it stops [m]
HAT_ON_THORAX = (0.0, 0.0, 0.24)                                         #: the helmet's frame on the thorax (the head's top at ~0.3)
NOZZLE = {"length": 0.2, "r": 0.018, "grip": 0.06}                         #: the nozzle: along its z, the grip this far up from its root
HOLSTER_ON_THORAX = (np.array([0.11, -0.11, -0.06]), 0.0)               #: the nozzle's root in the holster (low on the pack's right side), upright


def hat_prop(p):
    red, yel, blk = (0.75, 0.08, 0.06, 1.0), (0.95, 0.78, 0.15, 1.0), (0.1, 0.1, 0.1, 1.0)
    geoms = [Geom("hat_dome", "sphere", (0.074,), pos=(0.0, 0.0, 0.025), mass=0.15, role="visual", rgba=red),   # worn: drawn on the head, not colliding with it
             Geom("hat_brim_disc", "cylinder", (0.105, 0.004), pos=(-0.02, 0.0, -0.01), mass=0.05, role="visual", rgba=red),
             Geom("hat_badge", "box", (0.006, 0.03, 0.028), pos=(0.075, 0.0, 0.03), mass=0.02, role="visual", rgba=yel),
             gs._box("hat_comb", (0.06, 0.008, 0.012), (0.0, 0.0, 0.1), rgba=blk, mass=0.02)]
    return Prop(Link("fire_hat", pos=(-6.0, 0.0, 4.0), geoms=geoms), free=True)


def nozzle_prop(p):
    brass, blk = (0.78, 0.62, 0.25, 1.0), (0.12, 0.12, 0.12, 1.0)
    L, r = NOZZLE["length"], NOZZLE["r"]
    geoms = [gs._cap("nozzle_grip", r, (0.0, 0.0, 0.0), (0.0, 0.0, 0.11), rgba=blk, mass=0.3),
             gs._cap("nozzle_barrel", r * 0.8, (0.0, 0.0, 0.11), (0.0, 0.0, L - 0.02), rgba=brass, mass=0.3),
             gs._cap("nozzle_tip", r * 0.5, (0.0, 0.0, L - 0.02), (0.0, 0.0, L), rgba=brass, mass=0.05)]
    return Prop(Link("nozzle", pos=(-6.0, 1.0, 4.0), geoms=geoms), free=True)


def fire_props(p):
    return heist_props(p) + (hat_prop(p), nozzle_prop(p))


def scenario_fire(kin, q_stand, p):
    """The heist by the jump with the table on fire: the walk in the room stops ``spray_range`` short of the fire; the
    right hand draws the nozzle from the holster over the shoulder; both hands aim it (the left braces the hose behind
    the nozzle) and sweep the jet across the flames until they are out; the nozzle lowered."""
    segs = scenario_heist(kin, q_stand, p)
    w = _heist_window(p)
    x_land = w["x"] + w["wall_thickness"] / 2 + 0.45
    fire_c = np.array([w["x"] + p["table_x"] - 0.15, p["table_y"] + 0.1, 0.75])
    x_stop = fire_c[0] - p["spray_range"]
    lane = w["y"]
    segs[-1] = ("gait", max(x_stop - x_land, 0.2) / 0.6, lane)
    x_stop = x_land + max(x_stop - x_land, 0.2)
    base = np.array([x_stop, lane, gm.STANCE["h"]])
    pt = gm.STANCE["pitch"]
    Rb = gm.base_rot(pt)
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j and not j.startswith("tail")]
    kin.set(q_stand)
    feet = [(bd, l, Rb @ kin.world(bd, l) + base) for sd in ("L", "R") for bd, l in gm._contact_points(kin)[sd]]
    th = kin.body("thorax")
    R_th, p_th = Rb @ kin.data.xmat[th].reshape(3, 3), Rb @ kin.data.xpos[th] + base
    root_w = R_th @ HOLSTER_ON_THORAX[0] + p_th
    R_noz = R_th.copy()                                                    # upright in the holster: the nozzle's z the thorax's z
    axis_h = R_noz[:, 2]
    grip_h = root_w + axis_h * NOZZLE["grip"]
    fr_h = gs.hand_frame_on(gs.bar(grip_h, axis=axis_h, n=-R_th[:, 1], r=NOZZLE["r"], wrist_dir=gs._unit(-R_th[:, 2] * 0.3 + R_th[:, 0] * 0.6 - R_th[:, 1] * 0.5)), "R")
    hold_h = (grip_h, fr_h)
    # the aim: the nozzle on the line from the chest to the fire's foot, held at the chest
    d = gs._unit(fire_c + np.array([0.0, 0.0, 0.05]) - (p_th + np.array([0.3, 0.0, 0.0])))
    def aim(yaw):
        c_, s_ = math.cos(yaw), math.sin(yaw)
        dd = np.array([c_ * d[0] - s_ * d[1], s_ * d[0] + c_ * d[1], d[2]])
        grip = p_th + np.array([0.28, -0.08, -0.12])
        return dd, (grip, gs.hand_frame_on(gs.bar(grip, axis=dd, n=(0.0, 0.0, 1.0), r=NOZZLE["r"], wrist_dir=gs._unit(-dd + np.array([0.0, -0.3, -0.5]))), "R"))
    def brace(yaw):
        dd, (grip, _) = aim(yaw)
        c = grip - 0.17 * dd + np.array([0.0, 0.06, 0.0])
        return (c, gs.hand_frame_on(gs.bar(c, axis=dd, n=(0.0, 0.0, 1.0), r=0.015, wrist_dir=gs._unit(-dd + np.array([0.0, 0.3, -0.5]))), "L"))
    kin.set(q_stand)
    hand0 = {sd: gs._hand_pose(kin, q_stand, base, pt, sd) for sd in ("L", "R")}
    up_over = p_th + R_th @ np.array([0.12, -0.2, 0.05])                   # out to the side, clear of the hip
    T = lambda tg: [(bd, l, Rb.T @ (np.asarray(wp) - base)) for bd, l, wp in tg]
    fixed = lambda u: (base[0], base[1], base[2], pt)

    def r_draw(u):
        Rh0, ph0 = hand0["R"]
        if u < 0.5:                                                        # up over the shoulder and down onto the grip
            f = mk.smooth(u / 0.5)
            c = (1 - f) * ph0 + f * grip_h + (up_over - 0.5 * (ph0 + grip_h)) * math.sin(math.pi * f)
            Rf = gs._slerp(Rh0, fr_h, f)
            return feet + gs._hand_at(kin, "R", c, Rf)
        f = mk.smooth((u - 0.5) / 0.5)                                     # out of the holster, over the shoulder, to the aim
        dd, (ga, fa) = aim(0.0)
        c = (1 - f) * grip_h + f * ga + (up_over - 0.5 * (grip_h + ga)) * math.sin(math.pi * f) * 0.7
        return feet + gs._hand_at(kin, "R", c, gs._slerp(fr_h, fa, f))

    def x_draw(u, q_):
        c = gs.CURL_OPEN if u < 0.45 else gs.CURL_CLOSED
        sr.set_curl(q_, kin, "R", c)
        return q_
    gt.SOLID and setattr(gt.SOLID, "ignore", {"glass"})
    draw = gt.contact_motion(kin, "draw_the_hose", 2.4, r_draw, fixed, q_init=q_stand, extra=x_draw, reg=gs.IK_REG, iters=gs.IK_ITERS,
                             q_ref=q_stand, source="gibbon_scenarios.fire: the nozzle drawn from the holster over the right shoulder")
    # the nozzle in the hand, from the grasp (u 0.5)
    q_g, _ = kin.ik(T(r_draw(0.5)), q_init=q_stand, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)   # the pose at the grasp
    R_hand_g, p_hand_g = gs._hand_pose(kin, q_g, base, pt, "R")
    R_h2m = R_hand_g.T @ R_noz
    p_h2m = R_hand_g.T @ (root_w - p_hand_g)
    draw.params = {**(draw.params or {}), "nozzle_T": (R_h2m.tolist(), p_h2m.tolist()), "grasp_u": 0.5,
                   "nozzle_carry": ("thorax", np.eye(3).tolist(), HOLSTER_ON_THORAX[0].tolist()), "hand": "R_hand"}
    q1 = gt._last(kin, draw, q_stand)
    SWEEP = math.radians(9.0)

    def t_spray(u):
        yaw = SWEEP * math.sin(2.0 * math.pi * 1.5 * max(0.0, u - 0.15) / 0.85) if u > 0.15 else 0.0
        dd, (ga, fa) = aim(yaw)
        out = feet + gs._hand_at(kin, "R", ga, fa)
        cL, fL = brace(yaw)
        Rl0, pl0 = hand0["L"]
        f = mk.smooth(min(1.0, u / 0.15))
        return out + gs._hand_at(kin, "L", (1 - f) * pl0 + f * cL, gs._slerp(Rl0, fL, f))

    def x_spray(u, q_):
        sr.set_curl(q_, kin, "R", gs.CURL_CLOSED)
        sr.set_curl(q_, kin, "L", gs.CURL_CLOSED if u > 0.12 else gs.CURL_OPEN)
        return q_
    spray = gt.contact_motion(kin, "spray_the_fire", 5.5, t_spray, fixed, q_init=q1, extra=x_spray, reg=gs.IK_REG, iters=gs.IK_ITERS,
                              q_ref=q_stand, source="gibbon_scenarios.fire: both hands on the hose, the jet swept across the flames")
    q_a, _ = kin.ik(T(t_spray(0.15)), q_init=q1, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)   # the aim: the nozzle along the jet
    Rha, pha = gs._hand_pose(kin, q_a, base, pt, "R")
    dd0, (ga0, _) = aim(0.0)
    zc = gs._unit(dd0)
    xc = gs._unit(np.cross([0.0, 0.0, 1.0], zc)) if abs(zc[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    R_na = np.stack([xc, np.cross(zc, xc), zc], axis=1)
    root_a = ga0 - zc * NOZZLE["grip"]
    draw.params["nozzle_T_aim"] = ((Rha.T @ R_na).tolist(), (Rha.T @ (root_a - pha)).tolist())
    draw.params["aim_t"] = 0.0
    spray.params = {**(spray.params or {}), "spray_u": (0.15, 0.95), "out_u": 0.8, "fire": fire_c.tolist()}
    q2 = gt._last(kin, spray, q1)

    def t_lower(u):
        f = mk.smooth(u)
        dd, (ga, fa) = aim(0.0)
        low = ga + np.array([-0.1, -0.05, -0.35])
        fl = gs.hand_frame_on(gs.bar(low, axis=gs._unit(np.array([0.5, 0.0, -0.85])), n=(1.0, 0.0, 0.0), r=NOZZLE["r"], wrist_dir=(0.0, -0.2, 1.0)), "R")
        return feet + gs._hand_at(kin, "R", (1 - f) * ga + f * low, gs._slerp(fa, fl, f))
    lower = gt.contact_motion(kin, "lower_the_hose", 1.4, t_lower, fixed, q_init=q2, extra=lambda u, q_: sr.set_curl(sr.set_curl(q_, kin, "R", gs.CURL_CLOSED), kin, "L", gs.CURL_OPEN),
                              reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the fire out, the nozzle lowered")
    gt.SOLID and setattr(gt.SOLID, "ignore", set())
    return segs + [("motion", draw), ("motion", spray), ("motion", lower)]



# ---- the firefighter, reworked: a two-storey house, the fire on the first floor; the monkey rides in on Chuck (the
# self-driving firefighting quad: tank, pump, hose reel, water cannon), hops off, takes the hose's nozzle from its
# hook and clips it to the backpack, climbs the ladder standing under the window with the hose paying out behind,
# over the sill between the ladder's rails, down onto the first floor, to the burning table; draws the nozzle,
# opens its valve with the left hand and sprays the fire out

FIRE = {"wall_x": 6.0, "wall_t": 0.25, "floor": 3.0, "sill_h": 0.85, "win_w": 0.9, "win_h": 1.3, "roof": 6.4, "house_w": 12.0,
        "room_d": 5.0, "rung_out": 0.14, "rung_sp": 0.3, "rung0": 0.3, "rung_r": 0.016, "rail_y": 0.22, "rail_above": 0.95,
        "table_x": 2.4, "table_y": 0.25, "quad_x0": -6.0, "quad_x": 2.0, "quad_y": 1.0, "spray_range": 1.1, "climb_s": 3.0}
RED = (0.72, 0.1, 0.08, 1.0)
CHUCK = {"seat": (-0.05, 0.0, 0.92), "hook": (-0.25, -0.78, 0.8), "reel": (-0.6, 0.0, 1.05), "bars": (0.5, 0.28, 1.05), "pegs": (0.12, 0.44, 0.46)}


def _fire_sill(p):
    return p["floor"] + p["sill_h"]


def house_prop(p):
    """A plain two-storey house: the front wall with the first floor's window (no glass: open), a door below, the
    first floor's slab, the back and side walls, the roof; the table inside on the first floor."""
    X, t, F, S, w, h, top, W, D = p["wall_x"], p["wall_t"], p["floor"], _fire_sill(p), p["win_w"], p["win_h"], p["roof"], p["house_w"], p["room_d"]
    wall, slab, dark = (0.78, 0.74, 0.68, 1.0), (0.6, 0.58, 0.55, 1.0), (0.3, 0.2, 0.14, 1.0)
    bx = lambda n, half, c, rgba=wall: gs._box(n, half, c, rgba=rgba, mass=500.0, friction=gs.SLIPPERY)
    geoms = [bx("wall_below", (t / 2, W / 2, S / 2), (X, 0.0, S / 2)),
             bx("wall_above", (t / 2, W / 2, (top - S - h) / 2), (X, 0.0, S + h + (top - S - h) / 2)),
             bx("wall_left", (t / 2, (W - w) / 4, h / 2), (X, w / 2 + (W - w) / 4, S + h / 2)),
             bx("wall_right", (t / 2, (W - w) / 4, h / 2), (X, -w / 2 - (W - w) / 4, S + h / 2)),
             gs._box("sill_plate", (t / 2 + 0.04, w / 2, 0.03), (X, 0.0, S + 0.03), rgba=gs.WOOD, mass=20.0),
             bx("first_floor", (D / 2, W / 2, 0.1), (X + t / 2 + D / 2, 0.0, F - 0.1), slab),
             bx("back_wall", (t / 2, W / 2, top / 2), (X + t + D, 0.0, top / 2)),
             bx("side_left", (D / 2 + t, t / 2, top / 2), (X + D / 2 + t / 2, W / 2, top / 2)),
             bx("side_right", (D / 2 + t, t / 2, top / 2), (X + D / 2 + t / 2, -W / 2, top / 2)),
             bx("roof", (D / 2 + t + 0.2, W / 2 + 0.2, 0.08), (X + D / 2 + t / 2, 0.0, top + 0.08), slab),
             Geom("door", "box", (0.01, 0.5, 1.05), pos=(X - t / 2 - 0.01, -2.5, 1.05), mass=1.0, role="visual", rgba=dark),
             Geom("ground_window", "box", (0.01, 0.6, 0.6), pos=(X - t / 2 - 0.01, 2.5, 1.5), mass=1.0, role="visual", rgba=(0.55, 0.7, 0.78, 1.0))]
    tx, ty, tt = X + t / 2 + p["table_x"], p["table_y"], F + 0.75
    geoms.append(gs._box("table_top", (0.4, 0.6, 0.02), (tx, ty, tt - 0.02), rgba=gs.WOOD, mass=15.0))
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        geoms.append(gs._box(f"table_leg_{i}", (0.03, 0.03, 0.355), (tx + sx * 0.35, ty + sy * 0.55, F + 0.355), rgba=gs.WOOD, mass=3.0))
    return Prop(Link("house", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def fire_ladder_prop(p):
    """An aluminium ladder standing under the window: rungs up to the sill, the rails on past it for the hands."""
    xl, S = p["wall_x"] - p["wall_t"] / 2 - p["rung_out"], _fire_sill(p)
    ry, top = p["rail_y"], S + 0.06 + p["rail_above"]
    alu = (0.75, 0.77, 0.8, 1.0)
    geoms = [gs._cap(f"rail_{tag}", 0.022, (xl, sy * ry, 0.0), (xl, sy * ry, top), rgba=alu, mass=4.0) for sy, tag in ((1, "l"), (-1, "r"))]
    for sy, tag in ((1, "l"), (-1, "r")):                                    # the standoffs to the wall at the top
        geoms.append(gs._cap(f"standoff_{tag}", 0.015, (xl, sy * ry, S - 0.1), (p["wall_x"] - p["wall_t"] / 2, sy * ry, S - 0.1), rgba=alu, mass=0.5))
    z, k = p["rung0"], 0
    while z < S - 0.15:
        geoms.append(gs._cap(f"rung_{k}", p["rung_r"], (xl, -ry, z), (xl, ry, z), rgba=alu, mass=0.5))
        z += p["rung_sp"]; k += 1
    return Prop(Link("fire_ladder", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def chuck_prop(p):
    """Chuck, simplified: the quad's frame and wheels, the water tank, the pump, the hose reel, the water cannon on the
    nose, the handlebars and seat; the nozzle's hook on its left side. Its frame at the ground under the middle."""
    blk, grey, steel = (0.08, 0.08, 0.09, 1.0), (0.62, 0.64, 0.66, 1.0), (0.4, 0.42, 0.45, 1.0)
    geoms = [gs._box("chuck_frame", (1.0, 0.34, 0.08), (0.0, 0.0, 0.5), rgba=RED, mass=200.0),
             gs._box("chuck_tank", (0.32, 0.42, 0.24), (-0.6, 0.0, 0.82), rgba=grey, mass=200.0),
             Geom("chuck_reel", "cylinder", (0.24, 0.16), pos=CHUCK["reel"], quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=20.0, role="link", rgba=RED),
             gs._box("chuck_pump", (0.14, 0.12, 0.1), (-0.15, 0.0, 0.68), rgba=steel, mass=30.0),
             gs._box("chuck_seat", (0.22, 0.17, 0.05), (CHUCK["seat"][0], 0.0, 0.75), rgba=blk, mass=5.0),
             gs._box("chuck_tankcover", (0.3, 0.3, 0.06), (0.35, 0.0, 0.66), rgba=RED, mass=10.0),
             gs._cap("chuck_column", 0.03, (0.45, 0.0, 0.65), (0.5, 0.0, 1.02), rgba=steel, mass=3.0),
             gs._cap("chuck_bars", 0.018, (0.5, -0.3, 1.04), (0.5, 0.3, 1.04), rgba=blk, mass=2.0),
             gs._cap("chuck_cannon_post", 0.04, (0.9, 0.0, 0.58), (0.9, 0.0, 0.8), rgba=steel, mass=4.0),
             gs._cap("chuck_cannon", 0.035, (0.9, 0.0, 0.84), (1.15, 0.0, 0.94), rgba=(0.78, 0.62, 0.25, 1.0), mass=5.0),
             gs._cap("chuck_hook_arm", 0.012, (CHUCK["hook"][0], -0.34, 0.55), (CHUCK["hook"][0], CHUCK["hook"][1], CHUCK["hook"][2] - 0.05), rgba=steel, mass=0.3)]
    for sy, tag in ((1, "l"), (-1, "r")):                                    # the foot pegs out beside the frame
        geoms.append(gs._cap(f"chuck_peg_{tag}", 0.015, (CHUCK["pegs"][0] - 0.08, sy * 0.34, CHUCK["pegs"][2] - 0.03), (CHUCK["pegs"][0] - 0.08, sy * 0.52, CHUCK["pegs"][2] - 0.03), rgba=steel, mass=0.5))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        geoms.append(Geom(f"chuck_wheel_{i}", "cylinder", (0.33, 0.12), pos=(sx * 0.72, sy * 0.6, 0.33), quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=20.0, role="link", rgba=blk))
        geoms.append(Geom(f"chuck_hub_{i}", "cylinder", (0.12, 0.125), pos=(sx * 0.72, sy * 0.6, 0.33), quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=2.0, role="visual", rgba=steel))
    return Prop(Link("chuck", pos=(-20.0, 5.0, 0.0), geoms=geoms), free=True)


def fire_props(p):
    return (house_prop(p), fire_ladder_prop(p), chuck_prop(p), backpack_prop(p), hat_prop(p), nozzle_prop(p))


def _fire_fight(kin, q_stand, base, fire_c, pt):
    """At ``base``, standing: the nozzle drawn from the holster, aimed at ``fire_c`` and braced by the left hand on
    the hose behind it - the left hand first opens the nozzle's valve (its bail lever) - the jet swept over the
    flames, the nozzle lowered. Returns the three motions."""
    Rb = gm.base_rot(pt)
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j and not j.startswith("tail")]
    kin.set(q_stand)
    feet = [(bd, l, Rb @ kin.world(bd, l) + base) for sd in ("L", "R") for bd, l in gm._contact_points(kin)[sd]]
    th = kin.body("thorax")
    R_th, p_th = Rb @ kin.data.xmat[th].reshape(3, 3), Rb @ kin.data.xpos[th] + base
    root_w = R_th @ HOLSTER_ON_THORAX[0] + p_th
    R_noz = R_th.copy()
    axis_h = R_noz[:, 2]
    grip_h = root_w + axis_h * NOZZLE["grip"]
    fr_h = gs.hand_frame_on(gs.bar(grip_h, axis=axis_h, n=-R_th[:, 1], r=NOZZLE["r"], wrist_dir=gs._unit(-R_th[:, 2] * 0.3 + R_th[:, 0] * 0.6 - R_th[:, 1] * 0.5)), "R")
    d = gs._unit(fire_c + np.array([0.0, 0.0, 0.05]) - (p_th + np.array([0.3, 0.0, 0.0])))

    def aim(yaw):
        c_, s_ = math.cos(yaw), math.sin(yaw)
        dd = np.array([c_ * d[0] - s_ * d[1], s_ * d[0] + c_ * d[1], d[2]])
        grip = p_th + np.array([0.28, -0.08, -0.12])
        return dd, (grip, gs.hand_frame_on(gs.bar(grip, axis=dd, n=(0.0, 0.0, 1.0), r=NOZZLE["r"], wrist_dir=gs._unit(-dd + np.array([0.0, -0.3, -0.5]))), "R"))

    def brace(yaw):
        dd, (grip, _) = aim(yaw)
        c = grip - 0.17 * dd + np.array([0.0, 0.06, 0.0])
        return (c, gs.hand_frame_on(gs.bar(c, axis=dd, n=(0.0, 0.0, 1.0), r=0.015, wrist_dir=gs._unit(-dd + np.array([0.0, 0.3, -0.5]))), "L"))

    def bail(yaw):                                                         # the valve's lever on top of the nozzle, ahead of the grip
        dd, (grip, _) = aim(yaw)
        c = grip - 0.17 * dd + np.array([0.0, 0.06, 0.0])                   # the shut-off at the hose's coupling, where the left hand braces
        return (c, gs.hand_frame_on(gs.bar(c, axis=gs._unit(np.cross(dd, [0.0, 0.0, 1.0])), n=(0.0, 0.0, 1.0), r=0.01, wrist_dir=gs._unit(np.array([-0.4, 0.3, 0.6]))), "L"))
    hand0 = {sd: gs._hand_pose(kin, q_stand, base, pt, sd) for sd in ("L", "R")}
    up_over = p_th + R_th @ np.array([0.12, -0.2, 0.05])
    T = lambda tg: [(bd, l, Rb.T @ (np.asarray(wp) - base)) for bd, l, wp in tg]
    fixed = lambda u: (base[0], base[1], base[2], pt)

    def r_draw(u):
        Rh0, ph0 = hand0["R"]
        if u < 0.5:
            f = mk.smooth(u / 0.5)
            return feet + gs._hand_at(kin, "R", (1 - f) * ph0 + f * grip_h + (up_over - 0.5 * (ph0 + grip_h)) * math.sin(math.pi * f), gs._slerp(Rh0, fr_h, f))
        f = mk.smooth((u - 0.5) / 0.5)
        dd, (ga, fa) = aim(0.0)
        return feet + gs._hand_at(kin, "R", (1 - f) * grip_h + f * ga + (up_over - 0.5 * (grip_h + ga)) * math.sin(math.pi * f) * 0.7, gs._slerp(fr_h, fa, f))
    draw = gt.contact_motion(kin, "draw_the_nozzle", 2.4, r_draw, fixed, q_init=q_stand, extra=lambda u, q_: sr.set_curl(q_, kin, "R", gs.CURL_OPEN if u < 0.45 else gs.CURL_CLOSED),
                             reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the nozzle drawn from the holster on the pack")
    q_g, _ = kin.ik(T(r_draw(0.5)), q_init=q_stand, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    Rhg, phg = gs._hand_pose(kin, q_g, base, pt, "R")
    draw.params = {**(draw.params or {}), "nozzle_T": ((Rhg.T @ R_noz).tolist(), (Rhg.T @ (root_w - phg)).tolist()), "grasp_u": 0.5, "hand": "R_hand"}
    q1 = gt._last(kin, draw, q_stand)
    SWEEP, V0, V1 = math.radians(9.0), 0.12, 0.24                          # the valve: the left hand on the bail from V0, opened by V1

    def t_spray(u):
        yaw = SWEEP * math.sin(2.0 * math.pi * 1.5 * (u - 0.3) / 0.7) if u > 0.3 else 0.0
        dd, (ga, fa) = aim(yaw)
        out = feet + gs._hand_at(kin, "R", ga, fa)
        Rl0, pl0 = hand0["L"]
        cb, fb = bail(0.0)
        if u < V0:
            f = mk.smooth(u / V0)
            return out + gs._hand_at(kin, "L", (1 - f) * pl0 + f * cb, gs._slerp(Rl0, fb, f))
        if u < V1:                                                         # the bail pulled back: open
            f = mk.smooth((u - V0) / (V1 - V0))
            return out + gs._hand_at(kin, "L", cb - f * 0.05 * dd + np.array([0.0, 0.0, 0.02 * f]), fb)
        cL, fL = brace(yaw)
        f = mk.smooth(min(1.0, (u - V1) / 0.08))
        return out + gs._hand_at(kin, "L", (1 - f) * (cb - 0.05 * dd + np.array([0.0, 0.0, 0.02])) + f * cL, gs._slerp(fb, fL, f))

    def x_spray(u, q_):
        sr.set_curl(q_, kin, "R", gs.CURL_CLOSED)
        sr.set_curl(q_, kin, "L", gs.CURL_CLOSED if u > 0.08 else gs.CURL_OPEN)
        return q_
    spray = gt.contact_motion(kin, "open_the_valve_and_spray", 6.5, t_spray, fixed, q_init=q1, extra=x_spray, reg=gs.IK_REG, iters=gs.IK_ITERS,
                              q_ref=q_stand, source="gibbon_scenarios.fire: the valve opened by the left hand, then braced on the hose, the jet swept over the flames")
    q_a, _ = kin.ik(T(t_spray(0.0) if False else feet + gs._hand_at(kin, "R", *aim(0.0)[1])), q_init=q1, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    Rha, pha = gs._hand_pose(kin, q_a, base, pt, "R")
    dd0, (ga0, _) = aim(0.0)
    zc = gs._unit(dd0)
    xc = gs._unit(np.cross([0.0, 0.0, 1.0], zc))
    R_na = np.stack([xc, np.cross(zc, xc), zc], axis=1)
    draw.params["nozzle_T_aim"] = ((Rha.T @ R_na).tolist(), (Rha.T @ (ga0 - zc * NOZZLE["grip"] - pha)).tolist())
    spray.params = {**(spray.params or {}), "valve_u": (V0, V1), "spray_u": (V1, 0.96), "out_u": 0.75, "fire": fire_c.tolist()}
    q2 = gt._last(kin, spray, q1)

    def t_lower(u):
        f = mk.smooth(u)
        dd, (ga, fa) = aim(0.0)
        low = ga + np.array([-0.1, -0.05, -0.35])
        fl = gs.hand_frame_on(gs.bar(low, axis=gs._unit(np.array([0.5, 0.0, -0.85])), n=(1.0, 0.0, 0.0), r=NOZZLE["r"], wrist_dir=(0.0, -0.2, 1.0)), "R")
        return feet + gs._hand_at(kin, "R", (1 - f) * ga + f * low, gs._slerp(fa, fl, f))
    lower = gt.contact_motion(kin, "lower_the_nozzle", 1.4, t_lower, fixed, q_init=q2, extra=lambda u, q_: sr.set_curl(sr.set_curl(q_, kin, "R", gs.CURL_CLOSED), kin, "L", gs.CURL_OPEN),
                              reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the valve shut, the nozzle lowered")
    return [draw, spray, lower]


def scenario_fire(kin, q_stand, p):
    """Chuck drives up to the house with the monkey on it; the monkey hops off, takes the nozzle from Chuck's hook and
    clips it to its pack; walks to the ladder and climbs it (hands on the rails, feet on the rungs), the hose paying
    out; over the sill between the rails into the first floor, a hop down, the walk to the burning table, the fire
    put out."""
    S_ = gm.scale()
    X, t, F = p["wall_x"], p["wall_t"], p["floor"]
    sill = _fire_sill(p)
    top = sill + 0.06
    xl = X - t / 2 - p["rung_out"]
    sp, z0, ry = p["rung_sp"], p["rung0"], p["rail_y"]
    v = gs.walk_speed()
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
    xq, yq = p["quad_x"], p["quad_y"]
    seat = np.array([xq + CHUCK["seat"][0], yq, CHUCK["seat"][2]])
    # the ride: sat on the seat, the feet on the pegs, the hands on the bars
    pr = 0.15
    Rr = gm.base_rot(pr)
    tg = []
    for sd, sy in (("L", 1.0), ("R", -1.0)):
        tg += gs.foot_targets(kin, sd, gs.point(np.array([xq + CHUCK["pegs"][0], yq + sy * CHUCK["pegs"][1], CHUCK["pegs"][2]]), pitch=0.0))
        tg += gs.hand_targets(kin, sd, gs.bar(np.array([xq + CHUCK["bars"][0], yq + sy * CHUCK["bars"][1], CHUCK["bars"][2]]), axis=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0), r=0.018, wrist_dir=(-0.6, 0.0, -0.8)))
    q_ride, err_ride = kin.ik([(bd, l, Rr.T @ (np.asarray(wp) - seat)) for bd, l, wp in tg], q_init=hang_ref(kin, q_stand), q_ref=hang_ref(kin, q_stand),
                              joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    for sd in ("L", "R"):
        sr.set_curl(q_ride, kin, sd, gs.CURL_CLOSED)
    print(f"fire: riding pose reach {err_ride:.3f} m")
    D_RIDE = 8.0
    x0 = p["quad_x0"]

    def quad_x(u):                                                         # accelerates, cruises, brakes to a stop
        return x0 + (xq - x0) * mk.smooth(u)
    ride = gs._joint_track(kin, "ride_in_on_chuck", D_RIDE, [(0.0, q_ride), (1.0, q_ride)],
                           lambda u: (quad_x(u) + CHUCK["seat"][0], yq, CHUCK["seat"][2], pr), frames=80,
                           source="gibbon_scenarios.fire: on Chuck's seat as it drives itself up to the house")
    ride.params = {"quad": {"x0": x0, "x1": xq, "y": yq}}
    # the hop off: a crouch, a push sideways off the seat, the flight to the ground beside Chuck, the landing
    land = np.array([xq - 0.25, 0.0, gm.STANCE["h"]])
    G_ = 9.81
    apex = seat[2] + 0.12
    t_up, t_dn = math.sqrt(2 * 0.12 / G_), math.sqrt(2 * (apex - land[2]) / G_)
    T_F = t_up + t_dn
    vel = np.array([(land[0] - seat[0]) / T_F, (land[1] - seat[1]) / T_F, G_ * t_up])
    T_C, T_L = 0.35, 0.35
    Dh = T_C + T_F + T_L
    tuck = q_ride.copy()
    for sd in ("L", "R"):
        sr.set_curl(tuck, kin, sd, gs.CURL_OPEN)

    def hop_base(u):
        tt = u * Dh
        if tt < T_C:
            return seat[0], seat[1], seat[2] - 0.04 * math.sin(math.pi * tt / T_C), pr
        if tt < T_C + T_F:
            s_ = tt - T_C
            c = seat + vel * s_ + np.array([0.0, 0.0, -0.5 * G_ * s_ * s_])
            return c[0], c[1], c[2], pr + (gm.STANCE["pitch"] - pr) * mk.smooth(s_ / T_F)
        s_ = (tt - T_C - T_F) / T_L
        return land[0], land[1], land[2] - 0.08 * math.sin(math.pi * s_), gm.STANCE["pitch"]
    hop_off = gs._joint_track(kin, "hop_off_chuck", Dh, [(0.0, q_ride), (T_C / Dh, tuck), ((T_C + 0.6 * T_F) / Dh, q_stand), (1.0, q_stand)], hop_base,
                              frames=int(Dh * 40), source="gibbon_scenarios.fire: off the seat, a ballistic hop to the ground beside Chuck")
    # the hose: the nozzle off Chuck's hook (the right hand), clipped into the holster on the pack
    base_g = land.copy()
    pt = gm.STANCE["pitch"]
    Rb = gm.base_rot(pt)
    kin.set(q_stand)
    feet = [(bd, l, Rb @ kin.world(bd, l) + base_g) for sd in ("L", "R") for bd, l in gm._contact_points(kin)[sd]]
    th = kin.body("thorax")
    R_th, p_th = Rb @ kin.data.xmat[th].reshape(3, 3), Rb @ kin.data.xpos[th] + base_g
    hook = np.array([xq + CHUCK["hook"][0], yq + CHUCK["hook"][1], CHUCK["hook"][2]])
    R_hookN = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])   # hung upright on the hook
    grip_k = hook + np.array([0.0, 0.0, NOZZLE["grip"] - 0.12])
    fr_k = gs.hand_frame_on(gs.bar(grip_k, axis=(0.0, 0.0, 1.0), n=(0.0, -1.0, 0.0), r=NOZZLE["r"], wrist_dir=gs._unit(np.array([-0.3, -0.8, -0.3]))), "R")
    root_hol = R_th @ HOLSTER_ON_THORAX[0] + p_th
    grip_hol = root_hol + R_th[:, 2] * NOZZLE["grip"]
    fr_hol = gs.hand_frame_on(gs.bar(grip_hol, axis=R_th[:, 2], n=-R_th[:, 1], r=NOZZLE["r"], wrist_dir=gs._unit(-R_th[:, 2] * 0.3 + R_th[:, 0] * 0.6 - R_th[:, 1] * 0.5)), "R")
    Rh0, ph0 = gs._hand_pose(kin, q_stand, base_g, pt, "R")

    def t_take(u):
        if u < 0.4:
            f = mk.smooth(u / 0.4)
            return feet + gs._hand_at(kin, "R", (1 - f) * ph0 + f * grip_k + np.array([0.0, 0.0, 0.1 * math.sin(math.pi * f)]), gs._slerp(Rh0, fr_k, f))
        if u < 0.5:
            return feet + gs._hand_at(kin, "R", grip_k, fr_k)
        f = mk.smooth((u - 0.5) / 0.4) if u < 0.9 else 1.0
        return feet + gs._hand_at(kin, "R", (1 - f) * grip_k + f * grip_hol + np.array([0.0, -0.1, 0.12]) * math.sin(math.pi * f), gs._slerp(fr_k, fr_hol, f))
    fixed = lambda u: (base_g[0], base_g[1], base_g[2], pt)
    take = gt.contact_motion(kin, "take_the_hose", 3.2, t_take, fixed, q_init=q_stand,
                             extra=lambda u, q_: sr.set_curl(q_, kin, "R", gs.CURL_CLOSED if 0.42 < u < 0.92 else gs.CURL_OPEN),
                             reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the nozzle off Chuck's hook, clipped to the pack")
    q_k, _ = kin.ik([(bd, l, Rb.T @ (np.asarray(wp) - base_g)) for bd, l, wp in t_take(0.45)], q_init=q_stand, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    Rhk, phk = gs._hand_pose(kin, q_k, base_g, pt, "R")
    take.params = {"hook": hook.tolist(), "take_u": 0.45, "clip_u": 0.9, "nozzle_T": ((Rhk.T @ R_hookN).tolist(), (Rhk.T @ (hook - np.array([0.0, 0.0, 0.12]) - phk)).tolist())}
    q_t = gt._last(kin, take, q_stand)
    # the ladder: hands on the rails (vertical bars), feet on the rungs
    n_rungs = int((sill - 0.15 - z0) / sp) + 1
    rail = lambda sy, z: gs.bar((xl, sy * ry, z), axis=(0.0, 0.0, 1.0), n=(-1.0, 0.0, 0.0), r=0.022)
    hold = lambda k: {"L_hand": rail(1.0, z0 + (k + 2) * sp + 0.05), "R_hand": rail(-1.0, z0 + (k + 2) * sp + 0.05),
                      "L": gs.bar((xl, 0.11, z0 + k * sp), r=p["rung_r"], tip=0.3), "R": gs.bar((xl, -0.11, z0 + k * sp), r=p["rung_r"], tip=0.3)}
    base_c = lambda k: np.array([xl - 0.3, 0.0, z0 + k * sp + 0.35])
    n = n_rungs - 1
    stop_x = xl - 0.25
    segs = gs._climb_sequence(kin, q_stand, hold, base_c, n, stop_x=stop_x, trials=gs.VERTICAL_TRIALS, duration=p["climb_s"])
    segs[0] = ("gait", max(stop_x - base_g[0], 0.2) / v, 0.0)
    q = q_stand.copy()
    for sg in segs:
        if sg[0] == "motion":
            q = gt._last(kin, sg[1], q)
    h_end = hold(n)
    frames = segs[1][1].params.get("frames", {})
    for g_, f_ in frames.items():
        h_end[g_]["sign"], h_end[g_]["phi"] = f_
    # over the sill: the hands up the rails, the body hoisted between them, each foot over the lip onto the sill, in
    b_end = base_c(n)
    pitch = gs.CLIMB_PITCH
    hands_hi = {sd: rail(1.0 if sd == "L" else -1.0, top + 0.6) for sd in ("L", "R")}
    for sd in ("L", "R"):
        hands_hi[sd]["sign"], hands_hi[sd]["phi"] = h_end[f"{sd}_hand"]["sign"], h_end[f"{sd}_hand"]["phi"]
    feet_sill = {sd: gs.point(np.array([X - 0.02, 0.1 if sd == "L" else -0.1, top]), pitch=0.0, over=True) for sd in ("L", "R")}
    crouch = np.array([X + gs.SILL_CROUCH[0] * S_, 0.0, top + gs.SILL_CROUCH[1] * S_])
    hoist = np.array([xl - 0.2, 0.0, top + 0.3])
    duck = gs.SILL_CROUCH[2]
    pts = gm._contact_points(kin)
    kin.set(q)
    world0 = lambda tg_: [(bd, l, gm.base_rot(pitch) @ kin.world(bd, l) + b_end) for bd, l, _ in tg_]
    A = {"L_hand": (0.0, 0.15), "R_hand": (0.15, 0.3)}
    Bh = (0.3, 0.5)
    C = {"L": (0.5, 0.66), "R": (0.66, 0.82)}
    Dc = (0.82, 1.0)

    def b_over(u):
        fb = mk.smooth(max(0.0, min(1.0, (u - Bh[0]) / (Bh[1] - Bh[0]))))
        fd = mk.smooth(max(0.0, min(1.0, (u - Dc[0]) / (Dc[1] - Dc[0]))))
        c = b_end + fb * (hoist - b_end) + fd * (crouch - hoist)
        return c[0], c[1], c[2], pitch + fd * (duck - pitch)

    def t_over(u):
        out = []
        for sd in ("L", "R"):
            g_ = f"{sd}_hand"
            a, b_ = A[g_]
            end = gs.hand_targets(kin, sd, hands_hi[sd])
            if u >= Dc[0] + 0.04:                                          # the feet on the sill: the hands let go of the rails
                pass
            elif u < a:
                out += gs.hand_targets(kin, sd, h_end[g_])
            elif u < b_:
                f = mk.smooth((u - a) / (b_ - a))
                st = gs.hand_targets(kin, sd, h_end[g_])
                out += [(e[0], e[1], (1 - f) * np.asarray(s_[2]) + f * np.asarray(e[2]) + np.array([-0.05, 0.0, 0.0]) * math.sin(math.pi * f)) for s_, e in zip(st, end)]
            else:
                out += end
            a, b_ = C[sd]
            if u < a:
                out += gs.foot_targets(kin, sd, h_end[sd])
            elif u < b_:
                ff = (u - a) / (b_ - a)
                c = gs.foot_swing(h_end[sd], feet_sill[sd], ff, lift=0.2, clear=0.12)
                out += gs.foot_targets(kin, sd, gs.point(feet_sill[sd]["c"], pitch=0.3 * (1.0 - mk.smooth(ff)), over=True), c)
            else:
                out += gs.foot_targets(kin, sd, feet_sill[sd])
        return out
    if gt.SOLID is not None:
        gt.SOLID.ignore = set()
    over = gt.contact_motion(kin, "over_the_sill", 6.0, t_over, b_over, q_init=q, extra=gs._spine_extra(kin, gs.CLIMB_SPINE), reg=gs.IK_REG, iters=gs.IK_ITERS,
                             q_ref=gs.climb_ref(kin, q), source="gibbon_scenarios.fire: up the rails, hoisted between them, the feet over the lip onto the sill, in")
    q1 = gt._last(kin, over, q)
    release = q1.copy()
    for sd in ("L", "R"):
        sr.set_curl(release, kin, sd, gs.CURL_OPEN)
    tuck2 = release.copy()
    for sd in ("L", "R"):
        tuck2[kin.index[f"{sd}_knee"]] = min(q1[kin.index[f"{sd}_knee"]] + 0.25, kin.hi[kin.index[f"{sd}_knee"]])
        tuck2[kin.index[f"{sd}_hip_pitch"]] = max(q1[kin.index[f"{sd}_hip_pitch"]] - 0.2, kin.lo[kin.index[f"{sd}_hip_pitch"]])
    x_land = X + t / 2 + 0.75 * S_
    x_clear = X + t / 2 + 0.25 * S_

    def b_in(u):
        f = mk.smooth(u)
        xb = crouch[0] + f * (x_land - crouch[0])
        fz = mk.smooth(max(0.0, min(1.0, (xb - x_clear) / (x_land - x_clear))))
        return xb, 0.0, (1 - fz) * crouch[2] + fz * (F + gm.STANCE["h"]) + 0.04 * math.sin(math.pi * fz), duck + fz * (gm.STANCE["pitch"] - duck)
    hop_in = gs._joint_track(kin, "down_into_the_room", 1.6, [(0.0, q1), (0.12, release), (0.3, tuck2), (0.62, tuck2), (1.0, q_stand)], b_in,
                             source="gibbon_scenarios.fire: the hop down off the sill onto the first floor")
    fire_c = np.array([X + t / 2 + p["table_x"] - 0.1, p["table_y"] + 0.05, F + 0.75])
    x_stop = fire_c[0] - p["spray_range"]
    walk_in = max(x_stop - x_land, 0.2)
    base_f = np.array([x_land + walk_in, 0.0, F + gm.STANCE["h"]])
    fight = _fire_fight(kin, q_stand, base_f, fire_c, gm.STANCE["pitch"])
    return ([("motion", ride), ("motion", hop_off), ("motion", take)] + segs + [("motion", over), ("motion", hop_in), ("gait", walk_in / v, 0.0, F)]
            + [("motion", m_) for m_ in fight])


gs.DEFAULTS["fire"] = FIRE
gs.SCENARIOS["fire"] = scenario_fire
gs.COMPOSITE["fire"] = fire_props
