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
        "room_d": 5.0, "table_x": 2.4, "table_y": 0.25, "quad_x0": -6.0, "quad_x": 2.6, "quad_y": 1.25, "spray_range": 1.1, "climb_s": 3.0,
        "lean_deg": 75.0, "carry_v": 0.25}
RED = (0.72, 0.1, 0.08, 1.0)
CHUCK = {"seat": (-0.05, 0.0, 0.92), "hook": (0.35, -0.78, 1.0), "reel": (-0.6, 0.0, 1.05), "bars": (0.5, 0.28, 1.05), "pegs": (0.12, 0.44, 0.46),
         "rack": (1.45, -0.86, 0.66)}                                     #: the ladder's butt on Chuck's rack: on its right side, the rungs upright, the butt forward
LADDER = {"len": 2.9, "ext": 2.1, "ry": 0.16, "r": 0.02, "rung_r": 0.016, "sp": 0.3, "rung0": 0.3, "fly_off": 0.05, "grip": 1.45, "carry_h": 1.30}
#: the two-section ladder: each section ``len`` long, the fly out ``ext`` (seven rungs) on its front ``fly_off`` ahead of the base's
#: rails; its frame at the butt, x up the rails, y along the rungs, z out of its front (the climbing side)
RADIO_ON_THORAX = (np.array([0.1, 0.1, -0.03]), 0.0)                     #: the walkie-talkie clipped on the chest strap, left of the sternum, upright
LADDER_GEOMS = set()                                                      #: the ladder's geom names (not checked while it is moving: it is modelled where it ends, leaning)
FACE = {"front": 0.075, "z": 0.195, "mask_r": 0.06, "eye_y": 0.026, "eye_r": 0.018}   #: the face on the thorax's frame: the head's front, the eyes' height [m]


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


def _lad_R(a, w):
    """The ladder's frame: x up the rails ``a``, y along the rungs ``w``, z out of its front."""
    a, w = gs._unit(np.asarray(a, dtype=float)), gs._unit(np.asarray(w, dtype=float))
    return np.stack([a, w, np.cross(a, w)], axis=1)


def _quat(R):
    """(w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def _lean_profile(p):
    """The points of the house the ladder can rest on, in the xz plane: the wall's face up to the sill, the sill
    plate's outer edge (its bottom and top corners)."""
    Xw, S = p["wall_x"] - p["wall_t"] / 2, _fire_sill(p)
    zs = np.arange(0.0, S, 0.01)
    return np.concatenate([np.stack([np.full_like(zs, Xw), zs], axis=1), [[Xw - 0.04, S], [Xw - 0.04, S + 0.06]]])


def ladder_angle(p, bx, length):
    """The ladder's angle from the ground [rad] butt at ``bx``, ``length`` out, leant on the house: tipped towards
    the wall until a rail's back rests on the first point it meets (the fly's rails sit ``fly_off`` further out)."""
    L = LADDER
    pts = _lean_profile(p)
    d = np.hypot(pts[:, 0] - bx, pts[:, 1] - L["r"])
    ok = d <= length
    if not ok.any():
        return math.pi / 2
    off = np.where(d > L["len"], L["r"] + L["fly_off"], L["r"])
    th = np.arctan2(pts[:, 1] - L["r"], pts[:, 0] - bx) + np.arcsin(np.minimum(off / d, 1.0))
    return float(th[ok].max())


def ladder_foot(p):
    """The butt's x for the ladder out full to lean at ``lean_deg`` on the sill's edge."""
    L, want = LADDER, math.radians(p["lean_deg"])
    Xw = p["wall_x"] - p["wall_t"] / 2
    lo, hi = Xw - 2.5, Xw - 0.3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ladder_angle(p, mid, L["len"] + L["ext"]) > want:          # too steep: the butt further out
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def ladder_pose_leant(p, bx, ext):
    """(origin, R) of the ladder leant on the house, butt at ``bx``, the fly out ``ext``."""
    th = ladder_angle(p, bx, LADDER["len"] + ext)
    return np.array([bx, 0.0, LADDER["r"]]), _lad_R((math.cos(th), 0.0, math.sin(th)), (0.0, 1.0, 0.0))


def ladder_final(p):
    """(origin, R, ext) where the ladder ends: out full, leant on the sill's edge."""
    bx = ladder_foot(p)
    O, R = ladder_pose_leant(p, bx, LADDER["ext"])
    return O, R, LADDER["ext"]


def ladder_on_chuck(chuck_xyz):
    """(origin, R) of the ladder in its rack on Chuck's right side: the rungs upright, the butt forward, its front
    out from Chuck."""
    return np.asarray(chuck_xyz, dtype=float) + np.array(CHUCK["rack"]), _lad_R((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))


def ladder_rail_x(p, z, front=True):
    """The x of the ladder's (fly's, ``front``) rail axis at height ``z`` where it ends."""
    O, R, _ = ladder_final(p)
    off = R[:, 2] * (LADDER["fly_off"] if front else 0.0)
    s = (z - O[2] - off[2]) / R[2, 0]
    return float(O[0] + off[0] + s * R[0, 0])


def ladder_props(p):
    """The two sections, each its own loose body (Chuck carries them, the monkey raises and extends them): the base
    section with its feet and the halyard's pulley at its top, the fly ahead of it with its rubber tips. Both modelled
    where they end, leant on the sill's edge: they are checked solid there (the climb), not while moving."""
    L = LADDER
    alu, rub, blk = (0.78, 0.8, 0.84, 1.0), (0.12, 0.12, 0.12, 1.0), (0.2, 0.2, 0.22, 1.0)
    O, R, ext = ladder_final(p)
    LADDER_GEOMS.clear()

    def section(tag, off):
        g = [gs._cap(f"lad_{tag}_rail_{t_}", L["r"], (0.0, sy * L["ry"], off), (L["len"], sy * L["ry"], off), rgba=alu, mass=3.0) for sy, t_ in ((1, "l"), (-1, "r"))]
        s, k = L["rung0"], 0
        while s < L["len"] - 0.1:
            g.append(gs._cap(f"lad_{tag}_rung_{k}", L["rung_r"], (s, -L["ry"], off), (s, L["ry"], off), rgba=alu, mass=0.4))
            s += L["sp"]; k += 1
        return g
    base = section("base", 0.0)
    for sy, t_ in ((1, "l"), (-1, "r")):                                    # the feet, the fly's guides at the base's top
        base.append(Geom(f"lad_foot_{t_}", "box", (0.04, 0.03, 0.025), pos=(0.02, sy * L["ry"], 0.0), mass=0.2, role="visual", rgba=rub))
        base.append(Geom(f"lad_guide_{t_}", "box", (0.03, 0.03, 0.04), pos=(L["len"] - 0.05, sy * L["ry"], 0.03), mass=0.1, role="visual", rgba=blk))
    base.append(Geom("lad_pulley", "cylinder", (0.035, 0.012), pos=(L["len"] - 0.05, 0.0, 0.09), quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=0.2, role="visual", rgba=blk))
    fly = section("fly", L["fly_off"])
    for sy, t_ in ((1, "l"), (-1, "r")):
        fly.append(Geom(f"lad_tip_{t_}", "sphere", (0.026,), pos=(L["len"], sy * L["ry"], L["fly_off"]), mass=0.05, role="visual", rgba=rub))
    LADDER_GEOMS.update(g_.name for g_ in base + fly)
    q = _quat(R)
    return (Prop(Link("ladder_base", pos=tuple(O), quat=q, geoms=base), free=True),
            Prop(Link("ladder_fly", pos=tuple(O + ext * R[:, 0]), quat=q, geoms=fly), free=True))


def radio_prop(p):
    blk, yel, grn = (0.1, 0.1, 0.11, 1.0), (0.95, 0.75, 0.1, 1.0), (0.2, 0.9, 0.3, 1.0)
    geoms = [Geom("radio_body", "box", (0.014, 0.024, 0.045), pos=(0.0, 0.0, 0.0), mass=0.2, role="visual", rgba=blk),
             Geom("radio_face", "box", (0.002, 0.018, 0.016), pos=(0.015, 0.0, 0.012), mass=0.01, role="visual", rgba=yel),
             Geom("radio_antenna", "capsule", (0.004, 0.025), pos=(0.0, 0.012, 0.07), mass=0.01, role="visual", rgba=blk),
             Geom("radio_led", "sphere", (0.005,), pos=(0.006, -0.012, 0.047), mass=0.01, role="visual", rgba=grn)]
    return Prop(Link("radio", pos=(-6.0, 2.0, 4.0), geoms=geoms), free=True)


FACE_ON_THORAX = np.array([0.0, 0.0, 0.0])                                 #: the face's frame on the thorax (its geoms placed from it)


def face_prop(p):
    """A sweet, brave little monkey face on the head: big round eyes with a sparkle, a pale muzzle with a smile, round
    ears, rosy cheeks and eyebrows set firm. Drawn on the head (visual only)."""
    skin, white, blk, pink, brow, ear_in = (0.86, 0.72, 0.56, 1.0), (0.98, 0.98, 0.98, 1.0), (0.07, 0.05, 0.05, 1.0), (0.95, 0.55, 0.55, 1.0), (0.3, 0.18, 0.1, 1.0), (0.93, 0.66, 0.6, 1.0)
    F = FACE
    v = lambda name, typ, size, pos, rgba, quat=(1.0, 0.0, 0.0, 0.0): Geom(name, typ, size, pos=tuple(pos), quat=quat, mass=0.01, role="visual", rgba=rgba)
    cx, cz = F["front"], F["z"]
    g = [v("face_mask", "sphere", (F["mask_r"],), (cx - F["mask_r"] + 0.012, 0.0, cz + 0.005), skin),
         v("face_muzzle", "sphere", (0.03,), (cx + 0.008, 0.0, cz - 0.03), skin),
         v("face_nose_l", "sphere", (0.005,), (cx + 0.036, 0.008, cz - 0.02), blk),
         v("face_nose_r", "sphere", (0.005,), (cx + 0.036, -0.008, cz - 0.02), blk)]
    for i in range(11):                                                      # the smile: a curve across the muzzle, its corners up
        a = math.radians(-60 + 12 * i)
        g.append(v(f"face_smile_{i}", "sphere", (0.003,), (cx + 0.036 - 0.008 * abs(math.sin(a)), 0.019 * math.sin(a), cz - 0.044 + 0.014 * (1 - math.cos(a))), blk))
    for sy, t_ in ((1, "l"), (-1, "r")):
        ey = sy * F["eye_y"]
        g += [v(f"face_eye_{t_}", "sphere", (F["eye_r"],), (cx, ey, cz + 0.012), white),
              v(f"face_pupil_{t_}", "sphere", (F["eye_r"] * 0.62,), (cx + F["eye_r"] * 0.55, ey, cz + 0.01), blk),
              v(f"face_spark_{t_}", "sphere", (F["eye_r"] * 0.22,), (cx + F["eye_r"] * 1.05, ey + sy * 0.004, cz + 0.018), white),
              v(f"face_cheek_{t_}", "sphere", (0.011,), (cx - 0.004, sy * 0.042, cz - 0.022), pink),
              v(f"face_ear_{t_}", "cylinder", (0.03, 0.006), (cx - 0.06, sy * 0.078, cz + 0.008), skin, quat=(0.70710678, 0.70710678, 0.0, 0.0)),
              v(f"face_ear_in_{t_}", "cylinder", (0.019, 0.007), (cx - 0.058, sy * 0.079, cz + 0.008), ear_in, quat=(0.70710678, 0.70710678, 0.0, 0.0)),
              # the brows: a firm slant, low at the middle - brave
              v(f"face_brow_{t_}", "box", (0.004, 0.017, 0.0035), (cx + 0.004, ey, cz + 0.037), brow, quat=(math.cos(sy * 0.18), math.sin(sy * 0.18), 0.0, 0.0))]
    return Prop(Link("monkey_face", pos=(-6.0, 3.0, 4.0), geoms=g), free=True)


def chuck_prop(p):
    """Chuck, simplified: the quad's frame and wheels, the water tank, the pump, the hose reel, the water cannon on the
    nose, the handlebars and seat; the nozzle's hook and the ladder's rack on its right side. Its frame at the ground
    under the middle."""
    blk, grey, steel = (0.08, 0.08, 0.09, 1.0), (0.62, 0.64, 0.66, 1.0), (0.4, 0.42, 0.45, 1.0)
    hk = CHUCK["hook"]
    geoms = [gs._box("chuck_frame", (1.0, 0.34, 0.08), (0.0, 0.0, 0.5), rgba=RED, mass=200.0),
             gs._box("chuck_tank", (0.32, 0.42, 0.24), (-0.6, 0.0, 0.82), rgba=grey, mass=200.0),
             Geom("chuck_reel", "cylinder", (0.24, 0.16), pos=CHUCK["reel"], quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=20.0, role="link", rgba=RED),
             gs._box("chuck_pump", (0.12, 0.1, 0.05), (-0.32, 0.0, 0.63), rgba=steel, mass=30.0),
             gs._box("chuck_seat", (0.22, 0.17, 0.05), (CHUCK["seat"][0], 0.0, 0.75), rgba=blk, mass=5.0),
             gs._box("chuck_tankcover", (0.3, 0.3, 0.06), (0.35, 0.0, 0.66), rgba=RED, mass=10.0),
             gs._cap("chuck_column", 0.03, (0.45, 0.0, 0.65), (0.5, 0.0, 1.02), rgba=steel, mass=3.0),
             gs._cap("chuck_bars", 0.018, (0.5, -0.3, 1.04), (0.5, 0.3, 1.04), rgba=blk, mass=2.0),
             gs._cap("chuck_cannon_post", 0.04, (0.9, 0.0, 0.58), (0.9, 0.0, 0.8), rgba=steel, mass=4.0),
             gs._cap("chuck_cannon", 0.035, (0.9, 0.0, 0.84), (1.15, 0.0, 0.94), rgba=(0.78, 0.62, 0.25, 1.0), mass=5.0),
             gs._cap("chuck_hook_post", 0.012, (hk[0], -0.34, 0.55), (hk[0], -0.6, hk[2] - 0.02), rgba=steel, mass=0.3),
             gs._cap("chuck_hook_arm", 0.012, (hk[0], -0.6, hk[2] - 0.02), (hk[0], hk[1], hk[2] - 0.02), rgba=steel, mass=0.3)]
    ry, rk = LADDER["ry"], CHUCK["rack"]
    for i, xr in enumerate((-0.9, 0.9)):                                     # the ladder's rack: two arms out under its lower rail, a lip outside it
        geoms.append(gs._box(f"chuck_rack_arm_{i}", (0.025, 0.3, 0.015), (xr, -0.34 - 0.3, rk[2] - ry - LADDER["r"] - 0.015), rgba=steel, mass=1.0))
        geoms.append(gs._box(f"chuck_rack_lip_{i}", (0.025, 0.01, 0.05), (xr, rk[1] - LADDER["r"] - 0.035, rk[2] - ry - LADDER["r"] + 0.02), rgba=steel, mass=0.3))
    for sy, tag in ((1, "l"), (-1, "r")):                                    # the foot pegs out beside the frame
        geoms.append(gs._cap(f"chuck_peg_{tag}", 0.015, (CHUCK["pegs"][0] - 0.08, sy * 0.34, CHUCK["pegs"][2] - 0.03), (CHUCK["pegs"][0] - 0.08, sy * 0.52, CHUCK["pegs"][2] - 0.03), rgba=steel, mass=0.5))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        geoms.append(Geom(f"chuck_wheel_{i}", "cylinder", (0.33, 0.12), pos=(sx * 0.72, sy * 0.6, 0.33), quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=20.0, role="link", rgba=blk))
        geoms.append(Geom(f"chuck_hub_{i}", "cylinder", (0.12, 0.125), pos=(sx * 0.72, sy * 0.6, 0.33), quat=(0.70710678, 0.70710678, 0.0, 0.0), mass=2.0, role="visual", rgba=steel))
    return Prop(Link("chuck", pos=(-20.0, 5.0, 0.0), geoms=geoms), free=True)


def fire_props(p):
    return (house_prop(p), chuck_prop(p), backpack_prop(p), hat_prop(p), nozzle_prop(p), radio_prop(p), face_prop(p)) + ladder_props(p)


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


def _grip_frame(c, axis, wrist_dir):
    """The hand's frame closed on a bar along ``axis`` at ``c``, the wrist towards ``wrist_dir`` from the grip (as a
    rung's grip, whatever the bar's slope: the ladder turns from flat to upright in the hands)."""
    y = gs._unit(np.asarray(axis, dtype=float))
    e1 = -gs._unit(np.asarray(wrist_dir, dtype=float))
    e1 = gs._unit(e1 - (e1 @ y) * y)
    gamma = math.atan2(gs.GRIP[2], gs.GRIP[0])
    best = None
    for g in (gamma, -gamma):
        x = gs._rot(y, g) @ e1
        R = np.stack([x, y, np.cross(x, y)], axis=1)
        score = float((R @ gs.GRIP) @ e1)
        if best is None or score > best[0]:
            best = (score, R)
    return best[1]


def _feet_at(kin, q_stand, base):
    """{side: [(body, local, world)]} of the feet standing at ``base`` (upright)."""
    kin.set(q_stand)
    return {sd: [(bd, l, kin.world(bd, l) + np.asarray(base, dtype=float)) for bd, l in gm._contact_points(kin)[sd]] for sd in ("L", "R")}


def _stepping(feet, D, n, f, lift=0.04):
    """The feet stepping ``D`` (xy) in ``n`` steps each, left first, ``f`` of the way: (targets, the base's share of D)."""
    out, done_ = [], []
    for si, sd in enumerate(("L", "R")):
        slots, done, prog = f * 2 * n, 0, 0.0
        for j in range(n):
            s0 = 2 * j + si
            if slots >= s0 + 1:
                done += 1
            elif slots > s0:
                prog = mk.smooth(slots - s0)
        share = (done + prog) / n
        done_.append(share)
        dz = lift * math.sin(math.pi * prog) if 0.0 < prog < 1.0 else 0.0
        out += [(bd, l, w + np.array([D[0] * share, D[1] * share, dz])) for bd, l, w in feet[sd]]
    return out, 0.5 * (done_[0] + done_[1])


def _ladder_hands(kin, O, R, grips, base):
    """Hand targets on the ladder at pose (O, R): ``grips`` {side: (rail +1/-1 or 0 for the halyard, s up the rails)}."""
    L = LADDER
    a, w, n = R[:, 0], R[:, 1], R[:, 2]
    out = []
    for sd, (rail, s) in grips.items():
        sy = 1.0 if sd == "L" else -1.0
        if rail == 0:                                                      # the halyard: down the ladder's front
            c = O + s * a + 0.1 * n
        else:
            c = O + s * a + rail * L["ry"] * w + (L["fly_off"] if s > L["len"] else 0.0) * n
        sh = np.asarray(base, dtype=float) + np.array([0.02, sy * 0.12, 0.25])
        out += gs._hand_at(kin, sd, c, _grip_frame(c, a, sh - c))
    return out


def scenario_fire(kin, q_stand, p):
    """Chuck drives up to the house with the monkey on it; the monkey hops off, takes the nozzle from Chuck's hook and
    clips it to its pack; lifts the extension ladder out of Chuck's rack, overhead, carries it to the house, plants
    its butt and walks it up hand over hand, leans it on the wall and hauls the halyard until the fly's tips rest on
    the sill's edge; climbs it, the hose paying out; over the sill between the rails, down onto the first floor, to
    the burning table; draws the nozzle, opens its valve and sprays the fire out; then radios that the mission is
    accomplished, the nozzle held high."""
    import os
    upto = os.environ.get("VEGETA_FIRE_UPTO")
    S_ = gm.scale()
    X, t, F = p["wall_x"], p["wall_t"], p["floor"]
    Lad = LADDER
    sill = _fire_sill(p)
    top = sill + 0.06
    v = gs.walk_speed()
    h0, pt = gm.STANCE["h"], gm.STANCE["pitch"]
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
    xq, yq = p["quad_x"], p["quad_y"]
    seat = np.array([xq + CHUCK["seat"][0] + 0.12, yq, 0.80 + gs.SILL_CROUCH[1] * S_])
    # the ride: crouched on the seat, the hands on the bars
    pr = 0.35
    Rr = gm.base_rot(pr)
    tg = []
    for sd, sy in (("L", 1.0), ("R", -1.0)):
        tg += gs.foot_targets(kin, sd, gs.point(np.array([xq + CHUCK["seat"][0] - 0.02, yq + sy * 0.09, 0.80]), pitch=0.0))
        tg += gs.hand_targets(kin, sd, gs.bar(np.array([xq + CHUCK["bars"][0], yq + sy * CHUCK["bars"][1], CHUCK["bars"][2]]), axis=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0), r=0.018, wrist_dir=(-0.6, 0.0, -0.8)))
    q_ride, err_ride = kin.ik([(bd, l, Rr.T @ (np.asarray(wp) - seat)) for bd, l, wp in tg], q_init=hang_ref(kin, q_stand), q_ref=hang_ref(kin, q_stand),
                              joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    for sd in ("L", "R"):
        sr.set_curl(q_ride, kin, sd, gs.CURL_CLOSED)
    print(f"fire: riding pose reach {err_ride:.3f} m")
    D_RIDE = 8.0
    x0 = p["quad_x0"]
    ride = gs._joint_track(kin, "ride_in_on_chuck", D_RIDE, [(0.0, q_ride), (1.0, q_ride)],
                           lambda u: (x0 + (xq - x0) * mk.smooth(u) + seat[0] - xq, yq, seat[2], pr), frames=80,
                           source="gibbon_scenarios.fire: on Chuck's seat as it drives itself up to the house")
    ride.params = {"quad": {"x0": x0, "x1": xq, "y": yq}}
    # the hop off: a crouch, a spring up and sideways over the racked ladder, the legs kept tucked till it is cleared
    land = np.array([xq + 0.05, 0.15, h0])                                 # beside Chuck, in reach of its hook and rack
    G_, rise = 9.81, 0.3
    apex = seat[2] + rise
    t_up, t_dn = math.sqrt(2 * rise / G_), math.sqrt(2 * (apex - land[2]) / G_)
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
            return c[0], c[1], c[2], pr + (pt - pr) * mk.smooth(s_ / T_F)
        s_ = (tt - T_C - T_F) / T_L
        return land[0], land[1], land[2] - 0.08 * math.sin(math.pi * s_), pt
    hop_off = gs._joint_track(kin, "hop_off_chuck", Dh, [(0.0, q_ride), (T_C / Dh, tuck), ((T_C + 0.72 * T_F) / Dh, tuck), ((T_C + T_F) / Dh, q_stand), (1.0, q_stand)], hop_base,
                              frames=int(Dh * 40), source="gibbon_scenarios.fire: off the seat, a ballistic hop over the racked ladder to the ground beside Chuck")
    # the hose: the nozzle off Chuck's hook (over the racked ladder's top rail), clipped into the holster on the pack
    base_g = land.copy()
    Rb = gm.base_rot(pt)
    kin.set(q_stand)
    feet = [(bd, l, Rb @ kin.world(bd, l) + base_g) for sd in ("L", "R") for bd, l in gm._contact_points(kin)[sd]]
    th = kin.body("thorax")
    R_th, p_th = Rb @ kin.data.xmat[th].reshape(3, 3), Rb @ kin.data.xpos[th] + base_g
    hook = np.array([xq + CHUCK["hook"][0], yq + CHUCK["hook"][1], CHUCK["hook"][2]])
    R_hookN = np.eye(3)
    grip_k = hook + np.array([0.0, 0.0, NOZZLE["grip"] - 0.12])
    fr_k = gs.hand_frame_on(gs.bar(grip_k, axis=(0.0, 0.0, 1.0), n=(0.0, -1.0, 0.0), r=NOZZLE["r"], wrist_dir=gs._unit(np.array([-0.3, -0.8, -0.3]))), "R")
    root_hol = R_th @ HOLSTER_ON_THORAX[0] + p_th
    grip_hol = root_hol + R_th[:, 2] * NOZZLE["grip"]
    fr_hol = gs.hand_frame_on(gs.bar(grip_hol, axis=R_th[:, 2], n=-R_th[:, 1], r=NOZZLE["r"], wrist_dir=gs._unit(-R_th[:, 2] * 0.3 + R_th[:, 0] * 0.6 - R_th[:, 1] * 0.5)), "R")
    Rh0, ph0 = gs._hand_pose(kin, q_stand, base_g, pt, "R")
    over_rail = np.array([0.0, 0.0, 0.14])

    def t_take(u):
        if u < 0.4:
            f = mk.smooth(u / 0.4)
            return feet + gs._hand_at(kin, "R", (1 - f) * ph0 + f * grip_k + over_rail * math.sin(math.pi * f), gs._slerp(Rh0, fr_k, f))
        if u < 0.5:
            return feet + gs._hand_at(kin, "R", grip_k, fr_k)
        f = mk.smooth((u - 0.5) / 0.4) if u < 0.9 else 1.0
        return feet + gs._hand_at(kin, "R", (1 - f) * grip_k + f * grip_hol + (over_rail + np.array([0.0, -0.1, 0.0])) * math.sin(math.pi * f), gs._slerp(fr_k, fr_hol, f))
    fixed = lambda u: (base_g[0], base_g[1], base_g[2], pt)
    if gt.SOLID is not None:
        gt.SOLID.ignore = set(LADDER_GEOMS)
    take = gt.contact_motion(kin, "take_the_hose", 3.2, t_take, fixed, q_init=q_stand,
                             extra=lambda u, q_: sr.set_curl(q_, kin, "R", gs.CURL_CLOSED if 0.42 < u < 0.92 else gs.CURL_OPEN),
                             reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the nozzle off Chuck's hook, clipped to the pack")
    q_k, _ = kin.ik([(bd, l, Rb.T @ (np.asarray(wp) - base_g)) for bd, l, wp in t_take(0.45)], q_init=q_stand, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    Rhk, phk = gs._hand_pose(kin, q_k, base_g, pt, "R")
    take.params = {"hook": hook.tolist(), "take_u": 0.45, "clip_u": 0.9, "nozzle_T": ((Rhk.T @ R_hookN).tolist(), (Rhk.T @ (hook - np.array([0.0, 0.0, 0.12]) - phk)).tolist())}
    q_t = gt._last(kin, take, q_stand)
    out = [("motion", ride), ("motion", hop_off), ("motion", take)]
    # ---- the ladder
    O_f, R_f, ext_f = ladder_final(p)
    bx = O_f[0]
    th_r = ladder_angle(p, bx, Lad["len"])                                  # the base section alone leant on the wall
    rung = lambda k: O_f + (Lad["rung0"] + Lad["sp"] * k) * R_f[:, 0] + (Lad["fly_off"] if Lad["rung0"] + Lad["sp"] * k >= ext_f + Lad["rung0"] - 1e-6 else 0.0) * R_f[:, 2]
    stop_x = float(rung(0)[0]) - 0.25
    sg, hc = Lad["grip"], Lad["carry_h"]
    O_r, R_r = ladder_on_chuck(np.array([xq, yq, 0.0]))
    c0 = O_r + sg * R_r[:, 0]
    x_m0 = base_g[0]
    c1 = np.array([x_m0 + 0.1, base_g[1], hc])

    def lad_take(u):
        if u < 0.25:
            c, phi = c0.copy(), 0.0
        elif u < 0.4:
            c, phi = c0 + np.array([0.0, 0.0, 0.1 * mk.smooth((u - 0.25) / 0.15)]), 0.0
        else:
            f = min(1.0, (u - 0.4) / 0.4)
            cl = lambda v_: max(0.0, min(1.0, v_))
            phi = 0.5 * math.pi * mk.smooth(cl(f / 0.8))
            c = np.array([c0[0] + (c1[0] - c0[0]) * mk.smooth(f), c0[1] + (c1[1] - c0[1]) * mk.smooth(cl((f - 0.35) / 0.65)),
                          c0[2] + 0.1 + (c1[2] - c0[2] - 0.1) * mk.smooth(cl(f / 0.6))])
        R = _lad_R((-1.0, 0.0, 0.0), (0.0, math.sin(phi), -math.cos(phi)))
        return c - sg * R[:, 0], R, 0.0
    hand_t = {sd: gs._hand_pose(kin, q_t, base_g, pt, sd) for sd in ("L", "R")}

    def blend_to(sd, tgt, f):
        """The hand from where it was at the start of the take to the target targets, ``f`` of the way."""
        R0, p0 = hand_t[sd]
        c_t = np.mean([np.asarray(x[2]) for x in tgt], axis=0)
        return [(b_, l_, (1 - f) * (p0 + R0 @ np.asarray(l_)) + f * np.asarray(w_) + np.array([0.0, 0.0, 0.08]) * math.sin(math.pi * f)) for b_, l_, w_ in tgt]

    def t_lad_take(u):
        O, R, _ = lad_take(u)
        gL = (-1, sg - 0.15) if u < 0.82 else (1, sg)
        gR = (-1, sg + 0.15 - 0.15 * mk.smooth(max(0.0, min(1.0, (u - 0.9) / 0.1))))
        tg_ = []
        for sd, gg in (("L", gL), ("R", gR)):
            hand = _ladder_hands(kin, O, R, {sd: gg}, base_g)
            if u < 0.25:
                hand = blend_to(sd, hand, mk.smooth(u / 0.25))
            elif sd == "L" and 0.8 <= u < 0.9:                              # the left hand off the top rail, over to the other
                hA = _ladder_hands(kin, O, R, {sd: (-1, sg - 0.15)}, base_g)
                hB = _ladder_hands(kin, O, R, {sd: (1, sg)}, base_g)
                f = mk.smooth((u - 0.8) / 0.1)
                hand = [(b_, l_, (1 - f) * np.asarray(a_[2]) + f * np.asarray(b2[2]) + np.array([0.0, 0.0, -0.06]) * math.sin(math.pi * f)) for b_, l_, a_, b2 in ((x[0], x[1], x, y) for x, y in zip(hA, hB))]
            tg_ += hand
        return feet + tg_

    def x_lad_take(u, q_):
        sr.set_curl(q_, kin, "R", gs.CURL_CLOSED if u > 0.22 else gs.CURL_OPEN)
        sr.set_curl(q_, kin, "L", gs.CURL_CLOSED if (0.22 < u < 0.8 or u > 0.88) else gs.CURL_OPEN)
        return q_
    take_l = gt.contact_motion(kin, "take_the_ladder", 4.5, t_lad_take, fixed, q_init=q_t, extra=x_lad_take, reg=gs.IK_REG, iters=gs.IK_ITERS,
                               q_ref=q_stand, source="gibbon_scenarios.fire: the ladder lifted off Chuck's rack and turned flat overhead")
    take_l.params = {"ladder": lad_take}
    q_l = gt._last(kin, take_l, q_t)
    out.append(("motion", take_l))
    # the carry: overhead, flat, butt first, to where its butt will be planted
    th1 = math.pi - math.asin((hc - Lad["r"]) / sg)
    x_g = bx + sg * math.cos(th1)                                           # the grips over the place: the butt swings down onto its spot
    x_m1 = x_g - 0.1
    D_c = np.array([x_m1 - x_m0, -base_g[1]])                              # back onto the ladder's line on the way
    n_c = max(1, math.ceil(np.linalg.norm(D_c) / 0.18))
    T_c = max(abs(D_c[0]) / p["carry_v"], 1.0)
    feet_c = _feet_at(kin, q_stand, base_g)
    flat = _lad_R((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    GR = {"L": (1, sg), "R": (-1, sg)}

    def carry_state(u):
        tg_, share = _stepping(feet_c, D_c, n_c, u)
        bxy = base_g[:2] + D_c * share
        c = np.array([bxy[0] + 0.1, bxy[1], hc + 0.01 * math.sin(2 * math.pi * n_c * u)])
        return tg_, bxy, c - sg * flat[:, 0], flat
    carry = gt.contact_motion(kin, "carry_the_ladder", T_c, lambda u: carry_state(u)[0] + _ladder_hands(kin, *carry_state(u)[2:], GR, np.r_[carry_state(u)[1], h0]),
                              lambda u: (*carry_state(u)[1], h0, pt), q_init=q_l, extra=lambda u, q_: sr.set_curl(sr.set_curl(q_, kin, "L", gs.CURL_CLOSED), kin, "R", gs.CURL_CLOSED),
                              reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the ladder carried overhead to the house")
    carry.params = {"ladder": lambda u: (*carry_state(u)[2:], 0.0)}
    q_c = gt._last(kin, carry, q_l)
    out.append(("motion", carry))
    # the raise: the butt swung down onto its spot, then walked up hand over hand past upright, leant on the wall
    x_p2 = bx - 0.3
    base_r = np.array([x_m1, 0.0, h0])
    feet_r = _feet_at(kin, q_stand, base_r)
    D2, D3 = np.array([x_p2 - x_m1, 0.0]), np.array([stop_x - x_p2, 0.0])
    n2 = max(1, math.ceil(abs(D2[0]) / 0.16))
    O_b = np.array([bx, 0.0, Lad["r"]])
    U1, U2 = 0.25, 0.72

    def raise_state(u):
        """(feet targets, base xy, ladder origin, R, grip s)."""
        if u < U1:
            f = mk.smooth(u / U1)
            th_ = math.pi + (th1 - math.pi) * f
            a = np.array([math.cos(th_), 0.0, math.sin(th_)])
            O = np.array([x_g, 0.0, hc]) - sg * a
            tg_, bxy = [x for sd in ("L", "R") for x in feet_r[sd]], base_r[:2]
            s = sg
        elif u < U2:
            f = (u - U1) / (U2 - U1)
            th_ = th1 + (math.pi / 2 - th1) * mk.smooth(f)
            O, s = O_b, sg + (1.33 - sg) * mk.smooth(f)
            tg_, share = _stepping(feet_r, D2, n2, f)
            bxy = base_r[:2] + D2 * share
        else:
            f = (u - U2) / (1.0 - U2)
            th_ = math.pi / 2 + (th_r - math.pi / 2) * mk.smooth(f)
            O, s = O_b, 1.33 + (1.0 - 1.33) * mk.smooth(f)
            feet2 = {sd: [(b_, l_, w_ + np.r_[D2, 0.0]) for b_, l_, w_ in feet_r[sd]] for sd in ("L", "R")}
            tg_, share = _stepping(feet2, D3, 1, min(1.0, f * 1.6))
            bxy = base_r[:2] + D2 + D3 * share
        R = _lad_R((math.cos(th_), 0.0, math.sin(th_)), (0.0, 1.0, 0.0))
        return tg_, bxy, O, R, s
    raise_ = gt.contact_motion(kin, "raise_the_ladder", 7.0, lambda u: raise_state(u)[0] + _ladder_hands(kin, raise_state(u)[2], raise_state(u)[3], {"L": (1, raise_state(u)[4]), "R": (-1, raise_state(u)[4])}, np.r_[raise_state(u)[1], h0]),
                               lambda u: (*raise_state(u)[1], h0, pt), q_init=q_c, extra=lambda u, q_: sr.set_curl(sr.set_curl(q_, kin, "L", gs.CURL_CLOSED), kin, "R", gs.CURL_CLOSED),
                               reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the butt planted, the ladder walked up hand over hand and leant on the wall")
    raise_.params = {"ladder": lambda u: (*raise_state(u)[2:4], 0.0)}
    q_r = gt._last(kin, raise_, q_c)
    out.append(("motion", raise_))
    # the extension: the halyard hauled hand over hand, a rung a pull, the fly sliding up till its tips are on the sill
    base_e = np.array([stop_x, 0.0, h0])
    feet_e = [x for sd in ("L", "R") for x in _feet_at(kin, q_stand, base_e)[sd]]
    NP = int(round(ext_f / Lad["sp"]))
    S_HI, S_LO, U0 = 1.15, 0.75, 0.12

    def ext_state(u):
        """(ladder origin, R, ext, {side: rope s or None (on a rail)}, the pulling hand)."""
        if u < U0:
            e, grips = 0.0, None
        else:
            k_f = (u - U0) / (1.0 - U0) * NP
            k = min(int(k_f), NP - 1)
            f = k_f - k
            e = Lad["sp"] * (k + mk.smooth(min(1.0, f / 0.8)))
            pull, free = ("R", "L") if k % 2 == 0 else ("L", "R")
            fp = mk.smooth(min(1.0, f / 0.8))
            ff = mk.smooth(max(0.0, min(1.0, (f - 0.1) / 0.8)))
            grips = {pull: S_HI + (S_LO - S_HI) * fp, free: S_LO + (S_HI - S_LO) * ff}
        e = min(e, ext_f)
        O, R = ladder_pose_leant(p, bx, e)
        return O, R, e, grips

    hand_r = {sd: gs._hand_pose(kin, q_r, base_e, pt, sd) for sd in ("L", "R")}

    def t_ext(u):
        O, R, e, grips = ext_state(u)
        if grips is None:                                                  # off the rails onto the halyard
            f = mk.smooth(u / U0)
            tg_ = []
            for sd, s_ in (("R", S_HI), ("L", S_LO)):
                hand = _ladder_hands(kin, O, R, {sd: (0, s_)}, base_e)
                R0, p0 = hand_r[sd]
                tg_ += [(b_, l_, (1 - f) * (p0 + R0 @ np.asarray(l_)) + f * np.asarray(w_)) for b_, l_, w_ in hand]
            return feet_e + tg_
        return feet_e + _ladder_hands(kin, O, R, {sd: (0, s_) for sd, s_ in grips.items()}, base_e)

    def rope_pts(u):
        O, R, e, grips = ext_state(u)
        a, n = R[:, 0], R[:, 2]
        fly_bottom, pulley = O + e * a + 0.07 * n, O + (Lad["len"] - 0.05) * a + 0.12 * n
        if grips is None:
            ss = sorted([S_HI, S_LO], reverse=True)
        else:
            ss = sorted(grips.values(), reverse=True)
        hands = [O + s_ * a + 0.1 * n for s_ in ss]
        return [fly_bottom, pulley] + hands + [np.array([hands[-1][0] - 0.05, 0.0, 0.02])]
    def x_ext(u, q_):
        _, _, _, grips = ext_state(u)
        for sd in ("L", "R"):
            closed = True
            if grips is not None:
                k_f = (u - U0) / (1.0 - U0) * NP
                f = k_f - min(int(k_f), NP - 1)
                k = min(int(k_f), NP - 1)
                free = "L" if k % 2 == 0 else "R"
                closed = not (sd == free and 0.05 < f < 0.85)
            elif u > 0.02:
                closed = u > 0.85 * U0
            sr.set_curl(q_, kin, sd, gs.CURL_CLOSED if closed else gs.CURL_OPEN)
        return q_
    extend = gt.contact_motion(kin, "extend_the_ladder", 1.0 + 0.85 * NP, t_ext, lambda u: (base_e[0], 0.0, h0, pt), q_init=q_r, extra=x_ext,
                               reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the halyard hauled hand over hand, the fly run up to the sill")
    extend.params = {"ladder": lambda u: ext_state(u)[:3], "rope": rope_pts}
    q_e = gt._last(kin, extend, q_r)
    tied = q_e.copy()
    for sd in ("L", "R"):
        sr.set_curl(tied, kin, sd, gs.CURL_OPEN)
    let_go = gs._joint_track(kin, "tie_off_the_halyard", 1.0, [(0.0, q_e), (0.3, tied), (1.0, q_stand)], lambda u: (base_e[0], 0.0, h0, pt),
                             source="gibbon_scenarios.fire: the halyard tied off, the hands off it")
    out += [("motion", extend), ("motion", let_go)]
    if gt.SOLID is not None:
        gt.SOLID.ignore = set()
    if upto == "ladder":
        return out
    # the climb: the hands on the rails, the feet on the rungs (the fly's, ahead, once past the base's top)
    a_f, w_f, n_f = R_f[:, 0], R_f[:, 1], R_f[:, 2]
    th_f = math.atan2(a_f[2], a_f[0])

    def rail(sy, s):
        return gs.bar(O_f + s * a_f + sy * Lad["ry"] * w_f + (Lad["fly_off"] if s >= ext_f + 0.05 else 0.0) * n_f, axis=a_f, n=n_f, r=Lad["r"])
    hold = lambda k: {"L_hand": rail(1.0, Lad["rung0"] + (k + 2) * Lad["sp"] + 0.05), "R_hand": rail(-1.0, Lad["rung0"] + (k + 2) * Lad["sp"] + 0.05),
                      "L": gs.bar(rung(k) + 0.11 * w_f, r=Lad["rung_r"], tip=0.3), "R": gs.bar(rung(k) - 0.11 * w_f, r=Lad["rung_r"], tip=0.3)}
    base_c = lambda k: rung(k) + np.array([-0.3 + 0.35 / math.tan(th_f), 0.0, 0.35])
    n_rungs = sum(1 for k in range(40) if rung(k)[2] < sill - 0.15)
    n = n_rungs - 1
    segs = gs._climb_sequence(kin, q_stand, hold, base_c, n, stop_x=stop_x, trials=gs.VERTICAL_TRIALS, duration=p["climb_s"])
    segs = segs[1:]                                                        # no walk up: it is stood there from the haul
    q = q_stand.copy()
    for sg_ in segs:
        q = gt._last(kin, sg_[1], q)
    h_end = hold(n)
    frames = segs[0][1].params.get("frames", {})
    for g_, f_ in frames.items():
        h_end[g_]["sign"], h_end[g_]["phi"] = f_
    # over the sill: the hands up the rails, the body hoisted between them, each foot over the lip onto the sill, in
    b_end = base_c(n)
    pitch = gs.CLIMB_PITCH
    s_at = lambda z: (z - O_f[2] - Lad["fly_off"] * n_f[2]) / a_f[2]
    hands_hi = {sd: rail(1.0 if sd == "L" else -1.0, s_at(top + 0.6)) for sd in ("L", "R")}
    for sd in ("L", "R"):
        hands_hi[sd]["sign"], hands_hi[sd]["phi"] = h_end[f"{sd}_hand"]["sign"], h_end[f"{sd}_hand"]["phi"]
    feet_sill = {sd: gs.point(np.array([X - 0.02, 0.1 if sd == "L" else -0.1, top]), pitch=0.0, over=True) for sd in ("L", "R")}
    crouch = np.array([X + gs.SILL_CROUCH[0] * S_, 0.0, top + gs.SILL_CROUCH[1] * S_])
    hoist = np.array([ladder_rail_x(p, top) - 0.2, 0.0, top + 0.3])
    duck = gs.SILL_CROUCH[2]
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
        out_ = []
        for sd in ("L", "R"):
            g_ = f"{sd}_hand"
            a, b_ = A[g_]
            end = gs.hand_targets(kin, sd, hands_hi[sd])
            if u >= Dc[0] + 0.04:                                          # the feet on the sill: the hands let go of the rails
                pass
            elif u < a:
                out_ += gs.hand_targets(kin, sd, h_end[g_])
            elif u < b_:
                f = mk.smooth((u - a) / (b_ - a))
                st = gs.hand_targets(kin, sd, h_end[g_])
                out_ += [(e[0], e[1], (1 - f) * np.asarray(s_[2]) + f * np.asarray(e[2]) + np.array([-0.05, 0.0, 0.0]) * math.sin(math.pi * f)) for s_, e in zip(st, end)]
            else:
                out_ += end
            a, b_ = C[sd]
            if u < a:
                out_ += gs.foot_targets(kin, sd, h_end[sd])
            elif u < b_:
                ff = (u - a) / (b_ - a)
                c = gs.foot_swing(h_end[sd], feet_sill[sd], ff, lift=0.12, clear=0.1)
                out_ += gs.foot_targets(kin, sd, gs.point(feet_sill[sd]["c"], pitch=0.3 * (1.0 - mk.smooth(ff)), over=True), c)
            else:
                out_ += gs.foot_targets(kin, sd, feet_sill[sd])
        return out_
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
        return xb, 0.0, (1 - fz) * crouch[2] + fz * (F + h0) + 0.04 * math.sin(math.pi * fz), duck + fz * (pt - duck)
    hop_in = gs._joint_track(kin, "down_into_the_room", 1.6, [(0.0, q1), (0.12, release), (0.3, tuck2), (0.62, tuck2), (1.0, q_stand)], b_in,
                             source="gibbon_scenarios.fire: the hop down off the sill onto the first floor")
    fire_c = np.array([X + t / 2 + p["table_x"] - 0.1, p["table_y"] + 0.05, F + 0.75])
    x_stop = fire_c[0] - p["spray_range"]
    walk_in = max(x_stop - x_land, 0.2)
    base_f = np.array([x_land + walk_in, 0.0, F + h0])
    fight = _fire_fight(kin, q_stand, base_f, fire_c, pt)
    q_low = gt._last(kin, fight[-1], q_stand)
    radio = _radio_call(kin, q_stand, q_low, base_f, pt)
    return (out + segs + [("motion", over), ("motion", hop_in), ("gait", walk_in / v, 0.0, F)]
            + [("motion", m_) for m_ in fight] + [("motion", radio)])


def _radio_call(kin, q_stand, q0, base, pt):
    """The fire out: the left hand takes the walkie-talkie off the chest strap, holds it to the mouth - "mission
    accomplished" - and the right hand raises the nozzle high, brave."""
    Rb = gm.base_rot(pt)
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j and not j.startswith("tail")]
    kin.set(q_stand)
    feet = [(bd, l, Rb @ kin.world(bd, l) + base) for sd in ("L", "R") for bd, l in gm._contact_points(kin)[sd]]
    th = kin.body("thorax")
    R_th, p_th = Rb @ kin.data.xmat[th].reshape(3, 3), Rb @ kin.data.xpos[th] + base
    root = R_th @ RADIO_ON_THORAX[0] + p_th
    grip = root + R_th[:, 2] * 0.0
    sh_L = p_th + R_th @ np.array([0.0, 0.12, 0.1])
    fr_g = _grip_frame(grip, R_th[:, 2], sh_L - grip + R_th[:, 0] * 0.0 - R_th[:, 0] * 0.1)
    mouth = p_th + R_th @ np.array([0.14, 0.05, 0.16])
    fr_m = _grip_frame(mouth, gs._unit(R_th[:, 2] + 0.25 * R_th[:, 0]), sh_L - mouth + R_th @ np.array([0.0, 0.0, -0.2]))
    hand0 = {sd: gs._hand_pose(kin, q0, base, pt, sd) for sd in ("L", "R")}
    high = p_th + R_th @ np.array([0.1, -0.17, 0.55])
    fr_hi = _grip_frame(high, R_th[:, 2] + 0.0 * R_th[:, 0], R_th @ np.array([-0.1, -0.1, -1.0]))
    G0, G1, M0, M1, H0, H1 = 0.0, 0.18, 0.2, 0.34, 0.62, 0.78

    def t_call(u):
        RL0, pL0 = hand0["L"]
        if u < G1:
            f = mk.smooth(u / G1)
            L_ = gs._hand_at(kin, "L", (1 - f) * pL0 + f * grip, gs._slerp(RL0, fr_g, f))
        elif u < M0:
            L_ = gs._hand_at(kin, "L", grip, fr_g)
        elif u < M1:
            f = mk.smooth((u - M0) / (M1 - M0))
            L_ = gs._hand_at(kin, "L", (1 - f) * grip + f * mouth + R_th[:, 0] * 0.05 * math.sin(math.pi * f), gs._slerp(fr_g, fr_m, f))
        else:
            L_ = gs._hand_at(kin, "L", mouth, fr_m)
        RR0, pR0 = hand0["R"]
        if u < H0:
            R_ = gs._hand_at(kin, "R", pR0, RR0)
        else:
            f = mk.smooth(min(1.0, (u - H0) / (H1 - H0)))
            R_ = gs._hand_at(kin, "R", (1 - f) * pR0 + f * high, gs._slerp(RR0, fr_hi, f))
        return feet + L_ + R_
    fixed = lambda u: (base[0], base[1], base[2], pt)
    call = gt.contact_motion(kin, "call_mission_accomplished", 6.5, t_call, fixed, q_init=q0,
                             extra=lambda u, q_: sr.set_curl(sr.set_curl(q_, kin, "R", gs.CURL_CLOSED), kin, "L", gs.CURL_CLOSED if u > 0.15 else gs.CURL_OPEN),
                             reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_stand, source="gibbon_scenarios.fire: the walkie-talkie to the mouth, mission accomplished, the nozzle up")
    q_g, _ = kin.ik([(bd, l, Rb.T @ (np.asarray(wp) - base)) for bd, l, wp in t_call(0.19)], q_init=q0, q_ref=q_stand, joints=limbs, reg=gs.IK_REG, iters=gs.IK_ITERS)
    Rhg, phg = gs._hand_pose(kin, q_g, base, pt, "L")
    call.params = {"radio_T": ((Rhg.T @ R_th).tolist(), (Rhg.T @ (root - phg)).tolist()), "grab_u": 0.18, "talk_u": (0.36, 1.0), "high_u": H1}
    return call


gs.DEFAULTS["fire"] = FIRE
gs.SCENARIOS["fire"] = scenario_fire
gs.COMPOSITE["fire"] = fire_props
