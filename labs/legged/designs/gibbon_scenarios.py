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
    return {**gs.WINDOW, "x": p["x"] + p["wall_gap"], "y": p["window_y"], "glass": 1, "sash": 1, "table_x": p["table_x"], "table_y": p["table_y"]}


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
        m_ = gt.contact_motion(kin, "tail_off_the_trunk", 1.6, lambda u: T_settle + mv(u), shift(bc, crouch, pc_, 0.35),
                               q_init=qc, extra=x_tail_off, reg=gs.IK_REG, iters=gs.IK_ITERS, q_ref=q_crouch,
                               source="gibbon_scenarios.heist: on the sill on four, the tail unwound last and the body settled in the opening")
        steps.append(m_)
        q_sill = gt._last(kin, m_, qc)
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
