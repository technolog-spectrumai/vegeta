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
from vegeta.chiron import Link, Prop

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
