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

__all__ = ["LADDER", "ladder_prop", "scenario_brachiate", "half_period", "hang_ref"]

LADDER = {"x": 1.2, "height": 1.55, "spacing": 0.8, "rungs": 8, "rung_radius": 0.02, "rail_y": 0.38, "swings": 5}
REACH = 0.58                     #: the shoulders' distance from the holding rung's grip, the arm a little bent [m]
GRIP_Y = 0.12                    #: the hands' grips either side of the ladder's middle [m]
TRUNK_SWING = 0.25               #: the trunk's pitch against the arm's angle from straight down
SHOULDER_Z = 0.272               #: the shoulders above the pelvis, standing (``gibbon``) [m]
G = 9.81
GREY_STEEL = (0.5, 0.52, 0.56, 1.0)


def ladder_prop(p):
    """The overhead ladder: rungs across the gibbon's way between two side rails, on four posts."""
    x0, H, s, n, rr, ry = p["x"], p["height"], p["spacing"], int(p["rungs"]), p["rung_radius"], p["rail_y"]
    x1 = x0 + (n - 1) * s
    geoms = [gs._cap(f"rung_{k}", rr, (x0 + k * s, -ry, H), (x0 + k * s, ry, H), rgba=GREY_STEEL, mass=2.0) for k in range(n)]
    for sy in (-1, 1):
        geoms.append(gs._cap(f"rail_{'l' if sy > 0 else 'r'}", 0.025, (x0 - 0.3, sy * ry, H), (x1 + 0.3, sy * ry, H), rgba=GREY_STEEL, mass=10.0))
        for xp in (x0 - 0.3, x1 + 0.3):
            geoms.append(gs._cap(f"post_{xp:.1f}_{sy}", 0.03, (xp, sy * ry, 0.0), (xp, sy * ry, H + 0.03), rgba=GREY_STEEL, mass=10.0))
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


def _rung(p, k):
    return np.array([p["x"] + k * p["spacing"], 0.0, p["height"]])


def _hold(p, k, side, toward):
    """The overhead rung ``k`` as a bar hold for ``side``'s hand, the wrist towards ``toward`` (the shoulder)."""
    c = _rung(p, k) + np.array([0.0, GRIP_Y if side == "L" else -GRIP_Y, 0.0])
    wd = np.asarray(toward, dtype=float) - c
    wd[1] = 0.0
    return gs.bar(c, axis=(0.0, 1.0, 0.0), n=(0.0, 0.0, -1.0), r=p["rung_radius"], wrist_dir=gs._unit(wd))


def _swing_base(p, k, u):
    """(pelvis xyz, pitch, shoulders' midpoint) at ``u`` of the swing about rung ``k``: the shoulders on the reach's
    circle about the rung from midway behind it to midway ahead, at rest at both ends (a pendulum's turn)."""
    s = p["spacing"]
    xr = -0.5 * s * math.cos(math.pi * u)
    zr = -math.sqrt(max(REACH ** 2 - xr ** 2, 1e-6))
    theta = math.asin(xr / REACH)                                          # the arm's angle from straight down
    pitch = -TRUNK_SWING * theta
    sh = _rung(p, k) + np.array([xr, 0.0, zr])
    pelvis = sh - gm.base_rot(pitch) @ np.array([0.0, 0.0, SHOULDER_Z])
    return pelvis, pitch, sh


def _sides(i):
    """(holding, free) hand for swing ``i``: they alternate."""
    return ("R", "L") if i % 2 == 0 else ("L", "R")


def brachiate_swing(kin, q_from, p, k, i, duration):
    """One swing about rung ``k`` (swing ``i``): the holding hand on ``k``, the free hand from rung ``k - 1`` under
    rung ``k`` to rung ``k + 1``."""
    hold_side, free_side = _sides(i)

    def targets(u):
        pelvis, pitch, sh = _swing_base(p, k, u)
        out = gs.hand_targets(kin, hold_side, _hold(p, k, hold_side, sh))
        f0 = _hold(p, k - 1, free_side, sh)
        f1 = _hold(p, k + 1, free_side, sh)
        if u <= 0.1:
            out += gs.hand_targets(kin, free_side, f0)
        elif u >= 0.9:
            out += gs.hand_targets(kin, free_side, f1)
        else:                                                              # under the holding hand's rung, well below it
            f = mk.smooth((u - 0.1) / 0.8)
            c = (1 - f) * f0["c"] + f * f1["c"] - np.array([0.0, 0.0, 0.22 * math.sin(math.pi * f)])
            out += gs._hand_at(kin, free_side, c, gs._slerp(gs.hand_frame_on(f0, free_side), gs.hand_frame_on(f1, free_side), f))
        return out

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
    segs.append(("motion", drop_off(kin, q, q_stand, p, n)))
    segs.append(("gait", 1.5, 0.0))
    return segs


gs.DEFAULTS["brachiate"] = LADDER
gs.BUILDERS["brachiate"] = ladder_prop
gs.SCENARIOS["brachiate"] = scenario_brachiate
