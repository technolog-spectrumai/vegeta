"""Belisario's tasks on the way: the door. The scene (``door_prop``: a doorway in a wall at ``DOOR["x"]``, the door
hinged on the left and swinging away, a lever handle on the robot's side) and the sequence of motions from the
approach to the entry, for the sim (``sims/space_gorilla_door.py``):

1. **approach** — the knuckle walk up to the door (``gorilla_motions.knuckle_walk``), stopping with the handle in reach;
2. **reach** — up onto the feet: they stay planted while the trunk lifts (the pelvis to 0.76 m, nearly level) and
   the right hand goes to the handle, the left hand coming off the ground (IK on the planted feet and the knuckle);
3. **grasp** — the right fist closes on the lever;
4. **turn** — the lever turned down, the hand following its arc about the spindle;
5. **push** — the hand follows the handle's arc as the door swings open, the pelvis advancing 0.3 m over the planted
   feet (the legs extending, the arm reaching);
6. **release** — the fingers open, the wrist levels, the pelvis settles back over the feet and both hands come back
   down to their places on four;
7. **enter** — the knuckle walk on, through the doorway.

Every motion but the walk is solved frame by frame by IK in the world with the pelvis's height and pitch prescribed
(``contact_motion``), and carries ``base["height_m"]`` and ``base["pitch_rad"]`` per frame; the joints a motion does
not solve (the fingers, the wrist's roll) are set by ``extra``.
"""
from __future__ import annotations

import math

import numpy as np
from vegeta.chiron import Geom, Joint, Link, Prop

import gorilla_motions as gm
import humanoid_motions as hum
import motion_kit as mk

__all__ = ["DOOR", "door_prop", "handle_grip", "handle_world", "lever_hold", "contact_motion", "reach", "grasp", "turn", "push", "release",
           "door_sequence", "STOP_X"]

#: the doorway: at x [m], its clear width and height, the door's thickness, the hinge on the left (+y), the lever
#: handle's place on the door (y from the hinge's side, z) and length, how far the door swings open
DOOR = {"x": 3.0, "width": 1.3, "height": 2.1, "thickness": 0.05, "handle_dy": -1.15, "handle_z": 1.0, "handle_len": 0.14,
        "open_rad": 0.6, "turn_rad": -0.5, "mass_kg": 18.0, "damping": 6.0, "swung_rad": 1.6}   # swung_rad: where the shoved door coasts to
REACH_BASE = (0.76, 0.1)                                                  #: the pelvis (height, pitch) up on the feet at the door
PUSH_ADVANCE = 0.35                                                       #: how far the pelvis moves ahead over the feet in the push
STOP_X = 1.95                                                             #: where the approach stops (the pelvis), the handle in reach


def door_prop():
    """The doorway as a Chiron prop: the frame fixed to the world, the door on a damped hinge, the lever on a sprung
    hinge through the door, on the robot's side."""
    D = DOOR
    w, h, t = D["width"], D["height"], D["thickness"]
    grey, wood, brass = (0.45, 0.45, 0.47, 1.0), (0.55, 0.42, 0.28, 1.0), (0.8, 0.65, 0.3, 1.0)
    handle = Link("door_handle", pos=(-t / 2 - 0.045, D["handle_dy"], D["handle_z"]),
                  joints=[Joint("handle_hinge", axis=(1, 0, 0), range=(-1.2, 0.0), stiffness=6.0, damping=0.4)],
                  geoms=[Geom("handle_shaft", "cylinder", (0.012, 0.05), pos=(0.02, 0.0, 0.0), quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
                              mass=0.3, role="link", rgba=brass),
                         Geom("handle_lever", "capsule", (0.014,), fromto=(0.0, 0.0, 0.0, 0.0, D["handle_len"], 0.0), mass=0.4, role="link", rgba=brass)])
    door = Link("door", pos=(0.0, w / 2, 0.0),
                joints=[Joint("door_hinge", axis=(0, 0, 1), range=(0.0, 1.7), damping=D["damping"])],
                geoms=[Geom("door_panel", "box", (t / 2, w / 2 - 0.01, h / 2 - 0.01), pos=(0.0, -w / 2, h / 2), mass=D["mass_kg"], role="link", rgba=wood)],
                children=[handle])
    frame = Link("door_frame", pos=(D["x"], 0.0, 0.0),
                 geoms=[Geom("post_left", "box", (0.08, 0.08, h / 2 + 0.1), pos=(0.0, w / 2 + 0.08, h / 2 + 0.1), mass=50.0, role="link", rgba=grey),
                        Geom("post_right", "box", (0.08, 0.08, h / 2 + 0.1), pos=(0.0, -w / 2 - 0.08, h / 2 + 0.1), mass=50.0, role="link", rgba=grey),
                        Geom("lintel", "box", (0.08, w / 2 + 0.16, 0.1), pos=(0.0, 0.0, h + 0.1), mass=40.0, role="link", rgba=grey),
                        Geom("wall_left", "box", (0.08, 0.9, h / 2 + 0.1), pos=(0.0, w / 2 + 0.16 + 0.9, h / 2 + 0.1), mass=100.0, role="link", rgba=grey),
                        Geom("wall_right", "box", (0.08, 0.9, h / 2 + 0.1), pos=(0.0, -w / 2 - 0.16 - 0.9, h / 2 + 0.1), mass=100.0, role="link", rgba=grey)],
                 children=[door])
    return Prop(frame, free=False, priority=2)


def handle_world(phi=0.0, turn=0.0):
    """The grip point on the lever (its middle) in the world with the door swung ``phi`` and the lever turned
    ``turn`` (down, about the door's x)."""
    D = DOOR
    w, t = D["width"], D["thickness"]
    hinge = np.array([D["x"], w / 2, 0.0])
    lever_mid = np.array([0.0, D["handle_len"] / 2 * math.cos(turn), -D["handle_len"] / 2 * math.sin(-turn)])
    p = np.array([-t / 2 - 0.045, D["handle_dy"], D["handle_z"]]) + lever_mid     # in the door's frame
    c, s = math.cos(phi), math.sin(phi)
    return hinge + np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def handle_grip(phi=0.0, turn=0.0):
    """Where the knuckle sits when the fingers hold the lever: just under and behind the grip point."""
    return handle_world(phi, turn) + np.array([-0.05, 0.0, -0.035])


def _contacts_world(kin, q, base_x, h, pitch):
    """The four contacts and the wrists in the world for a pose at a base place."""
    R = gm.base_rot(pitch)
    base = np.array([base_x, 0.0, h])
    kin.set(q)
    out = {}
    for g, plist in gm._contact_points(kin).items():
        out[g] = [R @ kin.world(body, local) + base for body, local in plist]
    return out


def contact_motion(kin, name, duration, targets_fn, base_fn, *, q_init, frames=30, extra=None, reg=2e-4, iters=600, source=""):
    """A motion solved frame by frame: ``targets_fn(u)`` → [(body, local, world xyz)] (the planted contacts and the
    hand), ``base_fn(u)`` → (base x, height, pitch) in the world; ``extra(u, q)`` sets joints the IK does not (the
    fingers, the wrist's roll). The motion's ``base`` carries the base track per frame."""
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
    ts = [duration * i / frames for i in range(frames + 1)]
    Q, xs, ys, hs, ps, q, worst = [], [], [], [], [], np.asarray(q_init, dtype=float).copy(), 0.0
    for t in ts:
        u = t / duration
        b = base_fn(u)
        x, y, h, p = (b[0], 0.0, b[1], b[2]) if len(b) == 3 else b          # (x, h, pitch) or (x, y, h, pitch)
        R, base = gm.base_rot(p), np.array([x, y, h])
        targets = [(body, local, R.T @ (np.asarray(w) - base)) for body, local, w in targets_fn(u)]
        q, err = kin.ik(targets, q_init=q, q_ref=q_init, joints=limbs, reg=reg, iters=iters)
        if extra:
            q = np.clip(extra(u, q.copy()), kin.lo, kin.hi)
        worst = max(worst, err)
        Q.append(q.copy())
        xs.append(round(float(x), 4)); ys.append(round(float(y), 4)); hs.append(round(float(h), 4)); ps.append(round(float(p), 4))
    moving = mk._moving(kin, Q)
    cols = [kin.index[j] for j in moving]
    return mk.Motion(name, "transition", moving, ts, np.asarray(Q)[:, cols], loop=False, period=duration,
                     base={"x_m": xs, "y_m": ys, "height_m": hs, "pitch_rad": ps, "note": "the pelvis's x, y, height and pitch per frame"},
                     source=source or f"gorilla_tasks.{name}",
                     check={"ik_error_max_m": round(worst, 6), "limit_margin_min_rad": round(kin.margin(Q, moving), 4)})


def _planted(kin, q_quad, base_x, which=("L", "R", "L_knuckle")):
    """The planted contacts' targets, from the quad pose at ``base_x``."""
    w = _contacts_world(kin, q_quad, base_x, gm.PELVIS_H, gm.PELVIS_PITCH)
    pts = gm._contact_points(kin)
    return [(body, local, w[g][k]) for g in which for k, (body, local) in enumerate(pts[g])]


LEVER_WRIST = (-0.94, 0.0, 0.342)                                        #: the wrist from the lever's grip: behind, 20° above (the forearm along the spindle: the turn is a roll)
LEVER_CURL = 1.9                                                          #: the fist on the 28 mm lever


def lever_hold(phi=0.0, turn=0.0):
    """The lever as a bar hold (``gorilla_scenarios.bar``) with the door swung ``phi`` and the lever turned ``turn``:
    its middle, its axis and the wrist's direction turned with it."""
    import gorilla_scenarios as gs
    c, sn = math.cos(phi), math.sin(phi)
    Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    ct, st = math.cos(turn), math.sin(turn)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    R = Rz @ Rx
    return gs.bar(handle_world(phi, turn), axis=R @ np.array([0.0, 1.0, 0.0]), n=Rz @ np.array([-1.0, 0.0, 0.0]), r=0.014,
                  wrist_dir=R @ np.asarray(LEVER_WRIST))


def reach(kin, q_quad, base_x=STOP_X, duration=2.5):
    """Up onto the feet: they stay planted while the trunk lifts (the pelvis to ``REACH_BASE``), the right hand goes
    to the lever on an arc (its grip point first, then the whole hand settling onto the lever from above, the
    fingers open) and the left hand comes off the ground to hang."""
    import gorilla_scenarios as gs
    import space_unit_robots as sr
    planted = _planted(kin, q_quad, base_x, which=("L", "R"))
    pts = gm._contact_points(kin)
    start = _contacts_world(kin, q_quad, base_x, gm.PELVIS_H, gm.PELVIS_PITCH)
    hold = lever_hold()
    end = gs.hand_targets(kin, "R", hold)
    kin.set(q_quad)
    Rb, b0 = gm.base_rot(gm.PELVIS_PITCH), np.array([base_x, 0.0, gm.PELVIS_H])
    grip0 = Rb @ kin.world(end[0][0], end[0][1]) + b0
    kin.set(kin.q0)
    hang = kin.world(*pts["L_knuckle"][0])                                 # the left hand hanging (the design pose, base-relative)

    def targets(u):
        f = mk.smooth(u)
        x, h, p = base(u)
        R, b = gm.base_rot(p), np.array([x, 0.0, h])
        arc = np.array([0.0, 0.0, 0.25 * math.sin(math.pi * f)])
        left = (1 - f) * start["L_knuckle"][0] + f * (R @ hang + b) + arc * 0.5
        out = planted + [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], left)]
        if u < 0.7:
            fr = mk.smooth(u / 0.7)
            out.append((end[0][0], end[0][1], (1 - fr) * grip0 + fr * hold["c"] + arc))
        else:
            out += end
        return out

    def base(u):
        f = mk.smooth(u)
        return base_x, gm.PELVIS_H + f * (REACH_BASE[0] - gm.PELVIS_H), gm.PELVIS_PITCH + f * (REACH_BASE[1] - gm.PELVIS_PITCH)

    def extra(u, q):
        f = mk.smooth(min(1.0, u / 0.4))                                   # the right fingers open on the way
        return sr.set_curl(q, kin, "R", (1 - f) * q_quad[kin.index["R_finger_1_mcp"]] / 0.8 + f * gs.CURL_OPEN)

    return contact_motion(kin, "reach", duration, targets, base, q_init=q_quad, extra=extra, reg=gs.IK_REG, iters=gs.IK_ITERS,
                          source="gorilla_tasks.reach: up on the feet, the right hand onto the lever")


def _joint_motion(kin, name, duration, q_from, changes, frames=20, base=None, source=""):
    """Joints eased from ``q_from`` to new values (``changes`` {joint: value}); the base held (``base`` = (x, h, p))."""
    ts = [duration * i / frames for i in range(frames + 1)]
    Q = []
    for t in ts:
        w = mk.smooth(t / duration)
        q = np.asarray(q_from, dtype=float).copy()
        for j, v in changes.items():
            q[kin.index[j]] = (1 - w) * q_from[kin.index[j]] + w * v
        Q.append(np.clip(q, kin.lo, kin.hi))
    moving = mk._moving(kin, Q) or list(changes)
    cols = [kin.index[j] for j in moving]
    x, h, p = base
    n = len(ts)
    return mk.Motion(name, "transition", moving, ts, np.asarray(Q)[:, cols], loop=False, period=duration,
                     base={"x_m": [x] * n, "y_m": [0.0] * n, "height_m": [h] * n, "pitch_rad": [p] * n}, source=source,
                     check={"limit_margin_min_rad": round(kin.margin(Q, moving), 4)})


def grasp(kin, q_from, base, duration=0.8):
    """The right fist closes on the lever."""
    import space_unit_robots as sr
    return _joint_motion(kin, "grasp", duration, q_from, sr.curl_values("R", LEVER_CURL), base=base, source="gorilla_tasks.grasp: the fist closes on the lever")


def _lever_motion(kin, name, duration, q_quad, q_from, base_x, phi_fn, turn_fn, base_fn, source):
    """The hand holding the lever as it moves: the hold turned by ``turn_fn(u)`` about the lever's spindle and swung
    by ``phi_fn(u)`` with the door, the feet planted, the left hand hanging where it is, the fist held closed."""
    import gorilla_scenarios as gs
    import space_unit_robots as sr
    planted = _planted(kin, q_quad, base_x, which=("L", "R"))
    pts = gm._contact_points(kin)
    kin.set(q_from)
    x0, h0, p0 = base_fn(0.0)
    hang = gm.base_rot(p0) @ kin.world(*pts["L_knuckle"][0]) + np.array([x0, 0.0, h0])

    def targets(u):
        return planted + [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], hang)] + gs.hand_targets(kin, "R", lever_hold(phi_fn(u), turn_fn(u)))

    def extra(u, q):
        return sr.set_curl(q, kin, "R", LEVER_CURL)

    return contact_motion(kin, name, duration, targets, base_fn, q_init=q_from, extra=extra, reg=gs.IK_REG, iters=gs.IK_ITERS, source=source)


def turn(kin, q_quad, q_from, base_x=STOP_X, duration=1.0):
    """The lever turned down: the hand follows its arc about the spindle."""
    base = lambda u: (base_x, REACH_BASE[0], REACH_BASE[1])
    return _lever_motion(kin, "turn", duration, q_quad, q_from, base_x, lambda u: 0.0, lambda u: DOOR["turn_rad"] * mk.smooth(u), base,
                         "gorilla_tasks.turn: the lever turned down")


def push(kin, q_quad, q_from, base_x=STOP_X, duration=3.0):
    """The door swings open under the hand: the hold follows the lever's arc (the door to ``open_rad``), the feet
    planted, the pelvis advancing ``PUSH_ADVANCE`` over them."""
    base = lambda u: (base_x + PUSH_ADVANCE * mk.smooth(u), REACH_BASE[0], REACH_BASE[1])
    return _lever_motion(kin, "push", duration, q_quad, q_from, base_x, lambda u: DOOR["open_rad"] * mk.smooth(u), lambda u: DOOR["turn_rad"], base,
                         "gorilla_tasks.push: the door swung open, the hand on the lever")


def release(kin, q_quad, q_from, base_x=STOP_X, duration=2.0):
    """The fist opens and lets the lever go, the pelvis settles back over the feet and down, both hands come back
    down to their places on four (the right from the lever, the left from hanging)."""
    import gorilla_scenarios as gs
    import space_unit_robots as sr
    planted = _planted(kin, q_quad, base_x, which=("L", "R"))
    pts = gm._contact_points(kin)
    goal = _contacts_world(kin, q_quad, base_x, gm.PELVIS_H, gm.PELVIS_PITCH)
    kin.set(q_from)
    R0, b0 = gm.base_rot(REACH_BASE[1]), np.array([base_x + PUSH_ADVANCE, 0.0, REACH_BASE[0]])
    start_l = R0 @ kin.world(*pts["L_knuckle"][0]) + b0
    start_r = R0 @ kin.world(*pts["R_knuckle"][0]) + b0
    hold = lever_hold(DOOR["open_rad"], DOOR["turn_rad"])
    end_r = gs.hand_targets(kin, "R", hold)

    def targets(u):
        f = mk.smooth(max(0.0, (u - 0.25) / 0.75))
        arc = np.array([0.0, 0.0, 0.2 * math.sin(math.pi * f)])
        out = planted + [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], (1 - f) * start_l + f * goal["L_knuckle"][0] + arc)]
        if u < 0.25:
            out += end_r                                                   # the hand stays on the lever while the fist opens
        else:
            out.append((pts["R_knuckle"][0][0], pts["R_knuckle"][0][1], (1 - f) * (start_r + np.array([-0.1, 0.0, 0.05])) + f * goal["R_knuckle"][0] + arc))
        return out

    def base(u):
        f = mk.smooth(max(0.0, (u - 0.25) / 0.75))
        return base_x + PUSH_ADVANCE * (1 - f), REACH_BASE[0] + f * (gm.PELVIS_H - REACH_BASE[0]), REACH_BASE[1] + f * (gm.PELVIS_PITCH - REACH_BASE[1])

    def extra(u, q):
        f = mk.smooth(min(1.0, u / 0.25))                                  # the fist opens first, then the hand comes down to the walking curl
        g = mk.smooth(max(0.0, (u - 0.25) / 0.75))
        c = (1 - f) * LEVER_CURL + f * gs.CURL_OPEN if u < 0.25 else (1 - g) * gs.CURL_OPEN + g * q_quad[kin.index["R_finger_1_mcp"]] / 0.8
        return sr.set_curl(q, kin, "R", c)

    return contact_motion(kin, "release", duration, targets, base, q_init=q_from, extra=extra, reg=gs.IK_REG, iters=gs.IK_ITERS,
                          source="gorilla_tasks.release: the hand back down on four")


def _last(kin, m, q_prev):
    q = np.asarray(q_prev, dtype=float).copy()
    q[[kin.index[j] for j in m.joints]] = np.asarray(m.Q)[-1]
    return q


def door_sequence(kin, q_quad):
    """[reach, grasp, turn, push, release] in order, each starting where the last ended."""
    r = reach(kin, q_quad)
    q1 = _last(kin, r, q_quad)
    base = (STOP_X, REACH_BASE[0], REACH_BASE[1])
    g = grasp(kin, q1, base); q2 = _last(kin, g, q1)
    t = turn(kin, q_quad, q2); q3 = _last(kin, t, q2)
    p = push(kin, q_quad, q3); q4 = _last(kin, p, q3)
    rl = release(kin, q_quad, q4)
    return [r, g, t, p, rl]
