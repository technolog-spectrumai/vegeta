"""Belisario on all fours: the knuckle walk and the rise to standing, for the bare frame (``space_unit_robots.bare``).

* **quad_pose** — the gorilla's stance on four: the pelvis ``PELVIS_H`` above the ground and pitched nose-down
  ``PELVIS_PITCH`` (the back then slopes ~35°, the spine's modules only a little forward), the legs crouched under
  the hips, the long arms down to the ground ahead of the shoulders with the elbows nearly straight and the forearms
  standing on the knuckles (the curled fingers' middle joints, the wrist held above them: knuckle-walking) - an IK
  solve on the soles' three points, the knuckles and the wrists from a seed in that shape.
* **knuckle_walk** — the lateral-sequence walk every quadruped uses at a walk (a gorilla's: left hind, left fore,
  right hind, right fore, a quarter cycle apart), duty 0.72 (three contacts down most of the time), a 0.3 m stride at
  0.35 m/s, the swing 60 mm off the ground: ``motion_kit.legged_gait`` on the four contacts, from the quad pose.
* **rise** / **drop** — the transition to standing on two (the frame's design pose, as ``humanoid_motions``) and back,
  in joint space over 3 s: the gorilla rears up to look, beat its chest or carry something, then goes back down.
* **stand** — the standing hold (2 s, a slight sway).

    import gorilla_motions as gm, space_unit_robots as sr, motion_kit as mk
    kin = mk.Kin(sr.bare()); qq = gm.quad_pose(kin); walk = gm.knuckle_walk(kin, qq)
"""
from __future__ import annotations

import math

import numpy as np

import humanoid_motions as hum
import motion_kit as mk

__all__ = ["PELVIS_H", "PELVIS_PITCH", "CONTACTS", "base_rot", "base_quat", "quad_pose", "knuckle_walk", "rise", "drop", "stand", "motions", "parts"]

PELVIS_H = 0.62                                                           #: the pelvis above the ground on all fours [m]
PELVIS_PITCH = 0.45                                                       #: the pelvis pitched nose-down on all fours [rad] (~26°)
#: the four contacts and their touchdown phases: the lateral sequence LH, LF, RH, RF
CONTACTS = {"L": 0.0, "L_knuckle": 0.25, "R": 0.5, "R_knuckle": 0.75}
#: the seed for the quad pose's IK: the spine's modules a little forward (the back's slope comes mostly from the pelvis),
#: hips flexed, knees bent, the arms down and ahead, the elbows nearly straight
SEED = {"spine_lumbar_pitch": 0.25, "spine_thoracic_pitch": 0.25, "waist_yaw": 0.0,
        "L_hip_pitch": -1.2, "R_hip_pitch": -1.2, "L_knee": 1.7, "R_knee": 1.7, "L_ankle_pitch": -0.5, "R_ankle_pitch": -0.5,
        "L_shoulder_pitch": -0.9, "R_shoulder_pitch": -0.9, "L_elbow": -0.15, "R_elbow": -0.15}


def base_rot(pitch=PELVIS_PITCH):
    """The pelvis's orientation on all fours: pitched nose-down by ``pitch`` about y."""
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def base_quat(pitch=PELVIS_PITCH):
    """The same as a MuJoCo quaternion (w, x, y, z)."""
    return np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])


def _contact_points(kin):
    """{contact: [(body, local)]}: the soles' three points, the knuckles' one."""
    pts = {}
    for g in CONTACTS:
        body, c = kin.geom_point(g)
        offs = hum.SOLE_POINTS if g in ("L", "R") else [np.zeros(3)]
        pts[g] = [(body, c + np.asarray(o)) for o in offs]
    return pts


def quad_pose(kin, pelvis_h=PELVIS_H, pitch=PELVIS_PITCH, hands_ahead=0.3, feet_ahead=0.05, wrist=(-0.42, 0.52)):
    """The stance on four (joint angles) with the pelvis ``pelvis_h`` up and pitched nose-down by ``pitch``: the
    soles ``feet_ahead`` ahead of the hips (the knees far forward in the crouch, as a gorilla squats) and the
    knuckles ``hands_ahead`` ahead of the shoulders, all on the ground; the forearms leaning on the knuckles (a second
    target holds each wrist ``wrist`` = (back, up) of its knuckle: knuckle-walking's extended wrist, the forearm
    sloping down to the hand). World targets go into the pelvis frame through ``base_rot``."""
    R = base_rot(pitch)
    q = kin.q0.copy()
    for j, v in SEED.items():
        q[kin.index[j]] = v
    kin.set(q)

    def to_base(w):                                                       # a world point (the pelvis at the origin) in the base frame
        return R.T @ np.asarray(w, dtype=float)

    targets = []
    for g, plist in _contact_points(kin).items():
        if g in ("L", "R"):
            for k, (body, local) in enumerate(plist):
                o = hum.SOLE_POINTS[k]
                y = kin.world(body, local, q)[1]                            # the foot stays in its file
                targets.append((body, local, to_base([feet_ahead + o[0], y, -pelvis_h])))
        else:
            body, local = plist[0]
            side = "L" if g.startswith("L") else "R"
            sh = R @ kin.world(kin.body(f"{side}_upper_arm"), np.zeros(3), q)   # the shoulder, in the world
            w = R @ kin.world(body, local, q)
            targets.append((body, local, to_base([sh[0] + hands_ahead, w[1], -pelvis_h])))
            if wrist is not None:                                         # the wrist behind and above the knuckle: the forearm leaning on it
                fore = kin.body(f"{side}_forearm")
                targets.append((fore, np.zeros(3), to_base([sh[0] + hands_ahead + wrist[0], w[1], -pelvis_h + wrist[1]])))
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw"]
    q, err = kin.ik(targets, q_init=q, q_ref=q, joints=limbs, iters=400)
    assert err < 0.03, f"the quad pose's IK: {err:.3f} m"
    return q


def knuckle_walk(kin, q_quad, *, speed=0.35, stride=0.26, duty=0.72, step=0.06, reg=1e-3, iters=800, name="knuckle_walk"):
    """The lateral-sequence walk on four from the quad pose (``legged_gait`` sees the quad pose as the design pose)."""
    period = stride / (duty * speed)
    pts = _contact_points(kin)
    foot_points = {g: ([np.zeros(3)] if "knuckle" in g else hum.SOLE_POINTS) for g in CONTACTS}
    q0 = kin.q0
    kin.q0 = np.asarray(q_quad, dtype=float)                               # the gait's reference pose
    try:
        limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw"]
        gait = mk.legged_gait(kin, name, CONTACTS, period=period, duty=duty, stride=stride, step_height=step,
                              foot_points=foot_points, joints=limbs, base_rot=base_rot(), reg=reg, iters=iters,
                              source="gorilla_motions.knuckle_walk: the lateral-sequence walk on the knuckles",
                              params={"froude": round(speed ** 2 / (mk.G * PELVIS_H), 3), "sequence": "LH LF RH RF"})
    finally:
        kin.q0 = q0
    return gait


def _blend(kin, name, qa, qb, duration, kind, source):
    def fn(t):
        w = mk.smooth(min(1.0, t / duration))
        return {j: (1 - w) * qa[k] + w * qb[k] for k, j in enumerate(kin.names)}
    return mk.joint_cycle(kin, name, fn, period=duration, frames=36, loop=False, kind=kind, source=source)


def rise(kin, q_quad, duration=3.0):
    """From the quad pose up to the design (standing) pose, joint space, smooth."""
    return _blend(kin, "rise", q_quad, kin.q0, duration, "transition", "gorilla_motions.rise: up on two")


def drop(kin, q_quad, duration=3.0):
    """From standing back down on four."""
    return _blend(kin, "drop", kin.q0, q_quad, duration, "transition", "gorilla_motions.drop: back on four")


def stand(kin, period=2.0):
    """Standing on two: a slight sway of the trunk."""
    def fn(t):
        w = 0.05 * math.sin(2 * math.pi * t / period)
        return {"spine_thoracic_roll": w, "spine_lumbar_roll": -w * 0.5}
    return mk.joint_cycle(kin, "stand", fn, period=period, frames=24, loop=True, source="gorilla_motions.stand: on two, a sway")


def motions(robot=None):
    """(kin, [knuckle_walk, rise, stand, drop]) for the bare Belisario."""
    import space_unit_robots as sr
    kin = mk.Kin(robot or sr.bare())
    qq = quad_pose(kin)
    return kin, [knuckle_walk(kin, qq), rise(kin, qq), stand(kin), drop(kin, qq)]


def parts(machine) -> dict:
    kin, ms = motions()
    return mk.block(ms, base_height=hum.base_height(kin))
