"""The ork's motions (``motion_kit``), exported as ``data:motions`` (tag ``motion``); the ghoul's are in
``monster_motions`` (shamble, lunge with the pincers closing, pinch).

* **stomp** — Talia's walk at the ork's weight: 0.45 m/s, 0.30 m stance, 50 mm crouch, ±50 mm sway, the broom arm
  held still and the lid arm swinging a little (±0.1 rad), both elbows at their 35° stance.
* **sweep** — the broom swept side to side: the right shoulder rolling from −0.45 (out) to +0.15 rad with the
  waist turning ±0.3 rad with it, 1.6 s a stroke pair; the broom head's lift at the stroke ends is reported (``params.head_lift_max_m``).
* **block** — the lid brought up before the body: the guard pose by IK (the lid's centre 0.35 m ahead of the
  pelvis, facing forward and 30° out), the joints eased in, held and eased out.
"""
from __future__ import annotations

import math

import numpy as np

import fantasy_humanoid_specs as fh
import humanoid_motions as hum
import motion_kit as mk

__all__ = ["motions_ork", "base_height", "parts_ork"]

base_height = hum.base_height


def stomp(kin):
    e = math.radians(fh.ORK["elbow_bend_deg"])
    m = hum.walk(kin, speed=0.45, stance=0.30, duty=0.62, step=0.05, crouch=0.05, sway=0.05, arm=0.1, elbow=-e,
                 name="stomp", source="fantasy_humanoid_motions: humanoid_motions.walk at the ork's weight")
    return m


def sweep(kin, period=1.6, half_width=0.20, frames=24):
    """The broom head kept on the ground, swept ±``half_width`` sideways on a shallow arc (IK of the right arm and
    the waist, every frame on its own goal)."""
    head = kin.geom_point("broom_bristles")
    rest = kin.world(*head, kin.q0)
    arm = [j for j in kin.names if j.startswith(("R_shoulder", "R_elbow", "waist"))]
    keys = []
    for i in range(frames + 1):
        s = math.sin(2 * math.pi * i / frames)
        keys.append((i / frames, [(*head, rest + np.array([-0.06 * s * s, half_width * s, 0.0]))]))
    m = mk.ik_transition(kin, "sweep", keys, duration=period, frames=frames, loop=True, joints=arm,
                         source="fantasy_humanoid_motions.sweep: the broom head's path on the ground, by IK",
                         params={"half_width_m": half_width})
    return m


def block(kin, duration=2.5, face_deg=30.0):
    """The guard pose solved once by IK — the lid's centre 0.35 m ahead of the pelvis, 0.10 m left of centre, at
    0.25 m, its face turned ``face_deg`` out from straight ahead (a strapped shield faces past the forearm's twist
    range straight ahead) — seeded with the arm forward, then the joints eased stance → guard → held → stance."""
    lid, c = kin.geom_point("bin_lid")
    normal = c + np.array([0.0, 0.2, 0.0])                                # 0.2 m out along the lid's face normal
    a = math.radians(face_deg)
    centre = np.array([0.35, 0.10, 0.25])
    goal = [centre, centre + 0.2 * np.array([math.cos(a), math.sin(a), 0.0])]
    arm = [j for j in kin.names if j.startswith(("L_shoulder", "L_elbow", "waist"))]
    seed = kin.q0.copy()
    seed[kin.index["L_shoulder_pitch"]] = -1.3
    q_guard, err = kin.ik([(lid, c, goal[0]), (lid, normal, goal[1])], q_init=seed, joints=arm, iters=400)

    def fn(t):
        u = t / duration
        w = mk.smooth(u / 0.35) if u < 0.7 else 1 - mk.smooth((u - 0.7) / 0.3)
        return {j: kin.q0[kin.index[j]] + w * (q_guard[kin.index[j]] - kin.q0[kin.index[j]]) for j in arm}
    m = mk.joint_cycle(kin, "block", fn, period=duration, frames=30, loop=False, kind="transition",
                       source="fantasy_humanoid_motions.block: the guard by IK, eased in joint space",
                       params={"lid_centre_goal_m": goal[0].tolist(), "lid_face_deg_out": face_deg})
    m.check["ik_error_max_m"] = round(err, 6)
    return m


def motions_ork(robot=None):
    import fantasy_humanoid_robots as fhr
    kin = mk.Kin(robot or fhr.ork())
    return kin, [stomp(kin), sweep(kin), block(kin)]


def parts_ork(machine) -> dict:
    import fantasy_humanoid_robots as fhr
    return fhr.parts_ork(machine)
