"""The Myriapods' motions (``motion_kit``): Cleopatra (notebook 18, ``myropod_robot.cleopatra``) and Apheloria
(notebook 19, ``apheloria_robot.apheloria``), exported as ``data:motions`` (tag ``motion``). Persephone (notebook 17)
is CAD-only — the Chiron Myropod model has three segments — so it has no robot to carry motions.

Both walk the controller's own gait (``myropod_controller``, protocol §3): on each segment a lateral-sequence wave
(rear-left, front-left, rear-right, front-right at 0, ¼, ½, ¾ of the cycle, duty 0.75), segment i shifted by
(3 − i)/12 of a stride (notebook 18's metachronal wave), the feet sliding straight back in stance and returning on
a cosine with a 50 mm lift (Apheloria: 40 mm). The body joints are left alone: Cleopatra's are passive springs,
Apheloria's hold straight while it walks.

* **Cleopatra** — walk (0.2 m/s, 0.10 m stride: protocol §3) and turn (the left stride 30 % longer, the right
  30 % shorter: the heading controller's clip, §3) — a right turn at 0.2 m/s.
* **Apheloria** — walk (0.15 m/s, 0.10 m stride), curl (stand → the legs tucked, ``apheloria_robot.TUCK``, then
  every body joint to −40°, ``CURL_DEG``: the ball) and uncurl (back).
"""
from __future__ import annotations

import numpy as np

import motion_kit as mk

__all__ = ["motions_cleopatra", "motions_apheloria", "parts_cleopatra", "parts_apheloria", "metachronal_feet",
           "base_height"]

DUTY = 0.75
WAVE_SHIFT = 1.0 / 12.0


def metachronal_feet(robot):
    """{foot geom: phase} for the controller's gait (``myropod_controller.BASE_PHASES`` and the (3 − i)/12 shift)."""
    import myropod_controller as mc

    feet = {}
    for f in robot.feet:
        seg, key = f.name.split(" ", 1)
        i = int(seg[1:])
        touchdown = (mc.BASE_PHASES[key] + (3 - i) * WAVE_SHIFT) % 1.0
        feet[f.geom] = (1.0 - touchdown) % 1.0
    return feet


def _walk(kin, robot, name, *, speed, stride, lift, stride_scale=None, source):
    feet = metachronal_feet(robot)
    period = stride / speed                                                # the controller: frequency = v / stride
    sweep = DUTY * stride                                                  # its stance sweep L = duty × stride
    st = sweep
    if stride_scale:                                                       # {+1 (left) / -1 (right): factor}
        ys = {f.geom: kin.world(*kin.geom_point(f.geom), kin.q0)[1] for f in robot.feet}
        st = {g: sweep * stride_scale[1 if ys[g] > 0 else -1] for g in feet}
    legs = [j for j in kin.actuated if " hip_" in j or j.endswith(" knee")]
    return mk.legged_gait(kin, name, feet, period=period, duty=DUTY, stride=st, step_height=lift, frames=36,
                          joints=legs, source=source,
                          params={"wave_shift": "(3 − i)/12 of a stride per segment", "speed_target_m_s": speed})


def motions_cleopatra(robot=None):
    import myropod_robot as mr

    robot = robot or mr.cleopatra()
    kin = mk.Kin(robot)
    src = "myriapod_motions: myropod_controller's fixed gait (protocol §3)"
    return kin, [_walk(kin, robot, "walk", speed=0.2, stride=0.10, lift=0.05, source=src),
                 _walk(kin, robot, "turn", speed=0.2, stride=0.10, lift=0.05, stride_scale={1: 1.3, -1: 0.7},
                       source=src + " with the heading clip (±30 % stride)")]


def _curl(kin, ar, *, uncurl=False, duration=4.0, frames=40):
    """Stand → legs tucked (first 40 %) → body curled (the rest); ``uncurl`` plays it backwards."""
    stand, tuck = ar.pose(legs=ar.STAND, curl=0.0), ar.pose(legs=ar.TUCK, curl=0.0)
    ball = ar.pose(legs=ar.TUCK, curl=np.radians(ar.CURL_DEG))

    def at(u):
        if u < 0.4:
            a, b, w = stand, tuck, mk.smooth(u / 0.4)
        else:
            a, b, w = tuck, ball, mk.smooth((u - 0.4) / 0.6)
        return {j: (1 - w) * a[j] + w * b[j] for j in a}

    def fn(t):
        u = t / duration
        return at(1.0 - u if uncurl else u)
    m = mk.joint_cycle(kin, "uncurl" if uncurl else "curl", fn, period=duration, frames=frames, loop=False,
                       kind="transition", source="myriapod_motions: apheloria_robot.pose (STAND → TUCK → CURL_DEG)",
                       params={"curl_deg_per_joint": ar.CURL_DEG, "legs_tucked_by": 0.4})
    return m


def motions_apheloria(robot=None):
    import apheloria_robot as ar

    robot = robot or ar.apheloria()
    kin = mk.Kin(robot)
    src = "myriapod_motions: myropod_controller's gait on Apheloria's eight segments"
    return kin, [_walk(kin, robot, "walk", speed=0.15, stride=0.10, lift=0.04, source=src),
                 _curl(kin, ar), _curl(kin, ar, uncurl=True)]


def base_height(kin):
    """Height of the root above the ground with the feet down in the standing pose [m]."""
    kin.set(kin.q0)
    lows = [kin.data.geom_xpos[kin.model.geom(f.geom).id][2] - kin.model.geom_size[kin.model.geom(f.geom).id][0]
            for f in kin.robot.feet]
    return float(kin.data.xpos[1][2] - min(lows))


def _parts(fn):
    def parts(machine) -> dict:
        kin, ms = fn()
        return mk.block(ms, base_height(kin))
    return parts


parts_cleopatra = _parts(motions_cleopatra)
parts_apheloria = _parts(motions_apheloria)
