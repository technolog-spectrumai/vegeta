"""Talia's motions (``motion_kit``): a walk and a wave, exported as ``data:motions``.

* **walk** — 0.5 m/s, Froude number 0.04 on the 0.76 m leg (a slow robot walk; humans change to a run near 0.5):
  0.30 m stance, duty 0.6 (double support 20 %), 50 mm step height, the soles kept flat (three points tracked),
  the base 40 mm lower than the straight-legged design pose so the knees bend, ±35 mm sway over the stance foot,
  arms swinging against the legs (±0.35 rad) with the elbows at 0.3 rad;
* **wave** — the right arm raised to the side (shoulder roll 2.2 rad, elbow 0.8 rad), the forearm waving ±0.45 rad
  at 1 Hz.
"""
from __future__ import annotations

import math

import motion_kit as mk

__all__ = ["motions", "parts", "base_height"]

SOLE_POINTS = [(0.10, 0.0, 0.0), (-0.06, 0.04, 0.0), (-0.06, -0.04, 0.0)]


def base_height(kin):
    """Pelvis height with the soles on the ground at q = 0 [m]."""
    kin.set(kin.q0)
    g = kin.model.geom("L")
    return float(-(kin.data.geom_xpos[g.id][2] - kin.model.geom_size[g.id][2]))


def walk(kin, *, speed=0.5, stance=0.30, duty=0.6, step=0.05, crouch=0.04, sway=0.035, arm=0.35, arm_offset=0.0,
         elbow=-0.3, name="walk", source="humanoid_motions.walk"):
    """The walk above; ``arm_offset`` [rad] holds the arms forward (negative) around their swing, ``elbow`` [rad]."""
    period = stance / (duty * speed)
    feet = {"L": 0.0, "R": 0.5}

    def foot_dx(t, ph):
        s = (t / period + ph) % 1.0
        return 0.5 - s / duty if s < duty else -0.5 + mk.smooth((s - duty) / (1 - duty))

    def extra(t, q):
        for side, ph in feet.items():
            q[kin.index[f"{side}_shoulder_pitch"]] = arm_offset + 2 * arm * foot_dx(t, ph)   # back while its foot is ahead
            q[kin.index[f"{side}_elbow"]] = elbow
        return q

    return mk.legged_gait(kin, name, feet, period=period, duty=duty, stride=stance, step_height=step,
                          foot_points={g: SOLE_POINTS for g in feet}, crouch=crouch, extra=extra,
                          q_start={f"{s}_{j}": v for s in feet for j, v in (("hip_pitch", -0.3), ("knee", 0.6),
                                                                              ("ankle_pitch", -0.3))},
                          lateral=lambda t: -sway * math.sin(2 * math.pi * (t / period + 0.15)), source=source,
                          params={"froude": round(speed ** 2 / (mk.G * 0.76), 3), "sway_m": sway, "arm_swing_rad": arm,
                                  "arm_offset_rad": arm_offset})


def wave(kin):
    def fn(t):
        return {"R_shoulder_roll": -2.2, "R_elbow": -0.8 + 0.45 * math.sin(2 * math.pi * t),
                "R_shoulder_yaw": 0.3 * math.sin(2 * math.pi * t), "L_elbow": -0.2}
    return mk.joint_cycle(kin, "wave", fn, period=2.0, source="humanoid_motions.wave",
                          params={"frequency_hz": 1.0})


def motions(robot=None):
    import humanoid_robot as hr
    kin = mk.Kin(robot or hr.talia())
    return kin, [walk(kin), wave(kin)]


def parts(machine) -> dict:
    kin, ms = motions(machine.robot() if hasattr(machine, "robot") else None)
    return mk.block(ms, base_height(kin))
