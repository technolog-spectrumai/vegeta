"""The space game units' motions (``motion_kit``), exported as ``data:motions`` (tag ``motion``).

* **march** (every unit) — Talia's walk at the unit's weight: 0.55 m/s (the Heavy 0.4), 0.34 m stance, duty 0.64,
  a 20 mm crouch (near-straight knees: the modules' torque in single support, ``sims/space_squad_march.py``),
  ±60 mm sway; both arms held at their stance (the carbine at the low ready, the shield or the
  chainsaw forward, the maul up, the cannon ahead), the Heavy's free arm swinging a little.
* **slash** (Marine with a chainsaw) — the saw raised out to the side and swung across the body and down, the waist
  turning with it, the knees giving at the cut.
* **aim** (the Marines and the Centurion) — the forearm brought up level so the carbine (along it) points ahead: the shoulder pitched up by
  90° less the elbow's 60°, the waist turned in; exact in joint space, eased in, held, eased out.
* **fire** (Marine) — in the aim pose, the recoil: the right shoulder kicked back 0.04 rad at 8 Hz with the waist
  and the left shoulder taking a little of it.
* **rally** (Centurion) — the banner raised straight up and swept side to side, the waist turning with it, then
  lowered.
* **shield_up** (Breacher) — the guard in joint space: the shield arm forward with the elbow opened so the plate stays
  near upright, the waist turned behind it, the maul arm cocked back; eased in, held, eased out.
* **swing** (Breacher) — the maul swung overhead and down: the right shoulder from 0.5 rad back to −2.1 rad up and
  through, the waist turning with it, the knees giving at the blow.
* **suppress** (Heavy) — the cannon ahead at the hip, the waist sweeping ±0.4 rad at 0.2 Hz with the arm holding the
  line, the legs braced wide.
"""
from __future__ import annotations

import math

import numpy as np

import humanoid_motions as hum
import motion_kit as mk
import space_unit_specs as su

__all__ = ["motions_marine", "motions_marine_saw", "motions_centurion", "motions_breacher", "motions_heavy", "parts_marine",
           "parts_marine_saw", "parts_centurion", "parts_breacher", "parts_heavy", "base_height"]

base_height = hum.base_height
SOLE_POINTS = hum.SOLE_POINTS


def march(kin, unit, *, speed=0.55, stance=0.34, duty=0.64, step=0.05, crouch=0.02, sway=0.06, swing=0.15,
          hold=None, name="march"):
    """The walk with the weapon arm held (``hold``: {joint: rad}) and the other swinging ±``swing``."""
    e = math.radians(su.ROSTER[unit][0]["elbow_bend_deg"])
    hold = dict(hold or {})
    period = stance / (duty * speed)
    feet = {"L": 0.0, "R": 0.5}
    swinging = [s for s in feet if f"{s}_shoulder_pitch" not in hold]

    def foot_dx(t, ph):
        s = (t / period + ph) % 1.0
        return 0.5 - s / duty if s < duty else -0.5 + mk.smooth((s - duty) / (1 - duty))

    def extra(t, q):
        for side in swinging:
            q[kin.index[f"{side}_shoulder_pitch"]] = 2 * swing * foot_dx(t, feet[side])
        for side in feet:
            q[kin.index[f"{side}_elbow"]] = -e
        for j, v in hold.items():
            q[kin.index[j]] = v
        return q

    return mk.legged_gait(kin, name, feet, period=period, duty=duty, stride=stance, step_height=step,
                          foot_points={g: SOLE_POINTS for g in feet}, crouch=crouch, extra=extra,
                          q_start={f"{s}_{j}": v for s in feet for j, v in (("hip_pitch", -0.3), ("knee", 0.6), ("ankle_pitch", -0.3))},
                          lateral=lambda t: -sway * math.sin(2 * math.pi * (t / period + 0.15)),
                          source=f"space_unit_motions.march: humanoid_motions.walk at the {unit}'s weight",
                          params={"froude": round(speed ** 2 / (mk.G * 0.96), 3), "sway_m": sway, "arm_swing_rad": swing,
                                  "held": sorted(hold)})


def _hold(kin, q, joints):
    return {j: float(q[kin.index[j]]) for j in joints}


def aim_pose(kin):
    """The Marine's aim: the carbine lies along the forearm, so the forearm brought level points it ahead - the right
    shoulder pitched up by 90° less the elbow's bend, the waist turned 0.15 rad in to bring it before the chest; exact
    in joint space (no IK). Returns (q, error = 0, the joints)."""
    e = math.radians(su.MARINE["elbow_bend_deg"])
    arm = ["R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "waist_yaw"]
    q = kin.q0.copy()
    q[kin.index["R_shoulder_pitch"]] = -(math.pi / 2 - e)
    q[kin.index["waist_yaw"]] = -0.15
    return q, 0.0, arm


def aim(kin, duration=2.4):
    q_aim, err, arm = aim_pose(kin)

    def fn(t):
        u = t / duration
        w = mk.smooth(u / 0.35) if u < 0.7 else 1 - mk.smooth((u - 0.7) / 0.3)
        return {j: kin.q0[kin.index[j]] + w * (q_aim[kin.index[j]] - kin.q0[kin.index[j]]) for j in arm}
    m = mk.joint_cycle(kin, "aim", fn, period=duration, frames=30, loop=False, kind="transition",
                       source="space_unit_motions.aim: the forearm level, the carbine along it, eased in joint space",
                       params={"shoulder_pitch_rad": round(-(math.pi / 2 - math.radians(su.MARINE["elbow_bend_deg"])), 4)})
    m.check["ik_error_max_m"] = round(err, 6)
    return m


def fire(kin, rate_hz=8.0, kick=0.04, period=1.0):
    q_aim, err, arm = aim_pose(kin)

    def fn(t):
        k = 0.5 * (1 - math.cos(2 * math.pi * rate_hz * t)) if t < period else 0.0
        out = {j: float(q_aim[kin.index[j]]) for j in arm}
        out["R_shoulder_pitch"] = out["R_shoulder_pitch"] + kick * k
        out["waist_yaw"] = out["waist_yaw"] + 0.25 * kick * k
        out["L_shoulder_pitch"] = -0.3 - 0.3 * kick * k
        return out
    m = mk.joint_cycle(kin, "fire", fn, period=period, frames=int(rate_hz * period * 6),
                       source="space_unit_motions.fire: the recoil kick on the aim pose", params={"rate_hz": rate_hz, "kick_rad": kick})
    m.check["ik_error_max_m"] = round(err, 6)
    return m


GUARD = {"L_shoulder_pitch": -0.55, "L_shoulder_roll": 0.25, "L_shoulder_yaw": 0.3, "L_elbow": -0.35, "waist_yaw": 0.2}
COCK = {"R_shoulder_pitch": 0.6, "R_shoulder_roll": -0.5}


def shield_up(kin, duration=2.0):
    """The guard, in joint space: the shield arm forward (the shoulder 0.55 rad up, 0.25 rad out, the elbow opened to
    0.35 rad so the plate stays near upright), the waist turned a little behind it, the maul arm cocked back; eased
    stance -> guard -> held -> stance."""
    def fn(t):
        u = t / duration
        w = mk.smooth(u / 0.35) if u < 0.7 else 1 - mk.smooth((u - 0.7) / 0.3)
        return {j: kin.q0[kin.index[j]] + w * (v - kin.q0[kin.index[j]]) for j, v in GUARD.items()} | \
               {j: kin.q0[kin.index[j]] + w * v for j, v in COCK.items()}
    return mk.joint_cycle(kin, "shield_up", fn, period=duration, frames=30, loop=False, kind="transition",
                          source="space_unit_motions.shield_up: the guard pose in joint space, the maul cocked",
                          params={"guard_rad": GUARD})


def swing(kin, period=1.8):
    e = math.radians(su.BREACHER["elbow_bend_deg"])

    def fn(t):
        u = t / period
        s = mk.smooth(u / 0.45) if u < 0.45 else (1 - mk.smooth((u - 0.45) / 0.25)) if u < 0.7 else 0.0
        blow = math.sin(math.pi * min(max((u - 0.4) / 0.3, 0.0), 1.0)) if 0.4 <= u <= 0.7 else 0.0
        return {"R_shoulder_pitch": 0.5 - 2.6 * s, "R_shoulder_roll": -0.25 * s, "R_elbow": -e - 0.4 * s,
                "waist_yaw": -0.35 * s + 0.2 * blow, "L_shoulder_pitch": -0.9 + 0.3 * s,
                "L_knee": 0.25 * blow, "R_knee": 0.25 * blow, "L_hip_pitch": -0.12 * blow, "R_hip_pitch": -0.12 * blow,
                "L_ankle_pitch": -0.13 * blow, "R_ankle_pitch": -0.13 * blow}
    return mk.joint_cycle(kin, "swing", fn, period=period, frames=36,
                          source="space_unit_motions.swing: the maul overhead and down, the knees giving at the blow",
                          params={"period_s": period})


def suppress(kin, period=5.0, sweep=0.4):
    e = math.radians(su.HEAVY["elbow_bend_deg"])

    def fn(t):
        return {"waist_yaw": sweep * math.sin(2 * math.pi * t / period), "R_shoulder_pitch": -0.35, "R_elbow": -e,
                "L_shoulder_pitch": -0.5, "L_elbow": -e - 0.3, "L_shoulder_roll": 0.35,
                "L_hip_roll": 0.12, "R_hip_roll": -0.12, "L_ankle_roll": -0.12, "R_ankle_roll": 0.12,
                "L_knee": 0.2, "R_knee": 0.2, "L_hip_pitch": -0.1, "R_hip_pitch": -0.1, "L_ankle_pitch": -0.1, "R_ankle_pitch": -0.1}
    return mk.joint_cycle(kin, "suppress", fn, period=period, frames=40,
                          source="space_unit_motions.suppress: the cannon swept across by the waist, the stance braced wide",
                          params={"sweep_rad": sweep, "frequency_hz": 1 / period})


def slash(kin, period=1.6):
    """The chainsaw swung across: the left arm raised out and brought across the body and down, the waist turning
    with it, the right arm (the carbine) kept clear."""
    e = math.radians(su.MARINE_SAW["elbow_bend_deg"])

    def fn(t):
        u = t / period
        s = mk.smooth(u / 0.4) if u < 0.4 else (1 - mk.smooth((u - 0.4) / 0.35)) if u < 0.75 else 0.0
        cut = math.sin(math.pi * min(max((u - 0.35) / 0.3, 0.0), 1.0)) if 0.35 <= u <= 0.65 else 0.0
        return {"L_shoulder_pitch": -0.4 - 1.4 * s, "L_shoulder_roll": 1.1 * s - 0.9 * cut, "L_shoulder_yaw": 0.4 * s,
                "L_elbow": -e - 0.5 * s, "waist_yaw": 0.3 * s - 0.5 * cut, "R_shoulder_pitch": -0.3 + 0.3 * s,
                "R_elbow": -e, "L_knee": 0.15 * cut, "R_knee": 0.15 * cut, "L_hip_pitch": -0.07 * cut, "R_hip_pitch": -0.07 * cut,
                "L_ankle_pitch": -0.08 * cut, "R_ankle_pitch": -0.08 * cut}
    return mk.joint_cycle(kin, "slash", fn, period=period, frames=36,
                          source="space_unit_motions.slash: the chainsaw raised out and swung across", params={"period_s": period})


def rally(kin, period=3.0):
    """The banner raised high and swept side to side: the left arm straight up (the shoulder to -2.6 rad), the pole
    waved ±0.35 rad about the shoulder's roll with the waist turning with it, then lowered."""
    e = math.radians(su.CENTURION["elbow_bend_deg"])

    def fn(t):
        u = t / period
        up = mk.smooth(u / 0.25) if u < 0.25 else 1.0 if u < 0.8 else 1 - mk.smooth((u - 0.8) / 0.2)
        wave = math.sin(2 * math.pi * 2 * (u - 0.25) / 0.55) if 0.25 <= u <= 0.8 else 0.0
        return {"L_shoulder_pitch": -0.3 - 2.3 * up, "L_shoulder_roll": (0.3 + 0.3 * wave) * up, "L_elbow": -e + 0.6 * up,
                "waist_yaw": 0.2 * wave * up, "R_shoulder_pitch": -0.3, "R_elbow": -e}
    return mk.joint_cycle(kin, "rally", fn, period=period, frames=45, loop=False, kind="transition",
                          source="space_unit_motions.rally: the banner raised and waved", params={"period_s": period})


def motions_marine(robot=None, crouch=0.02):
    import space_unit_robots as sr
    kin = mk.Kin(robot or sr.marine())
    hold = {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.45}
    return kin, [march(kin, "marine", hold=hold, crouch=crouch), aim(kin), fire(kin)]


def motions_marine_saw(robot=None, crouch=0.02):
    import space_unit_robots as sr
    kin = mk.Kin(robot or sr.marine_saw())
    hold = {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.3}
    return kin, [march(kin, "marine_saw", hold=hold, crouch=crouch), aim(kin), slash(kin)]


def motions_centurion(robot=None, crouch=0.02):
    import space_unit_robots as sr
    kin = mk.Kin(robot or sr.centurion())
    hold = {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.3}
    return kin, [march(kin, "centurion", hold=hold, crouch=crouch), aim(kin), rally(kin)]


def motions_breacher(robot=None, crouch=0.02):
    import space_unit_robots as sr
    kin = mk.Kin(robot or sr.breacher())
    hold = {"L_shoulder_pitch": -0.7, "R_shoulder_pitch": 0.3}
    return kin, [march(kin, "breacher", hold=hold, crouch=crouch), shield_up(kin), swing(kin)]


def motions_heavy(robot=None, crouch=0.02):
    import space_unit_robots as sr
    kin = mk.Kin(robot or sr.heavy())
    hold = {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.5}
    return kin, [march(kin, "heavy", speed=0.4, stance=0.30, sway=0.07, hold=hold, crouch=crouch), suppress(kin)]


def parts_marine(machine) -> dict:
    import space_unit_robots as sr
    return sr.parts_marine(machine)


def parts_marine_saw(machine) -> dict:
    import space_unit_robots as sr
    return sr.parts_marine_saw(machine)


def parts_centurion(machine) -> dict:
    import space_unit_robots as sr
    return sr.parts_centurion(machine)


def parts_breacher(machine) -> dict:
    import space_unit_robots as sr
    return sr.parts_breacher(machine)


def parts_heavy(machine) -> dict:
    import space_unit_robots as sr
    return sr.parts_heavy(machine)
