"""The monsters' motions (``motion_kit``), exported as ``data:motions`` (tag ``motion``).

* **Fenrir** (wolf) — walk, trot, bound (``motion_kit.GAITS`` on its 0.71 m legs).
* **Leshy** — a slow, stately walk (Fr 0.05, duty 0.8) and a trot, on its 1.24 m legs.
* **Chiron-C** (centaur) — walk and trot on the horse half; reach (both grippers forward and down to a point
  0.45 m ahead of the chest, the waist pitching with them) and wave (the right arm raised to the side).
* **Hirudo** (leech) — inchworm (a vertical travelling wave, one per body: the grounded parts stay put while the
  lifted arch passes, so it advances by the arch's arc length less its span each period) and swim (a lateral
  serpenoid, ``zoo_motions.serpenoid``).
* **Wraith** — drift (every tentacle in a slow travelling wave, neighbours a quarter cycle apart) and grab
  (tentacle 0 reaches a point ahead of the gondola, its last three segments curling round it, then back).
* **Ghoul** — shamble (Talia's walk, slower and lower: 0.35 m/s, 80 mm crouch, ±60 mm sway, the arms held forward
  at 0.9 rad) and lunge (from a guard with the elbows bent 0.5 rad, both clawed hands thrust forward to 0.5 m
  ahead of the pelvis at chest height and back).
* **Wyvern** — hover (rotor spin rates: thrust = weight, ``T = C_T ρ n² D⁴`` with an assumed C_T 0.10, neighbouring
  rotors counter-rotating) and grab (the talons reach 0.15 m forward and 0.06 m down, the claws close, the
  talons lift 0.16 m with the load).
"""
from __future__ import annotations

import math

import numpy as np

import humanoid_motions as hum
import monster_robots as mr
import monster_specs as ms
import motion_kit as mk
import zoo_motions as zm

__all__ = ["motions_wolf", "motions_leshy", "motions_centaur", "motions_leech", "motions_wraith", "motions_ghoul",
           "motions_wyvern", "parts_wolf", "parts_leshy", "parts_centaur", "parts_leech", "parts_wraith",
           "parts_ghoul", "parts_wyvern", "hover_rate"]

RHO = 1.225          # air density at sea level [kg/m³]
C_T = 0.10           # rotor thrust coefficient (n in rev/s, D in m): assumed, typical of 15–20" props


def motions_wolf(robot=None):
    return zm._quadruped(robot or mr.wolf(), ms.WOLF, ("walk", "trot", "bound"))


def motions_leshy(robot=None):
    kin = mk.Kin(robot or mr.leshy())
    L = zm.leg_length(ms.LESHY)
    return kin, [mk.quadruped_gait(kin, "walk", L, froude=0.05, duty=0.8, source="monster_motions: a stately walk"),
                 mk.quadruped_gait(kin, "trot", L, source="monster_motions: motion_kit.GAITS['trot']")]


# ------------------------------------------------------------------------------------------------- centaur
def _grippers(kin):
    return [(kin.body(f"{s}_arm_elbow_link"), kin.geom_point(f"{s}_gripper")[1]) for s in ("L", "R")]


def centaur_reach(kin, p=ms.CENTAUR):
    grips = _grippers(kin)
    rest = [kin.world(b, l, kin.q0) for b, l in grips]
    waist, top, _ = ms.centaur_torso(p)
    goal = [np.array([waist[0] * 1e-3 + 0.45, y, (waist[2] + p["torso_height"] * 0.25) * 1e-3])
            for y in (0.15, -0.15)]
    upper = [j for j in kin.names if j.startswith(("waist", "L_arm", "R_arm"))]
    keys = [(0.0, [(b, l, r) for (b, l), r in zip(grips, rest)]), (0.45, [(b, l, g) for (b, l), g in zip(grips, goal)]),
            (0.65, [(b, l, g) for (b, l), g in zip(grips, goal)]), (1.0, [(b, l, r) for (b, l), r in zip(grips, rest)])]
    return mk.ik_transition(kin, "reach", keys, duration=3.0, joints=upper, source="monster_motions.centaur_reach",
                            params={"goals_m": [[round(float(x), 3) for x in g] for g in goal]})


def centaur_wave(kin):
    def fn(t):
        w = 2 * math.pi * t
        return {"R_arm_shoulder_b": -1.5, "R_arm_shoulder_a": -0.3, "R_arm_elbow": -0.9 + 0.4 * math.sin(w),
                "waist_yaw": 0.1 * math.sin(w / 2)}
    return mk.joint_cycle(kin, "wave", fn, period=2.0, source="monster_motions.centaur_wave",
                          params={"frequency_hz": 1.0})


def motions_centaur(robot=None):
    kin = mk.Kin(robot or mr.centaur())
    L = zm.leg_length(ms.CENTAUR)
    return kin, [mk.quadruped_gait(kin, "walk", L, source="monster_motions: motion_kit.GAITS['walk']"),
                 mk.quadruped_gait(kin, "trot", L, source="monster_motions: motion_kit.GAITS['trot']"),
                 centaur_reach(kin), centaur_wave(kin)]


# ------------------------------------------------------------------------------------------------- leech
def leech_inchworm(kin, p=ms.LEECH, pitch=0.35, period=3.0):
    m = zm.serpenoid(kin, p, "inchworm", alpha=0.0, waves=1.0, period=period, pitch=pitch,
                     source="monster_motions.leech_inchworm: a vertical travelling wave")
    n = int(p["segments"])
    spans = []
    for k in range(len(m.t)):                                             # the body's horizontal span per frame
        q = kin.q0.copy()
        for j, v in zip(m.joints, m.Q[k]):
            q[kin.index[j]] = v
        kin.set(q)
        o, R = kin.data.xpos[kin.body("head")], zm._ry(m.base["pitch_rad"][k])
        xs = [(o + R @ (kin.data.xpos[kin.body(f"seg_{i:02d}_link")] - o))[:2] for i in range(n)]
        spans.append(sum(float(np.linalg.norm(xs[i + 1] - xs[i])) for i in range(n - 1)))
    arc = (n - 1) * p["segment_length"] * 1e-3
    v = (arc - float(np.mean(spans))) / period
    m.base["velocity_m_s"] = [round(v, 4), 0.0, 0.0]
    m.base["note"] = ("inchworm: the grounded parts stay put, the body advances by its arc length less its "
                      "horizontal span once per period; z_offset_m keeps the lowest segment on the ground")
    return m


def motions_leech(robot=None):
    kin = mk.Kin(robot or mr.leech())
    return kin, [leech_inchworm(kin), zm.serpenoid(kin, ms.LEECH, "swim", alpha=0.45, waves=1.0, period=2.0,
                                                   source="monster_motions: a lateral serpenoid (swimming)")]


# ------------------------------------------------------------------------------------------------- wraith
def _tentacle(kin, i, axis):
    return sorted((n for n in kin.names if n.startswith(f"arm{i}_") and n.endswith(axis)),
                  key=lambda n: int(n.split("_s")[1].split("_")[0]))


def wraith_drift(kin, p=ms.WRAITH, period=4.0):
    n = int(p["arms"])

    def fn(t):
        out = {}
        for i in range(n):
            ph = 2 * math.pi * t / period + i * math.pi / 2
            for k, j in enumerate(_tentacle(kin, i, "_a")):
                out[j] = 0.12 * math.sin(ph - 0.6 * k)
            for k, j in enumerate(_tentacle(kin, i, "_b")):
                out[j] = 0.12 * math.cos(ph - 0.6 * k)
        return out
    return mk.joint_cycle(kin, "drift", fn, period=period, source="monster_motions.wraith_drift",
                          params={"amplitude_rad": 0.12, "lag_rad_per_segment": 0.6})


def wraith_grab(kin, p=ms.WRAITH):
    """The start of the last three segments is tracked; those segments then curl round on top of the IK."""
    segs = int(p["arm_segments"])
    hand = (kin.body(f"arm0_s{segs - 3}_link"), np.zeros(3))
    p0 = kin.world(*hand, kin.q0)
    goal = np.array([0.9, 0.3, 0.85])
    curl = [f"arm0_s{k}_a" for k in range(segs - 3, segs)]
    arm = [j for j in kin.names if j.startswith("arm0_") and j not in curl and not j.startswith(f"arm0_s{segs - 3}")
           and not j.startswith(f"arm0_s{segs - 2}") and not j.startswith(f"arm0_s{segs - 1}")]

    def close(t, q):                                                      # the last three segments wrap round
        w = mk.smooth((t / 4.0 - 0.45) / 0.15) * (1 - mk.smooth((t / 4.0 - 0.75) / 0.15))
        for j in curl:
            q[kin.index[j]] = kin.q0[kin.index[j]] + 0.7 * w
        return q
    keys = [(0.0, [(*hand, p0)]), (0.45, [(*hand, goal)]), (0.8, [(*hand, goal)]), (1.0, [(*hand, p0)])]
    return mk.ik_transition(kin, "grab", keys, duration=4.0, joints=arm, extra=close,
                            source="monster_motions.wraith_grab",
                            params={"hand_goal_m": goal.tolist(), "curl_rad": 0.7, "curling_segments": 3})


def motions_wraith(robot=None):
    kin = mk.Kin(robot or mr.wraith())
    return kin, [wraith_drift(kin), wraith_grab(kin)]


# ------------------------------------------------------------------------------------------------- ghoul
def ghoul_lunge(kin):
    hands = [(kin.body(f"{s}_forearm"), kin.geom_point(f"{s}_hand")[1]) for s in ("L", "R")]
    guard = kin.q0.copy()                                                 # from and back to a guard: elbows bent
    for s in ("L", "R"):
        guard[kin.index[f"{s}_elbow"]] = -0.5
    rest = [kin.world(b, l, guard) for b, l in hands]
    goal = [np.array([0.5, y, 0.3]) for y in (0.14, -0.14)]
    upper = [j for j in kin.names if j.startswith(("waist", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow"))]
    keys = [(0.0, [(b, l, r) for (b, l), r in zip(hands, rest)]), (0.35, [(b, l, g) for (b, l), g in zip(hands, goal)]),
            (0.5, [(b, l, g) for (b, l), g in zip(hands, goal)]), (1.0, [(b, l, r) for (b, l), r in zip(hands, rest)])]
    return mk.ik_transition(kin, "lunge", keys, duration=2.0, joints=upper, source="monster_motions.ghoul_lunge",
                            params={"hand_goals_m": [g.tolist() for g in goal]})


def motions_ghoul(robot=None):
    kin = mk.Kin(robot or mr.ghoul())
    shamble = hum.walk(kin, speed=0.35, stance=0.25, duty=0.65, step=0.04, crouch=0.08, sway=0.06, arm=0.2,
                       arm_offset=-0.9, elbow=-0.5, name="shamble", source="monster_motions: humanoid_motions.walk")
    return kin, [shamble, ghoul_lunge(kin)]


# ------------------------------------------------------------------------------------------------- wyvern
def hover_rate(robot, p=ms.WYVERN):
    """Each rotor's spin rate [rad/s] for thrust = weight: n = sqrt(T / (C_T ρ D⁴)) rev/s."""
    T = robot.total_mass() * mk.G / 6
    D = 2 * p["rotor_radius"] * 1e-3
    return 2 * math.pi * math.sqrt(T / (C_T * RHO * D ** 4))


def wyvern_hover(kin):
    w = hover_rate(kin.robot)
    rates = {f"rotor{k}_spin": w * (1 if k % 2 == 0 else -1) for k in range(6)}
    return mk.Motion("hover", "rates", [], [0.0], np.zeros((1, 0)), loop=True, period=None, rates=rates,
                     base={"velocity_m_s": [0.0, 0.0, 0.0]}, source="monster_motions.wyvern_hover",
                     params={"thrust_per_rotor_n": round(kin.robot.total_mass() * mk.G / 6, 2), "C_T": C_T,
                             "rho_kg_m3": RHO, "rpm": round(w * 60 / (2 * math.pi)),
                             "assumed": "C_T: no propeller map yet (Boreas computes one when the wyvern goes to Piccolo)"},
                     check={"limit_margin_min_rad": None})


def wyvern_grab(kin):
    """The wrists (the claws' base) are tracked; the claws close on top of the IK, so they stay out of it."""
    wr = [(kin.body(f"{s}_talon_wrist_link"), np.zeros(3)) for s in ("L", "R")]
    rest = [kin.world(*t, kin.q0) for t in wr]
    goal = [r + np.array([0.15, 0.0, -0.06]) for r in rest]              # the talons reach 0.87 m; they hang 0.72 m
    lift = [r + np.array([0.05, 0.0, 0.10]) for r in rest]
    talons = [j for j in kin.names if "talon" in j]
    wrists = [j for j in talons if j.endswith("wrist")]

    def close(t, q):
        w = mk.smooth((t / 4.0 - 0.45) / 0.15)
        for j in wrists:
            q[kin.index[j]] = kin.q0[kin.index[j]] + 0.8 * w
        return q
    keys = [(0.0, [(*t, r) for t, r in zip(wr, rest)]), (0.4, [(*t, g) for t, g in zip(wr, goal)]),
            (0.6, [(*t, g) for t, g in zip(wr, goal)]), (1.0, [(*t, g) for t, g in zip(wr, lift)])]
    return mk.ik_transition(kin, "grab", keys, duration=4.0, joints=[j for j in talons if j not in wrists], extra=close,
                            source="monster_motions.wyvern_grab",
                            params={"wrist_reach_m": [0.15, 0.0, -0.06], "lift_m": 0.10, "close_rad": 0.8})


def motions_wyvern(robot=None):
    kin = mk.Kin(robot or mr.wyvern())
    return kin, [wyvern_hover(kin), wyvern_grab(kin)]


# ------------------------------------------------------------------------------------------------- export
def _parts(fn, height=None):
    def parts(machine) -> dict:
        kin, mos = fn()
        h = height(kin) if callable(height) else height
        return mk.block(mos, h)
    return parts


parts_wolf = _parts(motions_wolf, zm.base_height(ms.WOLF))
parts_leshy = _parts(motions_leshy, zm.base_height(ms.LESHY))
parts_centaur = _parts(motions_centaur, zm.base_height(ms.CENTAUR))
parts_leech = _parts(motions_leech, ms.LEECH["body_radius"] * ms.LEECH["flatten"] * 1e-3)
parts_wraith = _parts(motions_wraith, ms.WRAITH["float_height"] * 1e-3)
parts_ghoul = _parts(motions_ghoul, hum.base_height)
parts_wyvern = _parts(motions_wyvern, ms.WYVERN["cruise_height"] * 1e-3)
