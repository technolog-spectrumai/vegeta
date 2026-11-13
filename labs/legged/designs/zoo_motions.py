"""The zoomorphic robots' motions (``motion_kit``), exported as ``data:motions`` (tag ``motion``).

* **Laika** (dog) — walk, trot, bound; **Ursa** (bear) — walk, trot (a heavy, short-legged amble: no bound);
  **Cerva** (deer) — walk, trot, bound. Speeds from the Froude number on each leg (``motion_kit.GAITS``).
* **Gaja** (elephant) — walk only (elephants never trot or gallop; a slow Fr 0.03 walk, ~1 s strides, duty ≥ 0.6, Hutchinson et al.
  2003), a trunk swing and a trunk lift (the tip raised over the head).
* **Naga** (snake) — slither (serpenoid lateral undulation, Hirose: 1.5 waves on the body, ±0.5 rad) and
  sidewind (the yaw wave plus a pitch wave a quarter period behind, lifting the body off the ground in arcs).
  Their speeds are the ideal no-slip values (the body follows its own track), an upper bound.
* **Noctis** (bat) — fold and unfold (the wing joints solved by IK through ``zoo_specs.bat_wing_points`` at
  fold = 0 … 1) and flap (2.5 Hz, ±0.6 rad stroke, the wing a quarter folded on the upstroke).
* **Kraken** (octopus) — crawl (arm pairs sweeping and lifting in a travelling wave around the mantle), pulse
  (all arms curling in and spreading, the jet-swim stroke) and reach (one arm's tip to a point ahead).
"""
from __future__ import annotations

import math

import numpy as np

import motion_kit as mk
import zoo_robots as zr
import zoo_specs as zs

__all__ = ["motions_dog", "motions_bear", "motions_deer", "motions_elephant", "motions_snake", "motions_bat",
           "motions_octopus", "parts_dog", "parts_bear", "parts_deer", "parts_elephant", "parts_snake", "parts_bat",
           "parts_octopus", "leg_length", "base_height"]

mm = 1e-3


def leg_length(p):
    return (p["upper_leg_length"] + p["lower_leg_length"] + p["pastern_length"]) * mm


def base_height(p):
    return zs.body_centre(p)[2] * mm


def _quadruped(robot, p, gaits, extra=()):
    kin = mk.Kin(robot)
    L = leg_length(p)
    out = [mk.quadruped_gait(kin, g, L, source=f"zoo_motions: motion_kit.GAITS[{g!r}]") for g in gaits]
    return kin, out + [f(kin) for f in extra]


def motions_dog(robot=None):
    return _quadruped(robot or zr.dog(), zs.DOG, ("walk", "trot", "bound"))


def motions_bear(robot=None):
    return _quadruped(robot or zr.bear(), zs.BEAR, ("walk", "trot"))


def motions_deer(robot=None):
    return _quadruped(robot or zr.deer(), zs.DEER, ("walk", "trot", "bound"))


# ------------------------------------------------------------------------------------------------- elephant
def _trunk_joints(kin, axis):
    return sorted(n for n in kin.names if n.startswith("trunk_") and n.endswith(axis))


def trunk_swing(kin):
    yaw = _trunk_joints(kin, "_b")
    pitch = _trunk_joints(kin, "_a")

    def fn(t):
        w = 2 * math.pi * t / 3.0
        out = {j: 0.12 * math.sin(w - 0.5 * i) for i, j in enumerate(yaw)}
        out.update({j: 0.04 * math.sin(2 * w - 0.5 * i) for i, j in enumerate(pitch)})
        return out
    return mk.joint_cycle(kin, "trunk_swing", fn, period=3.0, source="zoo_motions.trunk_swing",
                          params={"yaw_amplitude_rad": 0.12, "wave_lag_rad_per_segment": 0.5})


def trunk_lift(kin, p=zs.ELEPHANT):
    n = int(p["trunk_segments"])
    tip = kin.capsule_end(f"trunk_s{n - 1}_seg")
    p0 = kin.world(*tip, kin.q0)
    _, hc = zs.head_centre(p)
    up = np.array([hc[0] * mm + 0.25, 0.0, (hc[2] + p["head_height"] * 0.9) * mm]) - np.array([0, 0, 0])
    joints = [j for j in kin.names if j.startswith("trunk_")]
    keys = [(0.0, [(*tip, p0)]), (0.5, [(*tip, up)]), (0.7, [(*tip, up)]), (1.0, [(*tip, p0)])]
    return mk.ik_transition(kin, "trunk_lift", keys, duration=4.0, joints=joints, source="zoo_motions.trunk_lift",
                            params={"tip_goal_m": [round(float(x), 3) for x in up]})


def motions_elephant(robot=None):
    return _quadruped(robot or zr.elephant(), zs.ELEPHANT, (), extra=(
        lambda k: mk.quadruped_gait(k, "walk", leg_length(zs.ELEPHANT), froude=0.03, duty=0.7, stance=0.35,
                                    step=0.06, crouch=0.04, source="zoo_motions: an elephant walk (Fr 0.03, duty 0.7)"),
        trunk_swing, trunk_lift))


# ------------------------------------------------------------------------------------------------- snake
def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def serpenoid(kin, p, name, *, alpha=0.5, waves=1.5, period=2.0, pitch=0.0, pitch_lag=math.pi / 2, sideways=False,
              source="zoo_motions.serpenoid"):
    """Hirose's serpenoid: joint i turns α sin(ωt − iβ) relative to its neighbour (β = 2π · waves / n), measured
    from the design pose's own wave (``zoo_specs.slither_points``); ``pitch`` adds a vertical wave ``pitch_lag``
    behind (sidewinding)."""
    n = int(p["segments"])
    beta = 2 * math.pi * waves / n
    rest = [math.radians(p["wave_deg"]) * math.sin(1.1 * i) for i in range(n)]
    w = 2 * math.pi / period

    def fn(t):
        out = {}
        for i in range(n):
            out[f"seg_{i:02d}_a"] = alpha * math.sin(w * t - i * beta) - rest[i]
            if pitch:
                out[f"seg_{i:02d}_b"] = pitch * math.sin(w * t - i * beta - pitch_lag)
        return out
    wavelength = n * p["segment_length"] * mm / waves
    v = wavelength / period                                               # no slip: one wavelength per period
    base = {"velocity_m_s": [0.0, round(v * 0.5, 4), 0.0] if sideways else [round(v, 4), 0.0, 0.0],
            "note": "ideal (no-slip) speed, an upper bound: real ground slips; yaw_rad: the head's heading per "
                    "keyframe, keeping the body's mean axis still (the head wags, the body does not swing)"}
    m = mk.joint_cycle(kin, name, fn, period=period, base=base, source=source,
                       params={"alpha_rad": alpha, "waves_on_body": waves, "pitch_rad": pitch})
    yaw, pit = [], []
    for t in m.t:                                                         # segment headings relative to the head
        h, e, hs, es = 0.0, 0.0, [], []
        for i in range(n):
            h += alpha * math.sin(w * t - i * beta)
            e += pitch * math.sin(w * t - i * beta - pitch_lag)
            hs.append(h)
            es.append(e)
        yaw.append(-sum(hs) / n)
        pit.append(-sum(es) / n)
    m.base["yaw_rad"] = [round(v, 5) for v in yaw]
    if pitch:                                                             # level the body's mean, then lift the
        m.base["pitch_rad"] = [round(v, 5) for v in pit]                  # base so its lowest segment rests on
        r = [(p["body_radius"] + (p["tail_radius"] - p["body_radius"]) * i / max(n - 1, 1)) * mm for i in range(n)]
        dz = []                                                           # the ground, none below it
        for k in range(len(m.t)):
            q = kin.q0.copy()
            for j, v in zip(m.joints, m.Q[k]):
                q[kin.index[j]] = v
            kin.set(q)
            o = kin.data.xpos[kin.body("head")]
            R = _rz(yaw[k]) @ _ry(pit[k])
            z = [(o + R @ (kin.data.xpos[kin.body(f"seg_{i:02d}_link")] - o))[2] for i in range(n)]
            dz.append(round(max(ri - zi for ri, zi in zip(r, z)), 5))
        m.base["z_offset_m"] = dz
        m.base["note"] += ("; pitch_rad: the head's pitch per keyframe, levelling the body's mean; z_offset_m: the "
                           "base raised per keyframe so the lowest segment touches the ground (yaw, then pitch, about "
                           "the head)")
    return m


def motions_snake(robot=None, p=zs.SNAKE):
    kin = mk.Kin(robot or zr.snake())
    return kin, [serpenoid(kin, p, "slither"),
                 serpenoid(kin, p, "sidewind", alpha=0.35, waves=1.0, period=2.5, pitch=0.06, sideways=True)]


# ------------------------------------------------------------------------------------------------- bat
def _wing_targets(kin, p, fold):
    """IK targets putting each wing's elbow, wrist and finger tips where ``bat_wing_points`` puts them at ``fold``."""
    z = p["stand_height"] * mm
    out = []
    for side, sy in (("L", 1), ("R", -1)):
        sh, el, wr, tips = zs.bat_wing_points(dict(p, fold=fold), sy)
        w = lambda q: np.array([q[0] * mm, q[1] * mm, q[2] * mm])               # noqa: E731
        out.append((kin.body(f"{side}_wing_elbow_link"), np.zeros(3), w(el)))
        for i, tp in enumerate(tips):
            out.append((kin.body(f"{side}_finger{i}_wrist_link"), np.zeros(3), w(wr)))
            out.append((*kin.capsule_end(f"{side}_finger{i}_wrist_seg"), w(tp)))
    return out


def bat_fold(kin, p=zs.BAT, unfold=False):
    fs = [mk.smooth(i / 24) for i in range(25)]                           # every frame at its own fold value
    if unfold:
        fs = fs[::-1]
    keys = [(i / (len(fs) - 1), _wing_targets(kin, p, f)) for i, f in enumerate(fs)]
    wing = [j for j in kin.names if "wing" in j or "finger" in j]
    return mk.ik_transition(kin, "unfold" if unfold else "fold", keys, duration=1.2, joints=wing, frames=24,
                            source="zoo_motions.bat_fold: IK through zoo_specs.bat_wing_points")


def bat_flap(kin, p=zs.BAT, freq=2.5, stroke=0.6):
    wing = [j for j in kin.names if "wing" in j or "finger" in j]
    q_up, _ = kin.ik(_wing_targets(kin, p, 0.25), joints=wing)             # the upstroke's partly folded wing

    def fn(t):
        ph = 2 * math.pi * freq * t
        up = 0.5 * (1 - math.cos(ph))                                         # 0 at the top of the downstroke
        fold = max(0.0, -math.sin(ph))                                        # folds while the wing rises
        out = {j: fold * q_up[kin.index[j]] for j in wing}
        for side, sy in (("L", 1), ("R", -1)):
            out[f"{side}_wing_shoulder_a"] = sy * stroke * math.cos(ph)       # + lifts the left wing
            out[f"{side}_wing_shoulder_b"] = out.get(f"{side}_wing_shoulder_b", 0.0) + sy * 0.15 * up
        return out
    return mk.joint_cycle(kin, "flap", fn, period=1.0 / freq, frames=24, source="zoo_motions.bat_flap",
                          params={"frequency_hz": freq, "stroke_rad": stroke, "upstroke_fold": 0.25})


def motions_bat(robot=None):
    kin = mk.Kin(robot or zr.bat())
    return kin, [bat_fold(kin), bat_fold(kin, unfold=True), bat_flap(kin)]


# ------------------------------------------------------------------------------------------------- octopus
def _arm_joints(kin, i, axis):
    return sorted((n for n in kin.names if n.startswith(f"arm{i}_") and n.endswith(axis)),
                  key=lambda n: int(n.split("_s")[1].split("_")[0]))


def octopus_crawl(kin, p=zs.OCTOPUS, period=2.4):
    n = int(p["arms"])

    def fn(t):
        out = {}
        for i in range(n):
            ph = 2 * math.pi * t / period + math.pi * (i % 2) + 2 * math.pi * i / n
            for k, j in enumerate(_arm_joints(kin, i, "_a")):
                out[j] = 0.12 * math.sin(ph - 0.4 * k)
            for k, j in enumerate(_arm_joints(kin, i, "_b")):
                out[j] = -0.18 * max(0.0, math.sin(ph - 0.4 * k + math.pi / 2))     # negative lifts the arm
        return out
    return mk.joint_cycle(kin, "crawl", fn, period=period, source="zoo_motions.octopus_crawl",
                          params={"sweep_rad": 0.12, "lift_rad": 0.18, "alternate": "even/odd arms half a cycle apart"})


def octopus_pulse(kin, p=zs.OCTOPUS, period=1.6):
    n = int(p["arms"])

    def fn(t):
        s = math.sin(math.pi * t / period) ** 2                                   # 0 → 1 → 0
        out = {}
        for i in range(n):
            for k, j in enumerate(_arm_joints(kin, i, "_b")):
                out[j] = -0.22 * s * (k + 1) / 4
        return out
    return mk.joint_cycle(kin, "pulse", fn, period=period, source="zoo_motions.octopus_pulse",
                          params={"curl_rad_per_segment": "0.055·(k+1) at the peak"})


def octopus_reach(kin, p=zs.OCTOPUS):
    m_arm = int(p["arm_segments"])
    tip = kin.capsule_end(f"arm0_s{m_arm - 1}_seg")
    p0 = kin.world(*tip, kin.q0)
    goal = np.array([0.6, 0.2, 0.55])
    joints = [j for j in kin.names if j.startswith("arm0_")]
    keys = [(0.0, [(*tip, p0)]), (0.5, [(*tip, goal)]), (0.7, [(*tip, goal)]), (1.0, [(*tip, p0)])]
    return mk.ik_transition(kin, "reach", keys, duration=3.0, joints=joints, source="zoo_motions.octopus_reach",
                            params={"arm": 0, "tip_goal_m": goal.tolist()})


def motions_octopus(robot=None):
    kin = mk.Kin(robot or zr.octopus())
    return kin, [octopus_crawl(kin), octopus_pulse(kin), octopus_reach(kin)]


# ------------------------------------------------------------------------------------------------- export
def _parts(fn, height=None):
    def parts(machine) -> dict:
        _, ms = fn()
        return mk.block(ms, height)
    return parts


parts_dog = _parts(motions_dog, base_height(zs.DOG))
parts_bear = _parts(motions_bear, base_height(zs.BEAR))
parts_deer = _parts(motions_deer, base_height(zs.DEER))
parts_elephant = _parts(motions_elephant, base_height(zs.ELEPHANT))
parts_snake = _parts(motions_snake, zs.SNAKE["body_radius"] * mm)
parts_bat = _parts(motions_bat, zs.BAT["stand_height"] * mm)
parts_octopus = _parts(motions_octopus, (zs.OCTOPUS["arm_radius"] + 40.0 + zs.OCTOPUS["head_radius"] * 0.8) * mm)
