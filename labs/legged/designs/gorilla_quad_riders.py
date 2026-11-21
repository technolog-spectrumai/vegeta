"""Who rides Kerchak's quad (``gorilla_quad``): nobody (it drives itself), a human, or Kerchak. The quad adapts to the
rider the seat's load cell reports:

* **the handlebar** — the column telescopes and drops (``BAR_TRAVEL``): Kerchak's grips far forward for his long
  arms, a human's back and down (``human_bar_offset``, picked by IK: the most comfortable reach with the elbows bent);
* **the springs** — air springs at the corners and under the seat re-pressurised for the load: the same ride
  frequency and the same ride height whoever (or nobody) sits on it (``gorilla_quad.sizing(rider_mass=...)``);
* **the limits** — speed and acceleration by rider (``PROFILES``).

The human is a stand-in for an adult (1.78 m, 80 kg): Talia's frame (``humanoid_robot.talia``) at adult proportions,
its masses scaled to 80 kg, a helmet, its fists (gloves) on the bar.

    import gorilla_quad_riders as qr
    robot = qr.rider_robot("human")          # or "kerchak"
    q, err = qr.seated_pose(kin, "human", steer=0.2)
"""
from __future__ import annotations

import copy
import math

import numpy as np

import gorilla_quad as gq
from vegeta.chiron import Geom

__all__ = ["PROFILES", "HUMAN_DIMS", "HUMAN_MASS", "BAR_TRAVEL", "rider_robot", "passive_copy", "seated_pose", "human_bar_offset",
           "classify_seat_load", "grip"]

#: an adult rider's proportions [mm] on Talia's frame (the default Talia is a 1.56 m research humanoid)
HUMAN_DIMS = {"pelvis_width": 340.0, "pelvis_depth": 200.0, "pelvis_height": 120.0, "torso_width": 400.0, "torso_depth": 230.0,
              "torso_height": 500.0, "hip_spacing": 200.0, "thigh_length": 440.0, "shin_length": 430.0, "leg_width": 120.0,
              "ankle_height": 80.0, "foot_length": 270.0, "foot_width": 100.0, "upper_arm_length": 320.0, "forearm_length": 270.0,
              "arm_diameter": 90.0, "neck_length": 90.0, "head_width": 160.0, "head_depth": 200.0, "head_height": 230.0}
HUMAN_MASS = 80.0
#: the column's adjustment: the handlebar's travel back (along -x) and down [m]
BAR_TRAVEL = {"back": 0.30, "down": 0.15}

#: per rider: where the pelvis sits [m, vehicle frame], the grips' half spacing on the bar [m], the limits
PROFILES = {
    "none": {"label": "no rider - driving itself", "mass": 0.0, "v_max": 6.0, "a_max": 1.5, "lat_g": 0.35},
    "human": {"label": "human rider (1.78 m, 80 kg)", "mass": HUMAN_MASS, "pelvis": (-0.05, 0.0, 0.89), "grip_y": 0.40,
              "v_max": 6.0, "a_max": 1.5, "lat_g": 0.30},
    "kerchak": {"label": "Kerchak (384 kg)", "mass": gq.RIDER["mass"], "pelvis": tuple(gq.GEOM["pelvis"]), "grip_y": gq.GEOM["grip_y"],
                "v_max": 5.0, "a_max": 1.2, "lat_g": 0.25},
}

JACKET, PANTS, SKIN, BOOT, HELMET = (0.16, 0.36, 0.70, 1.0), (0.17, 0.18, 0.22, 1.0), (0.87, 0.70, 0.58, 1.0), (0.08, 0.08, 0.09, 1.0), (0.95, 0.95, 0.96, 1.0)


def classify_seat_load(kg: float) -> str:
    """The seat's load cell [kg] -> who sits: nobody, a human, or Kerchak."""
    if kg < 20.0:
        return "none"
    return "kerchak" if kg > 250.0 else "human"


def _human():
    import humanoid_robot as hr

    r = hr.talia("Rider", overrides=dict(HUMAN_DIMS), notes="an adult rider stand-in: Talia's frame at adult proportions, 80 kg")
    total = sum(lk.own_mass() for lk in r.links())
    f = HUMAN_MASS / total
    for lk in r.links():
        for g in lk.geoms:
            if g.mass:
                g.mass *= f
        for pm in lk.masses:
            pm.mass *= f
        for g in lk.geoms:
            n = g.name
            g.rgba = (SKIN if n in ("head", "neck") else JACKET if n == "torso_shell" or "arm" in n else BOOT if n in ("L", "R") or n.endswith("_hand") else PANTS)
    torso = next(lk for lk in r.links() if lk.name == "torso")
    head = next(g for g in torso.geoms if g.name == "head")
    torso.geoms.append(Geom("helmet", "sphere", (0.15,), pos=(head.pos[0] - 0.01, 0.0, head.pos[2] + 0.04), mass=0.0, role="visual", rgba=HELMET))
    torso.geoms.append(Geom("visor", "box", (0.02, 0.10, 0.04), pos=(head.pos[0] + 0.13, 0.0, head.pos[2] + 0.02), mass=0.0, role="visual",
                            rgba=(0.1, 0.1, 0.12, 1.0)))
    return r


def rider_robot(who: str):
    """The rider as a Chiron robot (servos on its joints): "human" or "kerchak"."""
    if who == "human":
        return _human()
    if who == "kerchak":
        import space_unit_robots as sr
        return sr.bare(casing="plastic")
    raise ValueError(who)


def passive_copy(robot):
    """The rider's link tree with the servos taken off (a Chiron prop is passive; the sim drives the joints itself)."""
    root = copy.deepcopy(robot.root)
    for lk, _ in root.walk():
        for j in lk.joints:
            j.servo = None
    return root


def grip(who: str, side: str, steer: float = 0.0, bar_offset=(0.0, 0.0)) -> np.ndarray:
    """Where the rider's fist holds the bar [m, vehicle frame], the bar turned ``steer`` and moved ``bar_offset``
    (back, down) by the column."""
    base = np.asarray(gq.GEOM["column_base"]) + np.array([-bar_offset[0], 0.0, -bar_offset[1]])
    p = np.array([gq.GEOM["grip_x"], (1 if side == "L" else -1) * PROFILES[who]["grip_y"], gq.GEOM["grip_z"]]) - np.asarray(gq.GEOM["column_base"])
    return base + gq._rot(gq._column_axis(), steer) @ p


def seated_pose(kin, who: str, steer: float = 0.0, q_init=None, bar_offset=(0.0, 0.0)):
    """The rider's joint angles astride the quad (soles on the footboards, fists on the bar turned ``steer``).
    Kerchak's is ``gorilla_quad.seated_pose`` (his fingers round the grips); the human's here, the forearm's end on the
    bar. Returns (q, worst IK error [m])."""
    if who == "kerchak":
        return gq.seated_pose(kin, steer, q_init=q_init)
    import humanoid_motions as hum  # noqa: F401  (kept for the sole convention)
    g = gq.GEOM
    pel = np.asarray(PROFILES["human"]["pelvis"])
    q = kin.q0.copy()
    for j, v in {"L_hip_pitch": -1.35, "R_hip_pitch": -1.35, "L_knee": 1.7, "R_knee": 1.7, "L_ankle_pitch": -0.3, "R_ankle_pitch": -0.3,
                 "L_shoulder_pitch": -0.9, "R_shoulder_pitch": -0.9, "L_elbow": -0.6, "R_elbow": -0.6}.items():
        if j in kin.index:
            q[kin.index[j]] = v
    q_ref = q.copy()
    if q_init is not None:
        q = np.array(q_init, dtype=float)
    kin.set(q)
    targets = []
    for side, sy in (("L", 1.0), ("R", -1.0)):
        body, c = kin.geom_point(side)
        half = kin.model.geom_size[kin.model.geom(side).id]
        for dx, dy in ((half[0] * 0.8, 0.0), (-half[0] * 0.8, half[1] * 0.7), (-half[0] * 0.8, -half[1] * 0.7)):
            sole = c + np.array([dx, dy, -half[2]])
            goal = np.array([g["footboard_x"] - 0.05 + dx, sy * g["footboard_y"] + dy, g["footboard_z"]]) - pel
            targets.append((body, sole, goal))
        hand = kin.geom_point(f"{side}_hand")                             # the gloved fist round the bar
        targets.append((hand[0], hand[1], grip("human", side, steer, bar_offset) - pel))
    joints = [n for n in kin.names if n != "waist_yaw"]
    return kin.ik(targets, q_init=q, q_ref=q_ref, joints=joints, iters=600, reg=3e-4)


def human_bar_offset(kin=None, steer_max=None):
    """The column's setting for the human: the least travel (back, down) at which the fists reach the bar across the
    steering range, the elbows bent (the reach kept 6 cm short of straight). Returns ((back, down), worst IK error)."""
    import motion_kit as mk

    kin = kin or mk.Kin(rider_robot("human"))
    steer_max = gq.GEOM["steer_max"] if steer_max is None else steer_max
    best = None
    for back in np.arange(0.0, BAR_TRAVEL["back"] + 1e-9, 0.05):
        for down in np.arange(0.0, BAR_TRAVEL["down"] + 1e-9, 0.05):
            off = (float(back), float(down))
            worst, q0 = 0.0, None
            for st in (0.0, steer_max, -steer_max):
                q, e = seated_pose(kin, "human", st, q_init=q0, bar_offset=off)
                q0 = q if st == 0.0 else q0
                kin.set(q)
                el = min(abs(q[kin.index["L_elbow"]]), abs(q[kin.index["R_elbow"]]))
                worst = max(worst, e + (0.06 if el < 0.25 else 0.0))       # a locked-straight elbow is no grip
            cost = worst + 0.02 * (back + down)
            if best is None or cost < best[0]:
                best = (cost, off, worst)
    return best[1], best[2]
