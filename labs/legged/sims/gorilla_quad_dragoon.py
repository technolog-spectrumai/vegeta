#!/usr/bin/env python3
"""Kerchak the dragoon: the quad is his horse (``gorilla_quad_autonomy``: it drives itself), the golf-ball bazooka
(``designs/gorilla_bazooka``) his carbine in the saddle scabbard.

The scenario, all in MuJoCo:

1. **ride out** - the quad drives him (AUTO, his fists on the bar) to the firing point, 18 m short of a stack of
   cardboard boxes, and stops facing them;
2. **draw** - his right fist goes down to the bazooka's rear handle in the scabbard beside his hip, closes, lifts it
   in front of his chest; the left fist leaves the bar for the fore handle;
3. **aim and fire** - the stock to his right shoulder, the tube on the boxes at the elevation the ball's flight
   needs (gravity and drag: ``gorilla_bazooka.elevation_for``); before each shot the tube's real direction is read
   and the aim corrected (the arms re-solved), and it fires when it is within ``AIM_TOL``; six balls, a second apart;
4. **stow** - the same moves back: the left fist to the bar, the bazooka down into the scabbard, the right fist back;
5. **ride back** - the quad U-turns and brings him home.

The arms move on keyframes solved by IK (``gorilla_scenarios.hand_targets``'s three points per fist, the D-handles
held as he holds the handlebar) and played through his joints' servos (PD and gravity compensation). The bazooka
follows his right fist (a mocap body placed from the fist's pose every step: its 6 kg are not on his arms, its
recoil not on his body - a golf ball's is 2 N s). The balls are real bodies: launched from the muzzle at the rims'
speed with the launcher's scatter, flying with gravity and the drag of ``balls.drag_coefficient``, bouncing, rolling,
knocking the boxes.

    python gorilla_quad_dragoon.py                   # the movie, sims/output/gorilla_quad_dragoon.mp4
    python gorilla_quad_dragoon.py --no-movie
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "legged" / "designs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import gorilla_bazooka as gb  # noqa: E402
import gorilla_quad as gq  # noqa: E402
import gorilla_quad_auto as qa  # noqa: E402
import gorilla_quad_autonomy as ga  # noqa: E402
import gorilla_scenarios as gs  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_unit_robots as sr  # noqa: E402
from vegeta.chiron import Geom, Link, Prop  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
FIRE_X = 22.0                                  #: the firing point [m]
RANGE = 18.0                                   #: the boxes this far ahead of the firing point
BOX = 0.20                                     #: the boxes' half size [m] (empty cardboard, 0.3 kg)
SHOTS = 6
AIM_TOL = math.radians(0.6)
HOME = (2.0, 8.0)
PAD_AT = np.array([-0.05, 0.15, -0.15])        #: the stock's pad from his right shoulder joint (pelvis frame) [m]
TURN_R = 4.0
#: the quadcopter the flak goes for: it hovers downrange and drifts across (a 1.6 kg drone holding its place on its
#: rotors' thrust); Armageddon's damage: under ``FLAK['crack_J']`` a hit cracks nothing, ``FLAK['drone_hp_J']`` brings it down
DRONE = {"home": (FIRE_X + RANGE + 4.0, 0.0, 5.5), "drift_m": 3.0, "drift_rad_s": 0.35, "kp": 6.0, "kd": 3.5}
FLAK_MAX_SHOTS = 30


def _path():
    """The ride out and the ride back: a polyline (x, y) with its arc length, and the index where it stops to fire."""
    out = [(x, 0.0) for x in np.arange(0.0, FIRE_X + 1e-9, 0.25)]
    back = [(x, 0.0) for x in np.arange(FIRE_X + 0.25, FIRE_X + 2.0 + 1e-9, 0.25)]
    cx, cy = FIRE_X + 2.0, TURN_R
    for a in np.linspace(-math.pi / 2, math.pi / 2, 40)[1:]:
        back.append((cx + TURN_R * math.cos(a), cy + TURN_R * math.sin(a)))
    back += [(x, 2 * TURN_R) for x in np.arange(cx - 0.25, HOME[0] - 1e-9, -0.25)]
    return np.array(out), np.array(back)


def scene():
    props = []
    k = 0
    for row, n in enumerate((3, 2, 1)):                                    # a pyramid of boxes, three rows
        for i in range(n):
            y = (i - (n - 1) / 2) * (2 * BOX + 0.02)
            props.append(Prop(Link(f"target_{k}", pos=(FIRE_X + RANGE, y, BOX + row * 2 * BOX),
                                   geoms=[Geom(f"target_{k}_box", "box", (BOX, BOX, BOX), mass=0.3, rgba=(0.78, 0.62, 0.40, 1.0),
                                               friction=(0.6, 0.005, 0.0001))]), free=True))
            k += 1
    balls = []
    for i in range(SHOTS):
        balls.append(f'''    <body name="ball_{i}" pos="{-20.0 - 0.3 * i} -12 {gb.BALL["radius"]}">
      <freejoint name="ball_{i}:free"/>
      <geom name="ball_{i}_g" type="sphere" size="{gb.BALL["radius"]}" mass="{gb.BALL["mass"]}" rgba="0.98 0.98 0.95 1" contype="1" conaffinity="1" condim="6" friction="0.5 0.005 0.0002" solref="0.004 0.6"/>
    </body>''')
    bazooka = '''    <body name="bazooka" mocap="true" pos="0 0 -5">
      <geom name="bazooka_ref" type="sphere" size="0.01" mass="0" contype="0" conaffinity="0" rgba="0 0 0 0"/>
    </body>
    <body name="flak" mocap="true" pos="0 0 -6">
      <geom name="flak_ref" type="sphere" size="0.01" mass="0" contype="0" conaffinity="0" rgba="0 0 0 0"/>
    </body>'''
    dark, rotor = (0.15, 0.15, 0.17, 1.0), (0.75, 0.10, 0.10, 1.0)
    drone = [Geom("drone_body", "box", (0.12, 0.12, 0.05), mass=1.0, rgba=dark)]
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        drone.append(Geom(f"drone_arm_{k}", "capsule", (0.015,), fromto=(0.0, 0.0, 0.0, 0.2 * sx, 0.2 * sy, 0.0), mass=0.05, rgba=dark))
        drone.append(Geom(f"drone_rotor_{k}", "cylinder", (0.10, 0.008), pos=(0.2 * sx, 0.2 * sy, 0.03), mass=0.05, rgba=rotor))
    props.append(Prop(Link("target_drone", pos=tuple(DRONE["home"]), geoms=drone), free=True))
    sc = {"rider": "kerchak", "start_x": 0.0, "seconds": 60.0}
    return {"props": props, "worldbody_xml": "\n".join(balls + [bazooka]), "sc": sc}


class DragoonRide(qa.AutoRide):
    def __init__(self):
        super().__init__("dragoon", scene=scene())
        m = self.model
        self.out_path, self.back_path = _path()
        self.path, self.path_i, self.leg = self.out_path, 0, "out"
        self.mocap = m.body_mocapid[m.body("bazooka").id]
        self.balls = [m.body(f"ball_{i}").id for i in range(SHOTS)]
        self.ball_q = [m.jnt_qposadr[m.joint(f"ball_{i}:free").id] for i in range(SHOTS)]
        self.ball_d = [m.jnt_dofadr[m.joint(f"ball_{i}:free").id] for i in range(SHOTS)]
        self.targets = [m.body(f"target_{k}").id for k in range(6)]
        self.target_q = [m.jnt_qposadr[m.joint(f"target_{k}:free").id] for k in range(6)]
        self.target0 = [np.array(self.data.qpos[a:a + 3]) for a in self.target_q]
        self.r_hand = m.body("R_hand").id
        self.pelvis = m.body("pelvis").id
        self.held = False
        self.mocap_flak = m.body_mocapid[m.body("flak").id]
        self.flak_holder = None                                             # which fist holds the flak, or None (in its scabbard)
        self.drone = m.body("target_drone").id
        self.drone_alive, self.drone_damage, self.drone_hits, self.drone_t0 = True, 0.0, [], None
        self.pellets, self.flak_shots, self.flak_aim_err = [], [], []
        self.blasts = []
        self.phase, self.phase_t0 = "ride out", 0.0
        self.keys = None                                                    # [(name, q, seconds)] of the stop's moves
        self.fired, self.next_shot, self.shots = 0, None, []
        self.trails = {i: [] for i in range(SHOTS)}
        self.ball_hits = set()
        self.rng = np.random.default_rng(3)
        self.aim_err = []
        self.wheelbase = gq.GEOM["wheelbase"]
        self.arbiter.mode = "AUTO"
        self.arbiter.events.append((0.0, "AUTO: the quad carries him out"))
        for g in range(m.ngeom):                                           # the lidars and the flak's rays do not see his weapons
            if m.geom_bodyid[g] in (m.body("bazooka").id, m.body("flak").id):
                m.geom_group[g] = 5

    # ---- the path follower (pure pursuit on a polyline)
    def follow(self, x, y, yaw, v):
        P = self.path
        lo = self.path_i
        d = np.hypot(P[lo:lo + 40, 0] - x, P[lo:lo + 40, 1] - y)
        self.path_i = lo + int(np.argmin(d))
        ld = max(2.0, 0.4 * abs(v) + 1.2)
        j = self.path_i
        while j < len(P) - 1 and math.hypot(P[j, 0] - x, P[j, 1] - y) < ld:
            j += 1
        tx, ty = P[j]
        alpha = math.atan2(ty - y, tx - x) - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steer = math.atan2(2 * self.wheelbase * math.sin(alpha), ld)
        remaining = float(np.sum(np.hypot(np.diff(P[self.path_i:, 0]), np.diff(P[self.path_i:, 1])))) + math.hypot(P[self.path_i, 0] - x, P[self.path_i, 1] - y)
        v_max = min(4.0, math.sqrt(max(0.0, 2 * 1.0 * (remaining - 0.15))))
        if self.leg == "back" and TURN_R > 0:
            v_max = min(v_max, math.sqrt(0.25 * 9.81 * TURN_R))
        return steer, v_max, remaining

    # ---- the stop's moves: IK keyframes in the pelvis frame
    def _kin_targets(self, W_rot, W_pos, sides):
        kin, G, GY = self.kin, gs.GRIP, gs.GRIP_Y
        out = []
        for s in sides:
            H_rot, H_pos = gb.weapon_to_hand(s, W_rot, W_pos)
            b = kin.body(f"{s}_hand")
            out += [(b, l, H_pos + H_rot @ l) for l in (G, np.zeros(3), G + np.array([0.0, GY, 0.0]))]
        return out

    def _bar_targets(self, side, q):
        kin, G, GY = self.kin, gs.GRIP, gs.GRIP_Y
        kin.set(q)
        b = kin.body(f"{side}_hand")
        return [(b, l, kin.world(b, l)) for l in (G, np.zeros(3), G + np.array([0.0, GY, 0.0]))]

    def _solve(self, targets, q_init, curl=None):
        kin = self.kin
        joints = [n for n in kin.actuated if ("shoulder" in n or "elbow" in n or "wrist" in n or "spine" in n or n == "waist_yaw")]
        q, e = kin.ik(targets, q_init=q_init, q_ref=self.Q[6], joints=joints, iters=800, reg=3e-4)
        for side, c in (curl or {}).items():
            sr.set_curl(q, kin, side, c)
        return q, e

    def _aim_pose(self, kin_q, muzzle=None, at=None, flat=False):
        """The weapon's pose in the pelvis frame for the aim: the pad at his shoulder, the tube on the boxes' middle at
        the elevation the flight needs (the quad's real pose now)."""
        kin = self.kin
        kin.set(kin_q)
        sh = kin.world(kin.body("R_upper_arm"), np.zeros(3))
        d = self.data
        Rp = d.xmat[self.pelvis].reshape(3, 3)
        pp = d.xpos[self.pelvis]
        aim_at = np.array([FIRE_X + RANGE - BOX, 0.0, 2 * BOX]) if at is None else np.asarray(at)   # the pyramid's front face, the middle row
        muzzle = gb.MUZZLE if muzzle is None else muzzle
        pad = sh + PAD_AT
        for _ in range(3):                                                 # the muzzle depends on the aim; two passes settle it
            W_rot = getattr(self, "_W_rot_guess", np.eye(3))
            muzzle_w = pp + Rp @ (pad + W_rot @ (muzzle - gb.PAD))
            dv = aim_at - muzzle_w
            az = math.atan2(dv[1], dv[0])
            el = math.atan2(dv[2], math.hypot(dv[0], dv[1])) if flat else gb.elevation_for(math.hypot(dv[0], dv[1]), dv[2])
            Rw = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]]) @ np.array(
                [[math.cos(el), 0, -math.sin(el)], [0, 1, 0], [math.sin(el), 0, math.cos(el)]])
            W_rot = Rp.T @ Rw
            self._W_rot_guess = W_rot
        return W_rot, pad - W_rot @ gb.PAD, Rw[:, 0]

    def plan_stop(self):
        """The draw-aim-stow keyframes, solved now that the quad stands at the firing point."""
        q_bar = self.Q[6].copy()
        Ws_rot, Ws_pos = gb.scabbard_pose()
        Ws_pos = Ws_pos - np.asarray(qr_pelvis())
        q_reach, e1 = self._solve(self._kin_targets(Ws_rot, Ws_pos, "R") + self._bar_targets("L", q_bar), q_bar, {"R": 0.5})
        q_grasp = q_reach.copy()
        sr.set_curl(q_grasp, self.kin, "R", gs.CURL_CLOSED)
        W_aim_rot, W_aim_pos, self.aim_dir = self._aim_pose(q_bar)
        W_mid_rot, W_mid_pos = W_aim_rot, 0.5 * (W_aim_pos + Ws_pos) + np.array([0.12, 0.08, 0.05])
        q_lift1, e2 = self._solve(self._kin_targets(W_mid_rot, W_mid_pos, "R") + self._bar_targets("L", q_bar), q_grasp, {"R": gs.CURL_CLOSED})
        q_lift2, e3 = self._solve(self._kin_targets(W_mid_rot, W_mid_pos, "RL"), q_lift1, {"R": gs.CURL_CLOSED, "L": gs.CURL_CLOSED})
        q_aim, e4 = self._solve(self._kin_targets(W_aim_rot, W_aim_pos, "RL"), q_lift2, {"R": gs.CURL_CLOSED, "L": gs.CURL_CLOSED})
        q_free = q_lift2.copy()
        sr.set_curl(q_free, self.kin, "L", 0.5)
        self.W_aim = (W_aim_rot, W_aim_pos)
        self.ik_errors = {"reach": round(e1, 4), "lift": round(e2, 4), "both hands": round(e3, 4), "aim": round(e4, 4)}
        self.keys = [("reach for the bazooka", q_reach, 1.4), ("grasp", q_grasp, 0.4), ("lift", q_lift1, 1.2),
                     ("left fist to the fore handle", q_lift2, 1.0), ("aim", q_aim, 1.0)]
        self.stow = [("lower it", q_lift2, 1.0), ("left fist to the bar", q_lift1, 1.0), ("into the scabbard", q_grasp, 1.2),
                     ("let go", q_reach, 0.4), ("fists on the bar", q_bar, 1.2)]
        self.q_aim = q_aim

    def plan_flak(self):
        """The flak's keyframes: the left fist to its fore handle in the left scabbard, lift it, the right fist to the
        rear handle, the stock to the shoulder and the tube on the drone."""
        q_bar = self.Q[6].copy()
        Wf_rot, Wf_pos = gb.flak_scabbard_pose()
        Wf_pos = Wf_pos - np.asarray(qr_pelvis())
        q_reach, e1 = self._solve(self._kin_targets(Wf_rot, Wf_pos, "L") + self._bar_targets("R", q_bar), q_bar, {"L": 0.5})
        q_grasp = q_reach.copy()
        sr.set_curl(q_grasp, self.kin, "L", gs.CURL_CLOSED)
        self._W_rot_guess = np.eye(3)
        W_aim_rot, W_aim_pos, self.aim_dir = self._aim_pose(q_bar, muzzle=gb.FLAK_MUZZLE, at=self._drone_aim_point(), flat=True)
        W_mid_rot, W_mid_pos = W_aim_rot, 0.5 * (W_aim_pos + Wf_pos) + np.array([0.12, -0.08, 0.05])
        q_lift1, e2 = self._solve(self._kin_targets(W_mid_rot, W_mid_pos, "L") + self._bar_targets("R", q_bar), q_grasp, {"L": gs.CURL_CLOSED})
        q_lift2, e3 = self._solve(self._kin_targets(W_mid_rot, W_mid_pos, "RL"), q_lift1, {"R": gs.CURL_CLOSED, "L": gs.CURL_CLOSED})
        q_aim, e4 = self._solve(self._kin_targets(W_aim_rot, W_aim_pos, "RL"), q_lift2, {"R": gs.CURL_CLOSED, "L": gs.CURL_CLOSED})
        self.W_aim = (W_aim_rot, W_aim_pos)
        self.ik_errors_flak = {"reach": round(e1, 4), "lift": round(e2, 4), "both hands": round(e3, 4), "aim": round(e4, 4)}
        self.keys = [("reach for the flak", q_reach, 1.4), ("grasp the flak", q_grasp, 0.4), ("lift the flak", q_lift1, 1.2),
                     ("right fist to the rear handle", q_lift2, 1.0), ("aim at the drone", q_aim, 1.0)]
        self.stow = [("lower the flak", q_lift2, 1.0), ("right fist to the bar", q_lift1, 1.0), ("flak into its scabbard", q_grasp, 1.2),
                     ("let go of the flak", q_reach, 0.4), ("fists on the bar", q_bar, 1.2)]
        self.q_aim = q_aim

    def _drone_aim_point(self, speed=None):
        """Where the drone will be when a flak ball gets there (its velocity held; a few iterations), the ball's drop added."""
        d = self.data
        p = np.array(d.xpos[self.drone])
        v = np.array(d.cvel[self.drone][3:6]) if False else self._drone_vel()
        speed = gb.FLAK["muzzle_speed"] if speed is None else speed
        mz = np.array(d.xpos[self.pelvis]) + np.array([0.6, -0.3, 0.6])
        aim = p.copy()
        for _ in range(3):
            tof = np.linalg.norm(aim - mz) / (0.93 * speed)
            aim = p + v * tof
        aim[2] += 0.5 * 9.81 * tof ** 2
        return aim

    def _drone_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.drone, v, 0)
        return v[3:6].copy()

    def _drone_control(self, t):
        """The drone holds its place on its rotors (a force and a torque on its body), drifting across; once down, nothing."""
        d = self.data
        b = self.drone
        d.xfrc_applied[b, :] = 0.0
        if not self.drone_alive:
            return
        m_d = float(self.model.body_subtreemass[b])
        h = DRONE["home"]
        p_des = np.array([h[0], h[1] + DRONE["drift_m"] * math.sin(DRONE["drift_rad_s"] * t), h[2]])
        v = self._drone_vel()
        acc = DRONE["kp"] * (p_des - d.xpos[b]) - DRONE["kd"] * v
        d.xfrc_applied[b, :3] = m_d * (acc + np.array([0.0, 0.0, 9.81]))
        R = d.xmat[b].reshape(3, 3)
        w = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, b, w, 0)
        d.xfrc_applied[b, 3:] = 0.4 * np.cross(R[:, 2], np.array([0, 0, 1.0])) - 0.05 * w[:3]

    def _fire_flak(self, t):
        self.move = f"flak: {len(self.flak_shots)} shots, the drone {'flying' if self.drone_alive else 'DOWN'}"
        if not self.drone_alive or len(self.flak_shots) >= FLAK_MAX_SHOTS:
            if t > self.next_shot + 1.5:
                self.phase = "stow flak"
                self._seq, self._seq_i, self._seq_t0, self._seq_from = self.stow, 0, t, self.q_fire
                self.arbiter.events.append((t, "the drone is down: stowing the flak" if not self.drone_alive else "out of time: stowing the flak"))
            return self.q_fire
        if t >= self.next_shot:
            W_rot, W_pos = self._weapon_world()
            mz = W_pos + W_rot @ gb.FLAK_MUZZLE
            self.aim_dir = self._drone_aim_point() - mz
            self.aim_dir /= np.linalg.norm(self.aim_dir)
            actual = W_rot[:, 0]
            err = math.acos(max(-1.0, min(1.0, float(actual @ self.aim_dir))))
            if err < math.radians(0.5) or t > self.next_shot + 1.0:
                self._shoot_flak(W_rot, W_pos, t)
                self.flak_aim_err.append(round(math.degrees(err), 2))
                self.next_shot = t + gb.FLAK["fire_every_s"]
            elif not getattr(self, "_corrected", False) or t > self._corrected + 0.25:
                self._correct(actual, err)
                self._corrected = t
        return self.q_fire

    def _correct(self, actual, err):
        """The tube points off: turn the target pose by the error, solve the arms again."""
        axis = np.cross(actual, self.aim_dir)
        if np.linalg.norm(axis) < 1e-9:
            return
        Rc = qa_rot(axis / np.linalg.norm(axis), err)
        Rp = self.data.xmat[self.pelvis].reshape(3, 3)
        W_rot_t, W_pos_t = self.W_aim
        pad = W_pos_t + W_rot_t @ gb.PAD
        W_rot_t = Rp.T @ Rc @ Rp @ W_rot_t
        self.W_aim = (W_rot_t, pad - W_rot_t @ gb.PAD)
        self.q_fire, _ = self._solve(self._kin_targets(*self.W_aim, "RL"), self.q_fire, {"R": gs.CURL_CLOSED, "L": gs.CURL_CLOSED})

    def _shoot_flak(self, W_rot, W_pos, t):
        F = gb.FLAK
        sp = F["muzzle_speed"] * (1.0 + F["scatter_speed"] * self.rng.standard_normal())
        dirn = W_rot[:, 0] + F["scatter_rad"] * (self.rng.standard_normal() * W_rot[:, 1] + self.rng.standard_normal() * W_rot[:, 2])
        dirn /= np.linalg.norm(dirn)
        p = W_pos + W_rot @ gb.FLAK_MUZZLE
        self.pellets.append({"p": p.copy(), "v": sp * dirn, "t0": t, "alive": True, "trail": [p.copy()]})
        self.flak_shots.append((round(t, 2), round(sp, 1)))
        self.blasts.append((t, W_pos + W_rot @ gb.FLAK_VENTURI, -W_rot[:, 0]))

    def _fly_pellets(self, t, dt):
        """The flak's balls: gravity and drag against Mach; a ray along each step's path finds what it hits."""
        m, d = self.model, self.data
        gid = np.zeros(1, dtype=np.int32)
        for pl in self.pellets:
            if not pl["alive"]:
                continue
            v = pl["v"]
            step = v * dt
            L = float(np.linalg.norm(step))
            dist = mujoco.mj_ray(m, d, pl["p"], step / L, self.geomgroup, 1, -1, gid)
            if 0.0 <= dist <= L and gid[0] >= 0:
                root = m.body(m.body_rootid[m.geom_bodyid[gid[0]]]).name
                pl["p"] = pl["p"] + step / L * dist
                pl["alive"] = False
                if root == "target_drone" and self.drone_alive:
                    E = 0.5 * gb.FLAK["ball_mass"] * float(v @ v)
                    if E > gb.FLAK["crack_J"]:
                        self.drone_damage += E
                        self.drone_hits.append((round(t, 2), round(E, 1)))
                        d.xfrc_applied[self.drone, :3] += gb.FLAK["ball_mass"] * v / dt      # the ball's momentum, this step
                        if self.drone_damage >= gb.FLAK["drone_hp_J"]:
                            self.drone_alive = False
                            self.arbiter.events.append((t, f"the drone is hit ({self.drone_damage:.0f} J): it falls"))
            else:
                pl["v"] = v + (gb.flak_drag_acc(v) + np.array([0.0, 0.0, -9.81])) * dt
                pl["p"] = pl["p"] + step
                if t - pl["t0"] > 2.5 or pl["p"][2] < 0.0:
                    pl["alive"] = False
            if np.linalg.norm(pl["p"] - pl["trail"][-1]) > 2.0 or not pl["alive"]:
                pl["trail"].append(pl["p"].copy())

    # ---- one control step
    def control(self, t):
        m, d = self.model, self.data
        self.t = t
        dt = m.opt.timestep
        x, y, yaw, v = self.pose()
        if t - self._t_scan >= 0.1 - 1e-9:
            self.lidar()
            self._t_scan = t
        steer, vt = 0.0, 0.0
        arms = None
        if self.phase in ("ride out", "ride back") and t >= qa.SETTLE_S:
            steer, vt, remaining = self.follow(x, y, yaw, v)
            if remaining < 0.2 and abs(v) < 0.05:
                if self.phase == "ride out":
                    self.phase, self.phase_t0 = "stopped", t
                    self.arbiter.events.append((t, "at the firing point: he draws"))
                else:
                    self.phase = "home"
                    self.arbiter.events.append((t, "home"))
        elif self.phase == "stopped" and t > self.phase_t0 + 0.4:
            self.plan_stop()
            self._seq, self._seq_i, self._seq_t0, self._seq_from = self.keys, 0, t, self._current_q()
            self.phase = "draw"
        if self.phase in ("draw", "stow", "draw flak", "stow flak"):
            arms = self._play(t)
        elif self.phase == "fire":
            arms = self._fire(t)
        elif self.phase == "fire flak":
            arms = self._fire_flak(t)
        self.aeb = self.guardian.check(x, y, yaw, v, float(d.qpos[self.col_q]), self.obstacles, self.wheelbase)
        rate = ga.STEER_BY_WIRE["rate_rad_s"] * dt
        self.steer_cmd += float(np.clip(steer - self.steer_cmd, -rate, rate))
        d.ctrl[self.col] = self.steer_cmd
        target = 0.0 if self.aeb else vt
        self.vt_cmd += float(np.clip(target - self.vt_cmd, -2.5 * dt, 1.2 * dt))
        R = gq.GEOM["wheel_r"]
        P = self.sz["motor"]["peak_power_kW"] * 1000.0
        T = self.sz["wheel_torque_Nm"]
        for act, dof in zip(self.motor, self.spin):
            w = float(d.qvel[dof])
            cap = min(T, P / max(abs(w), 1e-3))
            d.ctrl[act] = float(np.clip(120.0 * (self.vt_cmd / R - w), -cap, cap))
        d.qfrc_applied[:] = 0.0
        for dl, dr, k in self.arb:
            f = k * (d.qpos[m.jnt_qposadr[m.dof_jntid[dl]]] - d.qpos[m.jnt_qposadr[m.dof_jntid[dr]]])
            d.qfrc_applied[dl] -= f
            d.qfrc_applied[dr] += f
        q_t = self._arms(float(d.qpos[self.col_q])) if arms is None else arms
        tau = self.kp * (q_t - d.qpos[self.r_q]) - self.kd * d.qvel[self.r_d] + d.qfrc_bias[self.r_d]
        d.qfrc_applied[self.r_d] += np.clip(tau, -self.lim, self.lim)
        self._drone_control(t)
        self._place_bazooka()
        self._fly_balls()
        self._fly_pellets(t, dt)
        return self.arbiter.mode

    def _current_q(self):
        return np.array(self.data.qpos[self.r_q])

    def _play(self, t):
        name, q_to, dur = self._seq[self._seq_i]
        u = min(1.0, (t - self._seq_t0) / dur)
        q = self._seq_from + (q_to - self._seq_from) * mk.smooth(u)
        self.move = name
        if u >= 1.0:
            if name == "grasp":
                self.held = True
            if name == "let go":
                self.held = False
            if name == "grasp the flak":
                self.flak_holder = "L"
            if name == "right fist to the rear handle":
                self.flak_holder = "R"
            if name == "right fist to the bar":
                self.flak_holder = "L"
            if name == "let go of the flak":
                self.flak_holder = None
            self._seq_i += 1
            self._seq_t0, self._seq_from = t, q_to
            if self._seq_i >= len(self._seq):
                if self.phase == "draw":
                    self.phase, self.phase_t0 = "fire", t
                    self.next_shot, self.q_fire = t + 0.8, self.q_aim.copy()
                    self.arbiter.events.append((t, "aimed: firing"))
                elif self.phase == "stow":
                    self.plan_flak()
                    self.phase = "draw flak"
                    self._seq, self._seq_i, self._seq_t0, self._seq_from = self.keys, 0, t, q_to
                    self.arbiter.events.append((t, "the drone: he draws the flak"))
                elif self.phase == "draw flak":
                    self.phase, self.phase_t0 = "fire flak", t
                    self.next_shot, self.q_fire, self._corrected = t + 0.6, self.q_aim.copy(), False
                    self.arbiter.events.append((t, "flak: aimed at the drone"))
                else:
                    self.phase = "ride back"
                    self.path, self.path_i, self.leg = self.back_path, 0, "back"
                    self.arbiter.events.append((t, "stowed: the quad brings him back"))
        return q

    def _weapon_world(self, side="R"):
        d = self.data
        h = self.r_hand if side == "R" else self.model.body("L_hand").id
        return gb.hand_to_weapon(side, d.xmat[h].reshape(3, 3), np.array(d.xpos[h]))

    def _fire(self, t):
        self.move = f"firing {self.fired}/{SHOTS}"
        if self.fired >= SHOTS:
            if t > self.next_shot + 0.6:
                self.phase = "stow"
                self._seq, self._seq_i, self._seq_t0, self._seq_from = self.stow, 0, t, self.q_fire
                self.arbiter.events.append((t, "stowing the bazooka"))
            return self.q_fire
        if t >= self.next_shot:
            W_rot, W_pos = self._weapon_world()
            actual = W_rot[:, 0]
            err = math.acos(max(-1.0, min(1.0, float(actual @ self.aim_dir))))
            if err < AIM_TOL or t > self.next_shot + 1.5:
                self._shoot(W_rot, W_pos, t)
                self.aim_err.append(round(math.degrees(err), 2))
                self.next_shot = t + 1.0
            elif not getattr(self, "_corrected", False) or t > self._corrected + 0.5:
                # the tube points off: aim the target pose the other way by the error, solve the arms again
                axis = np.cross(actual, self.aim_dir)
                if np.linalg.norm(axis) > 1e-9:
                    Rc = qa_rot(axis / np.linalg.norm(axis), err)
                    Rp = self.data.xmat[self.pelvis].reshape(3, 3)
                    W_rot_t, W_pos_t = self.W_aim
                    W_rot_t = Rp.T @ Rc @ Rp @ W_rot_t
                    pad = W_pos_t + self.W_aim[0] @ gb.PAD
                    W_pos_t = pad - W_rot_t @ gb.PAD
                    self.W_aim = (W_rot_t, W_pos_t)
                    self.q_fire, _ = self._solve(self._kin_targets(W_rot_t, W_pos_t, "RL"), self.q_fire,
                                                 {"R": gs.CURL_CLOSED, "L": gs.CURL_CLOSED})
                self._corrected = t
        return self.q_fire

    def _shoot(self, W_rot, W_pos, t):
        d = self.data
        i = self.fired
        sp = gb.SPEC["rim_speed"] * gb.SPEC["grip"] * (1.0 + gb.SPEC["scatter_speed"] * self.rng.standard_normal())
        dirn = W_rot[:, 0] + gb.SPEC["scatter_rad"] * (self.rng.standard_normal() * W_rot[:, 1] + self.rng.standard_normal() * W_rot[:, 2])
        dirn /= np.linalg.norm(dirn)
        p = W_pos + W_rot @ gb.MUZZLE + 0.03 * dirn
        a = self.ball_q[i]
        d.qpos[a:a + 3] = p
        d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        b = self.ball_d[i]
        d.qvel[b:b + 3] = sp * dirn
        d.qvel[b + 3:b + 6] = 0.0
        self.fired += 1
        self.shots.append((round(t, 2), round(sp, 1)))

    def _place_bazooka(self):
        d = self.data
        if self.held:
            W_rot, W_pos = self._weapon_world()
        else:
            Rc = d.xmat[self.chassis].reshape(3, 3)
            Ws_rot, Ws_pos = gb.scabbard_pose()
            W_rot, W_pos = Rc @ Ws_rot, d.xpos[self.chassis] + Rc @ Ws_pos
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, W_rot.ravel())
        d.mocap_pos[self.mocap] = W_pos
        d.mocap_quat[self.mocap] = q
        if self.flak_holder is not None:
            W_rot, W_pos = self._weapon_world(self.flak_holder)
        else:
            Rc = d.xmat[self.chassis].reshape(3, 3)
            Wf_rot, Wf_pos = gb.flak_scabbard_pose()
            W_rot, W_pos = Rc @ Wf_rot, d.xpos[self.chassis] + Rc @ Wf_pos
        mujoco.mju_mat2Quat(q, W_rot.ravel())
        d.mocap_pos[self.mocap_flak] = W_pos
        d.mocap_quat[self.mocap_flak] = q

    def _fly_balls(self):
        d = self.data
        for i in range(self.fired):
            b = self.balls[i]
            v = d.qvel[self.ball_d[i]:self.ball_d[i] + 3]
            d.xfrc_applied[b, :3] = gb.drag_force(v)
            for c in d.contact[:d.ncon]:
                gid = (c.geom1, c.geom2)
                bodies = {self.model.geom_bodyid[g] for g in gid}
                if b in bodies and bodies & set(self.targets):
                    self.ball_hits.add(i)

    def step(self, t, dt):
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for k in range(n):
            self.control(t + k * self.model.opt.timestep)
            mujoco.mj_step(self.model, self.data)
        for i in range(self.fired):
            p = self.data.xpos[self.balls[i]].copy()
            if not self.trails[i] or np.linalg.norm(p - self.trails[i][-1]) > 0.25:
                self.trails[i].append(p)
        x, y, yaw, v = self.pose()
        self.log.append((t + dt, x, y, v, self.arbiter.mode, self.aeb, self.phase, 0.0))

    def report(self):
        moved = [round(float(np.linalg.norm(self.data.qpos[a:a + 3] - p0)), 2) for a, p0 in zip(self.target_q, self.target0)]
        x, y, yaw, v = self.pose()
        return {"shots": self.shots, "aim_error_deg_at_shot": self.aim_err, "balls_that_hit_a_box": sorted(self.ball_hits),
                "boxes_moved_m": moved, "boxes_knocked": sum(1 for v_ in moved if v_ > 0.1), "ik_errors_m": getattr(self, "ik_errors", None),
                "flak_shots": len(self.flak_shots), "flak_aim_error_deg_at_shot": self.flak_aim_err, "drone_hits": self.drone_hits,
                "drone_down": not self.drone_alive, "ik_errors_flak_m": getattr(self, "ik_errors_flak", None),
                "end": (round(x, 2), round(y, 2)), "home": HOME, "phase": self.phase,
                "events": [(round(t, 2), e) for t, e in self.arbiter.events]}


def qr_pelvis():
    return gq.GEOM["pelvis"]


def qa_rot(axis, ang):
    return gq._rot(axis, ang)


class DragoonDrawer(qa.Drawer):
    def __init__(self, ride, size="1280x720"):
        super().__init__(ride, size, extra_pieces=gb.scabbard_mesh() + gb.bazooka_mesh() + gb.flak_scabbard_mesh() + gb.flak_mesh())
        self.cam = None

    def frame(self, t):
        ride, br = self.ride, self.br
        x, y, yaw, v = ride.pose()
        b, tris, _ = self.mesh.pieces[self.beacon_i]
        self.mesh.pieces[self.beacon_i] = (b, tris, (*(qa.MODE_RGB["AUTO"] if ride.phase in ("ride out", "ride back") else (0.95, 0.75, 0.10)), 1.0))
        c = np.array(ride.data.xpos[ride.chassis])
        ph = ride.phase
        if ph in ("ride out", "home") or (ph == "ride back" and ride.path_i < 10):
            want = (np.array([c[0] + 1.0, c[1], 1.0]), 20.0, math.degrees(yaw) + 220.0, 7.5)
        elif ph in ("stopped", "draw", "stow"):
            want = (np.array([c[0] + 0.2, c[1] - 0.2, 1.2]), 12.0, math.degrees(yaw) - 70.0, 4.6)
        elif ph == "fire flak":
            dp = np.array(ride.data.xpos[ride.drone])
            want = (0.75 * c + 0.25 * dp + np.array([0.0, 0.0, 0.6]), 8.0, math.degrees(yaw) + 200.0, 8.5)   # over his shoulder, the drone ahead in the sky
        elif ph in ("draw flak", "stow flak"):
            want = (np.array([c[0] + 0.2, c[1] + 0.2, 1.2]), 12.0, math.degrees(yaw) + 70.0, 4.6)   # his left side
        elif ph == "fire":
            want = (np.array([c[0] + 5.0, c[1], 1.2]), 12.0, math.degrees(yaw) + 200.0, 9.0)       # over his right shoulder, downrange
        else:
            want = (np.array([c[0] + 1.0, c[1], 1.0]), 20.0, math.degrees(yaw) + 220.0, 7.5)
        if self.cam is None:
            self.cam = [want[0].copy(), want[1], want[2], want[3]]
        k = 0.12
        self.cam[0] += k * (want[0] - self.cam[0])
        for i in (1, 2, 3):
            if i == 2:
                dlt = (want[2] - self.cam[2] + 180.0) % 360.0 - 180.0
                self.cam[2] += k * dlt
            else:
                self.cam[i] += k * (want[i] - self.cam[i])
        cam = br.orbit(tuple(self.cam[0]), self.cam[1], self.cam[2], self.cam[3])
        lines, dots = [], []
        for i in range(ride.fired if ph in ("fire", "stow", "stopped", "draw") else 0):   # the golf balls' trails, while the bazooka is out
            tr = ride.trails[i]
            lines += [(tr[j], tr[j + 1], (0.95, 0.85, 0.20)) for j in range(len(tr) - 1)]
            dots.append((ride.data.xpos[ride.balls[i]], 4, (1.0, 1.0, 1.0)))
        if ph == "fire" and hasattr(ride, "aim_dir"):
            W_rot, W_pos = ride._weapon_world()
            mz = W_pos + W_rot @ gb.MUZZLE
            lines.append((mz, mz + 4.0 * W_rot[:, 0], (0.95, 0.30, 0.20)))
        if ph in ("draw flak", "fire flak", "stow flak"):                    # a ring round the drone
            dp = np.array(ride.data.xpos[ride.drone])
            dots.append((dp, 10, (0.95, 0.1, 0.1) if ride.drone_alive else (0.4, 0.4, 0.4)))
            dots.append((dp, 6, (1.0, 1.0, 1.0)))
        for pl in ride.pellets:
            tr = pl["trail"] + ([pl["p"]] if pl["alive"] else [])
            lines += [(tr[j], tr[j + 1], (1.0, 0.45, 0.10)) for j in range(len(tr) - 1)]
        for tb, o, back in ride.blasts:
            if t - tb < 0.25:                                                   # the back-blast: the jet out of the venturi, 2.1 m
                side = np.cross(back, [0.0, 0.0, 1.0])
                side /= max(np.linalg.norm(side), 1e-9)
                for a in np.linspace(-0.35, 0.35, 7):
                    for b_ in (-0.2, 0.0, 0.2):
                        dvec = back + a * side + b_ * np.array([0.0, 0.0, 1.0])
                        lines.append((o, o + gb.FLAK["backblast_m"] * dvec / np.linalg.norm(dvec), (0.85, 0.85, 0.80)))
        what = {"draw flak": getattr(ride, "move", ""), "fire flak": getattr(ride, "move", ""), "stow flak": getattr(ride, "move", ""),
                "ride out": "the quad carries him out (AUTO)", "stopped": "at the firing point", "draw": getattr(ride, "move", ""),
                "fire": getattr(ride, "move", ""), "stow": getattr(ride, "move", ""), "ride back": "the quad brings him back (AUTO)",
                "home": "home"}[ph]
        knocked = sum(1 for a, p0 in zip(ride.target_q, ride.target0) if np.linalg.norm(ride.data.qpos[a:a + 3] - p0) > 0.1)
        hud = [(f"Kerchak the dragoon - {what}", (0.15, 0.15, 0.2)),
               (f"{3.6 * v:4.1f} km/h   bazooka {ride.fired}/{SHOTS}, boxes knocked {knocked}/6   flak {len(ride.flak_shots)} shots, "
                f"drone {ride.drone_damage:.0f}/{gb.FLAK['drone_hp_J']:.0f} J {'DOWN' if not ride.drone_alive else ''}", (0.2, 0.2, 0.25))]
        title = f"Kerchak the dragoon - the golf-ball bazooka and the flak   t = {t:4.1f} s"
        return self.renderer.render(ride.data, cam, title=title, lines=lines, dots=dots, hud=hud)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kerchak the dragoon: ride out, draw, fire, stow, ride back")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--stills", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ride = DragoonRide()
    drawer = None if a.no_movie else DragoonDrawer(ride, a.size)
    stills = [float(s) for s in a.stills.split(",") if s.strip()]
    tmp = tempfile.mkdtemp()
    dt = 1.0 / a.fps
    t, k = 0.0, 0
    while t <= ride.sc["seconds"]:
        ride.step(t, dt)
        t += dt
        if drawer is not None:
            img = drawer.frame(t)
            img.save(os.path.join(tmp, f"f{k:04d}.png"))
            for ts in stills:
                if abs(t - ts) < dt / 2:
                    img.save(str(OUT / f"gorilla_quad_dragoon_{ts:g}s.png"))
        k += 1
        if ride.phase == "home" and t > 1.0:
            for _ in range(int(1.0 / dt)):                                   # a second at home
                ride.step(t, dt)
                t += dt
                if drawer is not None:
                    drawer.frame(t).save(os.path.join(tmp, f"f{k:04d}.png"))
                k += 1
            break
    rep = ride.report()
    (OUT / "gorilla_quad_dragoon.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if drawer is not None:
        out = str(OUT / "gorilla_quad_dragoon.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        print("movie", out)
    return rep


if __name__ == "__main__":
    main()
