#!/usr/bin/env python3
"""Kerchak's quad, autonomous and drivable (``designs/gorilla_quad_autonomy``, ``designs/gorilla_quad_riders``), in
MuJoCo on the ride course (``gorilla_quad_ride``: bumps, the cone slalom) with a crate dropped on the route and the
table with the banana at the end.

Three scenarios, one movie:

* ``auto`` - nobody aboard: it drives itself (the seat's load cell reads empty, so AUTO); the lidars find the crate,
  the planner swings round it, it stops at the goal;
* ``human`` - a human rider drives (MANUAL) but is distracted and heads straight at the crate: the guardian brakes
  (AEB); the rider then steers round and drives on;
* ``kerchak`` - Kerchak drives the bumps himself, then presses AUTO and rides as a passenger through the slalom and
  round the crate; near the table he takes the bar back (MANUAL) and stops by the banana.

The quad is the Chiron robot here (its free root, four sprung corners, four motors, the steer-by-wire column); the
rider is a free body welded at the pelvis to the air-sprung seat, its joints on PD torques with gravity compensation
(the servos' stall torques), its arms on the seated IK for the bar's angle. The corner and seat air springs are set
for the rider's mass, the column for the rider's reach.

    python gorilla_quad_auto.py                          # all three, the movie
    python gorilla_quad_auto.py --scenario human --no-movie
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

import gorilla_quad as gq  # noqa: E402
import gorilla_quad_autonomy as ga  # noqa: E402
import gorilla_quad_ride as gr  # noqa: E402
import gorilla_quad_riders as qr  # noqa: E402
import motion_kit as mk  # noqa: E402
from vegeta.chiron import Geom, Link, Prop, Robot, SimOptions, Weld  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
CRATE = {"x": 67.0, "y": 0.0, "half": (0.40, 0.45, 0.35), "mass": 40.0}
GOAL_X = 78.0
gr.TABLE = (GOAL_X + 2.0, -2.4)                                          # the table with the banana beside the goal, clear of the lane
SETTLE_S = 0.8

#: the scenarios: the rider, the start, the rider's own intentions (where they steer, how fast, the buttons)
SCENARIOS = {
    "auto": {"rider": "none", "start_x": 0.0, "seconds": 34.0},
    "human": {"rider": "human", "start_x": 57.5, "seconds": 20.0},
    "kerchak": {"rider": "kerchak", "start_x": 0.0, "seconds": 36.0, "auto_at_x": 21.0, "takeover_at_x": 73.0},
}


def crate_prop():
    c = CRATE
    return Prop(Link("crate", pos=(c["x"], c["y"], c["half"][2]),
                     geoms=[Geom("crate_box", "box", c["half"], mass=c["mass"], rgba=(0.62, 0.45, 0.25, 1.0)),
                            Geom("crate_lid", "box", (c["half"][0] + 0.01, c["half"][1] + 0.01, 0.02), pos=(0, 0, c["half"][2] - 0.02),
                                 mass=0.0, role="visual", rgba=(0.50, 0.36, 0.20, 1.0))]), free=True)


def route_y(x):
    return gr.path_y(x)


def route_speed(x):
    """The route's own speed [m/s]: the course's (slow over the bumps, the slalom), then stopping at the goal."""
    v = gr.speed_target(x) if x < 61.0 else 4.5
    return min(v, math.sqrt(max(0.0, 2 * 1.2 * (GOAL_X - x))))


class AutoRide:
    def __init__(self, scenario, scene=None):
        """``scene`` (optional): {"props": [...] instead of the course, "worldbody_xml": extra bodies (mocap, balls),
        "sc": the scenario's dict} for a scenario of another module (``gorilla_quad_dragoon``)."""
        sc = self.sc = SCENARIOS[scenario] if scene is None else scene["sc"]
        self.scenario, self.who = scenario, sc["rider"]
        self.profile = qr.PROFILES[self.who]
        self.bar_offset = (0.0, 0.0)
        # the rider: its kinematics, the seated poses over the bar's range (the arms follow the bar)
        self.rider = None if self.who == "none" else qr.rider_robot(self.who)
        if self.rider is not None:
            self.kin = mk.Kin(self.rider)
            if self.who == "human":
                self.bar_offset, _ = qr.human_bar_offset(self.kin)
            self.steers = np.linspace(-gq.GEOM["steer_max"], gq.GEOM["steer_max"], 13)
            q0, self.ik_err = qr.seated_pose(self.kin, self.who, 0.0, bar_offset=self.bar_offset)
            poses = {6: q0}
            for direction in (1, -1):
                q = q0
                for i in (range(7, 13) if direction > 0 else range(5, -1, -1)):
                    q, e = qr.seated_pose(self.kin, self.who, self.steers[i], q_init=q, bar_offset=self.bar_offset)
                    poses[i], self.ik_err = q, max(self.ik_err, e)
            self.Q = np.array([poses[i] for i in range(13)])
            self.kin.set(q0)
            mujoco.mj_comPos(self.kin.model, self.kin.data)
            rider_cg = self.kin.data.subtree_com[1] + np.asarray(self.profile["pelvis"])
        else:
            rider_cg = (0.0, 0.0, 0.9)
        self.sz = gq.sizing(rider_cg=tuple(rider_cg), rider_mass=self.profile["mass"])
        x0 = sc["start_x"]
        quad = gq.quad_prop(x0=x0, sz=self.sz, bar_offset=self.bar_offset, seat_free=self.rider is not None)
        robot = Robot(name="Kerchak's quad", root=quad.root)
        props = [gr.course_prop(), *gr.cone_props(), crate_prop()] if scene is None else list(scene["props"])
        welds = []
        if self.rider is not None:
            root = qr.passive_copy(self.rider)
            p = self.profile["pelvis"]
            root.pos = (p[0] + x0, p[1], p[2])
            props.insert(0, Prop(root, free=True, priority=1))
            welds.append(Weld("seat", "pelvis", "quad_seat"))
        opts = SimOptions()
        opts.course_extent = (-5.0, 90.0, -8.0, 8.0)
        xml = robot.to_mjcf(options=opts, props=props, welds=welds)
        if scene is not None and scene.get("worldbody_xml"):
            xml = xml.replace("</worldbody>", scene["worldbody_xml"] + "\n  </worldbody>")
        a = gq.GEOM["ackermann"]
        extra = ["  <equality>",
                 f'    <joint name="tie_FL" joint1="steer_FL" joint2="steer_column" polycoef="0 1 {a} 0 0"/>',
                 f'    <joint name="tie_FR" joint1="steer_FR" joint2="steer_column" polycoef="0 1 {-a} 0 0"/>',
                 "  </equality>", "  <actuator>"]
        for c in gq.CORNERS:
            extra.append(f'    <motor name="motor_{c}" joint="spin_{c}" ctrlrange="-{self.sz["wheel_torque_Nm"]} {self.sz["wheel_torque_Nm"]}"/>')
        extra += ['    <position name="column" joint="steer_column" kp="8000" kv="400" ctrlrange="-0.6 0.6"/>', "  </actuator>"]
        xml = xml.replace("</mujoco>", "\n".join(extra) + "\n</mujoco>")
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(m)
        self._collisions()
        # the rider's joints in this model, by name; their servos
        if self.rider is not None:
            jid = [m.joint(n).id for n in self.kin.names]
            self.r_q = np.array([m.jnt_qposadr[j] for j in jid])
            self.r_d = np.array([m.jnt_dofadr[j] for j in jid])
            servos = {j.name: j.servo for j in self.rider.joints() if j.servo is not None}
            self.kp = np.array([servos[n].kp if n in servos else 50.0 for n in self.kin.names])
            self.kd = np.zeros(len(self.kin.names))                       # set at reset: 0.7 of critical on each joint's inertia
            self.lim = np.array([servos[n].stall_torque if n in servos else 50.0 for n in self.kin.names])
            # the servos' rotors seen through their gears (reflected inertia, kg m²): a finger's phalanx alone is
            # 2e-5 kg m², which a stiff servo rings at the step's rate; the gearbox's rotor is what a real one has
            for n, dof in zip(self.kin.names, self.r_d):
                arm = (0.002 if ("finger" in n or "thumb" in n) else 0.02 if "wrist" in n else 0.1 if ("shoulder" in n or "elbow" in n) else 0.3)
                m.dof_armature[dof] = max(m.dof_armature[dof], arm)
        self.motor = [m.actuator(f"motor_{c}").id for c in gq.CORNERS]
        self.spin = [m.jnt_dofadr[m.joint(f"spin_{c}").id] for c in gq.CORNERS]
        self.col = m.actuator("column").id
        self.col_q = m.jnt_qposadr[m.joint("steer_column").id]
        self.col_d = m.jnt_dofadr[m.joint("steer_column").id]
        self.chassis = m.body("quad_chassis").id
        self.free_q = m.jnt_qposadr[m.joint("quad_chassis:free").id]
        self.free_d = m.jnt_dofadr[m.joint("quad_chassis:free").id]
        self.arb = [(m.jnt_dofadr[m.joint(f"susp_{a}L").id], m.jnt_dofadr[m.joint(f"susp_{a}R").id], self.sz["anti_roll_bar_N_m"][axle])
                    for a, axle in (("F", "front"), ("R", "rear"))]
        # the lidars: the rays in the vehicle frame, the quad's and the rider's own geoms out of their sight
        self.rays = {n: ga.lidar_rays(n) for n in ga.LIDARS}
        for g in range(m.ngeom):
            root = m.body(m.body_rootid[m.geom_bodyid[g]]).name
            if root in ("quad_chassis", "pelvis"):
                m.geom_group[g] = 5
        self.geomgroup = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)
        self.scan = {n: (np.zeros(3), np.zeros((0, 3)), np.zeros(0)) for n in ga.LIDARS}
        self.obstacles = np.zeros((0, 3))
        self.perception = ga.Perception()
        self.planner = ga.Planner(route_y, dict(self.profile))
        self.guardian = ga.Guardian()
        self.arbiter = ga.Arbiter(self.who)
        self.plan = None
        self.steer_cmd, self.vt_cmd = 0.0, 0.0
        self.aeb = False
        self.log = []
        self.events = []
        self._t_scan = -1.0
        self._auto_pressed = self._took_over = False
        self.human_phase, self.human_mark = "distracted", None
        self.reset()

    # ---- the model's set-up
    def _collisions(self):
        m = self.model
        TERRAIN, TYRE, COURSE, BODY = 1, 2, 4, 8
        for g in range(m.ngeom):
            name, root = m.geom(g).name, m.body(m.body_rootid[m.geom_bodyid[g]]).name
            if name == "chiron_terrain":
                continue
            solid = m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0
            if name.startswith("tyre_"):
                m.geom_contype[g], m.geom_conaffinity[g] = TYRE, TERRAIN | COURSE
            elif root in ("course", "crate") or root.startswith(("cone_", "target_", "ball_")):
                m.geom_contype[g], m.geom_conaffinity[g] = (COURSE | TYRE, TYRE | BODY | COURSE) if solid else (0, 0)
            elif name.startswith("q_") and name not in ("q_bar", "q_seat"):
                m.geom_contype[g], m.geom_conaffinity[g] = BODY, COURSE
            elif root == "pelvis" and solid:
                m.geom_contype[g], m.geom_conaffinity[g] = BODY, COURSE
            else:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
        for b in range(m.nbody):
            gs_ = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            m.body_contype[b] = int(np.bitwise_or.reduce([int(m.geom_contype[g]) for g in gs_])) if gs_ else 0
            m.body_conaffinity[b] = int(np.bitwise_or.reduce([int(m.geom_conaffinity[g]) for g in gs_])) if gs_ else 0

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.rider is not None:
            d.qpos[self.r_q] = self.Q[6]
        mujoco.mj_forward(m, d)
        if self.rider is not None:
            # the rider's servos damped at 0.7 of critical on each joint's own inertia (the mass matrix's diagonal):
            # a servo's own small damping leaves a 100 kg leg ringing on its stiffness
            M = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, M)
            I = np.maximum(np.diag(M)[self.r_d], 1e-4)
            dt = m.opt.timestep
            self.kp = np.minimum(self.kp, 0.09 * I / dt ** 2)             # stable at the step: sqrt(kp/I) dt <= 0.3
            self.kd = 2 * 0.7 * np.sqrt(self.kp * I)

    # ---- state
    def pose(self):
        d = self.data
        q = d.qpos[self.free_q:self.free_q + 7]
        R = d.xmat[self.chassis].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        v = float(d.qvel[self.free_d] * math.cos(yaw) + d.qvel[self.free_d + 1] * math.sin(yaw))
        return float(q[0]), float(q[1]), yaw, v

    def lidar(self):
        """Both lidars' scans: (origin, hit points, distances) in the world; the obstacle points they see."""
        m, d = self.model, self.data
        R = d.xmat[self.chassis].reshape(3, 3)
        pts = []
        for n, L in ga.LIDARS.items():
            o = d.xpos[self.chassis] + R @ np.asarray(L["pos"])
            vec = (self.rays[n] @ R.T).astype(np.float64)
            nray = len(vec)
            geomid = np.zeros(nray, dtype=np.int32)
            dist = np.zeros(nray, dtype=np.float64)
            mujoco.mj_multiRay(m, d, o.astype(np.float64), vec.ravel(), self.geomgroup, 1, -1, geomid, dist, None, nray, L["range"])
            ok = (dist > 0) & (dist < L["range"])
            hits = o + vec[ok] * dist[ok, None]
            self.scan[n] = (o, o + vec * np.where(dist > 0, np.minimum(dist, L["range"]), L["range"])[:, None], ok)
            pts.append(hits)
        self.obstacles = self.perception.obstacles(np.concatenate(pts))
        return self.obstacles

    # ---- the rider's intentions
    def rider_intent(self, x, y, yaw, v):
        """(steer, speed, wants AUTO, bar torque, throttle) of the rider - a driver model, by scenario."""
        L = gq.GEOM["wheelbase"]
        sc = self.sc
        if self.who == "human":
            # distracted: straight down the lane at 4 m/s at the crate; the guardian stops the quad; the rider looks up,
            # backs off 2.5 m, then swings wide round the crate and drives on
            if self.human_phase == "distracted" and self.aeb and v < 0.05:
                self.human_phase, self.human_mark = "wait", (self.t, x)
            if self.human_phase == "wait" and self.t > self.human_mark[0] + 1.0:
                self.human_phase = "reverse"
            if self.human_phase == "reverse":
                if x > self.human_mark[1] - 2.5:
                    return 0.0, -1.0, False, 0.0, True
                self.human_phase, self.human_mark = "around", (self.t, x)
            if self.human_phase == "around":
                x0 = self.human_mark[1]
                x1 = CRATE["x"] + CRATE["half"][0] + 1.5

                def path(xx):
                    if xx < x0:
                        return 0.0
                    u = min((xx - x0) / 5.0, 1.0) if xx < x1 else max(1.0 - (xx - x1) / 5.0, 0.0)
                    return 2.4 * (0.5 - 0.5 * math.cos(math.pi * u))
                return ga.pure_pursuit(x, y, yaw, v, path, L), min(2.0, route_speed(x)), False, 0.0, True
            return ga.pure_pursuit(x, y, yaw, v, lambda xx: 0.0, L), (4.0 if self.human_phase == "distracted" else 0.0), False, 0.0, True
        # Kerchak: drives the route himself, presses AUTO at auto_at_x, takes the bar back at takeover_at_x
        wants_auto = sc.get("auto_at_x") is not None and x >= sc["auto_at_x"] and not self._took_over
        torque = 0.0
        if self.arbiter.mode == "AUTO" and x >= sc.get("takeover_at_x", 1e9):
            torque = 25.0                                                  # his hands turn the bar against the motor
            self._took_over = True
        return ga.pure_pursuit(x, y, yaw, v, route_y, L), route_speed(x), wants_auto, torque, False

    # ---- one control step
    def control(self, t):
        m, d = self.model, self.data
        self.t = t
        x, y, yaw, v = self.pose()
        dt = m.opt.timestep
        if t - self._t_scan >= 1.0 / ga.LIDARS["front"]["rate_hz"] - 1e-9:
            self.lidar()
            self._t_scan = t
            self.plan = self.planner.update(x, v, self.obstacles, t)
        plan = self.plan
        driving = t >= SETTLE_S
        if self.who == "none":
            r_steer, r_speed, wants_auto, torque, throttle = 0.0, 0.0, False, 0.0, False
        else:
            r_steer, r_speed, wants_auto, torque, throttle = self.rider_intent(x, y, yaw, v)
            if self.arbiter.mode == "AUTO":
                throttle = False
        mode = self.arbiter.update(t, dt, bar_torque=torque, throttle=throttle if self.arbiter.mode == "AUTO" and torque else False,
                                   auto_button=wants_auto and not self._auto_pressed)
        if wants_auto:
            self._auto_pressed = True
        if mode == "AUTO":
            steer = ga.pure_pursuit(x, y, yaw, v, plan.steer_path, gq.GEOM["wheelbase"])
            vt = min(plan.v_max, route_speed(x))
        else:
            steer, vt = r_steer, r_speed
        steer = float(np.clip(steer, -0.55, 0.55)) if driving else 0.0
        vt = vt if driving else 0.0
        # the guardian: automatic emergency braking whoever drives
        self.aeb = self.guardian.check(x, y, yaw, v, float(d.qpos[self.col_q]), self.obstacles, gq.GEOM["wheelbase"])
        rate = ga.STEER_BY_WIRE["rate_rad_s"] * dt
        self.steer_cmd += float(np.clip(steer - self.steer_cmd, -rate, rate))
        d.ctrl[self.col] = self.steer_cmd
        a_up, a_dn = self.profile["a_max"] * dt, (ga.SAFETY["a_aeb"] if self.aeb else 2.5) * dt
        target = 0.0 if self.aeb else vt
        self.vt_cmd += float(np.clip(target - self.vt_cmd, -a_dn, a_up))
        if self.aeb:
            self.vt_cmd = min(self.vt_cmd, max(0.0, v - ga.SAFETY["a_aeb"] * dt))
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
        if self.rider is not None:                                         # the rider's joints: PD on the seated pose for the bar's angle
            q_t = self._arms(float(d.qpos[self.col_q]))
            tau = self.kp * (q_t - d.qpos[self.r_q]) - self.kd * d.qvel[self.r_d] + d.qfrc_bias[self.r_d]
            d.qfrc_applied[self.r_d] += np.clip(tau, -self.lim, self.lim)
        return mode

    def _arms(self, steer):
        s = np.clip(steer, self.steers[0], self.steers[-1])
        h = self.steers[1] - self.steers[0]
        i = min(int((s - self.steers[0]) / h), len(self.steers) - 2)
        f = (s - self.steers[i]) / h
        return (1 - f) * self.Q[i] + f * self.Q[i + 1]

    def step(self, t, dt):
        m = self.model
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            mode = self.control(t + k * m.opt.timestep)
            mujoco.mj_step(m, self.data)
        x, y, yaw, v = self.pose()
        self.log.append((t + dt, x, y, v, mode, self.aeb, self.plan.status if self.plan else "", float(self.plan.v_max if self.plan else 0.0)))

    def report(self):
        L = self.log
        xs = np.array([r[1] for r in L])
        ys = np.array([r[2] for r in L])
        d = self.data
        cr = d.qpos[self.model.jnt_qposadr[self.model.joint("crate:free").id]:][:3]
        gap = np.min(np.hypot(xs - CRATE["x"], ys - CRATE["y"]))
        cones = []
        for k, x in enumerate(gr.CONES):
            a = self.model.jnt_qposadr[self.model.joint(f"cone_{k}:free").id]
            cones.append(round(float(math.hypot(d.qpos[a] - x, d.qpos[a + 1])), 3))
        return {"scenario": self.scenario, "rider": self.profile["label"], "bar_offset_m": self.bar_offset,
                "seconds": round(L[-1][0], 2), "x_end_m": round(float(xs[-1]), 2), "y_end_m": round(float(ys[-1]), 2),
                "crate_moved_m": round(float(math.hypot(cr[0] - CRATE["x"], cr[1] - CRATE["y"])), 3), "cones_moved_m": cones,
                "closest_centre_to_crate_m": round(float(gap), 2), "aeb_s": round(sum(1 for r in L if r[5]) * (L[1][0] - L[0][0]), 2),
                "modes": sorted({r[4] for r in L}), "events": [(round(t, 2), e) for t, e in self.arbiter.events],
                "seated_ik_error_m": round(float(getattr(self, "ik_err", 0.0)), 4),
                "springs": {"corner_k_N_m": {c: self.sz["corners"][c]["k_N_m"] for c in gq.CORNERS}, "seat_k_N_m": self.sz["seat"]["k_N_m"]}}


MODE_RGB = {"AUTO": (0.15, 0.75, 0.30), "MANUAL": (0.15, 0.45, 0.90), "STOP": (0.90, 0.15, 0.10)}


class Drawer:
    """The movie's frames for a ride: the quad's CAD and the autonomy kit (the beacon in the mode's colour), Kerchak's
    CAD or the human's shapes, the lidars' beams and hits, the planned path, the mode and status."""

    def __init__(self, ride, size="1280x720", extra_pieces=()):
        import kerchak_mesh as bm
        import kerchak_render as br
        self.br, self.ride = br, ride
        pieces, struts = gq.quad_mesh()
        kit = ga.kit_mesh()
        self.beacon_i = len(pieces) + list(ga.KIT_COLORS).index("beacon")
        allp, st = list(pieces) + kit + list(extra_pieces), list(struts)
        if ride.who == "kerchak":
            km = bm.load("plastic")
            allp, st = list(km.pieces) + allp, list(km.struts) + st
            self.beacon_i += len(km.pieces)
        self.mesh = bm.Mesh(pieces=allp, struts=st)
        w, h = (int(v) for v in size.split("x"))
        self.renderer = br.Renderer(ride.model, self.mesh, w, h)

    def frame(self, t):
        ride, br = self.ride, self.br
        x, y, yaw, v = ride.pose()
        mode = ride.arbiter.mode
        rgb = (0.95, 0.15, 0.10) if ride.aeb else MODE_RGB[mode]
        b, tris, _ = self.mesh.pieces[self.beacon_i]
        self.mesh.pieces[self.beacon_i] = (b, tris, (*rgb, 1.0))
        lines, dots = [], []
        for n, (o, ends, ok) in ride.scan.items():
            per = len(ends) // len(ga.LIDARS[n]["layers"])
            for i in range(per):                                            # the horizontal layer: the beams that hit, every 6th that does not
                if ok[i]:
                    lines.append((o, ends[i], (1.0, 0.50, 0.50)))
                elif i % 6 == 0:
                    lines.append((o, o + 0.35 * (ends[i] - o), (0.93, 0.80, 0.80)))
        for p in ride.obstacles:
            dots.append((p, 3, (0.95, 0.05, 0.05)))
        if ride.plan is not None and mode == "AUTO":                       # the planned path, green, 15 m ahead
            xs = np.linspace(x, x + 15.0, 31)
            pts = [(xx, ride.plan.steer_path(xx), 0.03) for xx in xs]
            lines += [(pts[i], pts[i + 1], (0.1, 0.8, 0.2)) for i in range(len(pts) - 1)]
        c = np.array(ride.data.xpos[ride.chassis])
        az = math.degrees(yaw) + 215.0 + 25.0 * math.sin(0.08 * t)
        cam = br.orbit((c[0] + 1.2, c[1], 1.0), 22.0, az, 8.0)
        status = ("EMERGENCY BRAKE - obstacle ahead" if ride.aeb else ride.plan.status if mode == "AUTO" and ride.plan else "the rider drives")
        hud = [(f"mode {mode}   rider: {ride.profile['label']}   {3.6 * v:4.1f} km/h", (0.9, 0.1, 0.1) if ride.aeb else MODE_RGB[mode]),
               (f"autonomy: {status}", (0.2, 0.2, 0.25))]
        ev = [e for e in ride.arbiter.events if t - e[0] < 3.0]
        if ev:
            hud.append((ev[-1][1], (0.85, 0.45, 0.05)))
        title = f"Kerchak's quad - autonomous, drivable   [{ride.scenario}]   t = {t:4.1f} s"
        return self.renderer.render(ride.data, cam, title=title, lines=lines, dots=dots, hud=hud)


def run(scenario, movie=True, fps=15, size="1280x720", frames_dir=None, stills=()):
    ride = AutoRide(scenario)
    drawer = Drawer(ride, size) if movie else None
    dt = 1.0 / fps
    t, k = 0.0, 0
    while t <= ride.sc["seconds"] + 1e-9:
        ride.step(t, dt)
        t += dt
        if drawer is not None:
            img = drawer.frame(t)
            img.save(os.path.join(frames_dir, f"{scenario}_{k:04d}.png"))
            for ts in stills:
                if abs(t - ts) < dt / 2:
                    img.save(str(OUT / f"gorilla_quad_auto_{scenario}_{ts:g}s.png"))
        k += 1
        x, y, yaw, v = ride.pose()
        if x > GOAL_X - 0.6 and abs(v) < 0.03 and t > 3.0:
            break
    return ride, k


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kerchak's quad: autonomous, drivable by a human or Kerchak")
    ap.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--stills", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    tmp = tempfile.mkdtemp()
    stills = [float(s) for s in a.stills.split(",") if s.strip()]
    reports, lists = {}, []
    for n in names:
        ride, frames = run(n, movie=not a.no_movie, fps=a.fps, size=a.size, frames_dir=tmp, stills=stills)
        reports[n] = ride.report()
        print(json.dumps(reports[n]), flush=True)
        if not a.no_movie:
            lists.append((n, frames))
    tag = "gorilla_quad_auto" + ("" if a.scenario == "all" else f"_{a.scenario}")
    (OUT / f"{tag}.json").write_text(json.dumps(reports, indent=1))
    if not a.no_movie:
        parts = []
        for n, frames in lists:
            part = os.path.join(tmp, f"{n}.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, f"{n}_%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", part], check=True)
            parts.append(part)
        lst = os.path.join(tmp, "list.txt")
        Path(lst).write_text("".join(f"file '{p}'\n" for p in parts))
        out = str(OUT / f"{tag}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
        print("movie", out)
    return reports


if __name__ == "__main__":
    main()
