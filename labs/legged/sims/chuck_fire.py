#!/usr/bin/env python3
"""Chuck puts out a car fire: Kerchak the firefighter rides Chuck (``designs/chuck``), the quad driving itself to a
car the bandits set alight six minutes ago, and fights it with the hose.

1. **ride out** - AUTO to 10 m short of the burning car;
2. **the hose** - his right fist takes the hose's branch from the rack beside his hip (its hose runs back to the reel
   on the tank), his left the fore handle; the brace to his shoulder (the jet pushes back with ~110 N);
3. **attack** - the jet on the car: its drops fly as the design's jet (``chuck.throw``'s slowing length), aimed by the
   same closed loop as the bazooka (the tube's real direction read, the aim corrected), swept across the car; water
   that lands on the car cools the fire at ``chuck.COOLING_J_KG`` x ``chuck.EFFICIENCY``;
4. **the fire** - the game's model (``chuck.FIRES['car']``): t-squared growth; an intensity ``k`` that the water's
   cooling beats down (``dk/dt = -KNOCK * cooling / HRR(t)``) and that grows back when the water stops
   (``REGROW``); out when ``k`` < 2 % - then he shuts the branch, stows it, and the quad backs off.

The water and Chuck's gear (311 kg) ride on the quad as a body welded to it, the air springs and the motors set for
the load (``gorilla_quad.sizing``). The branch follows his right fist (a mocap body), the flames, smoke and steam are
drawn (dots), not simulated.

    python chuck_fire.py               # sims/output/chuck_fire.mp4
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

import chuck  # noqa: E402
import gorilla_bazooka as gb  # noqa: E402
import gorilla_quad as gq  # noqa: E402
import gorilla_quad_auto as qa  # noqa: E402
import gorilla_quad_autonomy as ga  # noqa: E402
import gorilla_quad_dragoon as D  # noqa: E402
import gorilla_scenarios as gs  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_unit_robots as sr  # noqa: E402
from vegeta.chiron import Geom, Link, Prop, Weld  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
CAR = {"x": 24.0, "y": 0.0, "half": (2.1, 0.9, 0.7), "burning_s": 360.0}
STOP_X = CAR["x"] - CAR["half"][0] - 10.0
KNOCK = 0.02          #: how fast the water beats the fire down (per s, at cooling = heat release): ~40 s and ~80 L for a 1.7 MW car fire
REGROW = 0.03         #: how fast it comes back without water (per s)
EMIT_S = 0.005        #: a drop every 5 ms (each carries the flow of that time)
SPREAD_RAD = math.radians(1.2)
NOZZLE_MUZZLE = gb.MUZZLE          #: the branch uses the bazooka's frame: the rear and fore D-handles, the brace, the tip


def scene():
    c = CAR
    burnt, glass, tyre = (0.30, 0.10, 0.08, 1.0), (0.12, 0.12, 0.14, 1.0), (0.05, 0.05, 0.05, 1.0)
    hx, hy, hz = c["half"]
    geoms = [Geom("car_body", "box", (hx, hy, hz * 0.45), pos=(0, 0, 0.30 + hz * 0.45), mass=1200.0, rgba=burnt),
             Geom("car_cabin", "box", (hx * 0.55, hy * 0.92, hz * 0.32), pos=(-0.15, 0, 0.30 + hz * 0.9 + hz * 0.32), mass=50.0, rgba=glass)]
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        geoms.append(Geom(f"car_wheel_{k}", "cylinder", (0.32, 0.11), pos=(sx * hx * 0.7, sy * (hy - 0.05), 0.32),
                          quat=(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0, 0), mass=10.0, rgba=tyre))
    props = [Prop(Link("course", pos=(c["x"], c["y"], 0.0), geoms=geoms), free=False)]
    load_kg = chuck.sizing()["load_kg"]
    T = chuck.TANK
    props.append(Prop(Link("chuck_load", pos=(T["pos"][0], 0.0, T["pos"][2] + T["size_m"][2] * 0.45),
                           geoms=[Geom("chuck_load_g", "box", (0.25, 0.3, 0.15), mass=load_kg, role="visual", rgba=(0, 0, 0, 0))]), free=True))
    nozzle = '''    <body name="bazooka" mocap="true" pos="0 0 -5">
      <geom name="bazooka_ref" type="sphere" size="0.01" mass="0" contype="0" conaffinity="0" rgba="0 0 0 0"/>
    </body>'''
    return {"props": props, "worldbody_xml": nozzle, "welds": [Weld("load", "chuck_load", "quad_chassis")],
            "load": (load_kg, (T["pos"][0], 0.0, chuck.sizing()["load_cg_z_m"])),
            "sc": {"rider": "kerchak", "start_x": 0.0, "seconds": 70.0}}


class FireRide(D.DragoonRide):
    def __init__(self):
        qa.AutoRide.__init__(self, "chuck", scene=scene())
        m = self.model
        self.out_path = np.array([(x, 0.0) for x in np.arange(0.0, STOP_X + 1e-9, 0.25)])
        self.back_path = np.array([(x, 0.0) for x in np.arange(STOP_X, STOP_X - 6.0, -0.25)])
        self.path, self.path_i, self.leg = self.out_path, 0, "out"
        self.mocap = m.body_mocapid[m.body("bazooka").id]
        self.r_hand, self.pelvis = m.body("R_hand").id, m.body("pelvis").id
        self.held, self.phase, self.phase_t0 = False, "ride out", 0.0
        self.wheelbase = gq.GEOM["wheelbase"]
        self.rng = np.random.default_rng(5)
        self.arbiter.mode = "AUTO"
        self.arbiter.events.append((0.0, "AUTO: Chuck carries him to the fire"))
        for g in range(m.ngeom):                                           # the lidars see neither the branch nor the load
            if m.geom_bodyid[g] in (m.body("bazooka").id, m.body("chuck_load").id):
                m.geom_group[g] = 5
        self.k, self.t_fire0 = 1.0, -CAR["burning_s"]
        self.drops, self.water_on_car_kg, self.water_used_l = [], 0.0, 0.0
        self.spraying, self._emit_t, self.cooling_kW = False, 0.0, 0.0
        self._cool_window = []
        self.out_at = None
        self.steam = []
        self.log_fire = []

    # ---- the fire
    def hrr(self, t):
        return chuck.fire_hrr("car", t - self.t_fire0) * self.k

    def _fire_step(self, t, dt):
        h0 = chuck.fire_hrr("car", t - self.t_fire0)
        cutoff = t - 0.5
        self._cool_window = [(tt, kg) for tt, kg in self._cool_window if tt > cutoff]
        self.cooling_kW = sum(kg for _, kg in self._cool_window) / 0.5 * chuck.COOLING_J_KG * chuck.EFFICIENCY / 1e3
        if self.k > 0.0:
            if self.cooling_kW > 0.0:
                self.k = max(0.0, self.k - KNOCK * self.cooling_kW / max(h0, 1.0) * dt)
            else:
                self.k = min(1.0, self.k + REGROW * (1.0 - self.k) * dt)
        if self.out_at is None and self.k < 0.02:
            self.out_at, self.k = t, 0.0
            self.arbiter.events.append((t, "the fire is out"))

    # ---- the jet
    def _jet_aim(self, muzzle):
        """(azimuth, elevation) that puts the jet on the car's middle, top of the bonnet (the jet model, bisection)."""
        target = np.array([CAR["x"] - 0.4, CAR["y"], 0.30 + CAR["half"][2] * 0.9])
        dv = target - muzzle
        az = math.atan2(dv[1], dv[0])
        dx = math.hypot(dv[0], dv[1])
        d_mm, v0, _ = chuck.nozzle()
        L = chuck.JET_SLOWING_D * d_mm / 1000.0
        lo, hi = math.radians(-10), math.radians(40)
        for _ in range(30):
            el = 0.5 * (lo + hi)
            p, vel = np.array([0.0, 0.0]), v0 * np.array([math.cos(el), math.sin(el)])
            while p[0] < dx and p[1] > -5:
                s = np.linalg.norm(vel)
                vel = vel + (-s * vel / L - np.array([0, 9.81])) * 2e-3
                p = p + vel * 2e-3
            lo, hi = (el, hi) if p[1] < dv[2] else (lo, el)
        return az, 0.5 * (lo + hi)

    def _emit(self, t, W_rot, W_pos):
        d_mm, v0, _ = chuck.nozzle()
        dirn = W_rot[:, 0] + SPREAD_RAD * (self.rng.standard_normal() * W_rot[:, 1] + self.rng.standard_normal() * W_rot[:, 2])
        dirn /= np.linalg.norm(dirn)
        kg = chuck.PUMP["flow_l_min"] / 60.0 * EMIT_S
        self.drops.append({"p": W_pos + W_rot @ NOZZLE_MUZZLE, "v": v0 * dirn, "kg": kg, "alive": True})
        self.water_used_l += kg

    def _fly_drops(self, t, dt):
        d_mm, _, _ = chuck.nozzle()
        L = chuck.JET_SLOWING_D * d_mm / 1000.0
        c = CAR
        for dr in self.drops:
            if not dr["alive"]:
                continue
            s = np.linalg.norm(dr["v"])
            dr["v"] = dr["v"] + (-s * dr["v"] / L + np.array([0, 0, -9.81])) * dt
            dr["p"] = dr["p"] + dr["v"] * dt
            rel = dr["p"] - np.array([c["x"], c["y"], 0.0])
            on_car = abs(rel[0]) < c["half"][0] + 0.2 and abs(rel[1]) < c["half"][1] + 0.2 and rel[2] < 0.30 + 2 * c["half"][2] * 0.9
            if on_car or dr["p"][2] < 0.0:
                dr["alive"] = False
                if on_car and self.k > 0.0:
                    self._cool_window.append((t, dr["kg"]))
                    if self.rng.random() < 0.15:
                        self.steam.append([dr["p"].copy(), t])
        self.drops = [dr for dr in self.drops if dr["alive"]] + [dr for dr in self.drops if not dr["alive"]][-200:]

    # ---- the stop's moves (the bazooka's draw, with the branch)
    def plan_stop(self):
        D.DragoonRide.plan_stop(self)
        self.keys = [(n.replace("bazooka", "the hose's branch"), q, dt) for n, q, dt in self.keys]
        self.keys[-1] = ("aim the jet", self.keys[-1][1], self.keys[-1][2])

    def _aim_pose(self, kin_q, muzzle=None, at=None, flat=False):
        """The branch's pose in the pelvis frame: the brace at his shoulder, the jet's arc on the car."""
        kin = self.kin
        kin.set(kin_q)
        sh = kin.world(kin.body("R_upper_arm"), np.zeros(3))
        d = self.data
        Rp, pp = d.xmat[self.pelvis].reshape(3, 3), d.xpos[self.pelvis]
        pad = sh + D.PAD_AT
        W_rot = np.eye(3)
        for _ in range(3):
            mz = pp + Rp @ (pad + W_rot @ (NOZZLE_MUZZLE - gb.PAD))
            az, el = self._jet_aim(mz)
            Rw = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]]) @ np.array(
                [[math.cos(el), 0, -math.sin(el)], [0, 1, 0], [math.sin(el), 0, math.cos(el)]])
            W_rot = Rp.T @ Rw
        return W_rot, pad - W_rot @ gb.PAD, Rw[:, 0]

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
            self._seq_i += 1
            self._seq_t0, self._seq_from = t, q_to
            if self._seq_i >= len(self._seq):
                if self.phase == "draw":
                    self.phase, self.phase_t0, self.q_fire = "attack", t, self.q_aim.copy()
                    self._sweep0 = self.W_aim
                    self.arbiter.events.append((t, "the branch open: water on the car"))
                else:
                    self.phase = "back off"
                    self.path, self.path_i, self.leg = self.back_path, 0, "back"
                    self.arbiter.events.append((t, "the branch stowed: Chuck backs off"))
        return q

    def _attack(self, t):
        """The jet on the car, swept along it (the aim's target point moving), the aim kept by the closed loop."""
        tt = t - self.phase_t0
        if self.out_at is not None:
            self.spraying = False
            self.move = "the fire is out: shutting the branch"
            if t > self.out_at + 2.0:
                self.phase = "stow"
                self._seq, self._seq_i, self._seq_t0, self._seq_from = self.stow, 0, t, self.q_fire
                self.stow = [(n.replace("bazooka", "branch"), q, dt) for n, q, dt in self.stow]
            return self.q_fire
        self.spraying = tt > 0.4
        self.move = f"water on the car: {self.water_used_l:.0f} L, the fire at {self.k * 100:.0f} %"
        W_rot, W_pos = self._weapon_world()
        sweep = 0.6 * math.sin(0.8 * tt)                                    # across the car's length, a metre each way
        mz = W_pos + W_rot @ NOZZLE_MUZZLE
        tgt = np.array([CAR["x"] - 0.4 + 1.4 * math.sin(0.5 * tt), CAR["y"] + sweep, 0.30 + CAR["half"][2] * 0.9])
        dv = tgt - mz
        az = math.atan2(dv[1], dv[0])
        _, el = self._jet_aim(mz)
        el += math.atan2(-0.4 + 1.4 * math.sin(0.5 * tt) - (-0.4), math.hypot(dv[0], dv[1])) * 0.0
        self.aim_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        actual = W_rot[:, 0]
        err = math.acos(max(-1.0, min(1.0, float(actual @ self.aim_dir))))
        if err > math.radians(0.8) and (not getattr(self, "_corrected", False) or t > self._corrected + 0.3):
            self._correct(actual, err)
            self._corrected = t
        return self.q_fire

    def control(self, t):
        m, d = self.model, self.data
        self.t = t
        dt = m.opt.timestep
        x, y, yaw, v = self.pose()
        if t - self._t_scan >= 0.1 - 1e-9:
            self.lidar()
            self._t_scan = t
        steer, vt, arms = 0.0, 0.0, None
        if self.phase in ("ride out", "back off") and t >= qa.SETTLE_S:
            steer, vt, _ = self.follow(x, y, yaw, v)
            end = self.path[-1, 0]
            remaining = (end - x) if self.phase == "ride out" else (x - end)   # signed along the lane: past the end it is negative
            vt = min(vt, math.sqrt(max(0.0, 2 * 1.0 * (remaining - 0.15))))
            if self.phase == "back off":
                steer, vt = 0.0, -min(1.0, math.sqrt(max(0.0, 2 * 0.8 * (remaining - 0.1))))
            if remaining < 0.2 and abs(v) < 0.05:
                if self.phase == "ride out":
                    self.phase, self.phase_t0 = "stopped", t
                    self.arbiter.events.append((t, "at the fire: he takes the hose"))
                else:
                    self.phase = "done"
        elif self.phase == "stopped" and t > self.phase_t0 + 0.4:
            self.plan_stop()
            self._seq, self._seq_i, self._seq_t0, self._seq_from = self.keys, 0, t, self._current_q()
            self.phase = "draw"
        if self.phase in ("draw", "stow"):
            arms = self._play(t)
        elif self.phase == "attack":
            arms = self._attack(t)
        rate = ga.STEER_BY_WIRE["rate_rad_s"] * dt
        self.steer_cmd += float(np.clip(steer - self.steer_cmd, -rate, rate))
        d.ctrl[self.col] = self.steer_cmd
        self.vt_cmd += float(np.clip(vt - self.vt_cmd, -2.0 * dt, 1.0 * dt))
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
        # the branch: in the rack, or in his fist
        if self.held:
            W_rot, W_pos = self._weapon_world()
        else:
            Rc = d.xmat[self.chassis].reshape(3, 3)
            Ws_rot, Ws_pos = gb.scabbard_pose()
            W_rot, W_pos = Rc @ Ws_rot, d.xpos[self.chassis] + Rc @ Ws_pos
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, W_rot.ravel())
        d.mocap_pos[self.mocap], d.mocap_quat[self.mocap] = W_pos, q
        if self.spraying and t - self._emit_t >= EMIT_S - 1e-9:
            self._emit(t, W_rot, W_pos)
            self._emit_t = t
        if int(round(t / dt)) % 2 == 0:
            self._fly_drops(t, 2 * dt)
        self._fire_step(t, dt)
        return self.arbiter.mode

    def step(self, t, dt):
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for k in range(n):
            self.control(t + k * self.model.opt.timestep)
            mujoco.mj_step(self.model, self.data)
        x, y, yaw, v = self.pose()
        self.log.append((t + dt, x, y, v, self.arbiter.mode, False, self.phase, 0.0))
        self.log_fire.append((round(t + dt, 2), round(self.hrr(t + dt)), round(self.cooling_kW), round(self.k, 3)))

    def report(self):
        x, y, yaw, v = self.pose()
        hrr_arrival = next((h for tt, h, c, k in self.log_fire if c > 0), None)
        return {"fire_kW_when_the_water_reached_it": hrr_arrival, "out_at_s": self.out_at,
                "spraying_s": round(self.out_at - self.phase_t0, 1) if self.out_at else None, "water_used_l": round(self.water_used_l, 1),
                "tank_l": chuck.TANK["volume_l"], "ik_errors_m": getattr(self, "ik_errors", None), "end": (round(x, 2), round(y, 2)),
                "phase": self.phase, "events": [(round(t, 2), e) for t, e in self.arbiter.events]}


def branch_mesh():
    """The hose's branch on the bazooka's frame: a pipe from the brace to the tip, the shut-off bale, the two D-handles."""
    import cadquery as cq
    mm = 1000.0
    tz = gb.TUBE_Z * mm
    pipe = cq.Workplane("YZ").circle(22).extrude(gb.MUZZLE[0] * mm + 150).translate((-150, 0, tz))
    tip = cq.Workplane("YZ").circle(34).workplane(offset=120).circle(14).loft().translate((gb.MUZZLE[0] * mm - 120, 0, tz))
    bale = cq.Workplane("XY").box(30, 120, 20).translate((80, 0, tz + 40))
    brace = cq.Workplane("XY").box(-gb.PAD[0] * mm - 150, 40, 40).translate(((gb.PAD[0] * mm - 150) / 2, 0, tz - 20))
    pad = cq.Workplane("XY").box(40, 110, 160).translate((gb.PAD[0] * mm, 0, gb.PAD[2] * mm)).edges("|X").fillet(30)
    parts = [pipe, tip, bale, brace, pad]
    for hx in (gb.GRIP[0] * mm, gb.FORE[0] * mm):
        parts.append(cq.Workplane("XZ").circle(25).extrude(70, both=True).translate((hx, 0, 0)))
        for sy in (-1, 1):
            parts.append(cq.Workplane("XY").box(30, 14, tz - 20).translate((hx, sy * 77, (tz - 20) / 2 + 10)))
    shape = cq.Compound.makeCompound([w.val() for w in parts])
    return [("bazooka", gq._tess(shape), (0.75, 0.75, 0.78, 1.0))]


class FireDrawer(qa.Drawer):
    def __init__(self, ride, size="1280x720"):
        dz = chuck.ChuckDesign()
        P = dz.resolve()
        helmet = [("thorax", gq._tess(dz.helmet(P)) + np.array([0.0, 0.0, 0.573]), chuck.HELMET_RGBA)]
        super().__init__(ride, size, extra_pieces=chuck.chuck_mesh() + helmet + branch_mesh())
        drop = {gq.COLORS["trunk"], gq.COLORS["tools"]}
        self.mesh.pieces[:] = [p for p in self.mesh.pieces if not (p[0] == "quad_chassis" and p[2] in drop)]
        self.beacon_i = next(i for i, p in enumerate(self.mesh.pieces) if p[2] == ga.KIT_COLORS["beacon"])
        self.cam = None
        self.rng = np.random.default_rng(2)

    def frame(self, t):
        ride, br = self.ride, self.br
        x, y, yaw, v = ride.pose()
        b, tris, _ = self.mesh.pieces[self.beacon_i]
        self.mesh.pieces[self.beacon_i] = (b, tris, ((0.95, 0.2, 0.1, 1.0) if int(t * 4) % 2 else (0.2, 0.3, 0.95, 1.0)))   # blue and red
        c = np.array(ride.data.xpos[ride.chassis])
        car = np.array([CAR["x"], CAR["y"], 0.8])
        ph = ride.phase
        if ph in ("ride out", "done", "back off"):
            want = (0.5 * (c + car) * np.array([1, 1, 0]) + np.array([0, 0, 1.0]), 18.0, math.degrees(yaw) + 235.0, 13.0)
        elif ph in ("stopped", "draw", "stow"):
            want = (c + np.array([0.3, -0.2, 0.3]), 12.0, math.degrees(yaw) - 70.0, 4.8)
        else:
            want = (0.65 * c + 0.35 * car + np.array([0, 0, 0.6]), 10.0, math.degrees(yaw) - 120.0, 9.5)
        if self.cam is None:
            self.cam = [want[0].copy(), *want[1:]]
        kf = 0.1
        self.cam[0] = self.cam[0] + kf * (want[0] - self.cam[0])
        self.cam[1] += kf * (want[1] - self.cam[1])
        self.cam[2] += kf * ((want[2] - self.cam[2] + 180.0) % 360.0 - 180.0)
        self.cam[3] += kf * (want[3] - self.cam[3])
        cam = br.orbit(tuple(self.cam[0]), self.cam[1], self.cam[2], self.cam[3])
        lines, dots = [], []
        # the flames and the smoke over the car, by the fire's heat release
        h = ride.hrr(t)
        n = int(90 * min(1.0, h / 2000.0))
        hx, hy, hz = CAR["half"]
        flame_h = 0.6 + 2.2 * min(1.0, h / 2500.0)
        for _ in range(n):
            p = np.array([CAR["x"] + self.rng.uniform(-hx, hx) * 0.9, CAR["y"] + self.rng.uniform(-hy, hy) * 0.9, 0.30 + 2 * hz * 0.75])
            u = self.rng.random()
            p[2] += u * flame_h
            col = (1.0, 0.85, 0.2) if u < 0.35 else (1.0, 0.5, 0.05) if u < 0.75 else (0.85, 0.15, 0.05)
            dots.append((p, int(7 - 4 * u), col))
        for _ in range(int(40 * min(1.0, h / 1500.0)) + (15 if ride.out_at and t - ride.out_at < 6 else 0)):
            p = np.array([CAR["x"] + self.rng.normal(0, 0.8), CAR["y"] + self.rng.normal(0, 0.6), 0.30 + 2 * hz + flame_h + self.rng.uniform(0, 3.5)])
            g = 0.25 + 0.15 * self.rng.random() if h > 50 else 0.75
            dots.append((p, int(6 + 6 * self.rng.random()), (g, g, g + 0.02)))
        for s, ts in ride.steam:                                               # the steam where water meets fire
            if t - ts < 1.2:
                dots.append((s + np.array([0, 0, 1.2 * (t - ts)]), 5, (0.95, 0.95, 0.97)))
        for dr in ride.drops:                                                  # the jet
            if dr["alive"]:
                dots.append((dr["p"], 2, (0.25, 0.5, 0.95)))
        # the hose: from the reel to the branch's brace, sagging
        Rc = ride.data.xmat[ride.chassis].reshape(3, 3)
        reel = c + Rc @ np.array([chuck.TANK["pos"][0] + 0.12, -0.36, chuck.TANK["pos"][2] + chuck.TANK["size_m"][2] + 0.30])
        W = ride.data.mocap_pos[ride.mocap]
        Wq = ride.data.mocap_quat[ride.mocap]
        Rw = np.zeros(9)
        mujoco.mju_quat2Mat(Rw, Wq)
        tail = W + Rw.reshape(3, 3) @ np.array([-0.15, 0.0, gb.TUBE_Z])
        pts = [reel + (tail - reel) * s - np.array([0, 0, 0.35 * math.sin(math.pi * s)]) for s in np.linspace(0, 1, 12)]
        lines += [(pts[i], pts[i + 1], (0.75, 0.12, 0.10)) for i in range(11)]
        what = getattr(ride, "move", "") if ph in ("draw", "attack", "stow") else {"ride out": "Chuck drives him to the fire (AUTO)",
                                                                                 "stopped": "at the fire", "back off": "Chuck backs off",
                                                                                 "done": "done"}.get(ph, ph)
        hud = [(f"Kerchak the firefighter - {what}", (0.15, 0.15, 0.2)),
               (f"the car fire: {h / 1000:4.2f} MW   water's cooling {ride.cooling_kW / 1000:4.2f} MW   water used {ride.water_used_l:4.0f} / {chuck.TANK['volume_l']:.0f} L"
                + ("   OUT" if ride.out_at else ""), (0.75, 0.2, 0.1) if not ride.out_at else (0.1, 0.55, 0.2))]
        return self.renderer.render(ride.data, cam, title=f"Chuck - a car fire   t = {t:4.1f} s", lines=lines, dots=dots, hud=hud)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chuck and Kerchak put out a car fire")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stills", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ride = FireRide()
    drawer = None if a.no_movie else FireDrawer(ride)
    stills = [float(s) for s in a.stills.split(",") if s.strip()]
    tmp = tempfile.mkdtemp()
    dt = 1.0 / a.fps
    t, k = 0.0, 0
    while t <= ride.sc["seconds"] and ride.phase != "done":
        ride.step(t, dt)
        t += dt
        if drawer is not None:
            img = drawer.frame(t)
            img.save(os.path.join(tmp, f"f{k:04d}.png"))
            for ts in stills:
                if abs(t - ts) < dt / 2:
                    img.save(str(OUT / f"chuck_fire_{ts:g}s.png"))
        k += 1
    rep = ride.report()
    (OUT / "chuck_fire.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if drawer is not None:
        out = str(OUT / "chuck_fire.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        print("movie", out)
    return rep


if __name__ == "__main__":
    main()
