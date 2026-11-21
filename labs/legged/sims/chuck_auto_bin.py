#!/usr/bin/env python3
"""Chuck alone, fully autonomous, against a bin fire: nobody aboard (the seat's load cell reads empty: AUTO), the quad
drives to a burning wheelie bin, stops 12 m short, turns its water cannon on it (the monitor's pan and tilt slewing
at ``chuck.MONITOR['slew_deg_s']``, aimed by the jet model and corrected from where the jet lands), puts it out and
backs off.

The fire is the game's bin fire (``chuck.FIRES['bin']``, set 2 minutes before: at its 0.3 MW peak); the water knocks it
down as in ``chuck_fire`` (the same model). The water and the water system (311 kg) ride on the quad welded to it.

    python chuck_auto_bin.py             # sims/output/chuck_auto_bin.mp4
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

import cadquery as cq  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import chuck  # noqa: E402
import chuck_fire as F  # noqa: E402
import gorilla_quad as gq  # noqa: E402
import gorilla_quad_auto as qa  # noqa: E402
import gorilla_quad_autonomy as ga  # noqa: E402
from vegeta.chiron import Geom, Link, Prop, Weld  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
BIN = {"x": 20.0, "y": 0.8, "half": (0.29, 0.37, 0.54), "burning_s": 120.0}
STOP_X = BIN["x"] - 12.0
BARREL = 0.42                     #: the cannon's barrel, pivot to tip [m]
PIVOT = np.array(chuck.MONITOR["pos"]) + np.array([0.0, 0.0, 0.075])


def scene():
    b = BIN
    hx, hy, hz = b["half"]
    green, dark = (0.15, 0.40, 0.18, 1.0), (0.08, 0.10, 0.08, 1.0)
    geoms = [Geom("bin_body", "box", (hx, hy, hz), pos=(0, 0, hz + 0.05), mass=15.0, rgba=green),
             Geom("bin_lid", "box", (hx + 0.03, hy + 0.02, 0.02), pos=(-0.05, 0, 2 * hz + 0.08), mass=1.0, rgba=dark, quat=(math.cos(-0.35), 0, math.sin(-0.35), 0))]
    for sy in (-1, 1):
        geoms.append(Geom(f"bin_wheel_{sy}", "cylinder", (0.1, 0.03), pos=(-hx, sy * (hy - 0.05), 0.1),
                          quat=(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0, 0), mass=0.5, rgba=dark))
    props = [Prop(Link("course", pos=(b["x"], b["y"], 0.0), geoms=geoms), free=False)]
    load = chuck.sizing()["load_kg"] - chuck.sizing()["masses_kg"]["tools (axe, halligan, extinguisher) + rack"]
    T = chuck.TANK
    props.append(Prop(Link("chuck_load", pos=(T["pos"][0], 0.0, T["pos"][2] + T["size_m"][2] * 0.45),
                           geoms=[Geom("chuck_load_g", "box", (0.25, 0.3, 0.15), mass=load, role="visual", rgba=(0, 0, 0, 0))]), free=True))
    monitor = '''    <body name="monitor" mocap="true" pos="0 0 -5">
      <geom name="monitor_ref" type="sphere" size="0.01" mass="0" contype="0" conaffinity="0" rgba="0 0 0 0"/>
    </body>'''
    return {"props": props, "worldbody_xml": monitor, "welds": [Weld("load", "chuck_load", "quad_chassis")],
            "load": (load, (T["pos"][0], 0.0, chuck.sizing()["load_cg_z_m"])), "sc": {"rider": "none", "start_x": 0.0, "seconds": 50.0}}


class AutoBin(F.FireRide):
    def __init__(self):
        qa.AutoRide.__init__(self, "chuck_auto", scene=scene())
        m = self.model
        self.out_path = np.array([(x, 0.0) for x in np.arange(0.0, STOP_X + 1e-9, 0.25)])
        self.back_path = np.array([(x, 0.0) for x in np.arange(STOP_X, STOP_X - 5.0, -0.25)])
        self.path, self.path_i, self.leg = self.out_path, 0, "out"
        self.mocap = m.body_mocapid[m.body("monitor").id]
        self.phase, self.phase_t0 = "ride out", 0.0
        self.wheelbase = gq.GEOM["wheelbase"]
        self.rng = np.random.default_rng(5)
        self.arbiter.events.append((0.0, "AUTO, nobody aboard: Chuck drives to the fire"))
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in (m.body("monitor").id, m.body("chuck_load").id):
                m.geom_group[g] = 5
        self.k, self.t_fire0 = 1.0, -BIN["burning_s"]
        self.drops, self.water_used_l = [], 0.0
        self.spraying, self._emit_t, self.cooling_kW = False, 0.0, 0.0
        self._cool_window, self.out_at, self.steam, self.log_fire = [], None, [], []
        self.pan, self.tilt = 0.0, 0.0                                     # the monitor's joints [rad], stowed level and ahead
        self.aim_corr = np.zeros(2)                                        # the correction learnt from where the jet lands
        self.landed = []
        self.held = False
        self.move = ""

    # the fire's target and kind
    def hrr(self, t):
        return chuck.fire_hrr("bin", t - self.t_fire0) * self.k

    def _fire_step(self, t, dt):
        h0 = chuck.fire_hrr("bin", t - self.t_fire0)
        self._cool_window = [(tt, kg) for tt, kg in self._cool_window if tt > t - 0.5]
        self.cooling_kW = sum(kg for _, kg in self._cool_window) / 0.5 * chuck.COOLING_J_KG * chuck.EFFICIENCY / 1e3
        if self.k > 0.0:
            if self.cooling_kW > 0.0:
                self.k = max(0.0, self.k - F.KNOCK * self.cooling_kW / max(h0, 1.0) * dt)
            else:
                self.k = min(1.0, self.k + F.REGROW * (1.0 - self.k) * dt)
        if self.out_at is None and self.k < 0.02:
            self.out_at, self.k = t, 0.0
            self.arbiter.events.append((t, "the fire is out"))

    def _target(self):
        return np.array([BIN["x"], BIN["y"], 0.05 + 2 * BIN["half"][2] * 0.9])

    def _jet_aim_to(self, muzzle, target):
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

    def _monitor_pose(self):
        d = self.data
        Rc = d.xmat[self.chassis].reshape(3, 3)
        cp, sp, ct, st = math.cos(self.pan), math.sin(self.pan), math.cos(self.tilt), math.sin(self.tilt)
        Rl = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]]) @ np.array([[ct, 0, -st], [0, 1, 0], [st, 0, ct]])
        return Rc @ Rl, d.xpos[self.chassis] + Rc @ PIVOT

    def _fly_drops(self, t, dt):
        d_mm, _, _ = chuck.nozzle()
        L = chuck.JET_SLOWING_D * d_mm / 1000.0
        b = BIN
        for dr in self.drops:
            if not dr["alive"]:
                continue
            s = np.linalg.norm(dr["v"])
            dr["v"] = dr["v"] + (-s * dr["v"] / L + np.array([0, 0, -9.81])) * dt
            dr["p"] = dr["p"] + dr["v"] * dt
            rel = dr["p"] - np.array([b["x"], b["y"], 0.0])
            on = abs(rel[0]) < b["half"][0] + 0.15 and abs(rel[1]) < b["half"][1] + 0.15 and rel[2] < 0.05 + 2 * b["half"][2] + 0.15
            if on or dr["p"][2] < 0.0:
                dr["alive"] = False
                self.landed.append((t, dr["p"].copy()))
                if on and self.k > 0.0:
                    self._cool_window.append((t, dr["kg"]))
                    if self.rng.random() < 0.15:
                        self.steam.append([dr["p"].copy(), t])
        self.drops = [dr for dr in self.drops if dr["alive"]] + [dr for dr in self.drops if not dr["alive"]][-200:]
        self.landed = self.landed[-60:]

    def control(self, t):
        m, d = self.model, self.data
        self.t = t
        dt = m.opt.timestep
        x, y, yaw, v = self.pose()
        if t - self._t_scan >= 0.1 - 1e-9:
            self.lidar()
            self._t_scan = t
        steer, vt = 0.0, 0.0
        pan_t, tilt_t = 0.0, 0.0
        if self.phase in ("ride out", "back off") and t >= qa.SETTLE_S:
            steer, vt, _ = self.follow(x, y, yaw, v)
            end = self.path[-1, 0]
            remaining = (end - x) if self.phase == "ride out" else (x - end)
            vt = min(vt, math.sqrt(max(0.0, 2 * 1.0 * (remaining - 0.15))))
            if self.phase == "back off":
                steer, vt = 0.0, -min(1.0, math.sqrt(max(0.0, 2 * 0.8 * (remaining - 0.1))))
            if remaining < 0.2 and abs(v) < 0.05:
                if self.phase == "ride out":
                    self.phase, self.phase_t0 = "attack", t
                    self.arbiter.events.append((t, "at the fire: the cannon turns on it"))
                else:
                    self.phase = "done"
        if self.phase == "attack":
            if t - getattr(self, "_t_aim", -1.0) >= 0.1 - 1e-9:               # the fire control at 10 Hz
                self._t_aim = t
                W_rot, piv = self._monitor_pose()
                mz = piv + W_rot @ np.array([BARREL, 0.0, 0.0])
                az, el = self._jet_aim_to(mz, self._target())
                Rc = d.xmat[self.chassis].reshape(3, 3)
                yaw_c = math.atan2(Rc[1, 0], Rc[0, 0])
                tt = t - self.phase_t0
                sweep = 0.03 * math.sin(1.6 * tt)                          # a small sweep over the bin
                # the trim learnt from where the jet lands (the last 0.4 s), small steps, clamped to 3 degrees
                recent = [p for tl, p in self.landed if t - tl < 0.4]
                if recent and self.spraying:
                    mean = np.mean(recent, axis=0)
                    tg = self._target()
                    da = math.atan2(tg[1] - mz[1], tg[0] - mz[0]) - math.atan2(mean[1] - mz[1], mean[0] - mz[0])
                    dr_ = math.hypot(tg[0] - mz[0], tg[1] - mz[1]) - math.hypot(mean[0] - mz[0], mean[1] - mz[1])
                    self.aim_corr = np.clip(self.aim_corr + np.array([0.3 * da, 0.01 * dr_]), -math.radians(3), math.radians(3))
                self._pan_t = (az - yaw_c + sweep + self.aim_corr[0] + math.pi) % (2 * math.pi) - math.pi
                self._tilt_t = el + self.aim_corr[1]
            pan_t, tilt_t = self._pan_t, self._tilt_t
            err = abs(pan_t - self.pan) + abs(tilt_t - self.tilt)
            if self.out_at is None:
                self.spraying = err < math.radians(2.0) or (self.spraying and err < math.radians(5.0))
                self.move = "on target: water on the bin" if self.spraying else "the cannon slews onto the bin"
            else:
                self.spraying = False
                pan_t, tilt_t = 0.0, 0.0
                self.move = "the fire is out: the cannon stows"
                if t > self.out_at + 2.5:
                    self.phase = "back off"
                    self.path, self.path_i, self.leg = self.back_path, 0, "back"
                    self.arbiter.events.append((t, "Chuck backs off"))
        rate = math.radians(chuck.MONITOR["slew_deg_s"]) * dt
        self.pan += float(np.clip(pan_t - self.pan, -rate, rate))
        lo, hi = (math.radians(a) for a in chuck.MONITOR["tilt_deg"])
        self.tilt += float(np.clip(min(max(tilt_t, lo), hi) - self.tilt, -rate, rate))
        drate = ga.STEER_BY_WIRE["rate_rad_s"] * dt
        self.steer_cmd += float(np.clip(steer - self.steer_cmd, -drate, drate))
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
        W_rot, piv = self._monitor_pose()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, W_rot.ravel())
        d.mocap_pos[self.mocap], d.mocap_quat[self.mocap] = piv, q
        if self.spraying and t - self._emit_t >= F.EMIT_S - 1e-9:
            d_mm, v0, _ = chuck.nozzle()
            dirn = W_rot[:, 0] + F.SPREAD_RAD * (self.rng.standard_normal() * W_rot[:, 1] + self.rng.standard_normal() * W_rot[:, 2])
            dirn /= np.linalg.norm(dirn)
            kg = chuck.PUMP["flow_l_min"] / 60.0 * F.EMIT_S
            self.drops.append({"p": piv + W_rot @ np.array([BARREL, 0.0, 0.0]), "v": v0 * dirn, "kg": kg, "alive": True})
            self.water_used_l += kg
            self._emit_t = t
        if int(round(t / dt)) % 2 == 0:
            self._fly_drops(t, 2 * dt)
        self._fire_step(t, dt)
        return self.arbiter.mode

    def report(self):
        x, y, yaw, v = self.pose()
        first = next((tt for tt, h, c, k in self.log_fire if c > 0), None)
        return {"fire_kW_when_the_water_reached_it": next((h for tt, h, c, k in self.log_fire if c > 0), None),
                "water_first_on_it_s": first, "out_at_s": self.out_at, "water_used_l": round(self.water_used_l, 1),
                "end": (round(x, 2), round(y, 2)), "phase": self.phase, "events": [(round(t, 2), e) for t, e in self.arbiter.events]}


def monitor_mesh():
    """The cannon on its mocap body (the pivot at the origin, the barrel along x): the yoke, the barrel, the tip."""
    yoke = cq.Workplane("XY").box(70, 170, 100).translate((0, 0, -20)).cut(cq.Workplane("XY").box(80, 110, 80).translate((0, 0, 0)))
    base = cq.Workplane("XY").circle(80).extrude(80).translate((0, 0, -150))
    barrel = cq.Workplane("YZ").circle(26).extrude(BARREL * 1000 - 60).translate((-60, 0, 0))
    tip = cq.Workplane("YZ").circle(36).workplane(offset=80).circle(16).loft().translate((BARREL * 1000 - 120, 0, 0))
    shape = cq.Compound.makeCompound([w.val() for w in (yoke, base, barrel, tip)])
    return [("monitor", gq._tess(shape), chuck.COLORS["monitor"])]


class AutoBinDrawer(qa.Drawer):
    def __init__(self, ride, size="1280x720"):
        sys_pieces = [p for p in chuck.chuck_mesh() if p[2] not in (chuck.COLORS["monitor"], chuck.COLORS["tools"])]
        super().__init__(ride, size, extra_pieces=sys_pieces + monitor_mesh())
        drop = {gq.COLORS["trunk"], gq.COLORS["tools"], gq.COLORS["seat"]}
        self.mesh.pieces[:] = [p for p in self.mesh.pieces if not (p[0] == "quad_chassis" and p[2] in drop)]
        self.beacon_i = next(i for i, p in enumerate(self.mesh.pieces) if p[2] == ga.KIT_COLORS["beacon"])
        self.cam, self.rng = None, np.random.default_rng(2)

    def frame(self, t):
        ride, br = self.ride, self.br
        x, y, yaw, v = ride.pose()
        b, tris, _ = self.mesh.pieces[self.beacon_i]
        self.mesh.pieces[self.beacon_i] = (b, tris, ((0.95, 0.2, 0.1, 1.0) if int(t * 4) % 2 else (0.2, 0.3, 0.95, 1.0)))
        c = np.array(ride.data.xpos[ride.chassis])
        tgt = np.array([BIN["x"], BIN["y"], 0.6])
        ph = ride.phase
        if ph == "attack":
            want = (0.72 * c + 0.28 * tgt + np.array([0, 0, 0.5]), 14.0, math.degrees(yaw) - 145.0, 12.0)
        else:
            want = ((0.8 * c + 0.2 * tgt) * np.array([1, 1, 0]) + np.array([0, 0, 0.9]), 16.0, math.degrees(yaw) + 225.0, 9.0)
        if self.cam is None:
            self.cam = [want[0].copy(), *want[1:]]
        kf = 0.1
        self.cam[0] = self.cam[0] + kf * (want[0] - self.cam[0])
        self.cam[1] += kf * (want[1] - self.cam[1])
        self.cam[2] += kf * ((want[2] - self.cam[2] + 180.0) % 360.0 - 180.0)
        self.cam[3] += kf * (want[3] - self.cam[3])
        cam = br.orbit(tuple(self.cam[0]), self.cam[1], self.cam[2], self.cam[3])
        dots = []
        h = ride.hrr(t)
        hx, hy, hz = BIN["half"]
        flame_h = 0.3 + 1.4 * min(1.0, h / 300.0)
        for _ in range(int(50 * min(1.0, h / 300.0))):
            p = np.array([BIN["x"] + self.rng.uniform(-hx, hx) * 0.8, BIN["y"] + self.rng.uniform(-hy, hy) * 0.8, 0.05 + 2 * hz])
            u = self.rng.random()
            p[2] += u * flame_h
            dots.append((p, int(7 - 4 * u), (1.0, 0.85, 0.2) if u < 0.35 else (1.0, 0.5, 0.05) if u < 0.75 else (0.85, 0.15, 0.05)))
        for _ in range(int(25 * min(1.0, h / 250.0)) + (12 if ride.out_at and t - ride.out_at < 6 else 0)):
            p = np.array([BIN["x"] + self.rng.normal(0, 0.4), BIN["y"] + self.rng.normal(0, 0.4), 0.05 + 2 * hz + flame_h + self.rng.uniform(0, 2.5)])
            g = 0.25 + 0.15 * self.rng.random() if h > 20 else 0.75
            dots.append((p, int(5 + 5 * self.rng.random()), (g, g, g + 0.02)))
        for s, ts in ride.steam:
            if t - ts < 1.2:
                dots.append((s + np.array([0, 0, 1.2 * (t - ts)]), 5, (0.95, 0.95, 0.97)))
        for dr in ride.drops:
            if dr["alive"]:
                dots.append((dr["p"], 2, (0.25, 0.5, 0.95)))
        what = ride.move if ph == "attack" else {"ride out": "driving itself to the fire (AUTO, nobody aboard)", "back off": "backing off",
                                                "done": "done"}.get(ph, ph)
        hud = [(f"Chuck, autonomous - {what}", (0.15, 0.15, 0.2)),
               (f"the bin fire: {h:4.0f} kW   the water's cooling {ride.cooling_kW:4.0f} kW   water {ride.water_used_l:4.1f} / {chuck.TANK['volume_l']:.0f} L"
                f"   cannon pan {math.degrees(ride.pan):+5.1f} tilt {math.degrees(ride.tilt):+5.1f} deg" + ("   OUT" if ride.out_at else ""),
                (0.75, 0.2, 0.1) if not ride.out_at else (0.1, 0.55, 0.2))]
        return self.renderer.render(ride.data, cam, title=f"Chuck - a bin fire, nobody aboard   t = {t:4.1f} s", dots=dots, hud=hud)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chuck alone against a bin fire")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stills", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ride = AutoBin()
    drawer = None if a.no_movie else AutoBinDrawer(ride)
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
                    img.save(str(OUT / f"chuck_auto_bin_{ts:g}s.png"))
        k += 1
    rep = ride.report()
    (OUT / "chuck_auto_bin.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if drawer is not None:
        out = str(OUT / "chuck_auto_bin.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        print("movie", out)
    return rep


if __name__ == "__main__":
    main()
