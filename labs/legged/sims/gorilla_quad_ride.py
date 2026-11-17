#!/usr/bin/env python3
"""Kerchak rides his quad (``designs/gorilla_quad``) in MuJoCo: a test course - a run-up, a field of bumps (speed
bumps across and whoops under one side, to pitch and roll him), a slalom round four cones at ~0.5 g, then a stop by
the table with the banana.

Everything is dynamic: the quad is a free body on four sprung, damped corners (``susp_XX``), its four wheels each
driven by its own motor (a speed loop per wheel, the torque capped by the motor's peak torque and power), the front
knuckles coupled to the raked handlebar column (1:1 with Ackermann). Kerchak sits welded at the pelvis to the seat
(the seat's grip), his 57 joints on their servos (PD plus gravity compensation, the servos' stall torques), the
arms on the seated-pose IK for the handlebar's angle (``gorilla_quad.seated_pose``), so his arms steer the bar; the
fists curled round the grips. A pure-pursuit driver picks the bar's angle, the course a speed.

    python gorilla_quad_ride.py                      # the movie (CAD drawn) and the comfort comparison
    python gorilla_quad_ride.py --no-movie           # the numbers only
    python gorilla_quad_ride.py --rigid              # the same ride with the suspension locked

Writes ``sims/output/gorilla_quad_ride.mp4``, ``.json`` and stills.
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
import gorilla_scenarios as gs  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_unit_robots as sr  # noqa: E402
from vegeta.chiron import Custom, Geom, Link, Prop, SimOptions, Weld  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"

#: the course [m]: bumps (x, y from, y to, height), the slalom, the stop
BUMPS = [(9.0, -1.4, 1.4, 0.10), (11.0, 0.0, 1.4, 0.09), (12.6, -1.4, 0.0, 0.09), (14.2, 0.0, 1.4, 0.09), (15.8, -1.4, 0.0, 0.09),
         (18.0, -1.4, 1.4, 0.12)]
BUMP_R = 0.30
SLALOM = {"x0": 24.0, "wavelength": 14.0, "amplitude": 1.2, "periods": 2}
CONES = [SLALOM["x0"] + SLALOM["wavelength"] * (0.25 + 0.5 * k) for k in range(4)]     #: on the line, where the path swings widest round them
STOP_X = 62.0
TABLE = (64.5, -1.9)
SETTLE_S = 1.0


def speed_target(x):
    """The course's speed [m/s] at ``x``: run up, gentle over the bumps, the slalom, the stop."""
    if x < 0.5:
        return 2.0
    if x < 7.0:
        return 2.0 + 2.5 * min(1.0, x / 6.0)
    if x < 21.0:
        return 4.0
    if x < 54.0:
        return 4.5
    return max(0.0, 4.5 * math.sqrt(max(0.0, (STOP_X - x) / (STOP_X - 54.0))))


def path_y(x):
    s = SLALOM
    u = (x - s["x0"]) / s["wavelength"]
    if 0.0 <= u <= s["periods"]:
        return s["amplitude"] * math.sin(2 * math.pi * u)
    return 0.0


def bump_height(x, y):
    """The course's ground [m]: each bump a circular segment of radius ``BUMP_R`` (the drawn cylinder's top) across its
    span; a heightfield, so the tyres roll over a surface, not a convex shape's single contact point."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    z = np.zeros(np.broadcast(x, y).shape)
    for xc, y0, y1, h in BUMPS:
        seg = np.sqrt(np.clip(BUMP_R ** 2 - (x - xc) ** 2, 0.0, None)) - (BUMP_R - h)
        z = np.maximum(z, np.where((y >= y0) & (y <= y1), np.clip(seg, 0.0, None), 0.0))
    return z


def course_prop():
    geoms = []
    quat_y = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)
    for k, (x, y0, y1, h) in enumerate(BUMPS):
        geoms.append(Geom(f"bump_{k}", "cylinder", (BUMP_R, (y1 - y0) / 2), pos=(x, (y0 + y1) / 2, h - BUMP_R), quat=quat_y, mass=100.0,
                          role="visual", rgba=(0.95, 0.80, 0.15, 1.0)))         # drawn; the bump itself is in the terrain (``bump_height``)
    for k, x in enumerate(CONES):
        geoms += [Geom(f"cone_{k}", "cylinder", (0.12, 0.25), pos=(x, 0.0, 0.25), mass=2.0, role="visual", rgba=(0.98, 0.45, 0.05, 1.0)),
                  Geom(f"cone_{k}_band", "cylinder", (0.125, 0.04), pos=(x, 0.0, 0.32), mass=0.1, role="visual", rgba=(0.95, 0.95, 0.95, 1.0))]
    geoms += [g for g in gs._table_with_banana(*TABLE)]
    for g in geoms[-8:]:
        g.role = "visual"
    return Prop(Link("course", geoms=geoms), free=False)


class Ride:
    def __init__(self, rigid=False):
        self.robot = sr.bare(casing="plastic")
        self.kin = mk.Kin(self.robot)
        self.sz = gq.sizing()
        # the seated poses over the handlebar's range: the arms' targets for an angle
        self.steers = np.linspace(-gq.GEOM["steer_max"], gq.GEOM["steer_max"], 13)
        q0, self.ik_err = gq.seated_pose(self.kin, 0.0)
        poses = {6: q0}
        for direction in (1, -1):
            q = q0
            for i in (range(7, 13) if direction > 0 else range(5, -1, -1)):
                q, e = gq.seated_pose(self.kin, self.steers[i], q_init=q)
                poses[i], self.ik_err = q, max(self.ik_err, e)
        self.Q = np.array([poses[i] for i in range(13)])
        self.robot.root.pos = tuple(gq.GEOM["pelvis"])
        opts = SimOptions()
        opts.course_extent = (4.0, 24.0, -2.5, 2.5)                       # the heightfield under the bumps; flat beyond
        opts.heightfield_cell = 0.02
        xml = self.robot.to_mjcf(terrain=Custom(bump_height, name="kerchak_bumps"), options=opts, props=[gq.quad_prop(rigid=rigid, sz=self.sz), course_prop()],
                                 welds=[Weld("seat", "pelvis", "quad_seat")])
        i = xml.index('<geom name="chiron_terrain"')                     # the ground beyond the heightfield: a plane half a millimetre under it
        xml = xml[:i] + '<geom name="ground_plane" type="plane" size="0 0 1" contype="1" conaffinity="2" priority="0" pos="0 0 -0.0005"/>\n    ' + xml[i:]
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
        # contacts: only the tyres with the ground and the course (the rider and the frame ride above it)
        for g in range(m.ngeom):
            name = m.geom(g).name
            body_root = m.body_rootid[m.geom_bodyid[g]]
            if name.startswith("tyre_") or name in ("chiron_terrain", "ground_plane") or m.body(body_root).name == "course":
                continue
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
        self.qadr = self.kin.qadr                                           # the robot's hinges come first in the model, as in Kin's
        self.dadr = self.kin.dadr
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        self.act = [m.actuator(f"servo:{n}").id for n in self.kin.names]
        self.kp = np.array([servos[n].kp for n in self.kin.names])
        self.kd = np.array([max(servos[n].kd, 0.05 * servos[n].kp ** 0.5) for n in self.kin.names])
        self.lim = np.array([servos[n].stall_torque for n in self.kin.names])
        self.motor = [m.actuator(f"motor_{c}").id for c in gq.CORNERS]
        self.spin = [m.jnt_dofadr[m.joint(f"spin_{c}").id] for c in gq.CORNERS]
        self.col = m.actuator("column").id
        self.col_q = m.jnt_qposadr[m.joint("steer_column").id]
        self.chassis = m.body("quad_chassis").id
        self.thorax = m.body("thorax").id
        self.free_q = m.jnt_qposadr[m.joint("quad_chassis:free").id]
        self.steer_cmd = 0.0
        self.vt_cmd = 0.0
        self.log = []
        self.reset()

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = self.Q[6]
        mujoco.mj_forward(m, d)

    def arms_target(self, steer):
        s = np.clip(steer, self.steers[0], self.steers[-1])
        i = min(int((s - self.steers[0]) / (self.steers[1] - self.steers[0])), len(self.steers) - 2)
        f = (s - self.steers[i]) / (self.steers[1] - self.steers[0])
        return (1 - f) * self.Q[i] + f * self.Q[i + 1]

    def pose(self):
        """(x, y, yaw, speed) of the quad."""
        d = self.data
        q = d.qpos[self.free_q:self.free_q + 7]
        R = d.xmat[self.chassis].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        v = float(d.qvel[m_dof(self.model, "quad_chassis:free")] * math.cos(yaw) + d.qvel[m_dof(self.model, "quad_chassis:free") + 1] * math.sin(yaw))
        return float(q[0]), float(q[1]), yaw, v

    def control(self, t):
        m, d = self.model, self.data
        x, y, yaw, v = self.pose()
        driving = t >= SETTLE_S
        # the driver: pure pursuit on the course's line, the look-ahead growing with the speed
        ld = max(2.0, 0.4 * abs(v) + 1.2)
        xa = x + ld * math.cos(yaw)
        tx, ty = xa, path_y(xa)
        alpha = math.atan2(ty - y, tx - x) - yaw
        L = gq.GEOM["wheelbase"]
        steer = math.atan2(2 * L * math.sin(alpha), ld) if driving else 0.0
        steer = float(np.clip(steer, -0.55, 0.55))
        self.steer_cmd += float(np.clip(steer - self.steer_cmd, -1.2 * m.opt.timestep, 1.2 * m.opt.timestep))   # a gorilla's wrists: 1.2 rad/s at most
        d.ctrl[self.col] = self.steer_cmd
        # Kerchak's servos: PD on the seated pose for the bar's angle, gravity compensated
        q_t = self.arms_target(float(d.qpos[self.col_q]))
        q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
        tau = self.kp * (q_t - q) - self.kd * qd + d.qfrc_bias[self.dadr]
        d.ctrl[self.act] = np.clip(tau, -self.lim, self.lim)
        # the four motors: a speed loop each, the torque capped by the motor's peak torque and power
        vt = speed_target(x) if driving else 0.0
        dv = vt - self.vt_cmd                                                # the throttle eased in and out: 1.5 m/s² up, 2.5 down
        self.vt_cmd += float(np.clip(dv, -2.5 * m.opt.timestep, 1.5 * m.opt.timestep))
        vt = self.vt_cmd
        R = gq.GEOM["wheel_r"]
        P = self.sz["motor"]["peak_power_kW"] * 1000.0
        T = self.sz["wheel_torque_Nm"]
        for a, dof in zip(self.motor, self.spin):
            w = float(d.qvel[dof])
            cap = min(T, P / max(abs(w), 1e-3))
            d.ctrl[a] = float(np.clip(120.0 * (vt / R - w), -cap, cap))
        return vt

    def step(self, t, dt):
        m, d = self.model, self.data
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            vt = self.control(t + k * m.opt.timestep)
            v_prev = self._thorax_vel()
            mujoco.mj_step(m, d)
            acc = (self._thorax_vel() - v_prev) / m.opt.timestep
            x, y, yaw, v = self.pose()
            R = d.xmat[self.chassis].reshape(3, 3)
            self.log.append((t + k * m.opt.timestep, x, y, v, vt, float(acc[2]), math.degrees(math.atan2(R[2, 1], R[2, 2])),
                             math.degrees(math.asin(-max(-1.0, min(1.0, R[2, 0])))), float(d.qpos[self.col_q]),
                             float(d.qpos[self.free_q + 2]), *[self._wheel_load(c) for c in gq.CORNERS]))

    def _thorax_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.thorax, v, 0)
        return v[3:6].copy()

    def _wheel_load(self, c):
        m, d = self.model, self.data
        g = m.geom(f"tyre_{c}").id
        f = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            if g in (con.geom1, con.geom2):
                r = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, r)
                f += r[0]
        return f

    def report(self, x_from=8.0, x_to=21.0):
        L = np.array(self.log)
        t, x, acc = L[:, 0], L[:, 1], L[:, 5]
        dt = self.model.opt.timestep
        # the thorax's vertical acceleration weighted as ISO 2631-1 weights what a seated body feels (Wk)
        a_s = wk_filter(acc, 1.0 / dt)
        bump = (x > x_from) & (x < x_to)
        loads = L[:, 10:14]
        settled = t > SETTLE_S
        return {"rms_vertical_acc_bumps_m_s2": round(float(np.sqrt(np.mean(a_s[bump] ** 2))), 2),
                "peak_vertical_acc_bumps_m_s2": round(float(np.max(np.abs(a_s[bump]))), 2),
                "roll_max_deg": round(float(np.max(np.abs(L[settled, 6]))), 1), "pitch_max_deg": round(float(np.max(np.abs(L[settled, 7]))), 1),
                "wheel_lift_s": round(float(np.sum(np.min(loads[settled], axis=1) < 1.0) * dt), 3),
                "min_wheel_load_N": round(float(np.min(loads[settled])), 0),
                "speed_max_m_s": round(float(np.max(L[:, 3])), 2), "x_end_m": round(float(x[-1]), 2), "y_end_m": round(float(L[-1, 2]), 2),
                "steer_max_rad": round(float(np.max(np.abs(L[:, 8]))), 3), "seated_ik_error_max_m": round(float(self.ik_err), 4),
                "max_lateral_offset_from_line_m": round(float(np.max(np.abs(L[:, 2] - np.array([path_y(v) for v in x])))), 2)}


def wk_filter(a, fs):
    """ISO 2631-1's Wk weighting (vertical, seated) of an acceleration signal sampled at ``fs`` [Hz]: band limits
    0.4-100 Hz, the acceleration-velocity transition, the upward step (ISO 8041's parameters), bilinear."""
    from scipy import signal
    w = lambda f: 2 * math.pi * f
    q1 = 1 / math.sqrt(2)
    num, den = np.array([1.0, 0.0, 0.0]), np.array([1.0, w(0.4) / q1, w(0.4) ** 2])                       # high-pass
    num, den = np.polymul(num, [w(100.0) ** 2]), np.polymul(den, [1.0, w(100.0) / q1, w(100.0) ** 2])     # low-pass
    num = np.polymul(num, np.array([1 / w(12.5), 1.0]))                                                   # a-v transition
    den = np.polymul(den, np.array([1 / w(12.5) ** 2, 1 / (0.63 * w(12.5)), 1.0]))
    num = np.polymul(num, np.array([1 / w(2.37) ** 2, 1 / (0.91 * w(2.37)), 1.0]) * (w(2.37) / w(3.35)) ** 2)   # upward step
    den = np.polymul(den, np.array([1 / w(3.35) ** 2, 1 / (0.91 * w(3.35)), 1.0]))
    b, a_ = signal.bilinear(num, den, fs)
    return signal.lfilter(b, a_, a)


def m_dof(model, joint):
    return int(model.jnt_dofadr[model.joint(joint).id])


def iso_comfort(rms):
    """ISO 2631-1's comfort reactions for a frequency-weighted RMS acceleration [m/s²] (Annex C)."""
    for lim, what in ((0.315, "not uncomfortable"), (0.63, "a little uncomfortable"), (1.0, "fairly uncomfortable"), (1.6, "uncomfortable"),
                      (2.5, "very uncomfortable")):
        if rms < lim:
            return what
    return "extremely uncomfortable (the gorilla cries)"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kerchak rides his quad")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--rigid", action="store_true", help="lock the suspension")
    ap.add_argument("--compare", action="store_true", help="also ride with the suspension locked, for the numbers")
    ap.add_argument("--stills", default="3,8,13,18")
    ap.add_argument("--size", default="1280x720")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ride = Ride(rigid=a.rigid)
    seconds = a.seconds if a.seconds is not None else 20.0
    tag = "gorilla_quad_ride" + ("_rigid" if a.rigid else "")
    renderer = None
    if not a.no_movie:
        import kerchak_mesh as bm
        import kerchak_render as br
        km = bm.load("plastic")
        pieces, struts = gq.quad_mesh()
        mesh = bm.Mesh(pieces=list(km.pieces) + pieces, struts=list(km.struts) + struts)
        w, h = (int(v) for v in a.size.split("x"))
        renderer = br.Renderer(ride.model, mesh, w, h)
    stills = [float(s) for s in a.stills.split(",") if s.strip()]
    tmp = tempfile.mkdtemp()

    def draw(path, t):
        d = ride.data
        c = np.array(d.xpos[ride.chassis])
        x, y, yaw, v = ride.pose()
        az = math.degrees(yaw) + 235.0 - 70.0 * mk.smooth(min(1.0, max(0.0, (x - 20.0) / 8.0))) + 60.0 * mk.smooth(min(1.0, max(0.0, (x - 52.0) / 8.0)))
        cam = br.orbit((c[0], c[1], 0.9), 16.0, az, 6.4)
        L = ride.log[-1] if ride.log else (0,) * 14
        sect = "settling" if t < SETTLE_S else "bumps" if x < 21 else "slalom" if x < 53 else "stopping at the banana"
        title = f"Kerchak's quad   t = {t:4.1f} s   {3.6 * v:4.1f} km/h   steer {math.degrees(L[8]):+5.1f} deg   roll {L[6]:+5.1f} deg   -   {sect}"
        renderer.render(d, cam, title=title).save(path)

    import kerchak_render as br  # noqa: F811
    t, dt, k = 0.0, 1.0 / a.fps, 0
    while t <= seconds + 1e-9:
        if renderer is not None:
            draw(os.path.join(tmp, f"f{k:04d}.png"), t)
            for ts in stills:
                if abs(t - ts) < dt / 2:
                    draw(str(OUT / f"{tag}_{ts:g}s.png"), t)
        ride.step(t, dt)
        t += dt
        k += 1
        if ride.pose()[0] > STOP_X + 0.5 and abs(ride.pose()[3]) < 0.05:
            break
    rep = {"rigid": a.rigid, "seconds": round(t, 2), "sizing": ride.sz, **ride.report()}
    rep["comfort"] = iso_comfort(rep["rms_vertical_acc_bumps_m_s2"])
    if a.compare and not a.rigid:
        other = Ride(rigid=True)
        tt = 0.0
        while tt <= seconds and other.pose()[0] < 22.0:
            other.step(tt, dt)
            tt += dt
        ro = other.report()
        rep["rigid_comparison"] = {"rms_vertical_acc_bumps_m_s2": ro["rms_vertical_acc_bumps_m_s2"],
                                   "peak_vertical_acc_bumps_m_s2": ro["peak_vertical_acc_bumps_m_s2"], "comfort": iso_comfort(ro["rms_vertical_acc_bumps_m_s2"])}
    if renderer is not None:
        out = str(OUT / f"{tag}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        rep["movie"] = out
    (OUT / f"{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sizing"}, indent=1))
    return rep


if __name__ == "__main__":
    main()
