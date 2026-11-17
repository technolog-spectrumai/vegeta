#!/usr/bin/env python3
"""The space game units march in MuJoCo: the squad (the Centurion at the point, the Marines, the Breacher and the
Heavy behind) walks its ``march`` gait across flat ground, and a movie is drawn from the models' geoms.

    python space_squad_march.py                               # dynamics: servos track the gait, 8 s, the squad
    python space_squad_march.py --mode kinematic --seconds 6  # the keyframes played through MuJoCo's kinematics
    python space_squad_march.py --units marine --no-assist    # one Marine, no upright assist: does it fall?

Two modes. ``kinematic`` plays the gait's keyframes (``space_unit_motions.march``, IK-solved in the design
pose) through ``mj_kinematics``: the base glides at the gait's speed and the joints follow the frames - what the
motion looks like, no physics. ``physics`` (default) runs ``mj_step``: gravity, the feet's contacts and the joint
modules' torque limits decide what happens while a PD loop on each servo tracks the keyframes (``kp`` of the
modules, the stall torque as the clip). The keyframes were solved with no balance controller, so the base is
helped by an **upright assist** by default: an external torque on the pelvis towards level (a virtual gyro) and a
sideways spring holding its lane - the "training wheels" a game engine would also give these units (the Breacher
and the Heavy, the two heaviest, get twice the Marines': ``ASSIST``). ``--no-assist`` takes them off and the log
says how far the unit gets before it falls.

Writes ``sims/output/space_squad_march.mp4`` and ``space_squad_march.json`` (per unit: distance walked, the base's
height, the largest tilt, the servo torques' saturation, slip). No OpenGL is needed: the frames are drawn from the
geoms with matplotlib and ffmpeg makes the mp4.
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import humanoid_motions as hum  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_unit_motions as sm  # noqa: E402
import space_unit_robots as sr  # noqa: E402

OUT = ROOT / "labs" / "legged" / "sims" / "output"
UNITS = {"marine": (sr.marine, sm.motions_marine), "marine_saw": (sr.marine_saw, sm.motions_marine_saw),
         "centurion": (sr.centurion, sm.motions_centurion), "breacher": (sr.breacher, sm.motions_breacher),
         "heavy": (sr.heavy, sm.motions_heavy)}
#: the upright assist's strength per unit: the two heaviest (the tower shield, the cannon) get twice the Marines'
ASSIST = {"breacher": 2.0, "heavy": 2.0}                                 # multipliers on --assist
V_HOLD_C = 1500.0                                                           # the speed hold's damping [N s/m]
#: the squad's wedge (unit, x, y) [m], the Centurion at the point
SQUAD = [("centurion", 1.6, 0.0), ("marine", 0.4, 1.3), ("marine", 0.4, -1.3), ("marine_saw", -0.9, 2.4), ("marine_saw", -0.9, -2.4),
         ("breacher", -2.1, 1.1), ("heavy", -2.1, -1.1)]


class Unit:
    """One unit: its robot, its MuJoCo model, its march gait and the maps between them."""

    def __init__(self, name, x, y, mode, crouch=0.02, leg_kp=None, leg_torque=None, modules=None):
        build, motions = UNITS[name]
        self.name, self.mode = name, mode
        self.robot = build()
        if leg_kp or leg_torque or modules:                               # other leg modules than the design's, to try
            import dataclasses
            for j in self.robot.joints():
                if j.servo is None or j.tag in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
                    continue
                if modules:
                    kp, lim = modules[0] if j.tag in ("hip_yaw", "hip_roll", "hip_pitch", "knee") else modules[1]
                    j.servo = dataclasses.replace(j.servo, kp=kp, stall_torque=lim)
                else:
                    j.servo = dataclasses.replace(j.servo, kp=leg_kp or j.servo.kp, stall_torque=leg_torque or j.servo.stall_torque)
        self.kin, ms = motions(self.robot, crouch=crouch)
        self.gait = ms[0]
        self.model, self.data = self.kin.model, mujoco.MjData(self.kin.model)
        m = self.model
        self.hinges = self.kin.hinges
        self.qadr, self.dadr = self.kin.qadr, self.kin.dadr
        self.cols = [self.kin.index[j] for j in self.gait.joints]
        self.Q, self.T = np.asarray(self.gait.Q), float(self.gait.period)
        self.t_keys = np.asarray(self.gait.t)
        self.v = float(self.gait.base["velocity_m_s"][0])
        self.z0 = hum.base_height(self.kin) - float(self.gait.params.get("crouch_m", 0.0))
        self.p0 = np.array([x, y, self.z0])
        # the servos: actuator -> hinge (its ctrl is a torque; the model's bias adds the module's damping)
        self.act_joint = [int(m.actuator_trnid[i][0]) for i in range(m.nu)]
        self.act_col = [self.hinges.index(j) for j in self.act_joint]
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        names = [m.joint(j).name for j in self.act_joint]
        self.kp = np.array([servos[n].kp for n in names])
        self.lim = np.array([servos[n].stall_torque for n in names])
        self.sat, self.tilt_max, self.z_min, self.fell = 0.0, 0.0, self.z0, None
        self.park = None                                                   # an x to hold with the assist (a halt), or None
        self.v_hold = None                                                 # a forward speed to hold with the assist, or None
        self.reset()

    def q_target(self, t):
        """The hinges at time ``t``: the gait's frame (periodic, linearly interpolated), the rest at the design pose."""
        q = self.kin.q0.copy()
        s = (t % self.T) / self.T * len(self.t_keys)
        i, f = int(s) % len(self.t_keys), s - int(s)
        row = (1 - f) * self.Q[i] + f * self.Q[(i + 1) % len(self.t_keys)]
        q[self.cols] = row
        return q

    def reset(self):
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[0:3] = self.p0 + np.array([0.0, 0.0, 0.01 if self.mode == "physics" else 0.0])
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = self.q_target(0.0)
        d.qvel[:] = 0.0
        d.qvel[0] = self.v if self.mode == "physics" else 0.0
        mujoco.mj_forward(self.model, d)

    def tilt(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return math.degrees(math.acos(max(-1.0, min(1.0, R.reshape(3, 3)[2, 2]))))

    def step_kinematic(self, t):
        d = self.data
        d.qpos[0:3] = self.p0 + np.array([self.v * t, 0.0, 0.0])
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = self.q_target(t)
        mujoco.mj_kinematics(self.model, d)

    def step_physics(self, t, dt, assist):
        m, d = self.model, self.data
        assist = assist * ASSIST.get(self.name, 1.0)
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            tk = t + k * m.opt.timestep
            q_t = self.q_target(tk)
            q = d.qpos[self.qadr]
            tau = self.kp * (q_t[self.act_col] - q[self.act_col])
            tau = np.clip(tau, -self.lim, self.lim)
            self.sat = max(self.sat, float(np.max(np.abs(tau) / self.lim)))
            d.ctrl[:] = tau
            d.xfrc_applied[1][:] = 0.0
            if assist > 0.0:
                # the virtual gyro: a torque turning the pelvis level (its z axis onto the world's), damped on its rate;
                # a spring on the lane (y) and on the heading
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, d.qpos[3:7])
                R = R.reshape(3, 3)
                err = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))          # the rotation taking the body's up onto the world's
                w = d.qvel[3:6]
                yaw = math.atan2(R[1, 0], R[0, 0])
                torque = assist * (6000.0 * err - 400.0 * w) + np.array([0.0, 0.0, -assist * (3000.0 * yaw + 300.0 * w[2])])
                force = np.array([0.0, -assist * (2000.0 * (d.qpos[1] - self.p0[1]) + 400.0 * d.qvel[1]), 0.0])
                if self.park is not None:                                  # halted: the assist also parks the base
                    force[0] = -assist * (4000.0 * (d.qpos[0] - self.park) + 2000.0 * d.qvel[0])
                elif self.v_hold is not None:                              # a speed to hold: the formation's pace
                    force[0] = -assist * V_HOLD_C * (d.qvel[0] - self.v_hold)
                d.xfrc_applied[1][0:3] = force
                d.xfrc_applied[1][3:6] = torque
            mujoco.mj_step(m, d)
        self.tilt_max = max(self.tilt_max, self.tilt())
        self.z_min = min(self.z_min, float(d.qpos[2]))
        if self.fell is None and (self.tilt() > 45.0 or d.qpos[2] < 0.5 * self.z0):
            self.fell = round(t, 2)

    def report(self):
        d = self.data
        return {"unit": self.name, "mass_kg": round(self.robot.total_mass(), 1), "gait_speed_m_s": self.v,
                "walked_m": round(float(d.qpos[0] - self.p0[0]), 3), "base_z_m": round(float(d.qpos[2]), 3),
                "base_z_min_m": round(self.z_min, 3), "tilt_max_deg": round(self.tilt_max, 1),
                "servo_saturation_max": round(self.sat, 3), "fell_at_s": self.fell}


# ---- drawing from the geoms (matplotlib; no OpenGL)
def _box(c, R, h):
    s = np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * h
    v = s @ R.T + c
    return [v[k] for k in ([0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5])]


def _cyl(c, R, r, hl, n=10, caps=True):
    t = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.c_[r * np.cos(t), r * np.sin(t)]
    out = [np.array([[ring[k][0], ring[k][1], -hl], [ring[k + 1][0], ring[k + 1][1], -hl], [ring[k + 1][0], ring[k + 1][1], hl],
                     [ring[k][0], ring[k][1], hl]]) @ R.T + c for k in range(n)]
    if caps:
        out += [np.c_[ring, np.full(len(ring), z)] @ R.T + c for z in (-hl, hl)]
    return out


def _sphere(c, r, n=5):
    th, ph = np.linspace(0, np.pi, n + 1), np.linspace(0, 2 * np.pi, 2 * n + 1)
    out = []
    for i in range(n):
        for j in range(2 * n):
            pts = [(th[i], ph[j]), (th[i + 1], ph[j]), (th[i + 1], ph[j + 1]), (th[i], ph[j + 1])]
            out.append(np.array([[r * math.sin(a) * math.cos(b), r * math.sin(a) * math.sin(b), r * math.cos(a)] for a, b in pts]) + c)
    return out


def snapshot(units):
    out = []
    for u in units:
        m, d = u.model, u.data
        for i in range(m.ngeom):
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
                continue
            out.append((int(m.geom_type[i]), d.geom_xpos[i].copy(), d.geom_xmat[i].reshape(3, 3).copy(), m.geom_size[i].copy(), m.geom_rgba[i].copy()))
    centre = np.mean([u.data.qpos[0:3] for u in units], axis=0)
    return out, centre


def draw_frame(ax, geoms, centre, t, title, span, elev=16.0, azim=28.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys, cols = [], []
    for gtype, c, R, sz, rgba in geoms:
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            p = _box(c, R, sz[:3])
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            p = _cyl(c, R, sz[0], sz[1])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            p = _cyl(c, R, sz[0], sz[1], caps=False) + _sphere(c + R[:, 2] * sz[1], sz[0], 3) + _sphere(c - R[:, 2] * sz[1], sz[0], 3)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            p = _sphere(c, sz[0])
        else:
            continue
        p = [q for q in p if np.all(np.isfinite(q)) and np.max(np.abs(q - centre)) < 4 * span]  # a diverged body would break the axes
        polys += p
        cols += [rgba] * len(p)
    # painter's order: farthest from the camera first (the camera sits ahead and to the left, above)
    cam = np.array([math.cos(math.radians(azim)), math.sin(math.radians(azim)), math.tan(math.radians(elev))])
    depth = np.array([np.mean(q, axis=0) @ cam for q in polys])
    order = np.argsort(depth)
    polys = [polys[k] for k in order]
    cols = [cols[k] for k in order]
    # the ground as the first polygon of the same collection (its own surface would be sorted over the feet)
    g = 1.6 * span
    polys.insert(0, np.array([[centre[0] - g, centre[1] - g, -0.005], [centre[0] + g, centre[1] - g, -0.005],
                              [centre[0] + g, centre[1] + g, -0.005], [centre[0] - g, centre[1] + g, -0.005]]))
    cols.insert(0, np.array([0.545, 0.522, 0.47, 1.0]))
    ax.set_xlim(centre[0] - span, centre[0] + span); ax.set_ylim(centre[1] - span, centre[1] + span); ax.set_zlim(0, 0.6 * span)
    ax.autoscale(False)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=[c * [0.6, 0.6, 0.6, 1] for c in cols], linewidths=0.15, shade=True),
                        autolim=False)
    ax.set_box_aspect((1, 1, 0.6), zoom=1.3); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_proj_type("persp", focal_length=3.0)                            # a long lens: matplotlib's default distorts the near units
    ax.set_title(f"The space game units march in MuJoCo   -   t = {t:4.1f} s   -   {title}", fontsize=11, loc="left", y=0.96)


def run(units, seconds, mode, assist, fps=10):
    frames, t, dt = [], 0.0, 1.0 / fps
    while t <= seconds + 1e-9:
        for u in units:
            if mode == "kinematic":
                u.step_kinematic(t)
            else:
                u.step_physics(t, dt, assist)
        frames.append((t, *snapshot(units)))
        t += dt
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description="the space game units march in MuJoCo")
    ap.add_argument("--mode", choices=("physics", "kinematic"), default="physics")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--units", default="squad", help="'squad' (the wedge), or unit names comma-separated, one each in a row")
    ap.add_argument("--no-assist", action="store_true", help="physics without the upright assist")
    ap.add_argument("--assist", type=float, default=2.0, help="the assist's strength (2 = the gyro and lane spring the 360 mm hips need; 1 stumbles)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--crouch", type=float, default=0.02, help="the gait's crouch [m] (the design's 0.02; 0.05 folds the knees further)")
    ap.add_argument("--leg-kp", type=float, default=None, help="try stronger leg servos: their PD gain")
    ap.add_argument("--leg-torque", type=float, default=None, help="try stronger leg servos: their stall torque [N m]")
    ap.add_argument("--modules", choices=("design", "light"), default="design",
                    help="'light': the first design's 400 N m modules in every leg joint (kp 1500), which sag and fall")
    a = ap.parse_args(argv)
    if os.environ.get("VEGETA_DEBUG") == "1" and argv is None and len(sys.argv) == 1:      # tools/lab.py run legged sims: a short check
        a.units, a.seconds, a.no_movie = "marine", 1.0, True
    assist = 0.0 if a.no_assist else a.assist
    OUT.mkdir(parents=True, exist_ok=True)
    places = SQUAD if a.units == "squad" else [(n, 0.0, 1.6 * (k - (len(a.units.split(",")) - 1) / 2)) for k, n in enumerate(a.units.split(","))]
    modules = ((1500.0, 400.0), (1500.0, 400.0)) if a.modules == "light" else None
    units = [Unit(n, x, y, a.mode, a.crouch, a.leg_kp, a.leg_torque, modules) for n, x, y in places]
    frames = run(units, a.seconds, a.mode, assist, a.fps)
    report = {"mode": a.mode, "assist": assist, "seconds": a.seconds, "crouch_m": a.crouch, "modules": a.modules, "leg_kp": a.leg_kp, "leg_torque": a.leg_torque,
              "units": [u.report() for u in units]}
    tag = "" if a.units == "squad" else "_" + a.units.replace(",", "_")
    (OUT / f"space_squad_march{tag}.json").write_text(json.dumps(report, indent=1))
    if not a.no_movie:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tmp = tempfile.mkdtemp()
        title = f"{'dynamics, servos tracking the gait' + (' with the upright assist' if assist else ', no assist') if a.mode == 'physics' else 'the keyframes played'}"
        span = 4.2 if a.units == "squad" else 2.0 + 0.8 * len(units)
        for k, (t, geoms, centre) in enumerate(frames):
            fig = plt.figure(figsize=(12.8, 7.2))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            draw_frame(ax, geoms, centre, t, title, span)
            fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
            fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=72)
            plt.close(fig)
        out = a.out or str(OUT / f"space_squad_march{tag}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        report["movie"] = out
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
