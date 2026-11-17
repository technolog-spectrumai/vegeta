#!/usr/bin/env python3
"""Belisario walks like a gorilla: on all fours on its knuckles (``gorilla_motions.knuckle_walk``, the lateral
sequence LH LF RH RF), rears up on two to stand a moment, drops back on four and walks on - the bare frame, no
weapons (``space_unit_robots.bare``), in MuJoCo.

    python space_gorilla_walk.py                      # dynamics, the whole sequence (22 s), the movie
    python space_gorilla_walk.py --mode kinematic     # the keyframes played
    python space_gorilla_walk.py --seconds 8 --no-movie --no-assist

The timeline: walk ``--walk`` s, rise (3 s), stand (``--stand`` s), drop (3 s), walk again. As the squad sims
(``space_squad_march.py``): a PD loop on every servo tracks the keyframes within the modules' stall torque, and an
upright assist on the pelvis (a virtual gyro, a lane spring, a hold on the gait's pace) stands in for the balance
controller the design has not got. Writes ``sims/output/space_gorilla_walk.mp4`` and ``.json``.
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

import gorilla_motions as gm  # noqa: E402
import humanoid_motions as hum  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_squad_march as march_sim  # noqa: E402
import space_unit_robots as sr  # noqa: E402

OUT = march_sim.OUT
T_RISE, T_DROP = 3.0, 3.0


class Gorilla:
    """The bare frame with its timeline: walk on four, rise, stand on two, drop, walk on."""

    def __init__(self, mode, walk_s=8.0, stand_s=3.0):
        self.mode = mode
        self.robot = sr.bare()
        self.kin = mk.Kin(self.robot)
        kin = self.kin
        self.q_quad = gm.quad_pose(kin)
        self.walk = gm.knuckle_walk(kin, self.q_quad)
        self.q_stand = kin.q0.copy()
        self.h_quad = gm.PELVIS_H
        self.h_stand = hum.base_height(kin)
        self.v = float(self.walk.base["velocity_m_s"][0])
        self.t_rise = walk_s
        self.t_stand = self.t_rise + T_RISE
        self.t_drop = self.t_stand + stand_s
        self.t_walk2 = self.t_drop + T_DROP
        self.model, self.data = kin.model, mujoco.MjData(kin.model)
        m = self.model
        self.qadr = kin.qadr
        self.act_joint = [int(m.actuator_trnid[i][0]) for i in range(m.nu)]
        self.act_col = [kin.hinges.index(j) for j in self.act_joint]
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        names = [m.joint(j).name for j in self.act_joint]
        self.kp = np.array([servos[n].kp for n in names])
        self.lim = np.array([servos[n].stall_torque for n in names])
        self.sat, self.tilt_max, self.fell = 0.0, 0.0, None
        self.x_kin = 0.0
        self.reset()

    def _gait_q(self, t):
        g = self.walk
        Q, keys = np.asarray(g.Q), len(g.t)
        s = (t % g.period) / g.period * keys
        i, f = int(s) % keys, s - int(s)
        q = self.q_quad.copy()
        q[[self.kin.index[j] for j in g.joints]] = (1 - f) * Q[i] + f * Q[(i + 1) % keys]
        return q

    def phase(self, t):
        """(q target, base speed, base height, what) at t."""
        if t < self.t_rise:
            return self._gait_q(t), self.v, self.h_quad, "walking on four"
        if t < self.t_stand:
            w = mk.smooth((t - self.t_rise) / T_RISE)
            return (1 - w) * self.q_quad + w * self.q_stand, 0.0, self.h_quad + w * (self.h_stand - self.h_quad), "rising"
        if t < self.t_drop:
            return self.q_stand, 0.0, self.h_stand, "standing on two"
        if t < self.t_walk2:
            w = mk.smooth((t - self.t_drop) / T_DROP)
            return (1 - w) * self.q_stand + w * self.q_quad, 0.0, self.h_stand + w * (self.h_quad - self.h_stand), "dropping on four"
        return self._gait_q(t - self.t_walk2), self.v, self.h_quad, "walking on four"

    def reset(self):
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[0:3] = [0.0, 0.0, self.h_quad + (0.01 if self.mode == "physics" else 0.0)]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = self.phase(0.0)[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def tilt(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return math.degrees(math.acos(max(-1.0, min(1.0, R.reshape(3, 3)[2, 2]))))

    def step_kinematic(self, t, dt):
        q, v, z, _ = self.phase(t)
        self.x_kin += v * dt
        d = self.data
        d.qpos[0:3] = [self.x_kin, 0.0, z]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)

    def step_physics(self, t, dt, assist):
        m, d = self.model, self.data
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            tk = t + k * m.opt.timestep
            q_t, v, _, _ = self.phase(tk)
            q = d.qpos[self.qadr]
            tau = np.clip(self.kp * (q_t[self.act_col] - q[self.act_col]), -self.lim, self.lim)
            self.sat = max(self.sat, float(np.max(np.abs(tau) / self.lim)))
            d.ctrl[:] = tau
            d.xfrc_applied[1][:] = 0.0
            if assist > 0.0:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, d.qpos[3:7])
                R = R.reshape(3, 3)
                err = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
                w = d.qvel[3:6]
                yaw = math.atan2(R[1, 0], R[0, 0])
                torque = assist * (6000.0 * err - 400.0 * w) + np.array([0.0, 0.0, -assist * (3000.0 * yaw + 300.0 * w[2])])
                force = np.array([-assist * march_sim.V_HOLD_C * (d.qvel[0] - v), -assist * (2000.0 * d.qpos[1] + 400.0 * d.qvel[1]), 0.0])
                d.xfrc_applied[1][0:3] = force
                d.xfrc_applied[1][3:6] = torque
            mujoco.mj_step(m, d)
        self.tilt_max = max(self.tilt_max, self.tilt())
        if self.fell is None and (self.tilt() > 45.0 or d.qpos[2] < 0.25):
            self.fell = round(t, 2)

    def report(self):
        d = self.data
        return {"mass_kg": round(self.robot.total_mass(), 1), "walk_speed_m_s": self.v, "walked_m": round(float(d.qpos[0]), 3),
                "base_z_m": round(float(d.qpos[2]), 3), "tilt_max_deg": round(self.tilt_max, 1), "servo_saturation_max": round(self.sat, 3),
                "fell_at_s": self.fell, "timeline_s": {"rise": self.t_rise, "stand": self.t_stand, "drop": self.t_drop, "walk_again": self.t_walk2}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Belisario walks like a gorilla")
    ap.add_argument("--mode", choices=("physics", "kinematic"), default="physics")
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--walk", type=float, default=8.0, help="the first walk's length [s]")
    ap.add_argument("--stand", type=float, default=3.0, help="how long it stands on two [s]")
    ap.add_argument("--no-assist", action="store_true")
    ap.add_argument("--assist", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("VEGETA_DEBUG") == "1" and argv is None and len(sys.argv) == 1:
        a.seconds, a.no_movie = 1.0, True
    assist = 0.0 if a.no_assist else a.assist
    OUT.mkdir(parents=True, exist_ok=True)
    g = Gorilla(a.mode, a.walk, a.stand)
    frames, t, dt = [], 0.0, 1.0 / a.fps
    while t <= a.seconds + 1e-9:
        if a.mode == "kinematic":
            g.step_kinematic(t, dt)
        else:
            g.step_physics(t, dt, assist)
        frames.append((t, *march_sim.snapshot([g]), g.phase(t)[3]))
        t += dt
    report = {"mode": a.mode, "assist": assist, "seconds": a.seconds, **g.report()}
    (OUT / "space_gorilla_walk.json").write_text(json.dumps(report, indent=1))
    if not a.no_movie:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tmp = tempfile.mkdtemp()
        base = "dynamics with the upright assist" if a.mode == "physics" and assist else "dynamics, no assist" if a.mode == "physics" else "the keyframes played"
        for k, (t, geoms, centre, what) in enumerate(frames):
            fig = plt.figure(figsize=(12.8, 7.2))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            march_sim.draw_frame(ax, geoms, centre, t, f"{what}  ({base})", 2.6, elev=12.0, azim=40.0)
            ax.set_title(f"Belisario walks like a gorilla   -   t = {t:4.1f} s   -   {what}  ({base})", fontsize=11, loc="left", y=0.96)
            fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
            fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=72)
            plt.close(fig)
        out = a.out or str(OUT / "space_gorilla_walk.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        report["movie"] = out
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
