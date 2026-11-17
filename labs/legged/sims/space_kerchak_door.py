#!/usr/bin/env python3
"""Belisario opens a door: the whole sequence from the approach to the entry, in MuJoCo - the knuckle walk up to
the doorway (``kerchak_tasks.door_prop``), up onto the feet with the right hand to the lever, the fingers closing,
the wrist turning the lever down, the door pushed open as the pelvis advances, the hands back down on four, and the
knuckle walk on through (``kerchak_tasks``: reach, grasp, turn, push, release round ``kerchak_motions.knuckle_walk``).

    python space_kerchak_door.py                      # dynamics, the sequence (~30 s), the movie
    python space_kerchak_door.py --mode kinematic     # the keyframes played, the door's hinges driven to match

As the other Kerchak sim: a PD loop on every servo tracks the keyframes, an upright assist on the pelvis holds the
phase's pitch, lane and pace and parks the base where a motion wants it. In dynamics the door is a passive prop:
the lever turns and the door swings under the hand's contacts (or not: the report says how far the door opened).
Writes ``sims/output/space_kerchak_door.mp4`` and ``.json``.
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

import kerchak_motions as gm  # noqa: E402
import kerchak_tasks as gt  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_squad_march as march_sim  # noqa: E402
import space_unit_robots as sr  # noqa: E402

OUT = march_sim.OUT


class Segment:
    def __init__(self, name, motion, duration, kind, x0=0.0, lane=0.0):
        self.name, self.motion, self.duration, self.kind, self.x0, self.lane = name, motion, duration, kind, x0, lane


class DoorRun:
    """The bare frame, the doorway, and the timeline: approach (gait) -> reach, grasp, turn, push, release
    (transitions with their own base tracks) -> enter (gait)."""

    def __init__(self, mode, enter_s=11.0):
        self.mode = mode
        self.robot = sr.bare()
        self.kin = mk.Kin(self.robot)
        kin = self.kin
        self.q_quad = gm.quad_pose(kin)
        self.walk = gm.knuckle_walk(kin, self.q_quad)
        self.v = float(self.walk.base["velocity_m_s"][0])
        seq = gt.door_sequence(kin, self.q_quad)
        approach = gt.STOP_X / self.v
        self.segments = [Segment("approach", self.walk, approach, "gait", 0.0)]
        for m in seq:
            self.segments.append(Segment(m.name, m, m.period, "transition"))
        self.segments.append(Segment("enter", self.walk, enter_s, "gait", gt.STOP_X, lane=0.0))
        t = 0.0
        for sg in self.segments:
            sg.t0 = t
            t += sg.duration
        self.total = t
        # the sim's model: the robot with the doorway
        xml = self.robot.to_mjcf(props=[gt.door_prop()])
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = kin.qadr
        self.door_adr = m.jnt_qposadr[m.joint("door_hinge").id]
        self.handle_adr = m.jnt_qposadr[m.joint("handle_hinge").id]
        self.act_joint = [int(m.actuator_trnid[i][0]) for i in range(m.nu)]
        self.act_col = [kin.hinges.index(j) for j in self.act_joint]
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        names = [m.joint(j).name for j in self.act_joint]
        self.kp = np.array([servos[n].kp for n in names])
        self.lim = np.array([servos[n].stall_torque for n in names])
        self.sat, self.tilt_max, self.fell, self.door_max = 0.0, 0.0, None, 0.0
        self.x_kin = 0.0
        self.reset()

    def segment(self, t):
        for sg in self.segments:
            if t < sg.t0 + sg.duration:
                return sg, t - sg.t0
        sg = self.segments[-1]
        return sg, t - sg.t0

    def _gait_q(self, t):
        g = self.walk
        Q, keys = np.asarray(g.Q), len(g.t)
        s = (t % g.period) / g.period * keys
        i, f = int(s) % keys, s - int(s)
        q = self.q_quad.copy()
        q[[self.kin.index[j] for j in g.joints]] = (1 - f) * Q[i] + f * Q[(i + 1) % keys]
        return q

    def _transition(self, m, tl, q_base):
        Q, n = np.asarray(m.Q), len(m.t) - 1
        s = min(max(tl / m.period, 0.0), 1.0) * n
        i, f = min(int(s), n - 1), min(s - int(s), 1.0)
        q = q_base.copy()
        q[[self.kin.index[j] for j in m.joints]] = (1 - f) * Q[i] + f * Q[i + 1]
        b = m.base
        lerp = lambda key: (1 - f) * b[key][i] + f * b[key][i + 1]
        return q, lerp("x_m"), lerp("height_m"), lerp("pitch_rad")

    def phase(self, t):
        """(q target, base x (None: free, walking), base speed, height, pitch, lane y, door angle, lever angle, what)."""
        sg, tl = self.segment(t)
        if sg.kind == "gait":
            return (self._gait_q(tl), None, self.v, gm.PELVIS_H, gm.PELVIS_PITCH, sg.lane, self.door_angle(t), 0.0, f"{sg.name}: walking on four")
        q, x, h, p = self._transition(sg.motion, tl, self.q_quad)
        return (q, x, 0.0, h, p, 0.0, self.door_angle(t), self.lever_angle(t), sg.name)

    def door_angle(self, t):
        """What the door does in the keyframes' world (kinematic mode): opens in the push, swings on as the body goes through."""
        sg, tl = self.segment(t)
        if sg.name in ("approach", "reach", "grasp", "turn"):
            return 0.0
        if sg.name == "push":
            return gt.DOOR["open_rad"] * mk.smooth(tl / sg.duration)
        if sg.name == "release":
            return gt.DOOR["open_rad"]
        return gt.DOOR["open_rad"] + 0.8 * mk.smooth(min(1.0, tl / 5.0))     # shouldered open on the way through

    def lever_angle(self, t):
        sg, tl = self.segment(t)
        if sg.name == "turn":
            return gt.DOOR["turn_rad"] * mk.smooth(tl / sg.duration)
        if sg.name == "push":
            return gt.DOOR["turn_rad"]
        if sg.name == "release":
            return gt.DOOR["turn_rad"] * (1.0 - mk.smooth(min(1.0, tl / (0.4 * sg.duration))))
        return 0.0

    def reset(self):
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[0:3] = [0.0, 0.0, gm.PELVIS_H + (0.01 if self.mode == "physics" else 0.0)]
        d.qpos[3:7] = gm.base_quat()
        d.qpos[self.qadr] = self.phase(0.0)[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def tilt(self, pitch):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return math.degrees(math.acos(max(-1.0, min(1.0, float(R.reshape(3, 3)[:, 2] @ gm.base_rot(pitch)[:, 2])))))

    def step_kinematic(self, t, dt):
        q, x, v, h, p, lane, door, lever, _ = self.phase(t)
        if x is None:
            self.x_kin += v * dt
        else:
            self.x_kin = x
        d = self.data
        d.qpos[0:3] = [self.x_kin, lane, h]
        d.qpos[3:7] = gm.base_quat(p)
        d.qpos[self.qadr] = q
        d.qpos[self.door_adr] = door
        d.qpos[self.handle_adr] = lever
        mujoco.mj_kinematics(self.model, d)

    def step_physics(self, t, dt, assist):
        m, d = self.model, self.data
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            tk = t + k * m.opt.timestep
            q_t, x, v, h, p, lane, _, _, _ = self.phase(tk)
            q = d.qpos[self.qadr]
            tau = np.clip(self.kp * (q_t[self.act_col] - q[self.act_col]), -self.lim, self.lim)
            self.sat = max(self.sat, float(np.max(np.abs(tau) / self.lim)))
            d.ctrl[:] = tau
            d.xfrc_applied[1][:] = 0.0
            if assist > 0.0:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, d.qpos[3:7])
                R = R.reshape(3, 3)
                err = np.cross(R[:, 2], gm.base_rot(p)[:, 2])
                w = d.qvel[3:6]
                yaw = math.atan2(R[1, 0], R[0, 0])
                torque = assist * (6000.0 * err - 400.0 * w) + np.array([0.0, 0.0, -assist * (3000.0 * yaw + 300.0 * w[2])])
                fx = -assist * (4000.0 * (d.qpos[0] - x) + 2000.0 * d.qvel[0]) if x is not None else -assist * march_sim.V_HOLD_C * (d.qvel[0] - v)
                force = np.array([fx, -assist * (2000.0 * (d.qpos[1] - lane) + 400.0 * d.qvel[1]), 0.0])
                d.xfrc_applied[1][0:3] = force
                d.xfrc_applied[1][3:6] = torque
            mujoco.mj_step(m, d)
        self.door_max = max(self.door_max, float(d.qpos[self.door_adr]))
        self.tilt_max = max(self.tilt_max, self.tilt(self.phase(t)[4]))
        if self.fell is None and (self.tilt(self.phase(t)[4]) > 45.0 or d.qpos[2] < 0.25):
            self.fell = round(t, 2)

    def report(self):
        d = self.data
        return {"mass_kg": round(self.robot.total_mass(), 1), "total_s": round(self.total, 2), "x_m": round(float(d.qpos[0]), 3),
                "base_z_m": round(float(d.qpos[2]), 3), "tilt_off_phase_max_deg": round(self.tilt_max, 1), "servo_saturation_max": round(self.sat, 3),
                "door_open_max_rad": round(self.door_max, 3), "door_open_rad": round(float(d.qpos[self.door_adr]), 3), "fell_at_s": self.fell,
                "timeline_s": {sg.name: round(sg.t0, 2) for sg in self.segments}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Belisario opens a door")
    ap.add_argument("--mode", choices=("physics", "kinematic"), default="physics")
    ap.add_argument("--seconds", type=float, default=None, help="default: the whole sequence")
    ap.add_argument("--enter", type=float, default=11.0, help="the walk through the doorway [s]")
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
    g = DoorRun(a.mode, a.enter)
    seconds = a.seconds if a.seconds is not None else g.total
    frames, t, dt = [], 0.0, 1.0 / a.fps
    while t <= seconds + 1e-9:
        if a.mode == "kinematic":
            g.step_kinematic(t, dt)
        else:
            g.step_physics(t, dt, assist)
        geoms, centre = march_sim.snapshot([g])
        frames.append((t, geoms, centre, g.phase(t)[8]))
        t += dt
    report = {"mode": a.mode, "assist": assist, "seconds": seconds, **g.report()}
    (OUT / "space_kerchak_door.json").write_text(json.dumps(report, indent=1))
    if not a.no_movie:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tmp = tempfile.mkdtemp()
        base = "dynamics with the upright assist" if a.mode == "physics" and assist else "dynamics, no assist" if a.mode == "physics" else "the keyframes played"
        for k, (t, geoms, centre, what) in enumerate(frames):
            fig = plt.figure(figsize=(12.8, 7.2))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            c = np.array([max(centre[0], 1.2), 0.0, centre[2]])
            march_sim.draw_frame(ax, geoms, c, t, f"{what}  ({base})", 3.0, elev=14.0, azim=215.0)
            ax.set_title(f"Belisario opens a door   -   t = {t:4.1f} s   -   {what}  ({base})", fontsize=11, loc="left", y=0.96)
            fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
            fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=72)
            plt.close(fig)
        out = a.out or str(OUT / "space_kerchak_door.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        report["movie"] = out
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
