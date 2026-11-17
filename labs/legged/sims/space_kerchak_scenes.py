#!/usr/bin/env python3
"""Belisario in its scenarios (``kerchak_scenarios``): the pre-programmed sequence of a scene played in MuJoCo -
the climbing wall, the tree, the ship's hull, the rope, the window, the obstacles, the door.

    python space_kerchak_scenes.py --scene wall                       # the keyframes played (default), the movie and stills
    python space_kerchak_scenes.py --scene wall --mode physics        # dynamics with the assist (a climb needs grips the
                                                                      #   assist does not give: expect the report to say so)
    python space_kerchak_scenes.py --scene tree --stills 8,14,20

Kinematic mode drives the pelvis along the motions' tracks (x, y, height, pitch per frame) and the joints along the
keyframes; the door's lever and hinge follow the hand, the rope stays put. Writes ``sims/output/space_kerchak_<scene>.mp4``,
``.json`` and ``space_kerchak_<scene>_<t>s.png`` stills, drawn by ``belisario_render`` (the CAD on the bodies: the
skeleton, the modules, the struts, the bays, the plates) unless ``--render boxes``.
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
import kerchak_scenarios as gs  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_squad_march as march_sim  # noqa: E402
import space_unit_robots as sr  # noqa: E402
from vegeta.chiron import SimOptions  # noqa: E402

OUT = march_sim.OUT
VIEWS = {"wall": (12.0, 265.0), "tree": (12.0, 265.0), "hull": (12.0, 265.0), "rope": (12.0, 265.0), "window": (14.0, 230.0),
         "window_hammer": (14.0, 230.0), "obstacles": (12.0, 265.0), "door": (14.0, 215.0)}


class ScenePlayer:
    def __init__(self, scene, mode, params=None, casing="none"):
        self.scene, self.mode = scene, mode
        self.robot = sr.bare(casing=casing)
        self.kin = mk.Kin(self.robot)
        self.q_quad = gm.quad_pose(self.kin)
        self.walk = gm.knuckle_walk(self.kin, self.q_quad)
        self.v = float(self.walk.base["velocity_m_s"][0])
        self.segments = gs.sequence(scene, self.kin, self.q_quad, params)
        t = 0.0
        self.starts = []
        for seg in self.segments:
            self.starts.append(t)
            t += seg[1] if seg[0] == "gait" else seg[1].period
        self.total = t
        opts = SimOptions()
        opts.course_extent = (-2.0, 10.0, -3.0, 3.0)
        xml = self.robot.to_mjcf(options=opts, props=list(gs.props(scene, params)))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = self.kin.qadr
        m = self.model
        self.act_joint = [int(m.actuator_trnid[i][0]) for i in range(m.nu)]
        self.act_col = [self.kin.hinges.index(j) for j in self.act_joint]
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        names = [m.joint(j).name for j in self.act_joint]
        self.kp = np.array([servos[n].kp for n in names])
        self.lim = np.array([servos[n].stall_torque for n in names])
        self.x_kin, self.fell, self.tilt_max = 0.0, None, 0.0
        self.strike_t = None                                               # when the pane goes (window_hammer)
        self._hammer = None
        if scene == "window_hammer":
            m = self.model
            free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name == "hammer"]
            by_name = {seg[1].name: (s0, seg[1]) for s0, seg in zip(self.starts, self.segments) if seg[0] == "motion"}
            pick, smash, drop = by_name["pick_up_the_hammer"], by_name["smash_the_glass"], by_name["drop_the_hammer"]
            R_h2m, p_h2m = (np.array(v) for v in pick[1].params["hammer_T"])
            rest_p, rest_R = (np.array(v) for v in pick[1].params["hammer_rest"])
            self._hammer = {"adr": m.jnt_qposadr[free[0]], "R_h2m": R_h2m, "p_h2m": p_h2m, "rest": (rest_p, rest_R),
                            "t_grasp": pick[0] + pick[1].period, "t_release": drop[0] + drop[1].params["release_u"] * drop[1].period,
                            "released": None, "hand": m.body("R_hand").id}
            self.strike_t = smash[0] + smash[1].params["strike_u"] * smash[1].period
        self.reset()

    def segment(self, t):
        for s, seg in zip(reversed(self.starts), reversed(self.segments)):
            if t >= s:
                return seg, t - s
        return self.segments[0], 0.0

    def _gait_q(self, t):
        g = self.walk
        Q, keys = np.asarray(g.Q), len(g.t)
        s = (t % g.period) / g.period * keys
        i, f = int(s) % keys, s - int(s)
        q = self.q_quad.copy()
        q[[self.kin.index[j] for j in g.joints]] = (1 - f) * Q[i] + f * Q[(i + 1) % keys]
        return q

    def phase(self, t):
        """(q, base x or None, y, speed, height, pitch, what)."""
        seg, tl = self.segment(t)
        if seg[0] == "gait":
            return self._gait_q(tl), None, seg[2], self.v, gm.PELVIS_H, gm.PELVIS_PITCH, "walking on four"
        m = seg[1]
        Q, n = np.asarray(m.Q), len(m.t) - 1
        s = min(max(tl / m.period, 0.0), 1.0) * n
        i, f = min(int(s), n - 1), min(s - int(s), 1.0)
        q = self.kin.q0.copy()                                             # a motion's joints are offsets from the design pose
        q[[self.kin.index[j] for j in m.joints]] = (1 - f) * Q[i] + f * Q[i + 1]
        b = m.base
        lerp = lambda key: (1 - f) * b[key][i] + f * b[key][i + 1]
        return q, lerp("x_m"), lerp("y_m"), 0.0, lerp("height_m"), lerp("pitch_rad"), m.name

    def reset(self):
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[0:3] = [0.0, 0.0, gm.PELVIS_H + (0.01 if self.mode == "physics" else 0.0)]
        d.qpos[3:7] = gm.base_quat()
        d.qpos[self.qadr] = self.phase(0.0)[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def step_kinematic(self, t, dt):
        q, x, y, v, h, p, _ = self.phase(t)
        if x is None:
            self.x_kin += v * dt
        else:
            self.x_kin = x
        d = self.data
        d.qpos[0:3] = [self.x_kin, y, h]
        d.qpos[3:7] = gm.base_quat(p)
        d.qpos[self.qadr] = q
        self._scene_joints(t)
        mujoco.mj_kinematics(self.model, d)
        if self._hammer is not None:
            self._hammer_pose(t)
            mujoco.mj_kinematics(self.model, d)

    def _hammer_pose(self, t):
        """The hammer through the sequence: lying where the hand finds it, in the hand from the grasp to the release,
        then falling (no tumble) and lying where it lands."""
        H, d, m = self._hammer, self.data, self.model
        adr = H["adr"]

        def put(p, R):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
            d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = p, q

        if t < H["t_grasp"]:
            put(*H["rest"])
        elif t < H["t_release"]:
            Rh, ph = d.xmat[H["hand"]].reshape(3, 3), d.xpos[H["hand"]]
            put(ph + Rh @ H["p_h2m"], Rh @ H["R_h2m"])
            H["released"] = (ph + Rh @ H["p_h2m"], Rh @ H["R_h2m"])
        else:
            if H["released"] is None:
                H["released"] = H["rest"]
            p0, R0 = H["released"]
            import kerchak_scenarios as gs
            Lh, rz = gs.HAMMER["handle"], gs.HAMMER["rest_z"]
            low = min(p0[2] - 0.018, p0[2] + Lh * R0[2, 2] - 0.03)          # the hammer's lowest point at the release
            tau = t - H["t_release"]
            fall = 0.5 * 9.81 * tau * tau
            if fall < low:
                put(p0 - np.array([0.0, 0.0, fall]), R0)
            else:                                                          # landed: flat, the handle along the way it pointed
                ax = R0[:, 2].copy(); ax[2] = 0.0
                ax = ax / max(np.linalg.norm(ax), 1e-6)
                z = np.array([0.0, 0.0, 1.0])
                put(np.array([p0[0], p0[1], rz]), np.stack([np.cross(z, ax), z, ax], axis=1))

    def _scene_joints(self, t):
        """The scene's own joints along the sequence (kinematic playback: the door's lever and hinge follow the hand)."""
        if self.scene != "door":
            return
        import kerchak_tasks as gt
        m = self.model
        lever, hinge = m.jnt_qposadr[m.joint("handle_hinge").id], m.jnt_qposadr[m.joint("door_hinge").id]
        seg, tl = self.segment(t)
        name = seg[1].name if seg[0] == "motion" else ""
        u = min(max(tl / seg[1].period, 0.0), 1.0) if seg[0] == "motion" else 0.0
        done = [s[1].name for s in self.segments[:self.segments.index(seg)] if s[0] == "motion"]
        turned = gt.DOOR["turn_rad"] * (mk.smooth(u) if name == "turn" else 1.0 if "turn" in done else 0.0)
        if name == "release":
            turned *= 1.0 - mk.smooth(min(1.0, u / 0.25))
        elif "release" in done:
            turned = 0.0
        opened = gt.DOOR["open_rad"] * (mk.smooth(u) if name == "push" else 1.0 if "push" in done else 0.0)
        if name == "release":                                              # the shoved door swings on as the hand lets go
            opened = gt.DOOR["open_rad"] + (gt.DOOR["swung_rad"] - gt.DOOR["open_rad"]) * mk.smooth(u)
        elif "release" in done:
            opened = gt.DOOR["swung_rad"]
        self.data.qpos[lever], self.data.qpos[hinge] = turned, opened

    def step_physics(self, t, dt, assist):
        m, d = self.model, self.data
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            q_t, x, y, v, h, p, _ = self.phase(t + k * m.opt.timestep)
            q = d.qpos[self.qadr]
            tau = np.clip(self.kp * (q_t[self.act_col] - q[self.act_col]), -self.lim, self.lim)
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
                force = np.array([fx, -assist * (2000.0 * (d.qpos[1] - y) + 400.0 * d.qvel[1]), 0.0])
                d.xfrc_applied[1][0:3] = force
                d.xfrc_applied[1][3:6] = torque
            mujoco.mj_step(m, d)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R.reshape(3, 3)[:, 2] @ gm.base_rot(self.phase(t)[5])[:, 2])))))
        self.tilt_max = max(self.tilt_max, tilt)
        if self.fell is None and (tilt > 60.0 or d.qpos[2] < 0.2):
            self.fell = round(t, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Belisario in a scenario")
    ap.add_argument("--scene", choices=sorted(gs.SCENARIOS), default="wall")
    ap.add_argument("--mode", choices=("kinematic", "physics"), default="kinematic")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--assist", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stills", default="", help="times [s], comma-separated, for still images")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--param", action="append", default=[], help="scene parameter key=value")
    ap.add_argument("--render", choices=("mesh", "boxes"), default="mesh", help="the CAD on the bodies (belisario_render), or the collision shapes")
    ap.add_argument("--casing", choices=("none", "plastic"), default="plastic")
    ap.add_argument("--size", default="1280x720")
    a = ap.parse_args(argv)
    if os.environ.get("VEGETA_DEBUG") == "1" and argv is None and len(sys.argv) == 1:
        a.seconds, a.no_movie = 0.5, True
    params = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        params[k] = float(v)
    OUT.mkdir(parents=True, exist_ok=True)
    g = ScenePlayer(a.scene, a.mode, params, casing=a.casing)
    seconds = a.seconds if a.seconds is not None else g.total
    elev, azim = VIEWS.get(a.scene, (12.0, 265.0))
    span = 3.2
    stills = [float(x) for x in a.stills.split(",") if x.strip()]
    tmp = tempfile.mkdtemp()
    if a.render == "mesh":
        import belisario_mesh as bm
        import belisario_render as br
        w, h = (int(v) for v in a.size.split("x"))
        renderer = br.Renderer(g.model, bm.load(a.casing), w, h)

    def draw(path, t, what):
        c = np.array(g.data.qpos[0:3])
        if a.render == "mesh":
            az = azim
            if a.scene in ("window", "window_hammer"):                     # once through the wall the camera comes round to the inside
                f = mk.smooth(min(1.0, max(0.0, (c[0] - (gs.WINDOW["x"] - 0.1)) / 0.9)))
                az = azim + 60.0 * f
            renderer.hidden = {"glass"} if (g.strike_t is not None and t >= g.strike_t) else set()
            cam = br.orbit((c[0], c[1], max(c[2], 0.8)), elev, az, 4.6)
            renderer.render(g.data, cam, title=f"Belisario: {a.scene}   -   t = {t:4.1f} s   -   {what}  ({a.mode})").save(path)
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        geoms, centre = march_sim.snapshot([g])
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        cc = np.array([centre[0], centre[1], max(centre[2], 1.0)])
        march_sim.draw_frame(ax, geoms, cc, t, what, span, elev=elev, azim=azim)
        ax.set_zlim(max(0.0, cc[2] - 0.6 * span), cc[2] + 0.6 * span)
        ax.set_title(f"Belisario: {a.scene}   -   t = {t:4.1f} s   -   {what}  ({a.mode})", fontsize=11, loc="left", y=0.96)
        fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
        fig.savefig(path, dpi=72)
        plt.close(fig)

    t, dt, k = 0.0, 1.0 / a.fps, 0
    while t <= seconds + 1e-9:
        if a.mode == "kinematic":
            g.step_kinematic(t, dt)
        else:
            g.step_physics(t, dt, a.assist)
        what = g.phase(t)[6]
        if not a.no_movie:
            draw(os.path.join(tmp, f"f{k:04d}.png"), t, what)
        for ts in stills:
            if abs(t - ts) < dt / 2:
                draw(str(OUT / f"space_kerchak_{a.scene}_{ts:g}s.png"), t, what)
        t += dt
        k += 1
    report = {"scene": a.scene, "mode": a.mode, "seconds": seconds, "total_s": round(g.total, 2), "x_m": round(float(g.data.qpos[0]), 3),
              "z_m": round(float(g.data.qpos[2]), 3), "fell_at_s": g.fell, "tilt_off_phase_max_deg": round(g.tilt_max, 1),
              "segments": [(s[0], round(s[1] if s[0] == "gait" else s[1].period, 2), s[2] if s[0] == "gait" else s[1].name) for s in g.segments],
              "ik_error_max_m": {s[1].name: s[1].check.get("ik_error_max_m") for s in g.segments if s[0] == "motion"}}
    (OUT / f"space_kerchak_{a.scene}.json").write_text(json.dumps(report, indent=1))
    if not a.no_movie:
        out = str(OUT / f"space_kerchak_{a.scene}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        report["movie"] = out
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
