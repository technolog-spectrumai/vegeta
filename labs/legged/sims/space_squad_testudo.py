#!/usr/bin/env python3
"""The space game units form a testudo and run: from a standing formation (three loose ranks) the squad marches
in to a tight block (the ranks closing up) - the Breacher's tower shield and the two Marines' body shields in the front rank, the Centurion
between the chainsaw Marines behind them, the Heavy at the back - the front rank brings the shields to the guard
and the ranks behind crouch under them, then the whole block runs together, shields still up.

    python space_squad_testudo.py                    # dynamics, 15 s, the movie
    python space_squad_testudo.py --mode kinematic   # the keyframes played
    python space_squad_testudo.py --units marine,breacher --seconds 15 --no-movie

Built on ``space_squad_march.py`` (the ``Unit``: the robot, its MuJoCo model, the PD loop on the servos, the upright
assist on the pelvis, the drawing): a ``Script`` gives each unit its timeline - stand, march to its place in the
block while its lane slides inward, stand, the testudo pose (the guard or the crouch), then the run gait
(``space_unit_motions.run``: 0.85 m/s on a 0.45 m stance, the shields held in the guard). Writes
``sims/output/space_squad_testudo.mp4`` and ``.json``.
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

import humanoid_motions as hum  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_unit_motions as sm  # noqa: E402
import space_squad_march as march_sim  # noqa: E402

OUT = march_sim.OUT
#: the block (unit, x, y) [m]: the shields in the front rank, the Centurion between the chainsaws, the Heavy last
BLOCK = [("marine", 0.0, 0.95), ("breacher", 0.0, 0.0), ("marine", 0.0, -0.95),
         ("marine_saw", -0.95, 0.95), ("centurion", -0.95, 0.0), ("marine_saw", -0.95, -0.95),
         ("heavy", -1.9, 0.0)]
#: the standing formation the squad starts in: the same files, the ranks strung out 1.5-1.8 m apart (closing ranks
#: is a straight march: the keyframed gait cannot side-step, and dragging a walker sideways trips it)
STAND = [(u, {0.0: -1.5, -0.95: -3.3, -1.9: -4.5}[x], y) for u, x, y in BLOCK]
#: the Marines' guard: the shield arm up and in so the plate stands square ahead of the body
MARINE_GUARD = {"L_shoulder_pitch": -0.75, "L_shoulder_roll": 0.2, "L_shoulder_yaw": 0.35}
SHIELDS = ("marine", "breacher")
T_STAND, T_BLEND, T_HOLD, CROUCH_M = 1.0, 0.4, 2.0, 0.22
ELEV, AZIM = 24.0, 52.0                                                  # the camera: ahead and to the left, so the ranks show
T_BLEND_RUN = 0.4                                                        # out of the testudo pose into the run


def crouch_pose(kin, dz):
    """The standing pose with the pelvis ``dz`` lower (both soles where they are): IK on the sole points."""
    targets = []
    for g in ("L", "R"):
        body, c = kin.geom_point(g)
        for o in sm.SOLE_POINTS:
            local = c + np.asarray(o)
            targets.append((body, local, kin.world(body, local, kin.q0) + np.array([0.0, 0.0, dz])))
    q = kin.q0.copy()
    for s in ("L", "R"):
        for j, v in (("hip_pitch", -0.3), ("knee", 0.6), ("ankle_pitch", -0.3)):
            q[kin.index[f"{s}_{j}"]] = v
    legs = [j for j in kin.actuated if j.split("_", 1)[1] in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
    q, err = kin.ik(targets, q_init=q, joints=legs)
    assert err < 0.01, err
    return q


class Script(march_sim.Unit):
    """A unit with a timeline: stand -> march to its place in the block -> stand -> the testudo pose -> run. The
    march starts at the gait's mid-stance phase (one foot under the hips, the other mid-swing: the nearest frame
    to standing, so no foot has to slide) and halts at the first half-period boundary after the unit has reached
    its place (in physics these weights walk faster than the keyframes' nominal speed, so the halt is decided on
    the base's actual position, not on a schedule)."""

    def __init__(self, name, start, place, mode, run_at, **kw):
        super().__init__(name, start[0], start[1], mode, **kw)
        self.start, self.place = np.asarray(start, float), np.asarray(place, float)
        kin, unit = self.kin, name
        self.z_stand = hum.base_height(kin)
        guard = MARINE_GUARD if name == "marine" else sm.GUARD if name == "breacher" else {}
        self.hold_run = dict(self._hold_of(name)) | guard
        self.run = sm.run(kin, unit, hold=self.hold_run)                    # the roster's run, the shield held in the guard
        self.v_run = float(self.run.base["velocity_m_s"][0])
        self.z_run = self.z_stand - float(self.run.params.get("crouch_m", 0.03))
        self.q_stand = kin.q0.copy()
        for j, v in self._hold_of(name).items():
            self.q_stand[kin.index[j]] = v
        import space_unit_specs as su
        for s in ("L", "R"):
            self.q_stand[kin.index[f"{s}_elbow"]] = -math.radians(su.ROSTER[unit][0]["elbow_bend_deg"])
        if name in SHIELDS:                                                 # the front rank: the guard, standing
            self.q_testudo, self.z_testudo = self.q_stand.copy(), self.z_stand
            for j, v in guard.items():
                self.q_testudo[kin.index[j]] = v
        else:                                                               # the ranks behind: crouched under the shields
            self.q_testudo, self.z_testudo = crouch_pose(kin, CROUCH_M), self.z_stand - CROUCH_M
            for j, v in self._hold_of(name).items():
                self.q_testudo[kin.index[j]] = v
        self.t_go, self.t_halt, self.t_run = T_STAND, None, run_at
        self.p0 = np.array([self.start[0], self.start[1], self.z_stand])
        self.reset()

    @staticmethod
    def _hold_of(name):
        return {"marine": {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.45}, "marine_saw": {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.3},
                "centurion": {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.3}, "breacher": {"L_shoulder_pitch": -0.7, "R_shoulder_pitch": 0.3},
                "heavy": {"R_shoulder_pitch": -0.35, "L_shoulder_pitch": -0.5}}[name]

    @staticmethod
    def _gait_start(gait):
        """The gait time at mid-stance of the first foot (duty / 2 of the period): the frame nearest to standing."""
        return 0.5 * float(gait.params["duty"]) * float(gait.period)

    def _gait_q(self, gait, t):
        Q, keys = np.asarray(gait.Q), len(gait.t)
        s = (t % gait.period) / gait.period * keys
        i, f = int(s) % keys, s - int(s)
        q = self.kin.q0.copy()
        q[[self.kin.index[j] for j in gait.joints]] = (1 - f) * Q[i] + f * Q[(i + 1) % keys]
        return q

    @property
    def t_pose(self):
        return None if self.t_halt is None else self.t_halt + T_BLEND + 0.2

    def phase(self, t):
        """(pose a, pose b, weight of b, base speed, base height) at time t."""
        blend = lambda t0, dur=T_BLEND: mk.smooth(min(1.0, max(0.0, (t - t0) / dur)))
        march_q = lambda: self._gait_q(self.gait, t - self.t_go + self._gait_start(self.gait))
        run_q = lambda: self._gait_q(self.run, t - self.t_run + self._gait_start(self.run))
        if t < self.t_go:
            return self.q_stand, self.q_stand, 0.0, 0.0, self.z_stand
        if self.t_halt is None or t < self.t_halt:
            w = blend(self.t_go)
            return self.q_stand, march_q(), w, w * self.v, self.z_stand + w * (self.z0 - self.z_stand)
        if t < self.t_pose:
            w = blend(self.t_halt)
            return march_q(), self.q_stand, w, (1 - w) * self.v, self.z0 + w * (self.z_stand - self.z0)
        if t < self.t_run:
            w = blend(self.t_pose)
            return self.q_stand, self.q_testudo, w, 0.0, self.z_stand + w * (self.z_testudo - self.z_stand)
        w = blend(self.t_run, T_BLEND_RUN)
        return self.q_testudo, run_q(), w, w * self.v_run, self.z_testudo + w * (self.z_run - self.z_testudo)

    def q_target(self, t):
        if not hasattr(self, "t_go"):
            return super().q_target(t)
        a, b, w, _, _ = self.phase(t)
        return (1 - w) * a + w * b

    def decide_halt(self, t, dt, x):
        """Halt at the first half-period boundary of the march once the base is at its place (or when the run is due)."""
        if self.t_halt is not None or t < self.t_go + T_BLEND:
            return
        frac = ((t - self.t_go + self._gait_start(self.gait)) / self.T) % 0.5
        at_boundary = frac * self.T < dt + 1e-9
        if (x >= self.place[0] - 0.2 and at_boundary) or t >= self.t_run - T_BLEND - 1.0:
            self.t_halt = t

    def reset(self):
        super().reset()
        if hasattr(self, "t_go"):
            self.data.qpos[2] = self.z_stand + (0.01 if self.mode == "physics" else 0.0)
            self.data.qvel[:] = 0.0
            self.x_kin = float(self.start[0])
            mujoco.mj_forward(self.model, self.data)

    def step_kinematic(self, t, dt):
        self.decide_halt(t, dt, self.x_kin)
        _, _, _, v, z = self.phase(t)
        self.x_kin += v * dt
        d = self.data
        d.qpos[0:3] = [self.x_kin, self.start[1], z]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = self.q_target(t)
        mujoco.mj_kinematics(self.model, d)

    def step_physics(self, t, dt, assist):
        self.decide_halt(t, dt, float(self.data.qpos[0]))
        halted = self.t_halt is not None and t < self.t_run
        if halted and self.park is None:
            self.park = float(self.place[0])                                # the halt: the assist parks the base at its place
        elif not halted:
            self.park = None
        self.v_hold = None if halted else self.phase(t)[3]                  # marching or running: the formation's pace
        super().step_physics(t, dt, assist)

    def report(self):
        r = super().report()
        r.update({"at_block_s": None if self.t_halt is None else round(self.t_halt, 2), "run_speed_m_s": self.v_run,
                  "ran_m": round(float(self.data.qpos[0] - self.place[0]), 3), "lane_error_m": round(float(self.data.qpos[1] - self.place[1]), 3)})
        return r


def run(units, seconds, mode, assist, fps=10):
    frames, t, dt = [], 0.0, 1.0 / fps
    while t <= seconds + 1e-9:
        for u in units:
            if mode == "kinematic":
                u.step_kinematic(t, dt)
            else:
                u.step_physics(t, dt, assist)
        frames.append((t, *march_sim.snapshot(units)))
        t += dt
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description="the space game units form a testudo and run")
    ap.add_argument("--mode", choices=("physics", "kinematic"), default="physics")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--units", default="squad", help="'squad' (the block), or unit names comma-separated (their places in the block)")
    ap.add_argument("--no-assist", action="store_true")
    ap.add_argument("--assist", type=float, default=2.0, help="the upright assist's strength (as the march sim: 2 for the wide hips)")
    ap.add_argument("--run-at", type=float, default=10.0, help="when the block breaks into the run [s]")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("VEGETA_DEBUG") == "1" and argv is None and len(sys.argv) == 1:      # tools/lab.py run legged sims: a short check
        a.units, a.seconds, a.no_movie = "marine", 1.0, True
    assist = 0.0 if a.no_assist else a.assist
    OUT.mkdir(parents=True, exist_ok=True)
    if a.units == "squad":
        rows = list(zip(BLOCK, STAND))
    else:
        used, rows = set(), []
        for n in a.units.split(","):
            k = next(i for i, b in enumerate(BLOCK) if b[0] == n and i not in used)
            used.add(k)
            rows.append((BLOCK[k], STAND[k]))
    units = [Script(b[0], s[1:], b[1:], a.mode, a.run_at) for b, s in rows]
    frames = run(units, a.seconds, a.mode, assist, a.fps)
    report = {"mode": a.mode, "assist": assist, "seconds": a.seconds, "run_at_s": a.run_at, "units": [u.report() for u in units]}
    tag = "" if a.units == "squad" else "_" + a.units.replace(",", "_")
    (OUT / f"space_squad_testudo{tag}.json").write_text(json.dumps(report, indent=1))
    if not a.no_movie:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tmp = tempfile.mkdtemp()
        base = "dynamics with the upright assist" if a.mode == "physics" and assist else "dynamics, no assist" if a.mode == "physics" else "the keyframes played"
        span = 3.8 if a.units == "squad" else 2.0 + 0.8 * len(units)
        t_in = max(u.t_halt or a.run_at for u in units) + T_BLEND
        for k, (t, geoms, centre) in enumerate(frames):
            what = "standing" if t < T_STAND else "closing ranks" if t < t_in else "the testudo" if t < a.run_at else "running together"
            fig = plt.figure(figsize=(12.8, 7.2))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            march_sim.draw_frame(ax, geoms, centre, t, f"{what}  ({base})", span, elev=ELEV, azim=AZIM)
            ax.set_title(f"The space game units form a testudo and run   -   t = {t:4.1f} s   -   {what}  ({base})", fontsize=11, loc="left", y=0.96)
            fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
            fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=72)
            plt.close(fig)
        out = a.out or str(OUT / f"space_squad_testudo{tag}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        report["movie"] = out
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
