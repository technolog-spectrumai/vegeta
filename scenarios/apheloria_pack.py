#!/usr/bin/env python3
"""Apheloria packs into its ball and unpacks again — MuJoCo dynamics through Chiron, two movies.

    xvfb-run -a python3 scenarios/apheloria_pack.py            # headless (pyvista needs a display)
    python3 scenarios/apheloria_pack.py --only pack --fps 25

Writes ``scenarios/output/apheloria_pack.mp4``, ``apheloria_unpack.mp4`` and ``apheloria_pack_unpack.json``
(what the joints achieved). The robot is ``notebooks/designs/apheloria_robot.py`` (33.1 kg: head + 8 segments, 96
leg servos, 8 active body pitch joints of 60 N·m stall). Nothing is animated: a scripted controller only sets
servo targets, and gravity, contacts and the actuators' torque–speed limits decide what the body actually does —
including whether the curl closes (notebook 19 gives its curl-up joint a safety factor of only 1.13).

Pack (12 s): stand 1 s → legs tuck against the belly (2 s) → the 8 body joints curl to −40° each, neck first, a
joint every 0.4 s, 3 s per joint → hold.
Unpack (11 s): start curled in the ball, resting on the ground → hold 1 s → the joints open tail first → legs
unfold to standing → hold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import numpy as np  # noqa: E402

import apheloria_robot as ar  # noqa: E402
import myropod_robot as mr  # noqa: E402
from vegeta import chiron as ch  # noqa: E402

OUT = ROOT / "scenarios" / "output"
N = ar.N_SEGMENTS
CURL = math.radians(ar.CURL_DEG)
LEG_KINDS = ("hip_yaw", "hip_pitch", "knee")


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Script:
    """Servo targets as a function of walking time: legs blend STAND↔TUCK, body joint k blends a→b in its window."""

    name = "scripted"

    def __init__(self, legs_from, legs_to, legs_window, curl_from, curl_to, joint_windows):
        self.legs = (legs_from, legs_to, legs_window)
        self.curl = (curl_from, curl_to, joint_windows)

    def reset(self, lab, seed=None):
        self.joints = list(lab.actuated_joints)

    def targets(self, t: float) -> dict:
        a, b, (t0, t1) = self.legs
        u = smooth((t - t0) / (t1 - t0))
        q = {}
        for s in range(1, N + 1):
            for key in mr.LEG_KEYS:
                for name, kind in zip(mr.leg_joints(s, key), LEG_KINDS):
                    q[name] = a[kind] + (b[kind] - a[kind]) * u
        c0, c1, windows = self.curl
        for k, (w0, w1) in enumerate(windows):
            q[ar.body_joint(k)] = c0 + (c1 - c0) * smooth((t - w0) / (w1 - w0))
        return q

    def settle_command(self, obs):
        return ch.Command(q_target=self.targets(0.0))

    def __call__(self, obs):
        return ch.Command(q_target=self.targets(float(obs.t)))


def ball_base(lab, qpos) -> tuple:
    """Root position that puts the curled robot's lowest geom just on the ground (a static kinematics pass)."""
    import mujoco
    lab.reset(base_pos=(0.0, 0.0, 2.0), qpos=qpos)
    m, d = lab.model, lab.data
    mujoco.mj_kinematics(m, d)
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    low = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in robot)
    return (0.0, 0.0, float(2.0 - low + 0.005))


def summarise(ep, lab, label) -> dict:
    log = ep.log
    names = list(log["joints"])
    body = [names.index(ar.body_joint(k)) for k in range(N)]
    q = np.degrees(np.asarray(log["q"])[:, body])
    tau = np.abs(np.asarray(log["tau"])[:, body])
    stall = np.asarray(log["tau_stall"])[body]
    pos = np.asarray(log["body_pos"])
    gap = float(np.linalg.norm(pos[-1, 0] - pos[-1, -1]))
    return {"scene": label, "final_body_joint_deg": [round(float(x), 1) for x in q[-1]],
            "target_body_joint_deg": ar.CURL_DEG if label == "pack" else 0.0,
            "max_body_torque_Nm": [round(float(x), 1) for x in tau.max(axis=0)],
            "body_torque_saturated_fraction": round(float((tau >= 0.98 * stall).mean()), 3),
            "final_head_to_tail_m": round(gap, 3), "segment_pitch_m": round(ar.PARAMS.pitch, 3),
            "max_tilt_note": "the ball is meant to tilt: no failure rules apply in these scenes"}


def scene(label: str, args) -> dict:
    from vegeta.chiron import viz
    robot = ar.apheloria()
    lab = ch.ChironLab(robot, log_geoms=True, **ar.LAB)
    windows_pack = [(3.0 + 0.4 * k, 6.0 + 0.4 * k) for k in range(N)]                 # neck first
    windows_unpack = [(1.0 + 0.4 * (N - 1 - k), 4.0 + 0.4 * (N - 1 - k)) for k in range(N)]   # tail first
    if label == "pack":
        ctrl = Script(ar.STAND, ar.TUCK, (1.0, 3.0), 0.0, CURL, windows_pack)
        ep = lab.run(ctrl, duration=12.0, rules=None, settle=0.0, qpos=ar.pose(legs=ar.STAND, curl=0.0))
    else:
        start = ar.pose(legs=ar.TUCK, curl=CURL)
        base = ball_base(lab, start)
        ctrl = Script(ar.TUCK, ar.STAND, (7.0, 9.0), CURL, 0.0, windows_unpack)
        ep = lab.run(ctrl, duration=11.0, rules=None, settle=0.0, base_pos=base, qpos=start)
    info = summarise(ep, lab, label)
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = viz.frames(ep, camera=args.camera, every=1, size=(args.width, args.height))
    info["movie"] = str(viz.to_video(imgs, OUT / f"apheloria_{label}.mp4", fps=args.fps))
    print(json.dumps(info, indent=1), flush=True)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("pack", "unpack"))
    ap.add_argument("--camera", default="iso", help="chiron.viz camera: follow, side, front, top, iso")
    ap.add_argument("--fps", type=int, default=25, help="25 = real time at the 0.04 s log step")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args(argv)
    results = [scene(s, args) for s in (("pack", "unpack") if args.only is None else (args.only,))]
    (OUT / "apheloria_pack_unpack.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
