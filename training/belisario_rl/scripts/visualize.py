#!/usr/bin/env python3
"""Watch a policy in MuJoCo's viewer (``mujoco.viewer.launch_passive``), or record a movie headless with the
geoms drawn by matplotlib the way the lab's sims do.

    python -m belisario_rl.scripts.visualize --checkpoint training/runs/<run>/model_latest.pt --vx 0.3
    python -m belisario_rl.scripts.visualize --checkpoint ... --movie training/runs/<run>/walk.mp4 --seconds 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from belisario_rl.commands import Command  # noqa: E402
from belisario_rl.config import Config, _from_dict  # noqa: E402
from belisario_rl.controllers import PolicyController  # noqa: E402
from belisario_rl.env import BelisarioEnv  # noqa: E402
from belisario_rl.export import build_actor_from_checkpoint  # noqa: E402
from belisario_rl.ppo.runner import load_checkpoint  # noqa: E402
from belisario_rl.scenes import build_scene  # noqa: E402


def record_movie(env, ctrl, cmd, seconds, path, fps=20):
    sys.path.insert(0, str(ROOT.parent / "labs" / "legged" / "sims"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import space_squad_march as march_sim
    obs, info = env.reset(options={"command": cmd})
    frames, t = [], 0.0
    while t < seconds:
        a = ctrl.act(obs, cmd, t)
        obs, r, term, trunc, info = env.step(a)
        t += env.dt
        class _U:
            model, data = env.model, env.data
        geoms, centre = march_sim.snapshot([_U])
        frames.append((t, geoms, centre))
        if term or trunc:
            break
    tmp = tempfile.mkdtemp()
    for k, (t, geoms, centre) in enumerate(frames):
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        march_sim.draw_frame(ax, geoms, centre, t, "the learned policy", 2.6, elev=12.0, azim=62.0)
        fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
        fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=72)
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(path)], check=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="visualize a Belisario policy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=0.62)
    ap.add_argument("--movie", default=None, help="record to this mp4 instead of opening the viewer")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    ck = load_checkpoint(a.checkpoint, a.device)
    cfg = _from_dict(Config, ck["config"])
    env = BelisarioEnv(cfg, scene=build_scene(a.scene, cfg), seed=7, eval_mode=True, render_mode=None if a.movie else "human")
    ac, norm = build_actor_from_checkpoint(ck, a.device)
    ctrl = PolicyController(ac, norm, deterministic=True, device=a.device)
    cmd = Command(a.vx, a.vy, a.yaw, a.height, False)
    if a.movie:
        print("wrote", record_movie(env, ctrl, cmd, a.seconds, a.movie))
        return
    obs, info = env.reset(options={"command": cmd})
    t = 0.0
    while t < a.seconds:
        act = ctrl.act(obs, cmd, t)
        obs, r, term, trunc, info = env.step(act)
        t += env.dt
        time.sleep(env.dt)
        if term or trunc:
            obs, info = env.reset(options={"command": cmd})
    env.close()


if __name__ == "__main__":
    main()
