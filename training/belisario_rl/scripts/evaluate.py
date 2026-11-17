#!/usr/bin/env python3
"""Evaluate a checkpoint: deterministic episodes on fixed commands, with the fall rate, the tracking errors and
the distance walked, as JSON.

    python -m belisario_rl.scripts.evaluate --checkpoint training/runs/<run>/model_latest.pt --episodes 10
    python -m belisario_rl.scripts.evaluate --checkpoint ... --vx 0.3 --yaw 0.2 --scene rough
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from belisario_rl.commands import Command  # noqa: E402
from belisario_rl.config import _from_dict, Config, apply_overrides  # noqa: E402
from belisario_rl.controllers import PolicyController  # noqa: E402
from belisario_rl.env import BelisarioEnv  # noqa: E402
from belisario_rl.export import build_actor_from_checkpoint  # noqa: E402
from belisario_rl.ppo.runner import load_checkpoint  # noqa: E402
from belisario_rl.scenes import build_scene  # noqa: E402


def run_episodes(env, controller, episodes: int, command: Command, seed: int = 0) -> dict:
    """``episodes`` deterministic episodes on ``command``; the summary."""
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep, options={"command": command})
        controller.reset()
        t, ret, done, steps = 0.0, 0.0, False, 0
        x0 = env.base_state()[0][0]
        vx_err, wz_err, heights = [], [], []
        while not done:
            a = controller.act(obs, command, t)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            t += env.dt
            steps += 1
            vx_err.append(info["lin_vel"][0] - command.effective().vx)
            heights.append(info["height"])
            done = term or trunc
        x1 = env.base_state()[0][0]
        results.append({"return": ret, "steps": steps, "fell": bool(info.get("fell")), "success": bool(info.get("success")),
                        "walked_m": float(x1 - x0), "vx_rmse": float(np.sqrt(np.mean(np.square(vx_err)))), "height_mean": float(np.mean(heights))})
    return {"episodes": results, "fall_rate": float(np.mean([r["fell"] for r in results])),
            "mean_return": float(np.mean([r["return"] for r in results])), "mean_walked_m": float(np.mean([r["walked_m"] for r in results])),
            "mean_vx_rmse": float(np.mean([r["vx_rmse"] for r in results]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="evaluate a Belisario policy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=0.62)
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the JSON here")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    ck = load_checkpoint(a.checkpoint, a.device)
    cfg = apply_overrides(_from_dict(Config, ck["config"]), a.set)
    env = BelisarioEnv(cfg, scene=build_scene(a.scene, cfg), seed=cfg.run.seed + 1000, eval_mode=True)
    ac, norm = build_actor_from_checkpoint(ck, a.device)
    ctrl = PolicyController(ac, norm, deterministic=not a.stochastic, device=a.device)
    cmd = Command(a.vx, a.vy, a.yaw, a.height, a.stop)
    summary = run_episodes(env, ctrl, a.episodes, cmd, seed=cfg.run.seed + 1000)
    summary["command"] = cmd.__dict__
    summary["checkpoint"] = a.checkpoint
    text = json.dumps(summary, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    env.close()


if __name__ == "__main__":
    main()
