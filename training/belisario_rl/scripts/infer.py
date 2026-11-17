#!/usr/bin/env python3
"""Deterministic inference: run a checkpoint in the environment on a command for a number of seconds, headless,
printing the base's state and the joint targets along the way (what a deployment loop does, without the robot).

    python -m belisario_rl.scripts.infer --checkpoint training/runs/<run>/model_latest.pt --vx 0.3 --seconds 10
    python -m belisario_rl.scripts.infer --checkpoint ... --onnx training/exports/belisario_actor.onnx   # through onnxruntime
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from belisario_rl.commands import Command  # noqa: E402
from belisario_rl.config import Config, _from_dict  # noqa: E402
from belisario_rl.controllers import Controller, PolicyController  # noqa: E402
from belisario_rl.env import BelisarioEnv  # noqa: E402
from belisario_rl.export import build_actor_from_checkpoint  # noqa: E402
from belisario_rl.ppo.runner import load_checkpoint  # noqa: E402
from belisario_rl.scenes import build_scene  # noqa: E402


class OnnxController(Controller):
    """The exported actor through onnxruntime (the normaliser is folded into the graph)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def act(self, obs, command, t):
        out = self.sess.run(None, {self.name: np.asarray(obs, dtype=np.float32)[None, :]})[0]
        return np.clip(out[0], -1.0, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="deterministic inference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", default=None, help="run this exported actor instead of the checkpoint's network")
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=0.62)
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--print-every", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    ck = load_checkpoint(a.checkpoint, a.device)
    cfg = _from_dict(Config, ck["config"])
    env = BelisarioEnv(cfg, scene=build_scene(a.scene, cfg), seed=123, eval_mode=True)
    if a.onnx:
        ctrl = OnnxController(a.onnx)
    else:
        ac, norm = build_actor_from_checkpoint(ck, a.device)
        ctrl = PolicyController(ac, norm, deterministic=True, device=a.device)
    cmd = Command(a.vx, a.vy, a.yaw, a.height, a.stop)
    obs, info = env.reset(seed=123, options={"command": cmd})
    t, nxt = 0.0, 0.0
    while t < a.seconds:
        act = ctrl.act(obs, cmd, t)
        obs, r, term, trunc, info = env.step(act)
        t += env.dt
        if t >= nxt:
            pos, vel = env.base_state()
            print(f"t {t:5.2f}  x {pos[0]:6.3f}  z {pos[2]:5.3f}  vx {info['lin_vel'][0]:6.3f}  contacts {info['contacts'].astype(int)}  reward {r:6.3f}")
            nxt += a.print_every
        if term or trunc:
            print("episode ended:", "fell" if info.get("fell") else "time limit" if trunc else "done")
            break
    print("final joint targets [rad]:", np.round(env.joint_targets(), 3).tolist())
    env.close()


if __name__ == "__main__":
    main()
