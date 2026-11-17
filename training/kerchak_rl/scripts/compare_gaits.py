#!/usr/bin/env python3
"""Compare the programmed knuckle walk with a learned policy on the same commands and scene: the distance walked,
the fall rate, the speed tracking error, the torque saturation and the energy per metre, side by side.

    python -m kerchak_rl.scripts.compare_gaits --checkpoint training/runs/<run>/model_latest.pt --episodes 3
    python -m kerchak_rl.scripts.compare_gaits --programmed-only                                   # the reference alone
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

from kerchak_rl.commands import Command  # noqa: E402
from kerchak_rl.config import Config, _from_dict, load_config  # noqa: E402
from kerchak_rl.controllers import PolicyController, ProgrammedGaitController  # noqa: E402
from kerchak_rl.env import KerchakEnv  # noqa: E402
from kerchak_rl.export import build_actor_from_checkpoint  # noqa: E402
from kerchak_rl.ppo.runner import load_checkpoint  # noqa: E402
from kerchak_rl.reference import ReferenceGait  # noqa: E402
from kerchak_rl.scenes import build_scene  # noqa: E402


def run(env, ctrl, cmd, episodes, seed):
    out = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep, options={"command": cmd})
        ctrl.reset()
        t, done, x0 = 0.0, False, env.base_state()[0][0]
        energy, sat, vx = 0.0, [], []
        while not done:
            a = ctrl.act(obs, cmd, t)
            obs, r, term, trunc, info = env.step(a)
            t += env.dt
            q = env.data.qvel[env.joints.dof_adr]
            energy += float(np.sum(np.abs(env.pd.last_torque * q))) * env.dt
            sat.append(float(np.max(env.pd.saturation(q))))
            vx.append(info["lin_vel"][0])
            done = term or trunc
        walked = env.base_state()[0][0] - x0
        out.append({"walked_m": float(walked), "fell": bool(info.get("fell")), "seconds": t, "energy_J_per_m": float(energy / max(walked, 0.05)),
                    "saturation_max": float(np.max(sat)), "vx_mean": float(np.mean(vx)), "vx_rmse": float(np.sqrt(np.mean((np.array(vx) - cmd.effective().vx) ** 2)))})
    return {"episodes": out, "walked_m": float(np.mean([o["walked_m"] for o in out])), "fall_rate": float(np.mean([o["fell"] for o in out])),
            "energy_J_per_m": float(np.mean([o["energy_J_per_m"] for o in out])), "vx_rmse": float(np.mean([o["vx_rmse"] for o in out]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="programmed gait vs learned policy")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--programmed-only", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.checkpoint:
        ck = load_checkpoint(a.checkpoint, a.device)
        cfg = _from_dict(Config, ck["config"])
    else:
        cfg = load_config(a.config)
    env = KerchakEnv(cfg, scene=build_scene(a.scene, cfg), seed=99, eval_mode=True)
    ref = ReferenceGait(env.robot)
    cmd = Command(a.vx if a.vx is not None else ref.speed, 0.0, 0.0, ref.height, False)
    result = {"command": cmd.__dict__, "programmed": run(env, ProgrammedGaitController(ref, env.default["q"], cfg.action.scale), cmd, a.episodes, 99)}
    if a.checkpoint and not a.programmed_only:
        ac, norm = build_actor_from_checkpoint(ck, a.device)
        result["learned"] = run(env, PolicyController(ac, norm, True, a.device), cmd, a.episodes, 99)
    text = json.dumps(result, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    env.close()


if __name__ == "__main__":
    main()
