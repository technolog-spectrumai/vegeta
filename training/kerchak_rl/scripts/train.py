#!/usr/bin/env python3
"""Train Kerchak's knuckle-walking policy with PPO.

    python -m kerchak_rl.scripts.train --session training/configs/session.yaml --stage 1_stance
    python -m kerchak_rl.scripts.train --session training/configs/session.yaml --stage 3_walk --resume training/runs/<run>/model_latest.pt
    python -m kerchak_rl.scripts.train --config training/configs/base.yaml --stages all      # every stage in turn
    python -m kerchak_rl.scripts.train ... --set ppo.num_envs=4 --set run.device=cpu

A session (``--session``) names the scenes and their weights; ``--config`` alone trains on the flat scene. Each
stage gets its own run directory under ``run.log_dir``: ``<name>/<stage>_<timestamp>/`` with the configuration
snapshot, the session summary, TensorBoard events and the checkpoints. When several stages run in turn, each
starts from the previous stage's last checkpoint.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerchak_rl.config import apply_overrides, load_config  # noqa: E402
from kerchak_rl.curriculum import STAGE_ORDER, apply_stage  # noqa: E402
from kerchak_rl.env import KerchakEnv  # noqa: E402
from kerchak_rl.ppo.runner import Runner, pick_device  # noqa: E402
from kerchak_rl.scenes import build_scene  # noqa: E402
from kerchak_rl.session import assign_scenes, load_session, session_summary  # noqa: E402
from kerchak_rl.vec_env import make_vec_env  # noqa: E402


def make_env_fns(cfg, session, num_envs, seed):
    """One environment per index: its scene from the session's weights (flat when there is no session)."""
    entries = assign_scenes(session, num_envs) if session else [None] * num_envs

    def fn(i, entry):
        def make():
            scene = build_scene(entry.name, cfg, entry.params) if entry else build_scene(cfg.terrain.kind if cfg.terrain.kind in ("flat", "rough", "steps") else "flat", cfg)
            return KerchakEnv(cfg, scene=scene, seed=seed + i)
        return make
    return [fn(i, e) for i, e in enumerate(entries)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO training for Kerchak")
    ap.add_argument("--session", default=None, help="session.yaml: the scenes and their weights")
    ap.add_argument("--config", default=None, help="a Config YAML (when there is no session)")
    ap.add_argument("--stage", default=None, help="a curriculum stage (default: the session's stages, or 1_stance)")
    ap.add_argument("--stages", default=None, help="'all' or a comma-separated list, run in turn, each from the last checkpoint")
    ap.add_argument("--resume", default=None, help="a checkpoint to start from")
    ap.add_argument("--iterations", type=int, default=None, help="override the stage's iteration count")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-subprocess", action="store_true", help="environments in the main process (debugging)")
    ap.add_argument("--set", action="append", default=[], help="override section.key=value")
    a = ap.parse_args(argv)
    if a.session:
        session = load_session(a.session)
        cfg = session.config
    else:
        session = None
        cfg = load_config(a.config)
    cfg = apply_overrides(cfg, a.set)
    if a.num_envs:
        cfg = apply_overrides(cfg, [f"ppo.num_envs={a.num_envs}"])
    if a.device:
        cfg = apply_overrides(cfg, [f"run.device={a.device}"])
    if a.seed is not None:
        cfg = apply_overrides(cfg, [f"run.seed={a.seed}"])
    if a.stages:
        stages = STAGE_ORDER if a.stages == "all" else a.stages.split(",")
    elif a.stage:
        stages = [a.stage]
    elif session and session.stages:
        stages = session.stages
    else:
        stages = [cfg.run.stage]
    device = pick_device(cfg.run.device)
    print(f"device {device}; stages {stages}; scenes {[s.name for s in session.scenes] if session else ['flat']}")
    resume = a.resume
    for stage in stages:
        scfg = apply_stage(cfg, stage)
        if a.iterations:
            scfg = apply_overrides(scfg, [f"ppo.iterations={a.iterations}"])
        run_dir = Path(scfg.run.log_dir) / scfg.run.name / f"{stage}_{time.strftime('%Y%m%d-%H%M%S')}"
        fns = make_env_fns(scfg, session, scfg.ppo.num_envs, scfg.run.seed)
        env = make_vec_env(fns, subprocess=not a.no_subprocess)
        try:
            runner = Runner(scfg, env, run_dir, device, session_summary(session) if session else None)
            if resume:
                ck = runner.load(resume, load_optimizer=False)
                runner.iteration = 0
                print(f"resumed from {resume} (stage {ck.get('stage')}, iteration {ck.get('iteration')})")
            runner.learn(stage=stage)
            resume = str(run_dir / "model_latest.pt")
        finally:
            env.close()
    print(f"last checkpoint: {resume}")


if __name__ == "__main__":
    main()
