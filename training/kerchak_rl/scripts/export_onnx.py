#!/usr/bin/env python3
"""Export a checkpoint's actor to ONNX (the observation normaliser folded in) with a JSON of the observation
layout and the action's meaning.

    python -m kerchak_rl.scripts.export_onnx --checkpoint training/runs/<run>/model_latest.pt --out training/exports/kerchak_actor.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerchak_rl.config import Config, _from_dict  # noqa: E402
from kerchak_rl.env import ObsLayout  # noqa: E402
from kerchak_rl.export import export_onnx  # noqa: E402
from kerchak_rl.ppo.runner import load_checkpoint  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="export the actor to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="training/exports/kerchak_actor.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args(argv)
    ck = load_checkpoint(a.checkpoint, "cpu")
    cfg = _from_dict(Config, ck["config"])
    from kerchak_rl.commands import COMMAND_DIM
    from kerchak_rl.robot_model import CONTACT_GEOMS
    n_joints = (ck["dims"]["obs"] - ck["dims"]["act"] - 3 - 3 - 3 - len(CONTACT_GEOMS) - COMMAND_DIM) // 2
    layout = ObsLayout(n_joints, ck["dims"]["act"])
    meta = {"scale_rad": cfg.action.scale, "clip": cfg.action.clip, "default_pose": cfg.action.default_pose,
            "note": "joint target = default pose + scale * clip(action); the PD (kp, kd, torque and speed limits) is the servo's"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = export_onnx(ck, out, layout_blocks=layout.blocks, action_meta=meta, opset=a.opset)
    print("wrote", path, "and", path.with_suffix(".json"))


if __name__ == "__main__":
    main()
