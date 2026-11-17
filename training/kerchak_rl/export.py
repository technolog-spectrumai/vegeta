"""The actor to ONNX: a module that folds the observation normaliser in front of the actor's MLP, so the exported
graph takes the raw observation vector (``env.ObsLayout`` order) and gives the action mean (normalised joint
offsets); the layout and the PD's scales go into a JSON next to it so a deployment can rebuild the targets."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .ppo.networks import ActorCritic
from .ppo.normalizer import RunningMeanStd

__all__ = ["ActorForExport", "export_onnx", "build_actor_from_checkpoint"]


class ActorForExport(nn.Module):
    def __init__(self, ac: ActorCritic, norm: RunningMeanStd | None):
        super().__init__()
        self.actor = ac.actor
        self.norm = norm

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            obs = self.norm(obs)
        return self.actor(obs)


def build_actor_from_checkpoint(ck: dict, device="cpu"):
    dims, p = ck["dims"], ck["config"]["ppo"]
    ac = ActorCritic(dims["obs"], dims["obs"] + dims["critic"], dims["act"], tuple(p["actor_hidden"]), tuple(p["critic_hidden"]),
                     p["activation"], p["init_noise_std"]).to(device)
    ac.load_state_dict(ck["actor_critic"])
    norm = None
    if ck.get("normalizer") is not None:
        norm = RunningMeanStd(dims["obs"]).to(device)
        norm.load_state_dict(ck["normalizer"])
        norm.frozen = True
    ac.eval()
    return ac, norm


def export_onnx(ck: dict, path: str | Path, layout_blocks=None, action_meta: dict | None = None, opset: int = 17) -> Path:
    """Write ``path`` (.onnx) and ``path.json`` (the observation layout and the action's meaning)."""
    ac, norm = build_actor_from_checkpoint(ck, "cpu")
    module = ActorForExport(ac, norm).eval()
    dummy = torch.zeros(1, ck["dims"]["obs"])
    path = Path(path)
    torch.onnx.export(module, dummy, str(path), input_names=["obs"], output_names=["action"], opset_version=opset,
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
    meta = {"obs_dim": ck["dims"]["obs"], "act_dim": ck["dims"]["act"], "obs_blocks": layout_blocks, "action": action_meta or {},
            "stage": ck.get("stage"), "iteration": ck.get("iteration"), "normalizer_folded": norm is not None}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path
