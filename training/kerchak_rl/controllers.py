"""One controller interface for the programmed gaits and the learned policies: ``act(obs, command, t)`` → a
normalised action for the environment (joint-position offsets about the default pose, what ``pd.PDController``
turns into targets), so either can drive ``KerchakEnv`` and both can be compared on the same commands.

* ``ProgrammedGaitController`` plays ``reference.ReferenceGait``: the knuckle walk when the command moves, the quad
  pose when it says stop (it cannot turn or go sideways; the comparison reports that honestly).
* ``PolicyController`` runs the actor (``ppo.networks.ActorCritic``) with its observation normaliser; deterministic
  (the mean) or sampled.
"""
from __future__ import annotations

import numpy as np

from .commands import Command

__all__ = ["Controller", "ProgrammedGaitController", "PolicyController"]


class Controller:
    def reset(self):
        pass

    def act(self, obs: np.ndarray, command: Command, t: float) -> np.ndarray:
        raise NotImplementedError


class ProgrammedGaitController(Controller):
    def __init__(self, reference, default_pose: np.ndarray, action_scale: float):
        self.ref, self.default, self.scale = reference, np.asarray(default_pose, dtype=float), float(action_scale)
        self._t0 = 0.0

    def reset(self):
        self._t0 = 0.0

    def act(self, obs, command: Command, t: float) -> np.ndarray:
        c = command.effective()
        q = self.ref.quad() if c.is_still() else self.ref.walk_at(t)
        return np.clip((q - self.default) / self.scale, -1.0, 1.0).astype(np.float32)


class PolicyController(Controller):
    def __init__(self, actor_critic, normalizer=None, deterministic: bool = True, device: str = "cpu"):
        import torch
        self.torch = torch
        self.ac, self.norm, self.deterministic, self.device = actor_critic, normalizer, deterministic, device
        self.ac.eval()

    def act(self, obs, command: Command, t: float) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
            if self.norm is not None:
                x = self.norm(x)
            a = self.ac.act(x, deterministic=self.deterministic)
        return a.squeeze(0).cpu().numpy()
