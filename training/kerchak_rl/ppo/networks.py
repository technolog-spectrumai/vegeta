"""The actor-critic: the actor an MLP 256 → 256 → 128 with ELU on the deployable observations, giving the mean of
a Gaussian over normalised joint-position offsets (a learned, state-independent log-std); the critic a separate
MLP of the same shape on the actor's observations plus the privileged ones."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

__all__ = ["ActorCritic", "mlp"]

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


def mlp(inp: int, hidden, out: int, activation: str = "elu") -> nn.Sequential:
    act = ACTIVATIONS[activation.lower()]
    layers, last = [], inp
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, critic_obs_dim: int, act_dim: int, actor_hidden=(256, 256, 128), critic_hidden=(256, 256, 128),
                 activation: str = "elu", init_noise_std: float = 0.8):
        super().__init__()
        self.obs_dim, self.critic_obs_dim, self.act_dim = obs_dim, critic_obs_dim, act_dim
        self.actor = mlp(obs_dim, actor_hidden, act_dim, activation)
        self.critic = mlp(critic_obs_dim, critic_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_noise_std)))))
        self.distribution = None

    def update_distribution(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.mean if deterministic else self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
