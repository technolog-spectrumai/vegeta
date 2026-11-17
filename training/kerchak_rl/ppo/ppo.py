"""The PPO update: a rollout buffer of ``steps_per_env`` × ``num_envs`` transitions with actor and critic
observations, GAE advantages, the clipped surrogate loss with a clipped value loss and an entropy bonus, several
epochs over shuffled minibatches, the learning rate adapted to the measured KL (as RSL-RL does), gradients clipped."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import ActorCritic

__all__ = ["RolloutBuffer", "PPO"]


class RolloutBuffer:
    def __init__(self, steps: int, num_envs: int, obs_dim: int, critic_dim: int, act_dim: int, device):
        self.steps, self.num_envs, self.device = steps, num_envs, device
        z = lambda *shape: torch.zeros(steps, num_envs, *shape, device=device)
        self.obs, self.critic_obs, self.actions = z(obs_dim), z(critic_dim), z(act_dim)
        self.rewards, self.dones, self.values, self.log_probs = z(), z(), z(), z()
        self.mu, self.sigma = z(act_dim), z(act_dim)
        self.advantages, self.returns = z(), z()
        self.step = 0

    def add(self, obs, critic_obs, actions, rewards, dones, values, log_probs, mu, sigma):
        i = self.step
        self.obs[i], self.critic_obs[i], self.actions[i] = obs, critic_obs, actions
        self.rewards[i], self.dones[i], self.values[i], self.log_probs[i] = rewards, dones, values, log_probs
        self.mu[i], self.sigma[i] = mu, sigma
        self.step += 1

    def compute_returns(self, last_values, gamma: float, lam: float):
        adv = torch.zeros(self.num_envs, device=self.device)
        for i in reversed(range(self.steps)):
            next_values = last_values if i == self.steps - 1 else self.values[i + 1]
            not_done = 1.0 - self.dones[i]
            delta = self.rewards[i] + gamma * next_values * not_done - self.values[i]
            adv = delta + gamma * lam * not_done * adv
            self.advantages[i] = adv
        self.returns = self.advantages + self.values
        a = self.advantages
        self.advantages = (a - a.mean()) / (a.std() + 1e-8)
        self.step = 0

    def minibatches(self, n: int, epochs: int):
        total = self.steps * self.num_envs
        size = total // n
        flat = lambda t: t.reshape(total, *t.shape[2:])
        obs, cobs, act = flat(self.obs), flat(self.critic_obs), flat(self.actions)
        val, lp, adv, ret = flat(self.values), flat(self.log_probs), flat(self.advantages), flat(self.returns)
        mu, sigma = flat(self.mu), flat(self.sigma)
        for _ in range(epochs):
            idx = torch.randperm(total, device=self.device)
            for k in range(n):
                b = idx[k * size:(k + 1) * size]
                yield obs[b], cobs[b], act[b], val[b], lp[b], adv[b], ret[b], mu[b], sigma[b]


class PPO:
    def __init__(self, ac: ActorCritic, cfg, device):
        self.ac, self.cfg, self.device = ac, cfg, device
        self.lr = cfg.learning_rate
        self.optimizer = torch.optim.Adam(ac.parameters(), lr=self.lr)
        self.buffer = None

    def init_buffer(self, obs_dim, critic_dim, act_dim):
        self.buffer = RolloutBuffer(self.cfg.steps_per_env, self.cfg.num_envs, obs_dim, critic_dim, act_dim, self.device)

    @torch.no_grad()
    def act(self, obs, critic_obs):
        actions = self.ac.act(obs)
        self._last = (actions, self.ac.evaluate(critic_obs), self.ac.log_prob(actions), self.ac.action_mean.clone(), self.ac.action_std.clone())
        return actions

    def record(self, obs, critic_obs, rewards, dones):
        actions, values, lp, mu, sigma = self._last
        self.buffer.add(obs, critic_obs, actions, rewards, dones, values, lp, mu, sigma)

    @torch.no_grad()
    def finish_rollout(self, last_critic_obs):
        self.buffer.compute_returns(self.ac.evaluate(last_critic_obs), self.cfg.gamma, self.cfg.lam)

    def update(self) -> dict:
        c = self.cfg
        stats = {"value_loss": 0.0, "surrogate_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_fraction": 0.0}
        count = 0
        for obs, cobs, act, old_val, old_lp, adv, ret, old_mu, old_sigma in self.buffer.minibatches(c.minibatches, c.epochs):
            self.ac.update_distribution(obs)
            lp = self.ac.log_prob(act)
            value = self.ac.evaluate(cobs)
            entropy = self.ac.entropy
            if c.adaptive_lr:
                with torch.no_grad():
                    mu, sigma = self.ac.action_mean, self.ac.action_std
                    kl = torch.sum(torch.log(sigma / old_sigma + 1e-5) + (old_sigma ** 2 + (old_mu - mu) ** 2) / (2.0 * sigma ** 2) - 0.5, dim=-1).mean()
                    if kl > c.desired_kl * 2.0:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < c.desired_kl / 2.0 and kl > 0.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.lr
                    stats["kl"] += float(kl)
            ratio = torch.exp(lp - old_lp)
            surr = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - c.clip, 1.0 + c.clip) * adv).mean()
            v_clipped = old_val + (value - old_val).clamp(-c.value_clip, c.value_clip)
            v_loss = torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
            loss = surr + c.value_coef * v_loss - c.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
            self.optimizer.step()
            stats["value_loss"] += float(v_loss)
            stats["surrogate_loss"] += float(surr)
            stats["entropy"] += float(entropy.mean())
            stats["clip_fraction"] += float(((ratio - 1.0).abs() > c.clip).float().mean())
            count += 1
        for k in stats:
            stats[k] /= max(count, 1)
        stats["lr"] = self.lr
        stats["action_std"] = float(torch.exp(self.ac.log_std).mean())
        return stats

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "lr": self.lr}

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d["optimizer"])
        self.lr = d.get("lr", self.lr)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr
