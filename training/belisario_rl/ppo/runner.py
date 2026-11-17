"""The training loop: the vectorised environments, the actor-critic, PPO, the observation normaliser; rollouts,
updates, TensorBoard, checkpoints every ``save_every`` iterations and at the end, the configuration snapshot
and the seeds, resuming from a checkpoint (its networks, optimiser, normaliser and iteration count)."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, save_snapshot, to_dict
from .networks import ActorCritic
from .normalizer import RunningMeanStd
from .ppo import PPO

__all__ = ["Runner", "pick_device", "seed_everything", "load_checkpoint"]


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


class Runner:
    def __init__(self, cfg: Config, vec_env, run_dir: str | Path, device=None, session_summary: dict | None = None):
        self.cfg, self.env = cfg, vec_env
        self.device = device or pick_device(cfg.run.device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg.run.seed)
        obs, critic = vec_env.reset()
        self.obs_dim, self.critic_dim, self.act_dim = obs.shape[1], critic.shape[1], vec_env.act_dim
        p = cfg.ppo
        self.ac = ActorCritic(self.obs_dim, self.obs_dim + self.critic_dim, self.act_dim, tuple(p.actor_hidden), tuple(p.critic_hidden),
                              p.activation, p.init_noise_std).to(self.device)
        self.norm = RunningMeanStd(self.obs_dim).to(self.device) if p.normalize_obs else None
        self.cnorm = RunningMeanStd(self.critic_dim).to(self.device) if p.normalize_obs else None
        self.algo = PPO(self.ac, p, self.device)
        self.algo.init_buffer(self.obs_dim, self.obs_dim + self.critic_dim, self.act_dim)
        self.iteration = 0
        self.writer = None
        if cfg.run.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:
                print("tensorboard not installed: no event logs")
        save_snapshot(cfg, self.run_dir)
        if session_summary:
            (self.run_dir / "session.json").write_text(json.dumps(session_summary, indent=1))
        self._obs, self._critic = obs, critic

    # ---- observations
    def _tensors(self, obs, critic):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        c = torch.as_tensor(critic, dtype=torch.float32, device=self.device)
        if self.norm is not None:
            self.norm.update(o)
            self.cnorm.update(c)
            o_n, c_n = self.norm(o), self.cnorm(c)
        else:
            o_n, c_n = o, c
        return o_n, torch.cat([o_n, c_n], dim=-1)

    # ---- training
    def learn(self, iterations: int | None = None, stage: str = ""):
        p = self.cfg.ppo
        iterations = p.iterations if iterations is None else iterations
        obs, critic = self._obs, self._critic
        ep_ret = np.zeros(self.env.num_envs)
        ep_len = np.zeros(self.env.num_envs, dtype=int)
        done_returns, done_lengths, term_counts = [], [], {"fell": 0, "success": 0, "time_limit": 0}
        reward_terms = {}
        t_start = time.time()
        for it in range(self.iteration, self.iteration + iterations):
            t0 = time.time()
            for _ in range(p.steps_per_env):
                o_n, c_n = self._tensors(obs, critic)
                actions = self.algo.act(o_n, c_n)
                obs, critic, rewards, dones, infos = self.env.step(actions.cpu().numpy())
                r = torch.as_tensor(rewards, device=self.device)
                d = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
                # a time-limit end is not a terminal state: bootstrap through it
                for i, info in enumerate(infos):
                    if dones[i] and info.get("time_limit"):
                        with torch.no_grad():
                            fo = torch.as_tensor(info["final_obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
                            fc = torch.as_tensor(info["final_critic_obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
                            if self.norm is not None:
                                fo, fc = self.norm(fo), self.cnorm(fc)
                            r[i] += p.gamma * self.ac.evaluate(torch.cat([fo, fc], dim=-1)).squeeze(0)
                    for k, v in info.get("reward_terms", {}).items():
                        reward_terms[k] = reward_terms.get(k, 0.0) + v
                self.algo.record(o_n, c_n, r, d)
                ep_ret += rewards
                ep_len += 1
                for i in range(self.env.num_envs):
                    if dones[i]:
                        done_returns.append(ep_ret[i])
                        done_lengths.append(ep_len[i])
                        ep_ret[i], ep_len[i] = 0.0, 0
                        info = infos[i]
                        key = "success" if info.get("success") else "fell" if info.get("fell") else "time_limit"
                        term_counts[key] += 1
            o_n, c_n = self._tensors(obs, critic)
            self.algo.finish_rollout(c_n)
            stats = self.algo.update()
            self.iteration = it + 1
            steps = p.steps_per_env * self.env.num_envs
            if self.iteration % p.log_every == 0:
                self._log(stats, done_returns, done_lengths, term_counts, reward_terms, steps, time.time() - t0, stage)
                done_returns, done_lengths = [], []
                term_counts = {k: 0 for k in term_counts}
                reward_terms = {}
            if self.iteration % p.save_every == 0:
                self.save(self.run_dir / f"model_{self.iteration}.pt")
        self.save(self.run_dir / f"model_{self.iteration}.pt")
        self.save(self.run_dir / "model_latest.pt")
        self._obs, self._critic = obs, critic
        print(f"done: {iterations} iterations in {time.time() - t_start:.0f} s; checkpoints in {self.run_dir}")

    def _log(self, stats, returns, lengths, terms, reward_terms, steps, dt, stage):
        it = self.iteration
        fps = steps / max(dt, 1e-6)
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        mean_len = float(np.mean(lengths)) if lengths else float("nan")
        print(f"[{stage}] it {it:5d}  return {mean_ret:8.2f}  length {mean_len:6.1f}  std {stats['action_std']:.3f}  "
              f"vloss {stats['value_loss']:.3f}  kl {stats['kl']:.4f}  lr {stats['lr']:.1e}  {fps:.0f} steps/s  "
              f"fell {terms['fell']} success {terms['success']} timeout {terms['time_limit']}")
        if self.writer is not None:
            w = self.writer
            for k, v in stats.items():
                w.add_scalar(f"ppo/{k}", v, it)
            w.add_scalar("episode/return", mean_ret, it)
            w.add_scalar("episode/length", mean_len, it)
            w.add_scalar("perf/steps_per_s", fps, it)
            for k, v in terms.items():
                w.add_scalar(f"episode/{k}", v, it)
            n = max(steps, 1)
            for k, v in reward_terms.items():
                w.add_scalar(f"reward/{k}", v / n, it)

    # ---- checkpoints
    def save(self, path):
        ck = {"iteration": self.iteration, "actor_critic": self.ac.state_dict(), "algo": self.algo.state_dict(),
              "normalizer": self.norm.state_dict() if self.norm is not None else None,
              "critic_normalizer": self.cnorm.state_dict() if self.cnorm is not None else None,
              "dims": {"obs": self.obs_dim, "critic": self.critic_dim, "act": self.act_dim}, "config": to_dict(self.cfg),
              "stage": self.cfg.run.stage}
        torch.save(ck, path)

    def load(self, path, load_optimizer: bool = True):
        ck = load_checkpoint(path, self.device)
        dims = ck["dims"]
        if (dims["obs"], dims["critic"], dims["act"]) != (self.obs_dim, self.critic_dim, self.act_dim):
            raise ValueError(f"checkpoint dims {dims} do not match the environment ({self.obs_dim}, {self.critic_dim}, {self.act_dim})")
        self.ac.load_state_dict(ck["actor_critic"])
        if self.norm is not None and ck.get("normalizer") is not None:
            self.norm.load_state_dict(ck["normalizer"])
            self.cnorm.load_state_dict(ck["critic_normalizer"])
        if load_optimizer and ck.get("algo"):
            self.algo.load_state_dict(ck["algo"])
        self.iteration = int(ck.get("iteration", 0))
        return ck
