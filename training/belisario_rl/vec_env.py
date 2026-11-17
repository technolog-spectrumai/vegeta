"""Vectorised environments: ``DummyVecEnv`` steps its environments in turn in one process (debugging, evaluation);
``SubprocVecEnv`` runs each in its own process (the CPU laptop's parallelism: MuJoCo is single-threaded per
instance). Both auto-reset an environment that ends and hand its last observation back in ``infos``.

The interface: ``reset()`` → (obs [N, obs_dim], critic_obs [N, priv_dim]); ``step(actions [N, act_dim])`` →
(obs, critic_obs, rewards [N], dones [N], infos [N dicts]); ``close()``; ``num_envs``, ``obs_dim``, ``critic_dim``,
``act_dim``."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

__all__ = ["DummyVecEnv", "SubprocVecEnv", "make_vec_env"]


class DummyVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        e = self.envs[0]
        self.obs_dim, self.act_dim = e.obs_dim, e.act_dim
        self.critic_dim = None

    def reset(self):
        obs, crit = [], []
        for e in self.envs:
            o, info = e.reset()
            obs.append(o)
            crit.append(info["critic_obs"])
        self.critic_dim = len(crit[0])
        return np.stack(obs), np.stack(crit)

    def step(self, actions):
        obs, crit, rews, dones, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            done = term or trunc
            info = dict(info)
            info["time_limit"] = bool(trunc and not term)
            if done:
                info["final_obs"] = o
                info["final_critic_obs"] = info["critic_obs"]
                o, ri = e.reset()
                info["critic_obs"] = ri["critic_obs"]
            obs.append(o)
            crit.append(info["critic_obs"])
            rews.append(r)
            dones.append(done)
            infos.append(info)
        return np.stack(obs), np.stack(crit), np.array(rews, dtype=np.float32), np.array(dones, dtype=bool), infos

    def close(self):
        for e in self.envs:
            e.close()


def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                o, r, term, trunc, info = env.step(data)
                done = term or trunc
                info = dict(info)
                info["time_limit"] = bool(trunc and not term)
                if done:
                    info["final_obs"] = o
                    info["final_critic_obs"] = info["critic_obs"]
                    o, ri = env.reset()
                    info["critic_obs"] = ri["critic_obs"]
                remote.send((o, info["critic_obs"], r, done, info))
            elif cmd == "reset":
                o, info = env.reset()
                remote.send((o, info["critic_obs"]))
            elif cmd == "dims":
                remote.send((env.obs_dim, env.act_dim))
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class SubprocVecEnv:
    def __init__(self, env_fns, start_method: str | None = None):
        ctx = mp.get_context(start_method or ("fork" if hasattr(mp, "get_context") else None))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
        self.procs = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(wr, r, fn), daemon=True)
            p.start()
            self.procs.append(p)
            wr.close()
        self.num_envs = len(env_fns)
        self.remotes[0].send(("dims", None))
        self.obs_dim, self.act_dim = self.remotes[0].recv()
        self.critic_dim = None

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [r.recv() for r in self.remotes]
        obs, crit = zip(*res)
        self.critic_dim = len(crit[0])
        return np.stack(obs), np.stack(crit)

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        res = [r.recv() for r in self.remotes]
        obs, crit, rews, dones, infos = zip(*res)
        return np.stack(obs), np.stack(crit), np.array(rews, dtype=np.float32), np.array(dones, dtype=bool), list(infos)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)


def make_vec_env(env_fns, subprocess: bool = True):
    if subprocess and len(env_fns) > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
