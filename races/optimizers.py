"""The optimisation strategies of the autopilot tuner (``races.tune``), interchangeable behind one interface (the
strategy pattern): ``--cma`` (the default) or ``--pso``.

A strategy searches the six autopilot settings (``races.converge.BOUNDS``, normalised to [0, 1]) to minimise an
objective (``races.laps.Laps``: the robust score of a candidate over the training scenarios). The driver only calls
``step`` until ``done``, saves ``state()`` after every step and rebuilds the strategy from it to resume; ``best()``
is the best candidate evaluated. Every strategy is deterministic for its seed, so a resumed run and ``-j N`` fly the
same candidates as an uninterrupted ``-j 1`` run.

* ``CMAStrategy`` — CMA-ES (Hansen & Ostermeier 2001; Hansen 2016, "The CMA evolution strategy: a tutorial"), the
  PyPI package ``cma``: a population sampled from a multivariate normal whose mean, step size and covariance adapt to
  the ranking of the candidates; the box handled by ``cma``'s boundary transformation. Suited to noisy, rugged,
  low-dimensional black boxes, which the robust lap score is. It stops on ``cma``'s own criteria (``tolfun``,
  ``tolx``, stagnation, …), ``--generations`` or the evaluation budget. It races each generation (``race``):
  candidates that cannot reach the best μ are not flown on every scenario.
* ``PSOStrategy`` — the earlier search: a particle swarm (``races.converge.PSO``), then a generalized pattern search
  polish (``races.converge.GPS``); ``--pso-iterations 0`` polishes the stored settings only, ``--polish-evals 0`` is
  the swarm only.

A strategy's options are the fields of its ``Config`` dataclass (``opt`` gives each its default and help); the
driver turns them into command-line options. Neither strategy proves a global optimum."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import pickle
from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

import numpy as np

from races import converge


class Objective(Protocol):
    """What a strategy may ask of the objective (``races.laps.Laps``): ``train`` the scenarios (index 0 the nominal
    lap), ``m`` the robust score's tail (the worst m laps), ``jobs`` the laps flown side by side."""

    train: list
    m: int
    jobs: int

    def prefetch(self, settings: list[dict], stage: str, iteration: int, cut: bool = False, js=None) -> int: ...

    def score(self, settings: dict, js=None) -> float: ...


def opt(default, help: str):
    """A strategy option: a dataclass field with its command-line help."""
    return dataclasses.field(default=default, metadata={"help": help})


@dataclasses.dataclass(frozen=True)
class Common:
    seed: int = opt(0, "the strategy's random seed (the same seed and options repeat a run exactly)")
    max_evals: int = opt(300, "the evaluation budget per course: candidates scored (each over the nominal lap and the training scenarios)")


def _x(v) -> dict:
    return {k: float(a) for k, a in zip(converge.KEYS, v)}


def race(objective: Objective, S: list[dict], keep: int, stage: str, iteration: int, log=print) -> tuple[list[float], int]:
    """Racing (after Birattari et al.'s F-race, 2002; Maron & Moore 1997) on the robust score's own bound: the scores
    of the candidates ``S`` — exact for the best ``keep`` (up to exact ties), and for the others the lower bound from
    the first scenarios, which is at least the ``keep``-th best exact score. Returns (scores, new evaluations).

    1. Every candidate flies the first ⌈(K+1)/3⌉ scenarios (the nominal lap first). The robust score is the mean of
       the worst ⌈(K+1)/3⌉ laps, and the worst m of a superset are never better than those of a subset, so this
       partial score is a lower bound of the full one (``races.validate.worst_mean``).
    2. In the order of these bounds, batches of candidates fly their remaining scenarios (as many laps as ``jobs``
       per batch); T is the ``keep``-th best full score so far. A candidate whose bound is ≥ T cannot be in the best
       ``keep``, nor can any after it: their remaining laps are not flown.

    Every candidate outside the best ``keep`` reports max(its stage-1 bound, the ``keep``-th best score) + 10⁻⁶: it
    ranks after the best ``keep`` (whose exact scores are all an update with zero weights beyond μ uses), and the
    value does not depend on the batches, so ``-j`` changes the laps flown, never a decision."""
    n, m = len(objective.train), objective.m
    first = list(range(min(m, n)))
    new = objective.prefetch(S, stage, iteration, js=first)
    if len(first) == n or keep >= len(S):
        new += objective.prefetch(S, stage, iteration)
        return [objective.score(s) for s in S], new
    lb = [objective.score(s, first) for s in S]
    order = sorted(range(len(S)), key=lambda i: lb[i])
    rest = list(range(len(first), n))
    per = max(1, -(-getattr(objective, "jobs", 1) // len(rest)))     # candidates per batch: about one lap per worker
    full, k = {}, 0
    while k < len(order):
        T = sorted(full.values())[keep - 1] if len(full) >= keep else float("inf")
        batch = [i for i in order[k:k + per] if lb[i] < T]
        if not batch:
            break
        objective.prefetch([S[i] for i in batch], stage, iteration, js=rest)
        full.update({i: objective.score(S[i]) for i in batch})
        k += len(batch)
        if len(batch) < per:
            break
    ranked = sorted(full, key=lambda i: full[i])
    best, t_keep = set(ranked[:keep]), full[ranked[keep - 1]]
    log(f"{stage} {iteration}: racing — {len(full)} of {len(S)} candidates flown on all {n} scenarios, "
        f"{(len(S) - len(full)) * len(rest)} laps saved")
    return [full[i] if i in best else max(lb[i], t_keep) + 1e-6 for i in range(len(S))], new


class Strategy(ABC):
    name: ClassVar[str]
    summary: ClassVar[str]
    Config: ClassVar[type]

    def __init__(self, start: dict, cfg):
        self.start, self.cfg = start, cfg
        self.done, self.reason, self.evals_used = False, "", 0

    @property
    def phase(self) -> str:
        """The current phase's name (the driver times each phase)."""
        return self.name

    @abstractmethod
    def step(self, objective: Objective, log=print) -> None:
        """One iteration (a generation, a swarm move, a poll); sets ``done`` and ``reason`` when it stops."""

    @abstractmethod
    def best(self) -> dict:
        """The best settings evaluated so far."""

    @abstractmethod
    def state(self) -> dict:
        """Everything needed to continue exactly here (JSON-compatible)."""

    @classmethod
    @abstractmethod
    def restore(cls, start: dict, cfg, state: dict) -> "Strategy":
        """The strategy rebuilt from ``state()``."""

    def _base_state(self) -> dict:
        return {"done": self.done, "reason": self.reason, "evals_used": self.evals_used}

    def _load_base(self, st: dict):
        self.done, self.reason, self.evals_used = st["done"], st["reason"], st["evals_used"]


# ---------------------------------------------------------------------------------------------------------- CMA-ES
@dataclasses.dataclass(frozen=True)
class CMAConfig(Common):
    popsize: int = opt(0, "CMA-ES candidates per generation (0 = cma's default, 4 + ⌊3 ln 6⌋ = 9)")
    sigma0: float = opt(0.2, "CMA-ES initial step size, a share of every setting's range")
    generations: int = opt(40, "CMA-ES hard maximum of generations")
    tolfun: float = opt(0.1, "CMA-ES stop: the recent generations' scores span less than this [score ≈ s]")
    tolx: float = opt(1e-3, "CMA-ES stop: the search steps are below this share of every range")
    race: bool = opt(True, "CMA-ES: race each generation — fly the remaining scenarios only for candidates that can still reach the best μ")


_STOPS = {"tolfun": "the recent generations' scores span less than {v}", "tolfunhist": "the recent best scores span less than {v}",
          "tolx": "the steps are below {v} of every range", "tolstagnation": "no progress in the median score for {v} generations",
          "maxiter": "the hard maximum of {v} generations", "maxfevals": "the evaluation budget ({v})", "tolflatfitness": "a flat score for {v} generations"}


class _Draws:
    """The normal draws of one generation, seeded by (seed, generation): ``cma`` would otherwise use NumPy's global
    generator, which a pickled state does not carry, so a resumed run would sample other candidates."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, *shape):
        return self.rng.standard_normal(shape)


class CMAStrategy(Strategy):
    name = "cma"
    summary = "CMA-ES (the PyPI package cma)"
    Config = CMAConfig

    def __init__(self, start: dict, cfg: CMAConfig):
        import cma
        super().__init__(start, cfg)
        x0 = [converge.to_x(start)[k] for k in converge.KEYS]
        opts = {"bounds": [0.0, 1.0], "seed": cfg.seed + 1, "verbose": -9, "maxiter": cfg.generations, "maxfevals": cfg.max_evals,
                "tolfun": cfg.tolfun, "tolx": cfg.tolx,                   # cma's seed 0 means "random": hence seed + 1
                "CMA_active": not cfg.race}                               # racing: the update must use only the best μ (exact ranks)
        if cfg.popsize:
            opts["popsize"] = cfg.popsize
        self.es = cma.CMAEvolutionStrategy(x0, cfg.sigma0, opts)
        self.gen, self.best_s, self.best_f = 0, None, None
        self.history = []                                                  # per generation: mean, σ, C (the plots of races.trainviz)

    def step(self, objective: Objective, log=print) -> None:
        self.es.opts["randn"] = _Draws([self.cfg.seed, self.gen])
        X = self.es.ask()
        S = [converge.to_settings(_x(x)) for x in X]
        if self.cfg.race:
            F, new = race(objective, S, int(self.es.sp.weights.mu), "cma", self.gen, log=log)
        else:
            new = objective.prefetch(S, "cma", self.gen)
            F = [objective.score(s) for s in S]
        self.evals_used += new
        self.es.tell(X, F)
        self.history.append({"gen": self.gen, "mean": [float(v) for v in self.es.mean], "sigma": float(self.es.sigma),
                             "C": np.asarray(self.es.C, dtype=float).round(10).tolist(), "best": float(min(F)), "median": float(np.median(F))})
        i = int(np.argmin(F))
        improved = self.best_f is None or F[i] < self.best_f - 1e-9
        if improved:
            self.best_s, self.best_f = S[i], float(F[i])
        log(f"cma generation {self.gen}: best {min(F):.2f} (overall {self.best_f:.2f}{', improved' if improved else ''}), "
            f"median {float(np.median(F)):.2f}{' (dropped candidates count with their bound)' if self.cfg.race else ''}, "
            f"σ {self.es.sigma:.3f}, evaluations {self.evals_used}")
        self.gen += 1
        stop = {k: v for k, v in self.es.stop().items() if k != "callback"}
        if stop:
            self.done = True
            why = "; ".join(_STOPS.get(k, k + " {v}").format(v=v) for k, v in stop.items())
            self.reason = f"CMA-ES stopped after {self.gen} generations: {why} — a local result, not a proven global optimum"
            log(f"cma: stopped — {self.reason}")

    def best(self) -> dict:
        return self.best_s if self.best_s is not None else converge.to_settings(converge.to_x(self.start))

    def state(self) -> dict:
        self.es.opts["randn"] = None                                       # the next step sets the generation's draws
        return {**self._base_state(), "gen": self.gen, "best_s": self.best_s, "best_f": self.best_f, "history": self.history,
                "es": base64.b64encode(pickle.dumps(self.es)).decode()}

    @classmethod
    def restore(cls, start, cfg, state) -> "CMAStrategy":
        s = cls.__new__(cls)
        Strategy.__init__(s, start, cfg)
        s._load_base(state)
        s.gen, s.best_s, s.best_f, s.history = state["gen"], state["best_s"], state["best_f"], state.get("history", [])
        s.es = pickle.loads(base64.b64decode(state["es"]))
        return s


# ----------------------------------------------------------------------------------------- PSO, then pattern search
@dataclasses.dataclass(frozen=True)
class PSOConfig(Common):
    particles: int = opt(12, "PSO swarm size")
    pso_iterations: int = opt(15, "PSO hard maximum of iterations, then the polish (0 = polish the stored settings only)")
    polish_evals: int = opt(120, "PSO: evaluations of --max-evals reserved for the pattern-search polish (0 = the swarm only)")
    delta_min: float = opt(1 / 64, "PSO polish convergence: the mesh size (a share of every range) below which it stops")
    patience: int = opt(0, "PSO polish: optional heuristic stop after this many iterations gaining < --tol (0 = off)")
    tol: float = opt(0.2, "PSO polish: see --patience")


class PSOStrategy(Strategy):
    name = "pso"
    summary = "a particle swarm, then a pattern-search polish"
    Config = PSOConfig

    def __init__(self, start: dict, cfg: PSOConfig):
        super().__init__(start, cfg)
        self.pso = (converge.PSO(start, particles=cfg.particles, iterations=cfg.pso_iterations, seed=cfg.seed,
                                 budget=max(cfg.particles, cfg.max_evals - cfg.polish_evals)) if cfg.pso_iterations else None)
        self.gps = None

    @property
    def phase(self) -> str:
        return "pso" if self.pso is not None and not self.pso.done else "polish"

    def step(self, objective: Objective, log=print) -> None:
        if self.phase == "pso":
            self.pso.step(objective, log=log)
            if self.pso.done and not self.cfg.polish_evals:
                self.done, self.reason = True, f"PSO only: {self.pso.reason}"
            return
        if self.gps is None:
            origin = self.pso.best() if self.pso else self.start
            used = self.pso.evals_used if self.pso else 0
            self.gps = converge.GPS(origin, delta_min=self.cfg.delta_min, budget=max(self.cfg.polish_evals, self.cfg.max_evals - used),
                                    patience=self.cfg.patience, tol=self.cfg.tol)
            log(f"polishing from {'the PSO best' if self.pso else 'the stored settings'}: {origin}")
        self.gps.step(objective, log=log)
        self.evals_used = (self.pso.evals_used if self.pso else 0) + self.gps.evals_used
        if self.gps.done:
            self.done, self.reason = True, self.gps.reason

    def best(self) -> dict:
        return self.gps.best() if self.gps else self.pso.best()

    def state(self) -> dict:
        return {**self._base_state(), "pso": self.pso.to_dict() if self.pso else None, "gps": self.gps.to_dict() if self.gps else None}

    @classmethod
    def restore(cls, start, cfg, state) -> "PSOStrategy":
        s = cls.__new__(cls)
        Strategy.__init__(s, start, cfg)
        s._load_base(state)
        s.pso = converge.PSO.from_dict(state["pso"]) if state["pso"] else None
        s.gps = converge.GPS.from_dict(state["gps"]) if state["gps"] else None
        return s


STRATEGIES: dict[str, type[Strategy]] = {c.name: c for c in (CMAStrategy, PSOStrategy)}
DEFAULT = CMAStrategy.name


def add_arguments(ap) -> None:
    """``--cma`` / ``--pso`` (one of them; CMA-ES by default) and every strategy's options."""
    g = ap.add_mutually_exclusive_group()
    for name, cls in STRATEGIES.items():
        g.add_argument(f"--{name}", dest="strategy", action="store_const", const=name,
                       help=f"{cls.summary}{' (the default)' if name == DEFAULT else ''}")
    ap.set_defaults(strategy=DEFAULT)
    seen = set()
    for cls in STRATEGIES.values():
        for f in dataclasses.fields(cls.Config):
            if f.name not in seen:
                seen.add(f.name)
                kind = {"action": argparse.BooleanOptionalAction} if isinstance(f.default, bool) else {"type": type(f.default)}
                ap.add_argument(f"--{f.name.replace('_', '-')}", default=f.default, help=f.metadata["help"], **kind)


def config(args) -> tuple[type[Strategy], object]:
    """The chosen strategy class and its configuration from parsed arguments."""
    cls = STRATEGIES[args.strategy]
    return cls, cls.Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls.Config)})
