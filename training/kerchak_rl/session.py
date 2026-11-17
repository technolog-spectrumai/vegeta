"""A training session: ``session.yaml`` names the scenes it trains on, each with a weight and parameters, the base
configuration they share and, optionally, the curriculum stages to run through. The vectorised environments are
shared out among the scenes by weight (``assign_scenes``), so one policy learns them all.

    name: knuckle_walk_and_doors
    base: base.yaml                      # the Config to start from (relative to this file)
    config: {ppo: {num_envs: 8}}         # overrides on it
    stages: [1_stance, 3_walk]           # optional: curriculum stages, in order (configs/stages/*.yaml)
    scenes:
      - {name: flat, weight: 0.5}
      - {name: rough, weight: 0.2, params: {rms: 0.03, correlation_length: 0.4}}
      - {name: door, weight: 0.3, params: {width: 1.3}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config, load_config, merge, to_dict
from .scenes import SCENES

__all__ = ["SceneEntry", "Session", "load_session", "assign_scenes"]


@dataclass
class SceneEntry:
    name: str
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class Session:
    name: str
    config: Config
    scenes: list
    stages: list = field(default_factory=list)
    path: str = ""

    def scene_weights(self) -> dict:
        total = sum(max(s.weight, 0.0) for s in self.scenes) or 1.0
        return {s.name: max(s.weight, 0.0) / total for s in self.scenes}


def load_session(path: str | Path, overrides: dict | None = None) -> Session:
    path = Path(path)
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    base = doc.get("base")
    cfg = load_config((path.parent / base).resolve() if base else None, overrides=merge(doc.get("config") or {}, overrides or {}))
    scenes = []
    for s in doc.get("scenes") or [{"name": "flat"}]:
        if isinstance(s, str):
            s = {"name": s}
        if s["name"] not in SCENES:
            raise KeyError(f"session {path.name}: unknown scene {s['name']!r} (one of {', '.join(SCENES)})")
        scenes.append(SceneEntry(s["name"], float(s.get("weight", 1.0)), dict(s.get("params") or {})))
    return Session(doc.get("name", path.stem), cfg, scenes, list(doc.get("stages") or []), str(path))


def assign_scenes(session: Session, num_envs: int) -> list:
    """Which scene each of ``num_envs`` environments gets: the weights' proportions, every scene at least once
    when there are enough environments (largest-remainder rounding)."""
    weights = session.scene_weights()
    names = list(weights)
    if num_envs < len(names):
        order = sorted(names, key=lambda n: -weights[n])
        return [session.scenes[names.index(order[i % len(order)])] for i in range(num_envs)]
    raw = {n: weights[n] * num_envs for n in names}
    counts = {n: max(1, int(raw[n])) for n in names}
    while sum(counts.values()) > num_envs:
        n = max((n for n in names if counts[n] > 1), key=lambda n: counts[n] - raw[n])
        counts[n] -= 1
    while sum(counts.values()) < num_envs:
        n = max(names, key=lambda n: raw[n] - counts[n])
        counts[n] += 1
    out = []
    for n in names:
        out += [session.scenes[names.index(n)]] * counts[n]
    return out


def session_summary(session: Session) -> dict:
    return {"name": session.name, "scenes": [{"name": s.name, "weight": s.weight, "params": s.params} for s in session.scenes],
            "stages": session.stages, "config": to_dict(session.config)}
