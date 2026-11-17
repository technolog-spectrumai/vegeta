"""Terrains for training, from Chiron's: flat, rough (a Gaussian random field of configurable RMS and correlation
length) and steps. A terrain is baked into the MuJoCo model, so an environment rebuilds its model when its terrain
changes (a curriculum stage, a new scene); the uneven ones take a seed so the field is reproducible."""
from __future__ import annotations

from vegeta.chiron import Flat, Rough, Steps

__all__ = ["make_terrain"]


def make_terrain(cfg, seed: int | None = None, kind: str | None = None):
    """A Chiron Terrain from a ``TerrainConfig``; ``seed`` and ``kind`` override the config's."""
    k = (kind or cfg.kind).lower()
    s = cfg.seed if seed is None else seed
    if k == "flat":
        return Flat()
    if k == "rough":
        return Rough(cfg.rms, cfg.correlation_length, start=0.0, seed=s, extent=tuple(cfg.extent), cell=cfg.cell, ramp=cfg.correlation_length)
    if k == "steps":
        return Steps(cfg.step_height, cfg.step_spacing, start=1.0, jitter=0.2, seed=s, mode="alternate")
    raise ValueError(f"terrain kind {k!r}: flat, rough or steps")
