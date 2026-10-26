"""Collision models: BGK and regularized BGK; :class:`CollisionModel` is the extension point (TRT, MRT, cumulant)."""
from __future__ import annotations

from typing import Any

from .base import CollisionModel
from .bgk import BGK, tau_from_viscosity, viscosity_from_tau
from .equilibrium import equilibrium, moments
from .regularized import RegularizedBGK

_MODELS: dict[str, type[CollisionModel]] = {BGK.kind: BGK, RegularizedBGK.kind: RegularizedBGK}


def from_parameters(params: dict[str, Any]) -> CollisionModel:
    """Rebuild a model from :meth:`CollisionModel.parameters` (used by checkpoints)."""
    params = dict(params)
    kind = params.pop("kind")
    if kind not in _MODELS:
        raise KeyError(f"unknown collision model {kind!r}; known: {sorted(_MODELS)}")
    return _MODELS[kind](**params)


__all__ = ["BGK", "CollisionModel", "RegularizedBGK", "equilibrium", "from_parameters", "moments", "tau_from_viscosity",
           "viscosity_from_tau"]
