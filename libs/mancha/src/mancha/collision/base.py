"""The collision-model interface.

A model is a small object carrying its parameters. The NumPy backend calls :meth:`CollisionModel.collide`; the
Triton backend dispatches on :attr:`CollisionModel.kind` to a fused kernel and raises for kinds it has none for.
New models (TRT, MRT, cumulant, an LES model with a per-cell tau) subclass this and add a kernel; the solver,
streaming and boundaries do not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np


class CollisionModel(ABC):
    kind: ClassVar[str]

    @property
    @abstractmethod
    def viscosity(self) -> float:
        """Kinematic shear viscosity in lattice units (dx²/dt)."""

    @abstractmethod
    def collide(self, f: np.ndarray, rho: np.ndarray, u: np.ndarray, c: np.ndarray, w: np.ndarray,
                force: np.ndarray | None = None, nu_t: np.ndarray | None = None) -> np.ndarray:
        """Post-collision populations from pre-collision ``f[q, ...]``, its density ``rho`` and velocity
        ``u[3, ...]``, an optional body-force density ``force`` (``[3]`` uniform or ``[3, ...]`` per node; ``u``
        then already includes ``F/2ρ``) and an optional eddy viscosity ``nu_t[...]`` added to the model's."""

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON-serialisable parameters (``kind`` included); :func:`mancha.collision.from_parameters` inverts it."""
