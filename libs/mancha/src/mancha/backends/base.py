"""The backend interface: a backend owns the populations and advances them; the Simulation owns everything else.

Populations are stored post-collision, ``f[q, nx, ny, nz]``. Host-side exchange (``populations``,
``set_populations``, ``density``, ``velocity``) is always NumPy; a GPU backend copies only when asked.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..boundaries import BoundaryConditions
from ..collision import CollisionModel
from ..lattice import D3Q19


class Backend(ABC):
    name: str = "abstract"

    def __init__(self, lattice: D3Q19, shape: tuple[int, int, int], collision: CollisionModel, dtype: np.dtype,
                 rest_density: float = 1.0, force: tuple[float, float, float] | None = None):
        self.lattice = lattice
        self.shape = shape
        self.collision = collision
        self.dtype = np.dtype(dtype)
        self.rest_density = float(rest_density)   # what solid voxels hold (a rest equilibrium), never read by fluid
        self.force = None if force is None or not any(force) else np.asarray(force, dtype=np.float64)

    @abstractmethod
    def configure(self, boundaries: BoundaryConditions, solid: np.ndarray | None) -> None:
        """Install the face conditions and solid mask; keeps the populations."""

    @abstractmethod
    def set_populations(self, f: np.ndarray) -> None:
        """Replace the post-collision populations (host array ``[q, nx, ny, nz]``)."""

    @abstractmethod
    def populations(self) -> np.ndarray:
        """A host copy of the post-collision populations."""

    @abstractmethod
    def step(self, steps: int = 1) -> None:
        """Advance ``steps`` time steps (stream, boundaries, collide)."""

    @abstractmethod
    def density(self) -> np.ndarray:
        """Density ``[nx, ny, nz]`` (host)."""

    @abstractmethod
    def velocity(self) -> np.ndarray:
        """Velocity ``[3, nx, ny, nz]`` (host)."""

    @abstractmethod
    def health(self) -> tuple[bool, float]:
        """``(all populations finite, max |u| over fluid nodes)``, computed on the device."""

    def synchronize(self) -> None:
        """Wait for queued device work (no-op on the CPU)."""

    def memory_bytes(self) -> int:
        """Bytes held by the solver state (two population buffers and the solid mask)."""
        n = int(np.prod(self.shape))
        return 2 * self.lattice.q * n * self.dtype.itemsize + n
