"""The NumPy reference backend: readable, vectorised over the grid, one direction at a time.

Per step, for each direction ``i``: pull ``f*_i`` from ``x − c_i`` with ``np.roll`` (periodic wrap), overwrite
the links whose source is a solid voxel (bounce-back), then the links whose source lies outside a non-periodic
face (the face conditions, in the precedence of :mod:`mancha.boundaries.conditions`). Then moments and
collision on every node; solid voxels are reset to a rest equilibrium.
"""
from __future__ import annotations

import numpy as np

from ..boundaries import BoundaryConditions, Kind, link_groups
from ..collision import CollisionModel, equilibrium, moments
from ..lattice import D3Q19
from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self, lattice: D3Q19, shape: tuple[int, int, int], collision: CollisionModel,
                 dtype: np.dtype = np.float64, rest_density: float = 1.0,
                 force: tuple[float, float, float] | None = None):
        super().__init__(lattice, shape, collision, dtype, rest_density, force)
        self._c, self._w, self._opp = lattice.c, lattice.w, lattice.opposite
        self.f = np.empty((lattice.q,) + shape, dtype=self.dtype)
        self._buf = np.empty_like(self.f)
        self.configure(BoundaryConditions(), None)

    def configure(self, boundaries: BoundaryConditions, solid: np.ndarray | None) -> None:
        boundaries.validate()
        codes, self._face_u, self._face_rho = boundaries.table()
        self._groups = link_groups(self.shape, self._c, codes)
        self._needs_local = any(g.kind == Kind.PRESSURE for groups in self._groups for g in groups)
        self.solid = None if solid is None or not solid.any() else solid.astype(bool)
        self._solid_links: list[np.ndarray] = []
        if self.solid is not None:
            for ci in self._c:
                src = np.roll(self.solid, shift=tuple(ci), axis=(0, 1, 2))
                self._solid_links.append(np.flatnonzero(src & ~self.solid))
            self._solid_flat = np.flatnonzero(self.solid)
            self._rest = (self._w * self.rest_density).astype(self.dtype)

    def set_populations(self, f: np.ndarray) -> None:
        self.f[...] = f

    def populations(self) -> np.ndarray:
        return self.f.copy()

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._stream()
            self._collide()
            self.f, self._buf = self._buf, self.f

    def _stream(self) -> None:
        c, w, opp, f, out = self._c, self._w, self._opp, self.f, self._buf
        if self._needs_local:
            u_b = self._moments(f)[1].reshape(3, -1)
        for i in range(self.lattice.q):
            out[i] = np.roll(f[i], shift=tuple(c[i]), axis=(0, 1, 2))
            dst = out[i].reshape(-1)
            back = f[opp[i]].reshape(-1)
            if self.solid is not None:
                n = self._solid_links[i]
                dst[n] = back[n]
            for g in self._groups[i]:
                n = g.nodes
                if g.kind == Kind.WALL:
                    dst[n] = back[n]
                elif g.kind in (Kind.MOVING_WALL, Kind.VELOCITY):
                    cu = float(c[i] @ self._face_u[g.face])
                    dst[n] = back[n] + 6.0 * w[i] * self.rest_density * cu
                elif g.kind == Kind.PRESSURE:
                    ub = u_b[:, n]
                    cu = c[i].astype(ub.dtype) @ ub
                    dst[n] = -back[n] + 2.0 * w[i] * self._face_rho[g.face] * (
                        1.0 + 4.5 * cu * cu - 1.5 * (ub * ub).sum(axis=0))

    def _collide(self) -> None:
        fin = self._buf
        rho, u = moments(fin, self._c)
        if self.force is not None:
            u = u + (0.5 * self.force).reshape(3, 1, 1, 1) / rho
        fin[...] = self.collision.collide(fin, rho, u, self._c, self._w, self.force)
        if self.solid is not None:
            flat = fin.reshape(self.lattice.q, -1)
            flat[:, self._solid_flat] = self._rest[:, None]

    def density(self) -> np.ndarray:
        return self.f.sum(axis=0)

    def velocity(self) -> np.ndarray:
        u = self._moments(self.f)[1]
        if self.solid is not None:
            u[:, self.solid] = 0.0
        return u

    def _moments(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Density and velocity of post-collision populations (they carry ``ρu + F/2`` under a force)."""
        rho, u = moments(f, self._c)
        if self.force is not None:
            u = u - (0.5 * self.force).reshape(3, 1, 1, 1) / rho
        return rho, u

    def health(self) -> tuple[bool, float]:
        if not np.isfinite(self.f).all():
            return False, float("nan")
        u = self.velocity()
        return True, float(np.sqrt((u * u).sum(axis=0)).max())


def rest_populations(lattice: D3Q19, rho: np.ndarray | float, u: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Equilibrium populations for an initial state (``u[3, nx, ny, nz]``)."""
    return equilibrium(rho, u, lattice.c, lattice.w).astype(dtype)
