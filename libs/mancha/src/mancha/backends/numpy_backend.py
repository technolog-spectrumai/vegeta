"""The NumPy reference backend: readable, vectorised over the grid, one direction at a time.

Per step, for each direction ``i``: pull ``f*_i`` from ``x − c_i`` with ``np.roll`` (periodic wrap), overwrite
the links whose source is a solid voxel (bounce-back), then the links whose source lies outside a non-periodic
face (the face conditions, in the precedence of :mod:`mancha.boundaries.conditions`). Then moments and
collision on every node; solid voxels are reset to a rest equilibrium.

Optional physics, each the reference for the Triton kernel: a uniform body force (Guo), a rotating frame
(:mod:`mancha.frames`: Coriolis and centrifugal forces, face velocities converted from the lab frame) and WALE
eddy viscosity (:mod:`mancha.turbulence`, from the previous step's velocity).
"""
from __future__ import annotations

import numpy as np

from ..boundaries import BoundaryConditions, Kind, Sponge, link_groups
from ..collision import CollisionModel, equilibrium, moments
from ..frames import RotatingFrame, fictitious_force, solve_velocity
from ..lattice import D3Q19
from ..turbulence import WALE, to_lab_gradient, velocity_gradient, wale_viscosity
from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self, lattice: D3Q19, shape: tuple[int, int, int], collision: CollisionModel,
                 dtype: np.dtype = np.float64, rest_density: float = 1.0,
                 force: tuple[float, float, float] | None = None, frame: RotatingFrame | None = None,
                 turbulence: WALE | None = None, sponge: Sponge | None = None):
        super().__init__(lattice, shape, collision, dtype, rest_density, force, frame, turbulence, sponge)
        self._nu_sponge = None if sponge is None else sponge.field(shape)
        self._c, self._w, self._opp = lattice.c, lattice.w, lattice.opposite
        self.f = np.empty((lattice.q,) + shape, dtype=self.dtype)
        self._buf = np.empty_like(self.f)
        if self.frame is not None:
            ry, rz = self.frame.radial(shape)
            self._ry = np.broadcast_to(ry, shape).astype(self.dtype)
            self._rz = np.broadcast_to(rz, shape).astype(self.dtype)
        self._u_prev: np.ndarray | None = None
        self.configure(BoundaryConditions(), None)

    def configure(self, boundaries: BoundaryConditions, solid: np.ndarray | None) -> None:
        boundaries.validate()
        codes, self._face_u, self._face_rho = boundaries.table()
        self._periodic = tuple(bool(codes[2 * a] == Kind.PERIODIC) for a in range(3))
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
        self._u_prev = None

    def set_populations(self, f: np.ndarray) -> None:
        self.f[...] = f
        self._u_prev = None

    def populations(self) -> np.ndarray:
        return self.f.copy()

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._stream()
            self._collide()
            self.f, self._buf = self._buf, self.f

    def _wall_velocity(self, i: int, face: int, nodes: np.ndarray) -> np.ndarray | float:
        """``c_i · u_rel`` of a face condition at the halfway point ``x_b − c_i/2`` of each node's link."""
        ci = self._c[i]
        cu = float(ci @ self._face_u[face])
        if self.frame is None:
            return cu
        om = self.frame.omega
        ry = self._ry.reshape(-1)[nodes] - 0.5 * ci[1]
        rz = self._rz.reshape(-1)[nodes] - 0.5 * ci[2]
        return cu + ci[1] * om * rz - ci[2] * om * ry      # c·(u_abs − Ω×r), Ω×r = (0, −ω r_z, ω r_y)

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
                if g.kind == Kind.WALL and self.frame is None:
                    dst[n] = back[n]
                elif g.kind in (Kind.WALL, Kind.MOVING_WALL, Kind.VELOCITY):
                    dst[n] = back[n] + 6.0 * w[i] * self.rest_density * self._wall_velocity(i, g.face, n)
                elif g.kind == Kind.PRESSURE:
                    ub = u_b[:, n]
                    cu = c[i].astype(ub.dtype) @ ub
                    dst[n] = -back[n] + 2.0 * w[i] * self._face_rho[g.face] * (
                        1.0 + 4.5 * cu * cu - 1.5 * (ub * ub).sum(axis=0))

    def _collide(self) -> None:
        fin = self._buf
        rho, j = moments(fin, self._c)            # moments() returns j/ρ
        nu_t = None
        if self.turbulence is not None:
            u_prev = self._u_prev if self._u_prev is not None else self.velocity()
            grad = velocity_gradient(u_prev, self.solid, self._periodic)
            if self.frame is not None:
                to_lab_gradient(grad, self.frame.omega)
            nu_t = wale_viscosity(grad, self.turbulence.c_w)
            if self.solid is not None:
                nu_t[self.solid] = 0.0
        if self._nu_sponge is not None:
            nu_t = self._nu_sponge if nu_t is None else nu_t + self._nu_sponge
        b = j.copy()
        if self.force is not None:
            b += (0.5 * self.force).reshape(3, 1, 1, 1) / rho
        if self.frame is None:
            u, force = b, self.force
        else:
            om = self.frame.omega
            b[1] += 0.5 * om * om * self._ry
            b[2] += 0.5 * om * om * self._rz
            u = solve_velocity(b, om, +1.0)
            force = fictitious_force(rho, u, om, self._ry, self._rz)
            if self.force is not None:
                force += self.force.reshape(3, 1, 1, 1)
        fin[...] = self.collision.collide(fin, rho, u, self._c, self._w, force, nu_t)
        if self.turbulence is not None:
            self._u_prev = u.copy()
            if self.solid is not None:
                self._u_prev[:, self.solid] = 0.0
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
        """Density and (frame-relative) velocity of post-collision populations, which carry ``ρu + F/2``."""
        rho, b = moments(f, self._c)
        if self.force is not None:
            b = b - (0.5 * self.force).reshape(3, 1, 1, 1) / rho
        if self.frame is None:
            return rho, b
        om = self.frame.omega
        b[1] -= 0.5 * om * om * self._ry
        b[2] -= 0.5 * om * om * self._rz
        return rho, solve_velocity(b, om, -1.0)

    def health(self) -> tuple[bool, float]:
        if not np.isfinite(self.f).all():
            return False, float("nan")
        u = self.velocity()
        return True, float(np.sqrt((u * u).sum(axis=0)).max())


def rest_populations(lattice: D3Q19, rho: np.ndarray | float, u: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Equilibrium populations for an initial state (``u[3, nx, ny, nz]``)."""
    return equilibrium(rho, u, lattice.c, lattice.w).astype(dtype)


def state_velocity(u: np.ndarray, rho: np.ndarray, force: np.ndarray | None, frame: RotatingFrame | None
                   ) -> np.ndarray:
    """The velocity to build the stored (post-collision) state from so that it reports ``u``: the inverse of
    ``NumpyBackend._moments`` — ``j*/ρ = u + F/(2ρ)`` with the frame's forces at ``u``."""
    out = np.array(u, dtype=np.float64, copy=True)
    if force is not None:
        out += np.asarray(force, dtype=np.float64).reshape(3, 1, 1, 1) / (2.0 * rho)
    if frame is not None:
        om = frame.omega
        ry, rz = frame.radial(u.shape[1:])
        out[1] += om * u[2] + 0.5 * om * om * ry
        out[2] += -om * u[1] + 0.5 * om * om * rz
    return out
