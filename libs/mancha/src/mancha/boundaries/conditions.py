"""Face boundary conditions and the rule deciding which one supplies a population.

Every face of the box (``x-``, ``x+``, ``y-``, ``y+``, ``z-``, ``z+``) carries one condition. During the pull,
the population arriving at node ``x`` along ``c_i`` comes from ``x − c_i``. When that source lies outside the
box, the condition of the face it leaves through supplies the population instead. A source leaving through two
or three faces (edge and corner nodes, diagonal directions) takes the condition with the highest *precedence*:

    wall (4) > moving wall (3) > velocity (2) > pressure (1) > periodic (0)

and, between equal kinds, the face on the lowest axis (x before y before z). Both backends use this rule.

The conditions, written for the incoming direction ``i`` at boundary node ``x_b`` with ``ī`` its opposite and
``f*`` the post-collision populations of the previous step (all halfway: the wall sits half a cell outside):

* wall:        ``f_i = f*_ī``                                             (bounce-back)
* moving wall: ``f_i = f*_ī + 6 w_i ρ_0 c_i·u_w``                         (Ladd; u_w tangential)
* velocity:    ``f_i = f*_ī + 6 w_i ρ_0 c_i·u_in``                        (bounce-back with a wall velocity)
* pressure:    ``f_i = −f*_ī + 2 w_i ρ_out [1 + 9/2 (c_i·u)² − 3/2 u·u]`` (anti-bounce-back, ``u = u*(x_b)``)

``ρ_0`` is the simulation's reference density (so the lid terms of a closed cavity cancel exactly and mass is
conserved); ``u*(x_b)`` is the velocity of ``f*`` at the boundary node itself (one step old). See
``docs/boundaries.md`` for what this costs in accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from ..errors import InvalidParameterError

FACES = ("x-", "x+", "y-", "y+", "z-", "z+")


class Kind(IntEnum):
    """Condition codes; a larger code wins at edges and corners."""

    PERIODIC = 0
    PRESSURE = 1
    VELOCITY = 2
    MOVING_WALL = 3
    WALL = 4


@dataclass(frozen=True)
class FaceCondition:
    """One face's condition: its kind, a velocity (moving wall, inlet) and a density (pressure outlet)."""

    kind: Kind
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.name.lower(), "velocity": list(self.velocity), "density": self.density}


def face_index(face: str) -> int:
    if face not in FACES:
        raise InvalidParameterError(f"unknown face {face!r}; faces are {FACES}")
    return FACES.index(face)


def _vector(v: Any, what: str) -> tuple[float, float, float]:
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise InvalidParameterError(f"{what} must be three finite numbers, got {v!r}")
    return tuple(float(x) for x in arr)


def make_condition(face: str, kind: str, *, velocity: Any = None, density: float | None = None) -> FaceCondition:
    """Parse ``set_boundary(face, kind, ...)``. Velocities and densities are in lattice units."""
    axis = face_index(face) // 2
    kind = kind.lower().replace("-", "_")
    if kind == "periodic":
        if velocity is not None or density is not None:
            raise InvalidParameterError("a periodic face takes no velocity or density")
        return FaceCondition(Kind.PERIODIC)
    if kind in ("wall", "no_slip", "moving_wall"):
        if density is not None:
            raise InvalidParameterError("a wall takes no density")
        u = _vector(velocity if velocity is not None else (0, 0, 0), "wall velocity")
        if u[axis] != 0.0:
            raise InvalidParameterError(f"a moving wall on {face} must move tangentially (velocity[{axis}] = 0), "
                                        f"got {u}; moving or rotating geometry is not supported yet")
        return FaceCondition(Kind.MOVING_WALL if any(u) else Kind.WALL, u)
    if kind in ("velocity", "inlet", "velocity_inlet"):
        if velocity is None:
            raise InvalidParameterError("a velocity boundary needs velocity=(ux, uy, uz)")
        if density is not None:
            raise InvalidParameterError("a velocity boundary takes no density (it follows the local density)")
        return FaceCondition(Kind.VELOCITY, _vector(velocity, "inlet velocity"))
    if kind in ("pressure", "outlet", "density", "pressure_outlet"):
        if velocity is not None:
            raise InvalidParameterError("a pressure boundary takes no velocity")
        rho = 1.0 if density is None else float(density)
        if not (math.isfinite(rho) and rho > 0):
            raise InvalidParameterError(f"the outlet density must be positive, got {density!r}")
        return FaceCondition(Kind.PRESSURE, density=rho)
    raise InvalidParameterError(f"unknown boundary kind {kind!r}; use periodic, wall, velocity or pressure")


class BoundaryConditions:
    """The six face conditions of a box; every face starts periodic."""

    def __init__(self) -> None:
        self.faces: list[FaceCondition] = [FaceCondition(Kind.PERIODIC) for _ in FACES]

    def set(self, face: str, kind: str, **params: Any) -> None:
        self.faces[face_index(face)] = make_condition(face, kind, **params)

    def __getitem__(self, face: str) -> FaceCondition:
        return self.faces[face_index(face)]

    def validate(self) -> None:
        """Periodicity must be paired: ``x-`` periodic if and only if ``x+`` is."""
        for axis in range(3):
            lo, hi = self.faces[2 * axis], self.faces[2 * axis + 1]
            if (lo.kind == Kind.PERIODIC) != (hi.kind == Kind.PERIODIC):
                raise InvalidParameterError(f"{FACES[2 * axis]} and {FACES[2 * axis + 1]} must both be periodic "
                                            "or both non-periodic")

    def table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``codes[6]``, ``velocities[6, 3]``, ``densities[6]`` in face order."""
        return (np.array([f.kind for f in self.faces], dtype=np.int64),
                np.array([f.velocity for f in self.faces], dtype=np.float64),
                np.array([f.density for f in self.faces], dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        return {name: f.to_dict() for name, f in zip(FACES, self.faces)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundaryConditions":
        bc = cls()
        for name, d in data.items():
            kind = d["kind"]
            if kind == "periodic":
                bc.set(name, "periodic")
            elif kind in ("wall", "moving_wall"):
                bc.set(name, "wall", velocity=d["velocity"])
            elif kind == "velocity":
                bc.set(name, "velocity", velocity=d["velocity"])
            else:
                bc.set(name, "pressure", density=d["density"])
        return bc

    def max_speed(self) -> float:
        return max(float(np.linalg.norm(f.velocity)) for f in self.faces)


@dataclass(frozen=True)
class LinkGroup:
    """Nodes (flat C-order indices) whose population along one direction is supplied by one face condition."""

    kind: Kind
    face: int
    nodes: np.ndarray


def link_groups(shape: tuple[int, int, int], c: np.ndarray, codes: np.ndarray) -> list[list[LinkGroup]]:
    """For each direction ``i``, the non-periodic face conditions that replace the pulled population.

    Implements the precedence rule of this module: the winning face maximises ``code * 4 + (2 − axis)``.
    """
    groups: list[list[LinkGroup]] = []
    for ci in c:
        key = np.full(shape, -1, dtype=np.int64)
        for axis in range(3):
            if ci[axis] == 0:
                continue
            n = shape[axis]
            face = 2 * axis + (0 if ci[axis] > 0 else 1)   # +c pulls from x−1: leaves through the low face
            line = np.full(n, -1, dtype=np.int64)
            line[0 if ci[axis] > 0 else n - 1] = codes[face] * 4 + (2 - axis)
            bshape = [1, 1, 1]
            bshape[axis] = n
            key = np.maximum(key, line.reshape(bshape))
        per_dir = []
        flat = key.reshape(-1)
        for k in np.unique(flat):
            if k < 0 or k // 4 == Kind.PERIODIC:
                continue
            axis = 2 - int(k % 4)
            face = 2 * axis + (0 if ci[axis] > 0 else 1)
            per_dir.append(LinkGroup(Kind(int(k // 4)), face, np.flatnonzero(flat == k)))
        groups.append(per_dir)
    return groups
