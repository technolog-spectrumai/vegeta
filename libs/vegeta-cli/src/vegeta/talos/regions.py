"""Named surface regions selected explicitly on the imported geometry."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise ValueError(f"region name {name!r} must start with a letter and contain only letters, digits, _")


@dataclass(frozen=True)
class Surfaces:
    """Surfaces by Gmsh tag (see ``talos.inspect_step`` for the tags)."""

    name: str
    tags: Sequence[int]

    def __post_init__(self):
        _check_name(self.name)
        if not self.tags:
            raise ValueError(f"region {self.name!r}: no surface tags given")

    def select(self, gmsh_model, diag: float) -> list[int]:
        existing = {t for _, t in gmsh_model.getEntities(2)}
        missing = sorted(set(self.tags) - existing)
        if missing:
            raise ValueError(f"region {self.name!r}: surface tags {missing} do not exist")
        return sorted(set(int(t) for t in self.tags))


@dataclass(frozen=True)
class SurfacesInBox:
    """Surfaces lying completely inside an axis-aligned box ``(xmin, ymin, zmin, xmax, ymax, zmax)``."""

    name: str
    box: Sequence[float]

    def __post_init__(self):
        _check_name(self.name)
        if len(self.box) != 6:
            raise ValueError(f"region {self.name!r}: box needs (xmin, ymin, zmin, xmax, ymax, zmax)")

    def select(self, gmsh_model, diag: float) -> list[int]:
        return sorted(t for _, t in gmsh_model.getEntitiesInBoundingBox(*self.box, dim=2))


@dataclass(frozen=True)
class SurfacesOnPlane:
    """Surfaces lying in the plane ``axis = value`` (within ``tol``; default 1e-6 × model size)."""

    name: str
    axis: str
    value: float
    tol: float | None = None

    def __post_init__(self):
        _check_name(self.name)
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"region {self.name!r}: axis must be 'x', 'y' or 'z'")

    def select(self, gmsh_model, diag: float) -> list[int]:
        i = "xyz".index(self.axis)
        tol = self.tol if self.tol is not None else 1e-6 * diag
        out = []
        for _, tag in gmsh_model.getEntities(2):
            bb = gmsh_model.getBoundingBox(2, tag)
            lo, hi = bb[i], bb[i + 3]
            if abs(lo - self.value) <= tol and abs(hi - self.value) <= tol:
                out.append(tag)
        return sorted(out)


Region = Surfaces | SurfacesInBox | SurfacesOnPlane
