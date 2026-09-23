"""Supports and loads. All values are explicit and in the model's unit system."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedSupport:
    """All three translations of the region's nodes fixed to zero."""

    region: str


@dataclass(frozen=True)
class Displacement:
    """Prescribed translations; ``None`` leaves that direction free."""

    region: str
    ux: float | None = None
    uy: float | None = None
    uz: float | None = None

    def __post_init__(self):
        if self.ux is None and self.uy is None and self.uz is None:
            raise ValueError(f"Displacement on {self.region!r} constrains nothing; give ux, uy and/or uz")

    def dofs(self):
        return [(i + 1, v) for i, v in enumerate((self.ux, self.uy, self.uz)) if v is not None]


@dataclass(frozen=True)
class Force:
    """Total force vector on a surface region, distributed as a uniform traction (consistent nodal loads)."""

    region: str
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0

    def __post_init__(self):
        if self.fx == 0 and self.fy == 0 and self.fz == 0:
            raise ValueError(f"Force on {self.region!r} is zero")

    @property
    def vector(self):
        return (self.fx, self.fy, self.fz)


@dataclass(frozen=True)
class Pressure:
    """Uniform pressure on a surface region. Positive pushes into the solid (CalculiX convention)."""

    region: str
    value: float

    def __post_init__(self):
        if self.value == 0:
            raise ValueError(f"Pressure on {self.region!r} is zero")


@dataclass(frozen=True)
class Acceleration:
    """Body acceleration on the whole solid (length unit / s^2), e.g. gravity (0, 0, -9810) in mm.
    Requires the material density."""

    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    def __post_init__(self):
        if self.ax == 0 and self.ay == 0 and self.az == 0:
            raise ValueError("Acceleration is zero")

    @property
    def vector(self):
        return (self.ax, self.ay, self.az)


Support = FixedSupport | Displacement
Load = Force | Pressure | Acceleration
