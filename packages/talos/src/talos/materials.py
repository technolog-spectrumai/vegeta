"""Explicit linear elastic materials. There are no built-in defaults."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Isotropic linear elastic material in the model's unit system.

    ``density`` is required only for acceleration loads; ``yield_strength`` only enables the
    yield safety factor. Both default to ``None`` — Talos never assumes them.
    """

    name: str
    youngs_modulus: float
    poissons_ratio: float
    density: float | None = None
    yield_strength: float | None = None
    units: str | None = None  # optional tag, checked against the model's unit system
    source: str = ""          # where the values come from (datasheet, standard, test)

    def __post_init__(self):
        if not self.name:
            raise ValueError("material needs a name")
        if not self.youngs_modulus or self.youngs_modulus <= 0:
            raise ValueError(f"material {self.name!r}: youngs_modulus must be > 0")
        if not (-1.0 < self.poissons_ratio < 0.5):
            raise ValueError(f"material {self.name!r}: poissons_ratio must be in (-1, 0.5)")
        if self.density is not None and self.density <= 0:
            raise ValueError(f"material {self.name!r}: density must be > 0 when given")
        if self.yield_strength is not None and self.yield_strength <= 0:
            raise ValueError(f"material {self.name!r}: yield_strength must be > 0 when given")
