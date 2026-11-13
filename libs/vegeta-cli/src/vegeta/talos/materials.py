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
    thermal_expansion: float | None = None    # 1/K, mean coefficient from reference_temperature (temperature loads)
    reference_temperature: float = 293.15     # K, stress-free temperature
    temperature_table: tuple | None = None    # rows (T [K], E, nu, yield strength or None), increasing T: the
                                              # stiffness and yield at temperature (replaces youngs_modulus /
                                              # poissons_ratio / yield_strength when a temperature load is applied)

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
        if self.thermal_expansion is not None and not self.thermal_expansion > 0:
            raise ValueError(f"material {self.name!r}: thermal_expansion must be > 0 when given")
        if self.temperature_table is not None:
            rows = [tuple(r) for r in self.temperature_table]
            if len(rows) < 2 or any(len(r) != 4 for r in rows):
                raise ValueError(f"material {self.name!r}: temperature_table needs at least 2 rows (T, E, nu, yield)")
            if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
                raise ValueError(f"material {self.name!r}: temperature_table temperatures must increase")
            if any(r[1] <= 0 or not -1 < r[2] < 0.5 or (r[3] is not None and r[3] <= 0) for r in rows):
                raise ValueError(f"material {self.name!r}: temperature_table needs E > 0, nu in (-1, 0.5), yield > 0 or None")
            object.__setattr__(self, "temperature_table", tuple(rows))

    def yield_at(self, temperature):
        """Yield strength at a temperature (K; scalar or array), from ``temperature_table`` when it gives yield values,
        else the constant ``yield_strength`` (``None`` when neither is given)."""
        import numpy as np

        rows = [r for r in (self.temperature_table or ()) if r[3] is not None]
        if len(rows) >= 2:
            return np.interp(temperature, [r[0] for r in rows], [r[3] for r in rows])
        return None if self.yield_strength is None else np.full_like(np.asarray(temperature, float), self.yield_strength)
