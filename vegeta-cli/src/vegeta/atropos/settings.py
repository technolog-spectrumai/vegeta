"""Explicit cutting settings: the sheet, the machine and one operation (power, feed, passes, kerf) per layer.

Nothing has a default that matters for the cut: power, feed, passes and kerf are stated by the engineer for the
material on the bench (a diode laser's kerf and the passes it needs vary with the balsa's density and the lens'
focus; cut a test strip first). ``vegeta.atropos.examples`` holds illustrations, not validated profiles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace as _replace
from typing import Any


@dataclass(frozen=True)
class Sheet:
    """A stock sheet as it lies on the bed: ``width_mm`` along the machine's X, ``height_mm`` along Y. ``grain`` is
    the direction of the wood's grain on the bed ("x" or "y"); ``margin_mm`` is kept free along every edge."""

    name: str
    width_mm: float
    height_mm: float
    thickness_mm: float
    density_g_cm3: float
    grain: str = "x"
    margin_mm: float = 5.0

    def __post_init__(self):
        for k in ("width_mm", "height_mm", "thickness_mm", "density_g_cm3"):
            if not getattr(self, k) > 0:
                raise ValueError(f"sheet {self.name!r}: {k} must be positive")
        if self.grain not in ("x", "y"):
            raise ValueError(f"sheet {self.name!r}: grain must be 'x' or 'y'")
        if self.margin_mm < 0 or 2 * self.margin_mm >= min(self.width_mm, self.height_mm):
            raise ValueError(f"sheet {self.name!r}: margin {self.margin_mm} mm leaves no usable area")

    @property
    def usable(self) -> tuple[float, float]:
        return self.width_mm - 2 * self.margin_mm, self.height_mm - 2 * self.margin_mm

    @property
    def mass_g(self) -> float:
        return self.width_mm * self.height_mm * self.thickness_mm * 1e-3 * self.density_g_cm3


@dataclass(frozen=True)
class Machine:
    """A GRBL laser: the bed (X by Y, origin at the front left), the spindle value that means full power (``$30``,
    1000 on an xTool D1 Pro) and the rapid feed used for the time estimate."""

    name: str
    bed_width_mm: float
    bed_height_mm: float
    travel_feed_mm_min: float
    s_max: int = 1000

    def __post_init__(self):
        if not (self.bed_width_mm > 0 and self.bed_height_mm > 0 and self.travel_feed_mm_min > 0 and self.s_max > 0):
            raise ValueError(f"machine {self.name!r}: bed, travel feed and s_max must be positive")


@dataclass(frozen=True)
class Operation:
    """One laser operation (a layer in the cutter software): power in percent, feed in mm/min, the number of passes,
    the kerf the operation leaves in this material (compensated: outer contours grow by half of it, holes shrink).
    ``order`` sorts the operations in the G-code (engrave and score before cut, so the parts stay put)."""

    name: str
    power_pct: float
    feed_mm_min: float
    passes: int
    kerf_mm: float
    order: int = 0
    colour: str | None = None     # SVG stroke; None: red for cut, blue for score, black for engrave

    def __post_init__(self):
        if not 0 < self.power_pct <= 100:
            raise ValueError(f"operation {self.name!r}: power_pct must be in (0, 100]")
        if not self.feed_mm_min > 0:
            raise ValueError(f"operation {self.name!r}: feed_mm_min must be positive")
        if not (isinstance(self.passes, int) and self.passes >= 1):
            raise ValueError(f"operation {self.name!r}: passes must be an integer >= 1")
        if self.kerf_mm < 0:
            raise ValueError(f"operation {self.name!r}: kerf_mm must be >= 0")

    @property
    def stroke(self) -> str:
        return self.colour or {"cut": "#ff0000", "score": "#0000ff", "engrave": "#000000"}.get(self.name, "#808080")


@dataclass
class CutSettings:
    """Everything a cut job needs: the sheet, the machine, the operations by layer name (``cut`` is required; parts
    with ``score`` or ``engrave`` paths need those too), the spacing between nested parts, whether parts may be turned
    by 90 degrees, whether part names are engraved."""

    sheet: Sheet
    machine: Machine
    operations: dict[str, Operation] = field(default_factory=dict)
    spacing_mm: float = 3.0
    rotate: bool = True
    labels: bool = False
    name: str = "settings"
    source: str = ""

    def __post_init__(self):
        if "cut" not in self.operations:
            raise ValueError(f"cut settings {self.name!r}: an operation named 'cut' must be given")
        for key, op in self.operations.items():
            if op.name != key:
                raise ValueError(f"cut settings {self.name!r}: operation {op.name!r} filed under {key!r}")
        if self.spacing_mm < 0:
            raise ValueError(f"cut settings {self.name!r}: spacing_mm must be >= 0")
        if self.sheet.width_mm > self.machine.bed_width_mm or self.sheet.height_mm > self.machine.bed_height_mm:
            raise ValueError(f"cut settings {self.name!r}: sheet {self.sheet.width_mm:g} x {self.sheet.height_mm:g} mm does "
                             f"not fit the bed {self.machine.bed_width_mm:g} x {self.machine.bed_height_mm:g} mm")

    def replace(self, **changes) -> "CutSettings":
        """Copy with fields changed, e.g. ``replace(sheet=other_sheet, spacing_mm=2.0)``."""
        return _replace(self, **changes)

    def operation(self, name: str) -> Operation:
        try:
            return self.operations[name]
        except KeyError:
            raise ValueError(f"cut settings {self.name!r} has no operation {name!r} (given: {sorted(self.operations)})") from None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
