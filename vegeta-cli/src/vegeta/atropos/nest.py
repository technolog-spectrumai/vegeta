"""Nesting on stock sheets: a deterministic shelf (row) packer. Parts lie with their long side along the sheet's X
unless their grain says otherwise (``Part.meta["grain"]``: the part axis, "x" or "y", that must follow the wood's
grain, matched against ``Sheet.grain``); a part may be turned by 90 degrees when ``CutSettings.rotate`` allows it.
No rotation search, no hole filling: what you see is what is cut."""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Part
from .settings import CutSettings, Sheet


@dataclass
class Placement:
    """A part placed on a sheet: ``part`` is already in sheet coordinates (origin at the sheet's front-left corner,
    X along the width); ``x, y`` is its bounding box's lower-left corner, ``angle_deg`` its rotation."""

    part: Part
    sheet_index: int
    x: float
    y: float
    angle_deg: float
    source: str


@dataclass
class Nesting:
    sheet: Sheet
    placements: list[Placement] = field(default_factory=list)
    n_sheets: int = 0
    utilisation: list[float] = field(default_factory=list)

    def parts_on(self, sheet_index: int) -> list[Part]:
        return [pl.part for pl in self.placements if pl.sheet_index == sheet_index]

    def table(self) -> list[dict]:
        return [{"part": pl.source, "sheet": pl.sheet_index + 1, "x_mm": round(pl.x, 3), "y_mm": round(pl.y, 3),
                 "angle_deg": pl.angle_deg} for pl in self.placements]


def _orientations(part: Part, settings: CutSettings) -> list[float]:
    grain = part.meta.get("grain")
    if grain in ("x", "y"):
        return [0.0] if grain == settings.sheet.grain else [90.0]
    w, h = part.size()
    first = 0.0 if w >= h else 90.0
    if not settings.rotate:
        return [0.0]
    return [first, 90.0 - first]


def nest(parts: list[Part], settings: CutSettings) -> Nesting:
    """Place ``parts`` on as many sheets as they need. Raises ``ValueError`` for a part larger than the usable sheet."""
    sheet, gap, m = settings.sheet, settings.spacing_mm, settings.sheet.margin_mm
    W, H = sheet.usable
    items = []
    for part in parts:
        opts = _orientations(part, settings)
        oriented = [(a, part.transformed(a).at_origin()) for a in opts]
        fits = [(a, p) for a, p in oriented if p.size()[0] <= W + 1e-9 and p.size()[1] <= H + 1e-9]
        if not fits:
            w, h = part.size()
            raise ValueError(f"part {part.name!r} ({w:.1f} x {h:.1f} mm) does not fit the usable sheet ({W:g} x {H:g} mm)")
        items.append((part.name, fits))
    # tallest first, in the preferred orientation; ties by name so the result is reproducible
    items.sort(key=lambda it: (-it[1][0][1].size()[1], -it[1][0][1].size()[0], it[0]))

    nesting = Nesting(sheet)
    sheet_i, x, y, shelf_h = 0, 0.0, 0.0, 0.0
    areas: dict[int, float] = {}
    for name, fits in items:
        placed = False
        for attempt in range(3):             # this shelf, a new shelf, a new sheet
            for angle, p in fits:
                w, h = p.size()
                if x + w <= W + 1e-9 and y + max(h, shelf_h) <= H + 1e-9:
                    moved = p.transformed(0.0, m + x, m + y)
                    nesting.placements.append(Placement(moved, sheet_i, m + x, m + y, angle, name))
                    areas[sheet_i] = areas.get(sheet_i, 0.0) + p.area()
                    x += w + gap
                    shelf_h = max(shelf_h, h)
                    placed = True
                    break
            if placed:
                break
            if attempt == 0:                 # new shelf
                y += shelf_h + gap if shelf_h else 0.0
                x, shelf_h = 0.0, 0.0
            else:                            # new sheet
                sheet_i += 1
                x, y, shelf_h = 0.0, 0.0, 0.0
        if not placed:                       # cannot happen: an empty sheet takes any part that fits the usable area
            raise ValueError(f"part {name!r} could not be placed")
    nesting.n_sheets = sheet_i + 1 if nesting.placements else 0
    nesting.utilisation = [areas.get(i, 0.0) / (sheet.width_mm * sheet.height_mm) for i in range(nesting.n_sheets)]
    return nesting
