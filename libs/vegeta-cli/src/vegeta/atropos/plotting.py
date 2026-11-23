"""Matplotlib previews: a sheet with its nested parts, a gallery of parts. They return figures and never call
``plt.show()``."""
from __future__ import annotations

import math

from .geometry import Part
from .svg import DEFAULT_STROKES


def _draw_part(ax, part: Part, operations=None, label=False, lw=0.8):
    import numpy as np

    for p in part.paths():
        stroke = operations[p.op].stroke if operations and p.op in operations else DEFAULT_STROKES.get(p.op, "#808080")
        pts = np.vstack([p.points, p.points[:1]]) if p.closed else p.points
        ax.plot(pts[:, 0], pts[:, 1], color=stroke, lw=lw)
    if label:
        cx, cy = part.outer.centroid()
        ax.text(cx, cy, part.name, ha="center", va="center", fontsize=6, color="#444444")


def plot_sheet(parts: list[Part], sheet=None, ax=None, operations=None, labels=True, title=""):
    """One sheet (``parts`` in sheet coordinates, e.g. ``nesting.parts_on(i)``) with its outline and margin."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        w = sheet.width_mm if sheet else 400.0
        h = sheet.height_mm if sheet else 100.0
        _, ax = plt.subplots(figsize=(10, max(2.0, 10 * h / w)))
    if sheet is not None:
        ax.add_patch(Rectangle((0, 0), sheet.width_mm, sheet.height_mm, fill=False, ec="#8d6e63", lw=1.2))
        m = sheet.margin_mm
        ax.add_patch(Rectangle((m, m), sheet.width_mm - 2 * m, sheet.height_mm - 2 * m, fill=False, ec="#bcaaa4", lw=0.6, ls="--"))
    for part in parts:
        _draw_part(ax, part, operations, labels)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if sheet is not None:
        ax.set_xlim(-2, sheet.width_mm + 2)
        ax.set_ylim(-2, sheet.height_mm + 2)
    ax.set_title(title or (f"{sheet.name}: {len(parts)} parts" if sheet else f"{len(parts)} parts"), fontsize=9)
    return ax.figure


def plot_nesting(nesting, operations=None, labels=True, max_sheets=None):
    """Every sheet of a nesting, one row each."""
    import matplotlib.pyplot as plt

    n = nesting.n_sheets if max_sheets is None else min(max_sheets, nesting.n_sheets)
    s = nesting.sheet
    fig, axes = plt.subplots(max(n, 1), 1, figsize=(10, max(2.2, 10 * s.height_mm / s.width_mm) * max(n, 1)), squeeze=False)
    for i in range(n):
        plot_sheet(nesting.parts_on(i), s, axes[i, 0], operations, labels,
                   title=f"sheet {i + 1}/{nesting.n_sheets}: {len(nesting.parts_on(i))} parts, {100 * nesting.utilisation[i]:.0f} % used")
    fig.tight_layout()
    return fig


def plot_parts(parts: list[Part], ncols: int = 4, operations=None, width_in: float = 12.0):
    """A gallery: every part at its own origin, same scale in every panel, named."""
    import matplotlib.pyplot as plt

    n = len(parts)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    sizes = [p.size() for p in parts]
    w_max = max(s[0] for s in sizes) if sizes else 1.0
    h_max = max(s[1] for s in sizes) if sizes else 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, width_in / ncols * nrows * max(0.25, min(1.0, h_max / w_max)) + 0.4 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, part in zip(axes.flat, parts):
        p = part.at_origin()
        _draw_part(ax, p, operations, False)
        ax.set_xlim(-2, w_max + 2)
        ax.set_ylim(-2, h_max + 2)
        ax.set_aspect("equal")
        w, h = p.size()
        ax.set_title(f"{part.name}  {w:.0f} x {h:.0f} mm", fontsize=7)
    fig.tight_layout()
    return fig
