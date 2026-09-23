"""Matplotlib helpers for Talos field results. They return figures and never call ``plt.show()``."""
from __future__ import annotations

import numpy as np


def _fields(obj):
    from .frd import FieldResults, read_frd

    if isinstance(obj, FieldResults):
        return obj
    if hasattr(obj, "artifacts"):  # a talos Result
        return read_frd(obj.artifacts["frd"])
    return read_frd(obj)


def plot_von_mises_histogram(results, bins: int = 40, units: str = "", ax=None):
    """Distribution of nodal von Mises stress. ``results``: Result, FieldResults or .frd path."""
    import matplotlib.pyplot as plt

    fr = _fields(results)
    vm = fr.von_mises
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(vm[np.isfinite(vm)], bins=bins, color="#4c78a8")
    ax.set_xlabel(f"von Mises stress {units}".strip())
    ax.set_ylabel("nodes")
    ax.grid(True, alpha=0.3)
    return ax.figure


def plot_along_axis(results, axis: str = "x", quantity: str = "displacement", component: int | None = None,
                    ax=None):
    """Scatter of a nodal quantity against a coordinate (e.g. deflection along a beam)."""
    import matplotlib.pyplot as plt

    fr = _fields(results)
    x = fr.coords[:, "xyz".index(axis)]
    if quantity == "displacement":
        y = fr.displacement_magnitude if component is None else fr.displacement[:, component]
        label = "|u|" if component is None else f"u{'xyz'[component]}"
    elif quantity == "von_mises":
        y, label = fr.von_mises, "von Mises"
    else:
        raise ValueError("quantity must be 'displacement' or 'von_mises'")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y, ".", ms=2)
    ax.set_xlabel(axis)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    return ax.figure


def plot_deformed(results, scale: float | None = None, color_by: str = "von_mises", ax=None):
    """3D scatter of deformed nodes coloured by von Mises or displacement magnitude.

    ``scale=None`` picks a visual magnification so the largest displacement is 10 % of the model size
    (the scale used is shown in the title; it is a display choice only).
    """
    import matplotlib.pyplot as plt

    fr = _fields(results)
    span = float(np.ptp(fr.coords, axis=0).max()) or 1.0
    umax = float(np.nanmax(fr.displacement_magnitude)) or 1.0
    if scale is None:
        scale = 0.1 * span / umax
    pts = fr.coords + scale * np.nan_to_num(fr.displacement)
    c = fr.von_mises if color_by == "von_mises" else fr.displacement_magnitude
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=2, cmap="viridis")
    ax.figure.colorbar(sc, ax=ax, shrink=0.6, label=color_by)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"deformed shape (x{scale:.3g})")
    return ax.figure
