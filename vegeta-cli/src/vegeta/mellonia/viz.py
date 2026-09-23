"""Visualisation for Mellonia: the sliced toolpath in 3D (pyvista) and per-layer views (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .gcode import GcodeInfo, read_gcode


def _pv():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("3D visualisation needs pyvista: pip install 'vegeta-cli[viz]'") from exc
    return pv


def _info(obj) -> GcodeInfo:
    if isinstance(obj, GcodeInfo) and obj.segments is not None:
        return obj
    path = obj.artifacts["gcode"] if hasattr(obj, "artifacts") else (obj.path if isinstance(obj, GcodeInfo) else obj)
    return read_gcode(Path(path), moves=True)


def toolpath_to_pyvista(gcode, extruding_only: bool = True):
    """Extrusion moves as a pyvista PolyData of lines with ``layer`` and ``z`` arrays."""
    pv = _pv()
    info = _info(gcode)
    seg = info.segments
    keep = info.segment_extruding if extruding_only else np.ones(len(seg), dtype=bool)
    seg, layer = seg[keep], info.segment_layer[keep]
    n = len(seg)
    points = np.vstack([seg[:, :3], seg[:, 3:]])
    lines = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()
    poly = pv.PolyData(points, lines=lines)
    poly.cell_data["layer"] = layer
    poly.cell_data["z"] = seg[:, 5]
    return poly


def plot_toolpath(gcode, plotter=None, line_width: float = 2.0, cmap: str = "viridis"):
    """3D toolpath coloured by layer."""
    pv = _pv()
    poly = toolpath_to_pyvista(gcode)
    pl = plotter or pv.Plotter()
    pl.add_mesh(poly, scalars="layer", cmap=cmap, line_width=line_width, scalar_bar_args={"title": "layer"})
    info = _info(gcode)
    m = info.metrics()
    pl.add_text(f"{info.layer_count} layers, {m.get('estimated_time')}, {m.get('filament_used_g')} g", font_size=10)
    pl.add_axes()
    return pl


def plot_layer(gcode, layer: int | None = None, ax=None, travel: bool = True):
    """One layer from above: extrusion moves (solid, coloured by order) and travel moves (dotted)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    info = _info(gcode)
    if layer is None:
        layer = info.layer_count // 2
    mask = info.segment_layer == layer
    seg = info.segments[mask]
    ext = info.segment_extruding[mask]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    lines = seg[ext][:, [0, 1, 3, 4]].reshape(-1, 2, 2)
    lc = LineCollection(lines, cmap="viridis", array=np.arange(len(lines)), linewidths=1.5)
    ax.add_collection(lc)
    if travel and (~ext).any():
        ax.add_collection(LineCollection(seg[~ext][:, [0, 1, 3, 4]].reshape(-1, 2, 2), colors="#bbbbbb",
                                         linestyles="dotted", linewidths=0.8))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(f"layer {layer} (z = {info.layer_z[layer]:.2f} mm), {int(ext.sum())} extrusion moves")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.figure.colorbar(lc, ax=ax, label="print order")
    return ax.figure


def plot_layer_grid(gcode, n: int = 6, cols: int = 3):
    """Several layers from bottom to top in one figure."""
    import matplotlib.pyplot as plt

    info = _info(gcode)
    idx = np.unique(np.linspace(0, info.layer_count - 1, n).astype(int))
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), idx):
        plot_layer(info, int(k), ax=ax, travel=False)
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show(plotter, **kwargs):
    """Show a plotter: interactive in a live Jupyter kernel, static image otherwise."""
    import os

    backend = os.environ.get("PYVISTA_JUPYTER_BACKEND")
    if backend:
        return plotter.show(jupyter_backend=backend, **kwargs)
    try:
        return plotter.show(**kwargs)
    except Exception:
        return plotter.show(jupyter_backend="static", **kwargs)
