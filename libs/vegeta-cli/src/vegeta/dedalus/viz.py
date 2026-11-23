"""Visualisation for Dedalus geometry: interactive 3D (pyvista) and 2D sections (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _pv():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("3D visualisation needs pyvista: pip install 'vegeta-cli[viz]'") from exc
    return pv


def to_pyvista(geometry, tolerance: float = 0.05, angular_tolerance: float = 0.2):
    """Tessellated geometry (or an STL path) as a pyvista PolyData."""
    pv = _pv()
    if isinstance(geometry, (str, Path)):
        return pv.read(str(geometry))
    v, t = geometry.tessellate(tolerance, angular_tolerance)
    faces = np.hstack([np.full((len(t), 1), 3), t]).ravel()
    return pv.PolyData(v, faces)


def plot3d(*geometries, colors=None, opacity: float = 1.0, show_edges: bool = False, plotter=None,
           labels: bool = True):
    """Interactive 3D view of one or more geometries (or STL paths)."""
    pv = _pv()
    pl = plotter or pv.Plotter()
    palette = colors or ["#9fb8d0", "#d0a29f", "#a3d09f", "#cfc39f", "#b39fd0", "#9fd0cb"]
    for i, g in enumerate(geometries):
        name = getattr(g, "name", Path(str(g)).stem)
        pl.add_mesh(to_pyvista(g), color=palette[i % len(palette)], opacity=opacity, show_edges=show_edges,
                    label=name, smooth_shading=True)
    if labels and len(geometries) > 1:
        pl.add_legend()
    pl.add_axes()
    pl.show_grid()
    return pl


def section_polylines(geometry, normal: str = "z", position: float = 0.0, tolerance: float = 0.05):
    """Section curves of the geometry on the plane ``normal = position`` as a list of (N,2) arrays."""
    axis = "xyz".index(normal)
    origin = [0.0, 0.0, 0.0]
    origin[axis] = position
    sl = to_pyvista(geometry, tolerance).slice(normal=normal, origin=origin)
    if sl.n_points == 0:
        return []
    sl = sl.strip()  # join the cut segments into connected polylines
    keep = [i for i in range(3) if i != axis]
    lines = sl.lines
    out = []
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        out.append(sl.points[ids][:, keep])
        i += n + 1
    return out


def plot_sections(geometry, normal: str = "z", positions=None, n: int = 4, tolerance: float = 0.05, cols: int = 2):
    """Grid of 2D sections through the geometry (matplotlib). ``positions`` default: n evenly spaced cuts."""
    import matplotlib.pyplot as plt

    axis = "xyz".index(normal)
    lo, hi = geometry.bounding_box if hasattr(geometry, "bounding_box") else _bbox(geometry)
    if positions is None:
        a, b = lo[axis], hi[axis]
        positions = [a + (b - a) * (k + 0.5) / n for k in range(n)]
    keep = [i for i in range(3) if i != axis]
    rows = (len(positions) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax, pos in zip(axes.ravel(), positions):
        for poly in section_polylines(geometry, normal, pos, tolerance):
            ax.plot(poly[:, 0], poly[:, 1], color="#1f3b5a", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlim(lo[keep[0]], hi[keep[0]])
        ax.set_ylim(lo[keep[1]], hi[keep[1]])
        ax.set_title(f"{normal} = {pos:.3g}")
        ax.set_xlabel("xyz"[keep[0]])
        ax.set_ylabel("xyz"[keep[1]])
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(positions):]:
        ax.axis("off")
    fig.suptitle(f"{getattr(geometry, 'name', 'geometry')} — sections normal to {normal}")
    fig.tight_layout()
    return fig


def _bbox(stl_path):
    m = to_pyvista(stl_path)
    b = m.bounds
    return [b[0], b[2], b[4]], [b[1], b[3], b[5]]


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
