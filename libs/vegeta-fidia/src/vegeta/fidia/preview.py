"""Interactive 3D preview of a revision's parts (pyvista): rotate and zoom in a live notebook, a static image
otherwise (``PYVISTA_JUPYTER_BACKEND`` chooses, as in the other Vegeta viewers)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .mesh import Part, load_parts


def _parts(target) -> list[Part]:
    if isinstance(target, (list, tuple)):
        return list(target)
    if hasattr(target, "parts") and callable(target.parts):  # a Revision
        return target.parts()
    path = Path(target)
    return load_parts(path if (path / "result.json").is_file() else path / "best")


def plotter(target, *, window_size=(900, 600), edges: bool = True, off_screen: bool | None = None):
    """A pyvista ``Plotter`` with every part in its colour (and its name in the legend)."""
    import pyvista as pv

    parts: Sequence[Part] = _parts(target)
    pl = pv.Plotter(window_size=window_size, off_screen=off_screen)
    pl.set_background("white")
    for p in parts:
        faces = np.hstack([np.full((len(p.triangles), 1), 3), p.triangles]).ravel()
        mesh = pv.PolyData(np.asarray(p.vertices, float), faces).clean(tolerance=1e-9)
        pl.add_mesh(mesh, color=p.color[:3], opacity=p.color[3], smooth_shading=True, split_sharp_edges=True,
                    label=p.name)
        if edges:
            fe = mesh.extract_feature_edges(boundary_edges=True, feature_edges=True, manifold_edges=False, feature_angle=35)
            if fe.n_cells:
                pl.add_mesh(fe, color="black", line_width=1)
    if len(parts) > 1:
        pl.add_legend(bcolor="white", face="rectangle", size=(0.18, min(0.05 * len(parts), 0.5)))
    pl.add_axes()
    pl.view_isometric()
    return pl


def show(target, **kwargs):
    """Show ``target`` (a Revision, a run or revision directory, or parts): interactive in a live kernel."""
    pl = plotter(target, **{k: kwargs.pop(k) for k in ("window_size", "edges", "off_screen") if k in kwargs})
    backend = os.environ.get("PYVISTA_JUPYTER_BACKEND")
    if backend:
        return pl.show(jupyter_backend=backend, **kwargs)
    try:
        return pl.show(**kwargs)
    except Exception:
        return pl.show(jupyter_backend="static", **kwargs)
