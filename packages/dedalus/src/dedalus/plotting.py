"""Matplotlib helpers. They return figures and never call ``plt.show()``."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_views(geometry, tolerance: float = 0.2, figsize=(10, 8)):
    """Front/side/top orthographic projections and an isometric view of the tessellated geometry."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, t = geometry.tessellate(tolerance)
    tris = v[t]
    u = geometry.units
    fig = plt.figure(figsize=figsize)
    for i, (title, (a, b)) in enumerate([("front (XZ)", (0, 2)), ("side (YZ)", (1, 2)), ("top (XY)", (0, 1))], 1):
        ax = fig.add_subplot(2, 2, i)
        ax.add_collection(PolyCollection(tris[:, :, [a, b]], facecolor="#9fb8d0", edgecolor="#40556b", linewidths=0.1))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(f"{'xyz'[a]} [{u}]")
        ax.set_ylabel(f"{'xyz'[b]} [{u}]")
        ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolor="#9fb8d0", edgecolor="#40556b", linewidths=0.1))
    lo, hi = v.min(axis=0), v.max(axis=0)
    centre, span = (lo + hi) / 2, max((hi - lo).max(), 1e-9) / 2
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=30, azim=-60)
    ax.set_title("iso")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    fig.suptitle(f"{geometry.name} [{u}]")
    fig.tight_layout()
    return fig


def plot_parameter_study(geometries: Sequence, parameter: str, metric: str = "volume", ax=None):
    """Plot a measurement against one parameter across already-generated geometries."""
    import matplotlib.pyplot as plt

    xs = np.array([g.parameters[parameter] for g in geometries], dtype=float)
    ys = np.array([np.nan if g.measure()[metric] is None else g.measure()[metric] for g in geometries])
    order = np.argsort(xs)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(xs[order], ys[order], "o-")
    ax.set_xlabel(parameter)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    return ax.figure


def measurements_table(geometries: Sequence):
    """``pandas.DataFrame`` of parameters and scalar measurements (requires the ``pandas`` extra)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("measurements_table needs pandas: pip install 'dedalus[pandas]'") from exc
    rows = []
    for g in geometries:
        m = g.measure()
        row = {"name": g.name, **g.parameters}
        row.update({k: m[k] for k in ("volume", "surface_area", "valid", "n_solids")})
        row.update(dict(zip(("dx", "dy", "dz"), m["dimensions"])))
        rows.append(row)
    return pd.DataFrame(rows)
