"""Matplotlib 3-D views of the CAD (no GPU, no display): each part is tessellated by OpenCascade and drawn as shaded
triangles. ``render(variant, path)`` writes a 4-view PNG (isometric, front, top, side)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .cad import Auseklis, _category  # noqa: E402

COLORS = {"structure": "#c9ccd1", "avionics": "#5b7fa6", "battery": "#b5651d", "gnc": "#6a9955", "prox_sensors": "#2b2b2b",
          "comms": "#d4a017", "arrays": "#1f3b73", "capture": "#c0392b", "propulsion_hw": "#8e8e93", "tanks": "#4aa3a2"}
# (title, elevation, azimuth, zoom on the bus?)
VIEWS = [("isometric, deployed", 22, -52, False), ("isometric zoom, bus shown transparent", 25, -38, True),
         ("top (+Z)", 90, -90, False), ("side (+Y)", 0, -90, True)]


def tessellate(shape, tol=4.0):
    v, t = shape.tessellate(tol, 0.3)
    v = np.array([[p.x, p.y, p.z] for p in v])
    return v, np.array(t)


def render(variant: str = "A", path: str = "auseklis.png", tol: float = 4.0, dpi: int = 130, **overrides) -> str:
    parts = Auseklis().generate_parts(variant=variant, **overrides)
    meshes = {n: tessellate(s, tol) for n, s in parts.items()}
    allv = np.vstack([m[0] for m in meshes.values()])
    lo, hi = allv.min(0), allv.max(0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(14, 10.5))
    bus = meshes["bus_shell"][0]
    bctr = np.array([(bus[:, 0].min() + 2600) / 2 + 300, 0.0, 0.0])
    for k, (title, elev, azim, zoom) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(2, 2, k, projection="3d", proj_type="ortho")
        for name, (v, t) in meshes.items():
            tri = v[t]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = 0.45 + 0.55 * np.abs(n @ light)
            base = np.array(matplotlib.colors.to_rgb(COLORS[_category(name)]))
            fc = np.clip(base * shade[:, None], 0, 1)
            if zoom and name == "bus_shell":                       # see-through bus in the zoom view: tanks, deck, equipment
                fc = np.column_stack([fc, np.full(len(fc), 0.12)])
            ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0))
        c, h = (bctr, 1500.0) if zoom else (ctr, half)
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_zlim(c[2] - h, c[2] + h)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev, azim)
        ax.set_title(title, fontsize=10)
        if elev in (0, 90):                                    # flat views: no grid, just a scale bar and the axes' meaning
            ax.set_axis_off()
            ax.text2D(0.02, 0.02, "X (toward the target) points right", transform=ax.transAxes, fontsize=7)
        else:
            ax.set_xlabel("X (to target) mm", fontsize=7)
            ax.set_ylabel("Y mm", fontsize=7)
            ax.set_zlabel("Z mm", fontsize=7)
            ax.tick_params(labelsize=6)
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=c, label=n) for n, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Auseklis variant {variant}: CadQuery model, matplotlib render ({len(parts)} parts)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
