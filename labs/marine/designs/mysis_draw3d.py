"""Mysis in 3D (matplotlib, from the CadQuery tessellation): one unit from three sides, and a group of four on a
ship's side, each on its lane.    python mysis_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mysis import BRUSH, Mysis

COLOURS = {"shell": "#3d7ea6", "housing": ("#9fc5e8", 0.5), "thrusters": "#222222", "brush": "#c0392b", "altimeters": "#7f8c8d",
           "lights": "#f1c40f", "top": "#e67e22", "handles": "#7f8c8d"}


def _tris(shape, tol=2.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def unit_polys(p, offset=(0, 0, 0), R=None):
    d = Mysis()
    out = []
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        tris = _tris(d.build({**p, "part": name}))
        if R is not None:
            tris = [t @ R.T for t in tris]
        out.append(([t + np.asarray(offset) for t in tris], col, alpha))
    return out


def draw_unit(ax, p, elev, azim, title):
    for tris, col, alpha in unit_polys(p):
        ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
    ax.set_xlim(-400, 400); ax.set_ylim(-400, 400); ax.set_zlim(-150, 650)
    ax.set_box_aspect((800, 800, 800)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def draw_group(ax, p, n=4):
    """Four units belly-to-hull on a ship's side (the hull a vertical plane at y = 0, the units' brushes on it),
    each on its lane, 0.8 m apart along the ship's length."""
    # a unit lying on its side: z (brush down) -> -y (towards the hull); x stays along the ship
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)         # (x, y, z) -> (x, -z, y)
    for k in range(n):
        off = (k * 900.0 - 1350.0, 0.0, 2100.0 - k * 600.0)                     # the lanes drawn 0.6 m apart for legibility
        for tris, col, alpha in unit_polys(p, off, R):
            ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
        ax.plot([-2400, 2400], [0, 0], [off[2], off[2]], color="#c0392b", lw=0.8, ls="--")            # the lane
    X, Z = np.meshgrid(np.linspace(-2600, 2600, 2), np.linspace(-100, 2900, 2))
    ax.plot_surface(X, np.zeros_like(X), Z, color="#5d6d7e", alpha=0.25)
    for z in np.linspace(0, 2800, 5):
        ax.plot([-2600, 2600], [0, 0], [z, z], color="#34495e", lw=0.5)                                    # plate seams
    ax.set_xlim(-2600, 2600); ax.set_ylim(-2600, 2600); ax.set_zlim(-1200, 4000)
    ax.set_box_aspect((5200, 5200, 5200)); ax.view_init(elev=10, azim=-40); ax.set_axis_off()
    ax.set_title("a group of four on a ship's side: belly to the hull, each on its lane (red), the seams as their ruler", fontsize=10)


def main(out="mysis_3d.png"):
    p = Mysis.p()
    fig = plt.figure(figsize=(18, 11))
    draw_unit(fig.add_subplot(2, 3, 1, projection="3d"), p, 20, -50, "the unit: dome, lights, four vectored and two vertical thrusters")
    draw_unit(fig.add_subplot(2, 3, 2, projection="3d"), p, -28, -120, "from below: the brush head (the magnetic roller pad round it) and the four altimeters")
    draw_unit(fig.add_subplot(2, 3, 3, projection="3d"), p, 90, -90, "from above: the USBL transponder, the modem, the handles")
    draw_group(fig.add_subplot(2, 1, 2, projection="3d"), p)
    s = Mysis.sizing()
    fig.suptitle(f"Mysis v0 — the group submarine for maritime maintenance: {s['mass_in_air_kg']:.1f} kg a unit, {Mysis.p()['length']:.0f} × {Mysis.p()['width']:.0f} × {Mysis.p()['height']:.0f} mm, "
                 f"{s['rate_m2_h']:.0f} m²/h grooming, {s['groom_h']:.1f} h a battery; six on a node groom a {Mysis.p()['ship_hull_m2']:.0f} m² hull in {s['shift_h']:.1f} h; "
                 f"€{s['unit_price_eur']/1e3:.0f}k a unit, €{s['fleet_price_eur']/1e3:.0f}k the fleet of six", fontsize=11)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93, wspace=0.0, hspace=0.0)
    fig.savefig(out, dpi=90)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
