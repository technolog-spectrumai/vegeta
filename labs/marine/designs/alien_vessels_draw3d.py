"""The two fictional alien vessels in 3D (matplotlib, from the CadQuery tessellation), each component in its colour,
and the emblem on its flag.   python alien_vessels_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import alien_vessels as av


def _draw(ax, D, parts, lim, elev, azim, z_sea=0.0, title=""):
    d, p = D(), D.p()
    tris, cols = [], []
    for k in parts:
        verts, tr = d.build(p | {"part": k}).val().tessellate(60.0, 0.3)
        v = np.array([(q.x, q.y, q.z) for q in verts]) / 1000
        t = [v[list(i)] for i in tr]
        tris += t
        cols += [av.COLOURS[k]] * len(t)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.05, shade=True, lightsource=None))
    (x0, x1), (y0, y1), (z0, z1) = lim
    if z_sea is not None:
        X, Y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
        ax.plot_surface(X, Y, np.full_like(X, z_sea), color="#5d8aa8", alpha=0.15)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=1.0); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def _emblem(ax):
    w, b = av.emblem_2d(1000.0, 10.0)
    ax.set_facecolor("#0a0a0a")
    for shp, col in ((w, "#f2f2f2"), (b, "#050505")):
        verts, tr = shp.val().tessellate(2.0, 0.2)
        v = np.array([(q.x, q.y) for q in verts])
        for t in tr:
            ax.fill(*v[list(t)].T, color=col, lw=0)
    ax.set_xlim(-750, 750); ax.set_ylim(-500, 500); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("the flag (3:2) and the hull emblem", fontsize=10)


def main(out="alien_vessels_3d.png"):
    s, u = av.AlienShip.sizing(), av.AlienSubmarine.sizing()
    fig = plt.figure(figsize=(20, 12), facecolor="white")
    _draw(fig.add_axes([-0.04, 0.44, 0.7, 0.5], projection="3d"), av.AlienShip, av.SHIP_PARTS, ((-72, 72), (-12, 12), (-6, 32)), 16, -58,
          title=f"alien ship: {s['length_m']:g} x {s['beam_m']:g} m, hull {s['freeboard_m']:g} m above the water, {s['height_m']:g} m overall; "
                f"superstructure {s['superstructure_m'][0]:g} x {s['superstructure_m'][1]:g} m; {s['displacement_t']:.0f} t")
    _draw(fig.add_axes([-0.04, -0.02, 0.7, 0.5], projection="3d"), av.AlienSubmarine, av.SUB_PARTS, ((-50, 50), (-8, 8), (-7, 16)), 14, -60, z_sea=4.0,
          title=f"alien submarine (surfaced, flag up): {u['length_m']:g} m, {u['diameter_m']:g} m hull, {u['height_m']:g} m with the sail; "
                f"sail {u['sail_m'][0]:g} x {u['sail_m'][1]:g} x {u['sail_m'][2]:g} m; {u['displacement_t']:.0f} t submerged")
    _emblem(fig.add_axes([0.66, 0.58, 0.3, 0.3]))
    _draw(fig.add_axes([0.62, 0.04, 0.38, 0.46], projection="3d"), av.AlienShip, av.SHIP_PARTS, ((0, 72), (-12, 12), (-5, 32)), 6, -110,
          title="the ship's bow quarter: the emblem on the hull, the bridge's window band, the sensor tower")
    fig.suptitle("Fictional extraterrestrial vessels - non-functional exterior models for the sea game", fontsize=13, y=0.995)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
