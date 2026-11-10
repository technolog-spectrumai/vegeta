"""Walrus in 3D (matplotlib, from the CadQuery tessellation): on a patch of clam bed from three sides; the harvest
numbers.    python walrus_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from walrus import Walrus

COLOURS = {"tracks": "#222222", "frame": "#7f8c8d", "housing": ("#9fc5e8", 0.6), "gantry": "#e67e22", "basket": "#c0392b",
           "thrusters": "#2c3e50", "lights": "#f1c40f", "arms": "#8e6b3e"}


def _tris(shape, tol=2.5):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev, azim, title, bed=True):
    d = Walrus()
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        ax.add_collection3d(Poly3DCollection(_tris(d.build({**p, "part": name})), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
    if bed:
        X, Y = np.meshgrid(np.linspace(-650, 650, 2), np.linspace(-650, 650, 2))
        ax.plot_surface(X, Y, np.zeros_like(X), color="#d9c89e", alpha=0.5)
        rng = np.random.default_rng(3)
        pts = rng.uniform([-620, -620], [620, 620], (60, 2))
        ax.scatter(pts[:, 0], pts[:, 1], np.full(60, 2.0), s=12, color="#6e5a3a", marker="o")             # the shows
    ax.set_xlim(-650, 650); ax.set_ylim(-650, 650); ax.set_zlim(-250, 1050)
    ax.set_box_aspect((1300, 1300, 1300)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main(out="walrus_3d.png"):
    p = Walrus.p()
    fig = plt.figure(figsize=(18, 6.5))
    draw(fig.add_subplot(1, 3, 1, projection="3d"), p, 22, -50, "on the bed: the digging arms, camera and lights ahead of the gantry")
    draw(fig.add_subplot(1, 3, 2, projection="3d"), p, 20, 130, "from behind: the basket, four vectored + four vertical ducted thrusters")
    draw(fig.add_subplot(1, 3, 3, projection="3d"), p, -30, -60, "from below: the gantry's grippers and jet in the swath", bed=False)
    s = Walrus.sizing()
    fig.suptitle(f"Walrus (round 2: 8 big slow ducted thrusters) — {s['mass_wet_vehicle_kg']:.0f} kg crawler ({s['mass_in_air_kg']:.0f} with the tether), sits {s['sink_kg']:.0f} kg on the bed; "
                 f"{s['rate_per_h']:.0f} clams/h with two pickers, {s['rate_kg_h']:.0f} kg/h, {s['day_kg']:.0f} kg a day in two shifts ({s['vs_diver']:.1f}× a diver); €{s['price_eur']/1e3:.0f}k, payback {s['payback_days']:.0f} days", fontsize=11)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.0)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
