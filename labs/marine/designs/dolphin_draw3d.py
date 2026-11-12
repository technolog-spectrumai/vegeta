"""Dolphin in 3D (matplotlib, from the CadQuery tessellation): surfaced with an Emerillion standing on the pad, from
three sides, and the day's numbers.    python dolphin_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dolphin import Dolphin

COLOURS = {"hull": ("#2c5f8a", 0.95), "sail": "#34495e", "hangar": ("#7f8c8d", 0.7), "pad": "#e67e22", "wetwell": "#c0392b",
           "propulsor": "#222222", "drone": "#f1c40f"}


def _tris(shape, tol=4.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev, azim, title, sea=True):
    d = Dolphin()
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        ax.add_collection3d(Poly3DCollection(_tris(d.build({**p, "part": name})), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
    if sea:
        wl = p["hull_d"] - p["freeboard"]
        X, Y = np.meshgrid(np.linspace(-2600, 2600, 2), np.linspace(-2600, 2600, 2))
        ax.plot_surface(X, Y, np.full_like(X, wl), color="#7fb3d5", alpha=0.35)
    ax.set_xlim(-2600, 2600); ax.set_ylim(-2600, 2600); ax.set_zlim(-1800, 3400)
    ax.set_box_aspect((5200, 5200, 5200)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main(out="dolphin_3d.png"):
    p = Dolphin.p()
    fig = plt.figure(figsize=(18, 6.5))
    draw(fig.add_subplot(1, 3, 1, projection="3d"), p, 18, -55, "surfaced: the sail and mast, the hatch open, an Emerillion nose-up on the pad")
    draw(fig.add_subplot(1, 3, 2, projection="3d"), p, 8, -100, "from the side at the waterline: the deck dry")
    draw(fig.add_subplot(1, 3, 3, projection="3d"), p, -22, -40, "from below: the wet-well and its hook for Walrus's basket lines, the ducted propulsor", sea=False)
    s = Dolphin.sizing()
    fig.suptitle(f"Dolphin v0 — the pearl-farm submarine: {Dolphin.p()['length']/1e3:.1f} m, Ø{Dolphin.p()['hull_d']:.0f} mm, {s['mass_kg']:.0f} kg, {Dolphin.p()['cruise']:.1f} m/s; "
                 f"a day of 40 km on {s['day_wh']/1e3:.1f} kWh (margin {s['energy_margin']:+.0%}); a {s['capsule_pearls']:.0f}-pearl capsule every {s['capsule_h']*60:.0f} min flown {Dolphin.p()['vessel_km']:.0f} km in {s['flight_min']:.0f} min; €{s['price_eur']/1e3:.0f}k", fontsize=11)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.0)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
