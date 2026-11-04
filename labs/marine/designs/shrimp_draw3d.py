"""Shrimp in 3D (matplotlib, from the CadQuery tessellation): with each tool, beside a patch of net, and the backpack
kit's numbers.    python shrimp_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from shrimp import REQUIREMENTS, TOOLS, Shrimp

COLOURS = {"shell": "#e8734a", "housing": ("#9fc5e8", 0.55), "thrusters": "#222222", "lights": "#f1c40f", "tool_body": "#2980b9",
           "handle": "#7f8c8d", "skids": "#7f8c8d"}


def _tris(shape, tol=1.5):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev=18, azim=-50, net=True):
    d = Shrimp()
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        shp = d.build({**p, "part": name})
        ax.add_collection3d(Poly3DCollection(_tris(shp), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
    if net:
        xn = p["length"] / 2 + 120
        for v in np.linspace(-220, 220, 12):
            ax.plot([xn, xn], [v, v], [-120, 280], color="#7f8c8d", lw=0.6)
        for h in np.linspace(-120, 280, 11):
            ax.plot([xn, xn], [-220, 220], [h, h], color="#7f8c8d", lw=0.6)
    ax.set_xlim(-240, 320); ax.set_ylim(-280, 280); ax.set_zlim(-150, 410)
    ax.set_box_aspect((560, 560, 560)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


def main(out="shrimp_3d.png"):
    fig = plt.figure(figsize=(18, 10))
    views = [("none", 18, -50, "camera only, beside the net"), ("sampler", 18, -50, "with the 250 mL syringe sampler"),
             ("probe", 18, -50, "with the O₂ / temperature / salinity pod"), ("clip", 18, -50, "with the six-clip magazine"),
             ("none", 90, -90, "from above: four vectored, two vertical thrusters, the handle"), ("none", -25, -120, "from below: the tool port under the nose, the skids")]
    for k, (tool, el, az, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        draw(ax, Shrimp.p({"tool": tool}), el, az, net=(k < 4))
        ax.set_title(title, fontsize=10)
    s = Shrimp.sizing()
    fig.suptitle(f"Shrimp v0 — the backpack aquaculture craft: {s['mass_in_air_kg']:.1f} kg in air ({s['backpack_kg']:.1f} kg kit), "
                 f"{Shrimp.p()['length']:.0f} × {Shrimp.p()['width']:.0f} × {Shrimp.p()['height']:.0f} mm, {s['v_max_m_s']:.1f} m/s, "
                 f"{s['dive_h']:.1f} h a battery, holds {s['current_max_m_s']:.2f} m/s on 25 m of tether; kit €{s['kit_price_eur']:.0f}", fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.0, hspace=0.0)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
