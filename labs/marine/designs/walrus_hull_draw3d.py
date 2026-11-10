"""Walrus's full hull: the fairing it swims in, over the frame, with every part in place — six views and a cut
through the centreline.    python walrus_hull_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from walrus import Walrus

COLOURS = {"hull": ("#1f5f8b", 0.9), "tracks": "#222222", "frame": "#7f8c8d", "housing": ("#9fc5e8", 0.7), "gantry": "#e67e22",
           "basket": "#c0392b", "thrusters": "#2c3e50", "lights": "#f1c40f", "arms": "#8e6b3e"}
LABELS = ["hull fairing (HDPE, two halves)", "tracks", "frame + deck", "electronics tube", "belly gantry: grippers + jet",
          "basket (release on its float line)", "thrusters: 2 vertical, 2 horizontal", "camera + lights", "digging arms"]


def _tris(shape, tol=2.0, cut=False):
    import cadquery as cq
    out = []
    if cut:
        keep = cq.Workplane("XY").box(3000, 1500, 3000).translate((0, -750, 0))
        shape = cq.Workplane("XY").add(cq.Compound.makeCompound([s.intersect(keep.val()) for s in shape.solids().vals()]))
    for s in shape.solids().vals():
        verts, tris = s.tessellate(tol, 0.5)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        out += [v[list(t)] for t in tris]
    return out


def draw(ax, p, elev, azim, title, cut=False, hull_alpha=None):
    d = Walrus()
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        if name == "hull" and hull_alpha is not None:
            alpha = hull_alpha
        tris = _tris(d.build({**p, "part": name}), cut=cut)
        if tris:
            ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
    ax.set_xlim(-600, 600); ax.set_ylim(-600, 600); ax.set_zlim(-250, 950)
    ax.set_box_aspect((1200, 1200, 1200)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main(out="walrus_hull_3d.png"):
    p = Walrus.p()
    fig = plt.figure(figsize=(18, 11))
    views = [(22, -50, "three-quarter front: the nose opening, arms, flared flanks", False, None),
             (22, 130, "three-quarter rear: the basket hatch, the thrusters' bores, the tether port", False, None),
             (0, -90, "side", False, None),
             (90, -90, "from above: the two vertical bores, the hatch", False, None),
             (22, -50, "the hull translucent: everything in its place", False, 0.25),
             (18, 90, "cut on the centreline: tube, gantry, basket under the shell", True, None)]
    for k, (el, az, title, cut, ha) in enumerate(views):
        draw(fig.add_subplot(2, 3, k + 1, projection="3d"), p, el, az, title, cut=cut, hull_alpha=ha)
    handles = [plt.Rectangle((0, 0), 1, 1, color=(c if isinstance(c, str) else c[0])) for c in COLOURS.values()]
    fig.legend(handles, LABELS, loc="lower center", ncol=5, fontsize=9, frameon=False)
    s = Walrus.sizing()
    fig.suptitle(f"Walrus — the full hull: {Walrus.p()['length']:.0f} × {Walrus.p()['width']:.0f} × 420 mm, {s['mass_wet_vehicle_kg']:.0f} kg, sits {s['sink_kg']:.1f} kg heavy, swims {s['v_swim_m_s']:.2f} m/s", fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.94, wspace=0.0, hspace=0.0)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
