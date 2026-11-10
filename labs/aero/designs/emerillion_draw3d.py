"""The X-wing Emerillion in 3D (matplotlib, from the CadQuery tessellation): level in flight from three sides, and
standing on its fins; the numbers of the pearl run.    python emerillion_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from emerillion import Emerillion, hover, pearl_run

COLOURS = {"fuselage": "#f1c40f", "wing": "#f1c40f", "tail": "#4d4d4d", "canopy": ("#8ec0e4", 0.8)}


def _tris(shape, tol=1.5):
    out = []
    for s in shape.solids().vals():
        verts, tris = s.tessellate(tol, 0.4)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        out += [v[list(t)] for t in tris]
    return out


def draw(ax, p, elev, azim, title, standing=False):
    e = Emerillion()
    L = p["length"]
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        shape = e.build({**p, "part": name})
        if standing:                                   # nose up on the fin pads: rotate +X to +Z about the tail
            shape = shape.rotate((0, 0, 0), (0, 1, 0), 90).translate((0, 0, L))
        ax.add_collection3d(Poly3DCollection(_tris(shape), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=True))
    h = L * 0.6
    cx = (0, 0, L / 2) if standing else (L / 2, 0, 0)
    ax.set_xlim(cx[0] - h, cx[0] + h); ax.set_ylim(cx[1] - h, cx[1] + h); ax.set_zlim(cx[2] - h, cx[2] + h)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main(out="emerillion_3d.png"):
    p = Emerillion.p()
    fig = plt.figure(figsize=(14, 9))
    for k, (elev, azim, title, standing) in enumerate([(28, -50, "from the front quarter", False), (90, -90, "from above", False),
                                                        (0, -90, "from the side", False), (18, -35, "standing on its fins (on the pad)", True)]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        draw(ax, p, elev, azim, title, standing)
    hv, r = hover(), pearl_run()
    fig.suptitle(f"Emerillion (X-wing) — Dolphin's pearl drone: {p['length']/1e3:.2f} m, span {p['span']/1e3:.1f} m with the two 80 mm fan pods, {hv['mass_kg']:.2f} kg with the capsule, "
                 f"T/W {hv['thrust_to_weight']:.2f}, hovers at {hv['hover_W']:.0f} W\ncruise 28 m/s (stall {r['stall_m_s']:.0f}); a 10 km pearl run {r['minutes']:.0f} min, {r['wh']:.0f} Wh, "
                 f"reserve {r['reserve']:+.0%}; {r['range_km_reserve']:.0f} km each way with 20 % kept", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
