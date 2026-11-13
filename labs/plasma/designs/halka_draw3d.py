"""Halka in 3D (matplotlib, from the CadQuery tessellation): the three presets from the front (the air's side), one
oblique view each, and a cut-away through the axis showing the convergence; the masses against the video's.

    python halka_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from halka import Halka

COLOURS = {"frame": ("#2f6db3", 0.28), "grounds": ("#b87333", 1.0), "emitters": ("#d4a017", 1.0),
           "supports": ("#9aa4ad", 1.0), "fasteners": ("#4d5560", 1.0)}
LABELS = ["printed frame (clear resin)", "ground rings, plated (cathodes, downstream)", "bronze emitter rings, serrated (anodes)",
          "standoffs", "screws and terminals"]


def _tris(shape, tol=0.5, half=False, box=None):
    """Triangles of every solid; ``half`` keeps y <= 0 (a cut through the axis), ``box`` keeps those inside it."""
    out = []
    for s in shape.solids().vals():
        verts, tris = s.tessellate(tol, 0.4)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        for t in tris:
            tri = v[list(t)]
            if half and tri[:, 1].min() > 0.0:
                continue
            if box is not None and not ((tri.min(0) >= box[0]).all() and (tri.max(0) <= box[1]).all()):
                continue
            out.append(tri)
    return out


def draw(ax, p, elev, azim, title=None, half=False, box=None, zoom=0.78, parts=COLOURS):
    d = Halka()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for name, (c, a) in parts.items():
        tris = _tris(d.build({**p, "part": name}), half=half, box=box)
        if not tris:
            continue
        lw = 0.4 if name == "emitters" else 0.0
        ax.add_collection3d(Poly3DCollection(tris, facecolors=c, edgecolors=c, linewidths=lw, alpha=a, shade=True))
        pts = np.concatenate(tris)
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * zoom
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)


def main(out="halka_3d.png"):
    base = {q.name: q.default for q in Halka.parameters}
    fig = plt.figure(figsize=(18, 13))
    rows = [("mark4", "Mark IV: 150 mm, 3 copper grounds, 2 emitters, base plate"),
            ("mark4b", "Mark 4B: 200 mm, 4 nickel grounds, 2 emitters, wall + spokes + hub"),
            ("mark4c", "Mark 4C: the 4B's rings, the lighter frame")]
    for i, (preset, title) in enumerate(rows):
        p = {**base, "preset": preset, "teeth": "full" if preset == "mark4" else "none"}
        m = Halka().masses({**p, "teeth": "none"})
        for j, (el, az, half, sub) in enumerate([(90, -90, False, "from the front (air comes towards you)"),
                                                 (35, -55, False, "oblique, the frame translucent"), (22, 90, True, "cut through the axis: grounds (copper) downstream of the emitters")]):
            ax = fig.add_subplot(3, 3, 3 * i + j + 1, projection="3d")
            draw(ax, p, el, az, f"{title if j == 0 else ''}\n{sub}" if j == 0 else sub, half=half, zoom=0.62 if half else 0.78)
            if j == 2:
                ax.text2D(0.02, 0.02, f"model {m['total']:.0f} g (frame {m['frame']:.0f}, grounds {m['grounds']:.0f}, emitters {m['emitters']:.0f})"
                          f" — video {Halka.bom(p)['video\'s mass g']} g", transform=ax.transAxes, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in COLOURS.values()]
    fig.legend(handles, LABELS, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle("Halka v0 — the ion-ring thruster: Plasma Channel's three generations as presets (geometry estimated from the video)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=90)
    # the detail: one sixth of the Mark IV, every tooth drawn, the gap to the grounds
    p = {**base, "preset": "mark4", "teeth": "full"}
    fig2 = plt.figure(figsize=(14, 7))
    for k, (el, az, sub) in enumerate([(30, -70, "a sector of the Mark IV: the serrated bronze emitters between the copper grounds"),
                                       (8, -80, "from the side: the teeth point downstream, 1 mm pitch, 1.5 mm tall")]):
        ax = fig2.add_subplot(1, 2, k + 1, projection="3d")
        draw(ax, p, el, az, sub, box=(np.array([20.0, -14.0, -60.0]), np.array([90.0, 36.0, 60.0])), zoom=0.55)
    fig2.suptitle("Halka — the Mark IV's electrodes up close (the standoffs hold the strips; the gap is what the spark margin watches)")
    fig2.tight_layout()
    out2 = out.replace(".png", "_detail.png")
    fig2.savefig(out2, dpi=100)
    return out, out2


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
