"""The glove in 3D (matplotlib, from the CadQuery tessellation): every part in its colour, from the back, the side
and the palm, in three poses; and the bill of materials.

    python glove_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from glove import Glove

COLOURS = {"fabric": ("#3b3f46", 0.55), "rings": ("#f39c12", 1.0), "sensors": ("#2e86c1", 1.0), "magnets": ("#c0392b", 1.0),
           "plate": ("#e67e22", 1.0), "cuff": ("#7f8c8d", 1.0), "cables": ("#27ae60", 1.0)}


def _tris(shape, tol=0.6):
    verts, tris = shape.val().tessellate(tol, 0.4)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev, azim, title=None):
    d = Glove()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for name, (c, a) in COLOURS.items():
        tris = _tris(d.build({**p, "part": name}))
        ax.add_collection3d(Poly3DCollection(tris, facecolors=c, edgecolors=c, linewidths=0.0, alpha=a, shade=True))
        pts = np.concatenate(tris)
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 0.72
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)


def main(out="glove_3d.png"):
    p = {q.name: q.default for q in Glove.parameters}
    fig = plt.figure(figsize=(20, 12))
    views = [(dict(), 55, -120, "relaxed, from the back of the hand"), (dict(), 8, -95, "relaxed, from the side (thumb towards you)"),
             (dict(), -60, -120, "relaxed, from the palm"), (dict(pose="flat"), 70, -90, "flat (calibration pose)"),
             (dict(pose="pinch"), 30, -140, "pinch: thumb tip on the index tip"), (dict(pose="fist"), 25, -110, "fist")]
    for k, (over, el, az, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        draw(ax, {**p, **over}, el, az, title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in COLOURS.values()]
    labels = ["stretch fabric glove", "TPU split rings (on the bones)", "Hall sensor pods (14)", "NdFeB magnets (14)",
              "back plate: IMU, MCU, battery, vibration motor", "wrist cuff + strap", "flat cable channels"]
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=10, frameon=False)
    bom = Glove.bom()
    fig.suptitle(f"vegeta.bionic data glove v0 (right, 190 mm hand: 50th percentile male) — {bom['Hall sensor pods']} sensed joints, "
                 f"{bom['bought mass g']:.0f} g of bought parts", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=90)
    fig2 = plt.figure(figsize=(14, 11))
    ax = fig2.add_subplot(1, 1, 1, projection="3d")
    draw(ax, p, 52, -115, "the glove, relaxed, from the back of the hand")
    fig2.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False)
    fig2.tight_layout(rect=(0, 0.06, 1, 1))
    out2 = out.replace(".png", "_back.png")
    fig2.savefig(out2, dpi=110)
    return out, out2


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
