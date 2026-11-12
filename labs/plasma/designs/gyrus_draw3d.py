"""Gyrus in 3D (matplotlib from the CadQuery tessellation): from the exit, oblique, a cut through the axis; the
masses.    python gyrus_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gyrus import Gyrus

COLOURS = {"channel": ("#e8e1cf", 0.95), "anode": ("#7f8c8d", 1.0), "core": ("#4d5560", 0.9), "coil_in": ("#c8711a", 1.0),
           "coils_out": ("#c8711a", 1.0), "screen": ("#6b7480", 0.35), "cathode": ("#2f6db3", 1.0), "feed": ("#9aa4ad", 1.0)}
LABELS = ["BN channel", "anode + distributor", "iron core + plate + inner pole", "inner coil", "outer coils on cores",
          "iron screen + outer pole", "hollow cathode (bought)", "propellant line"]


def _tris(shape, tol=0.3, half=False):
    """Triangles of every solid; ``half``: the solids cut by the plane y = 0 in CAD first (a real section)."""
    import cadquery as cq
    out = []
    if half:
        keep = cq.Workplane("XY").box(400, 200, 400).translate((0, -100, 0))
        shape = cq.Workplane("XY").add(cq.Compound.makeCompound([s.intersect(keep.val()) for s in shape.solids().vals()]))
    for s in shape.solids().vals():
        verts, tris = s.tessellate(tol, 0.3)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        for t in tris:
            out.append(v[list(t)])
    return out


def draw(ax, p, elev, azim, title=None, half=False, zoom=0.7):
    d = Gyrus()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for name, (c, a) in COLOURS.items():
        tris = _tris(d.build({**p, "part": name}), half=half)
        if not tris:
            continue
        ax.add_collection3d(Poly3DCollection(tris, facecolors=c, edgecolors=c, linewidths=0.0, alpha=a, shade=True))
        pts = np.concatenate(tris)
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * zoom
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)


def main(out="gyrus_3d.png"):
    p = {q.name: q.default for q in Gyrus.parameters}
    m = Gyrus().masses(p)
    fig = plt.figure(figsize=(18, 6.5))
    for k, (el, az, half, title) in enumerate([(70, -90, False, "from the exit plane (the plume comes towards you)"),
                                              (28, -50, False, "oblique: the four outer coils, the cathode at the side"),
                                              (22, 90, True, "cut through the axis: plate, core, inner coil (orange), channel (cream), outer coils, screen")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw(ax, p, el, az, title, half=half, zoom=0.62 if half else 0.72)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in COLOURS.values()]
    fig.legend(handles, LABELS, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Gyrus-300 v0 — a 300 W Hall thruster: channel Ø{Gyrus.section(p).ch.d*1e3:.0f} mm mean, {Gyrus.section(p).ch.h*1e3:.1f} mm wide, "
                 f"{Gyrus.section(p).ch.L*1e3:.1f} mm long; {m['total']:.0f} g (iron {m['core + plate (iron)'] + m['screen + pole (iron)']:.0f}, coils {m['inner coil (Cu)'] + m['outer coils + cores']:.0f}, BN {m['channel (BN)']:.0f})", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=100)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
