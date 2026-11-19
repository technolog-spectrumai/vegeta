"""ARIGO-1's pictures (notebook 39): 3D views from the CAD (matplotlib on the CadQuery tessellation) and the
STEP/STL export of all nine configurations.

    python arigo1_drawings.py [folder]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import arigo1 as a1
from gavilan_drawings import _tris

COL = {"fuselage": ("#8d99a6", 1.0), "wing": ("#c9a66b", 1.0), "tail": ("#6b4f2a", 1.0), "camera": ("#111111", 1.0),
       "motors": ("#3b3f45", 1.0), "props": ("#7fb2e5", 0.3), "hook": ("#c0392b", 1.0)}


def draw(ax, q, elev, azim, title):
    p = a1.resolve(q)
    for part, shape in a1.Arigo1().parts(p).items():
        c, al = COL[part]
        ax.add_collection3d(Poly3DCollection(_tris(shape, 1.0), facecolors=c, edgecolors=c, linewidths=0.0, alpha=al, shade=al == 1.0))
    h, cx = 0.52 * p["span"], p["fuselage_length"] / 2
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(-h, h); ax.set_zlim(-h, h)
    ax.set_box_aspect((1, 1, 1), zoom=1.5); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def render_views(out, variant="hybrid", propulsion="pusher"):
    q = a1.variant(variant, propulsion)
    L = a1.Arigo1.layout(a1.resolve(q))
    fig = plt.figure(figsize=(15, 9.5))
    for k, (e, az, t) in enumerate([(22, -130, "front quarter"), (22, -50, "rear quarter"), (90, -90, "from above"),
                                    (0, -90, "from the side"), (0, 180, "from the front"), (-30, -60, "from below")]):
        draw(fig.add_subplot(2, 3, k + 1, projection="3d"), q, e, az, t)
    p = a1.resolve(q)
    fig.suptitle(f"ARIGO-1 {variant}, {propulsion} — span {p['span']:.0f} mm, length {p['fuselage_length']:.0f} mm, wing {L['S_mm2'] / 1e6:.3f} m², "
                 f"AR {L['AR']:.1f}, LE sweep {p['le_sweep_deg']:.0f}°, tail volumes V_h {L['V_h']:.2f} V_v {L['V_v']:.3f}", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)
    return out


def render_matrix(out):
    fig = plt.figure(figsize=(15, 13))
    k = 0
    for v in ("goshawk", "hybrid", "zero"):
        for pr in a1.PROPULSION:
            k += 1
            draw(fig.add_subplot(3, 3, k, projection="3d"), a1.variant(v, pr), 28, -125, f"{v} — {pr}")
    fig.suptitle("ARIGO-1: three wings x three propulsion layouts (propeller discs translucent)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=95); plt.close(fig)
    return out


def export_all(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for v in a1.VARIANTS:
        for pr in a1.PROPULSION:
            g = a1.Arigo1().generate(**a1.variant(v, pr))
            g.export_step(str(folder / f"arigo1_{v}_{pr}.step")); g.export_stl(str(folder / f"arigo1_{v}_{pr}.stl"), tolerance=0.3)


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    render_views(d / "arigo1_hybrid_pusher_3d.png")
    render_matrix(d / "arigo1_matrix_3d.png")
    export_all(d / "arigo1")
