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
       "motors": ("#3b3f45", 1.0), "props": ("#7fb2e5", 0.3), "blades": ("#1f2a33", 1.0), "hook": ("#c0392b", 1.0)}


def _mesh(shape, tol=0.6):
    import pyvista as pv
    import numpy as np
    pts, faces, off = [], [], 0
    for so in shape.solids().vals():
        v, t = so.tessellate(tol, 0.2)
        pts += [(q.x, q.y, q.z) for q in v]
        faces += [x for tri in t for x in (3, tri[0] + off, tri[1] + off, tri[2] + off)]
        off += len(v)
    return pv.PolyData(np.array(pts), np.array(faces))


def gl_views(out, cls, q, views, title, size=(560, 420)):
    """Proper (z-buffered) renders with pyvista/VTK, off screen; one panel per (elevation, azimuth, title, overrides)."""
    import pyvista as pv
    import math
    imgs = []
    for elev, azim, t, over in views:
        p = cls().resolve(**{**q, **over})
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.set_background("white")
        for part, shape in cls().parts(p).items():
            c, al = COL.get(part, ("#888888", 1.0))
            pl.add_mesh(_mesh(shape), color=c, opacity=al, smooth_shading=False, specular=0.2)
        pl.enable_parallel_projection()
        cx = p["fuselage_length"] / 2
        e, a = math.radians(elev), math.radians(azim)
        d = 3000.0
        pos = (cx + d * math.cos(e) * math.cos(a), d * math.cos(e) * math.sin(a), d * math.sin(e))
        up = (0, 0, 1) if abs(elev) < 89 else (-1, 0, 0)
        pl.camera_position = [pos, (cx, 0, 0), up]
        pl.reset_camera(); pl.camera.zoom(1.25)
        imgs.append((pl.screenshot(return_img=True), t))
        pl.close()
    n = len(imgs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.1 * rows))
    for ax in axs.flat:
        ax.set_axis_off()
    for ax, (im, t) in zip(axs.flat, imgs):
        ax.imshow(im); ax.set_title(t, fontsize=11)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


SIX = [(22, -130, "front quarter", {}), (22, -50, "rear quarter", {}), (90, -90, "from above", {}),
       (0, -90, "from the side", {}), (0, 180, "from the front", {}), (-25, -125, "from below", {})]


def draw(ax, q, elev, azim, title, cls=None):
    cls = cls or a1.Arigo1
    p = cls().resolve(**q)
    for part, shape in cls().parts(p).items():
        c, al = COL[part]
        ax.add_collection3d(Poly3DCollection(_tris(shape, 1.0), facecolors=c, edgecolors=c, linewidths=0.0, alpha=al, shade=al == 1.0))
    h, cx = 0.52 * p["span"], p["fuselage_length"] / 2
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(-h, h); ax.set_zlim(-h, h)
    ax.set_box_aspect((1, 1, 1), zoom=1.5); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def render_views(out, variant="hybrid", propulsion="pusher"):
    q = a1.variant(variant, propulsion)
    p = a1.resolve(q)
    L = a1.Arigo1.layout(p)
    return gl_views(out, a1.Arigo1, {**q, "show_props": True}, SIX,
                    f"ARIGO-1 {variant}, {propulsion} — span {p['span']:.0f} mm, length {p['fuselage_length']:.0f} mm, wing {L['S_mm2'] / 1e6:.3f} m², "
                    f"AR {L['AR']:.1f}, LE sweep {p['le_sweep_deg']:.0f}°, tail volumes V_h {L['V_h']:.2f} V_v {L['V_v']:.3f}")


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


def render_arigo2(out, wing="hybrid"):
    import arigo2 as a2
    q = a2.variant(wing)
    p = a2.resolve(q)
    L = a2.Arigo2.layout(p)
    views = SIX[:5] + [(22, -130, "landing: motors stopped, blades folded", {"folded": True})]
    return gl_views(out, a2.Arigo2, q, views,
                    f"ARIGO-2 ({wing} wing) — two {p['twin_prop_diameter'] / 25.4:.0f}-inch folding tractors at y ±{p['nacelle_y']:.0f} mm, counter-rotating; "
                    f"span {p['span']:.0f} mm, wing {L['S_mm2'] / 1e6:.3f} m², nose camera unobstructed")


def render_arigo20(out):
    import arigo20 as a20
    q = a20.variant()
    p = a20.resolve(q)
    L = a20.Arigo20.layout(p)
    views = [SIX[0], (25, -40, "rear quarter, jets shown", {"show_jets": True}), SIX[2], SIX[3], SIX[4], (0, 0, "from behind", {})]
    return gl_views(out, a20.Arigo20, q, views,
                    f"ARIGO-20 — two {p['edf_diameter']:.0f} mm ducted fans over the rear fuselage, H tail; span {p['span']:.0f} mm, "
                    f"wing {L['S_mm2'] / 1e6:.3f} m², AR {L['AR']:.1f}, LE sweep {p['le_sweep_deg']:.0f}°, V_h {L['V_h']:.2f}, V_v {L['V_v']:.3f}")


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    render_views(d / "arigo1_hybrid_pusher_3d.png")
    render_arigo2(d / "arigo2_hybrid_3d.png")
    render_arigo20(d / "arigo20_3d.png")
    render_matrix(d / "arigo1_matrix_3d.png")
    export_all(d / "arigo1")
